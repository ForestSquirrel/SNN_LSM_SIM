#pragma once
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include <cuda_runtime.h>

/**
 * Defines how SpMV results are merged into the destination input buffer.
 * 
 * @ingroup sparse_propagators
 */
enum class InputBehavior : uint8_t {
    INPUT_OVERRIDE = 0,
    INPUT_ADD = 1
};

/**
 * Defines the physical equation used for propagation.
 * @ingroup sparse_propagators
 */
enum class PropagationMode : uint8_t {
    CHEMICAL_LINEAR_FORWARD         = 0,                        // I_post = V_pre * W
    INPUT                           = CHEMICAL_LINEAR_FORWARD,  // Dupe with another alias for user-friendliness
    CHEMICAL_LINEAR_BIDIRECTIONAL   = 1,                        // I_post = V_pre * W;  I_pre = V_post * W
    ELECTRICAL_LINEAR_FORWARD       = 2,                        // I_post = (V_pre - V_post) * W
    ELECTRICAL_LINEAR_BIDIRECTIONAL = 3                         // I_post = (V_pre - V_post) * W; I_pre = (V_post - V_pre) * W
};

// Forward decl of layer template
template <typename StateTypes>
class genericLayer;

// Functor for Electrical subtraction: I -= V_local * WeightSum
struct ElectricalLeakFunctor {
    __host__ __device__
        float operator()(const float& current_input, const thrust::tuple<float, float>& v_and_sum) const {
        // tuple: 0 = voltage (V_post), 1 = weight sum (RowSum)
        return current_input - (thrust::get<0>(v_and_sum) * thrust::get<1>(v_and_sum));
    }
};

/**
 * RAII wrapper for cuSPARSE SpMV propagation with prebuilt CSR matrices.
 * 
 * @ingroup sparse_propagators
 */
class SparsePropagator {
private:
    struct CsrMatrix {
        cusparseSpMatDescr_t matA = nullptr;

        // Descriptors for standard propagation
        cusparseDnVecDescr_t vecX = nullptr;
        cusparseDnVecDescr_t vecY = nullptr;

        // Descriptors for reverse propagation (Bidirectional)
        cusparseDnVecDescr_t vecX_rev = nullptr; // Wraps Post Layer State
        cusparseDnVecDescr_t vecY_rev = nullptr; // Wraps Pre Layer Input

        thrust::device_vector<int> rowPtr;
        thrust::device_vector<int> colInd;
        thrust::device_vector<float> values;

        // Pre-calculated weight sums for Electrical Equations
        thrust::device_vector<float> rowSums; // For Forward Electrical
        thrust::device_vector<float> colSums; // For Reverse Electrical

        void* dBuffer = nullptr;
        size_t bufferSize = 0;
        int nrows = 0, ncols = 0, nnz = 0;

        CsrMatrix()                             = default;
        CsrMatrix(const CsrMatrix&)             = delete;
        CsrMatrix& operator=(const CsrMatrix&)  = delete;

        CsrMatrix(CsrMatrix&& other) noexcept
            : matA(other.matA),
            vecX(other.vecX),
            vecY(other.vecY),
            vecX_rev(other.vecX_rev),
            vecY_rev(other.vecY_rev),
            rowPtr(std::move(other.rowPtr)),
            colInd(std::move(other.colInd)),
            values(std::move(other.values)),
            rowSums(std::move(other.rowSums)),
            colSums(std::move(other.colSums)),
            dBuffer(other.dBuffer),
            bufferSize( other.bufferSize),
            nrows(other.nrows),
            ncols(other.ncols),
            nnz(other.nnz)
        {
            other.matA      = nullptr;
            other.vecX      = nullptr;
            other.vecY      = nullptr;
            other.vecX_rev  = nullptr;
            other.vecY_rev  = nullptr;
            other.dBuffer   = nullptr;
        }

        CsrMatrix& operator=(CsrMatrix&& other) noexcept {
            if (this != &other) {
                if (matA)       cusparseDestroySpMat(matA);
                if (vecX)       cusparseDestroyDnVec(vecX);
                if (vecY)       cusparseDestroyDnVec(vecY);
                if (vecX_rev)   cusparseDestroyDnVec(vecX_rev);
                if (vecY_rev)   cusparseDestroyDnVec(vecY_rev);
                if (dBuffer)    cudaFree(dBuffer);

                matA        = other.matA;
                vecX        = other.vecX;
                vecY        = other.vecY;
                rowPtr      = std::move(other.rowPtr);
                colInd      = std::move(other.colInd);
                values      = std::move(other.values);
                rowSums     = std::move(other.rowSums),
                colSums     = std::move(other.colSums),
                dBuffer     = other.dBuffer;
                bufferSize  = other.bufferSize;
                nrows       = other.nrows;
                ncols       = other.ncols;
                nnz         = other.nnz;

                other.matA      = nullptr;
                other.vecX      = nullptr;
                other.vecY      = nullptr;
                other.vecX_rev  = nullptr;
                other.vecY_rev  = nullptr;
                other.dBuffer   = nullptr;
            }
            return *this;
        }

        ~CsrMatrix() {
            if (matA)       cusparseDestroySpMat(matA);
            if (vecX)       cusparseDestroyDnVec(vecX);
            if (vecY)       cusparseDestroyDnVec(vecY);
            if (vecX_rev)   cusparseDestroyDnVec(vecX_rev);
            if (vecY_rev)   cusparseDestroyDnVec(vecY_rev);
            if (dBuffer)    cudaFree(dBuffer);
        }
    };

    cusparseHandle_t handle_ = nullptr;
    std::unordered_map<std::string, CsrMatrix> mats_;
    bool initialized_ = false;

    // Helper to compute row/col sums for electrical equations
    void computeWeightSums(CsrMatrix& csr, int num_pre, int num_post) {
        // 1. Compute Row Sums (A * 1_vec)
        // We temporarily create dense vectors of 1s to do SpMV
        csr.rowSums.resize(num_post);
        csr.colSums.resize(num_pre);

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Temp vectors for ones
        thrust::device_vector<float> ones_pre(num_pre, 1.0f);
        thrust::device_vector<float> ones_post(num_post, 1.0f);

        cusparseDnVecDescr_t vecOnesPre, vecOnesPost, vecResRow, vecResCol;

        cusparseCreateDnVec(&vecOnesPre,  num_pre,  thrust::raw_pointer_cast(ones_pre.data()),    CUDA_R_32F);
        cusparseCreateDnVec(&vecOnesPost, num_post, thrust::raw_pointer_cast(ones_post.data()),   CUDA_R_32F);
        cusparseCreateDnVec(&vecResRow,   num_post, thrust::raw_pointer_cast(csr.rowSums.data()), CUDA_R_32F);
        cusparseCreateDnVec(&vecResCol,   num_pre,  thrust::raw_pointer_cast(csr.colSums.data()), CUDA_R_32F);

        size_t buffSize = 0;
        void* tempBuff = nullptr;

        // Row Sums: Result = A * Ones_Pre
        cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csr.matA, vecOnesPre, &beta, vecResRow, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffSize);
        cudaMalloc(&tempBuff, buffSize);
        cusparseSpMV(           handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csr.matA, vecOnesPre, &beta, vecResRow, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, tempBuff);
        cudaFree(tempBuff);

        // Col Sums: Result = A^T * Ones_Post
        cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE,     &alpha, csr.matA, vecOnesPost, &beta, vecResCol, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffSize);
        cudaMalloc(&tempBuff, buffSize);
        cusparseSpMV(           handle_, CUSPARSE_OPERATION_TRANSPOSE,     &alpha, csr.matA, vecOnesPost, &beta, vecResCol, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, tempBuff);
        cudaFree(tempBuff);

        cusparseDestroyDnVec(vecOnesPre);
        cusparseDestroyDnVec(vecOnesPost);
        cusparseDestroyDnVec(vecResRow);
        cusparseDestroyDnVec(vecResCol);
    }

public:
    SparsePropagator() = default;
    ~SparsePropagator() { destroy(); }

    /**
     * Initializes the cuSPARSE handle if it has not been created.
     */
    void init() {
        if (initialized_) return;
        cusparseCreate(&handle_);
        initialized_ = true;
    }

    /**
     * Releases allocated matrices and destroys the cuSPARSE handle.
     */
    void destroy() {
        if (!initialized_) return;
        for (auto& kv : mats_)
            ; // CsrMatrix destructor handles its own cleanup
        mats_.clear();
        cusparseDestroy(handle_);
        initialized_ = false;
    }

    /**
     * Build CSR from (Xn, X, W) triplets.
     */
    /**
     * Build CSR from (Xn, X, W) triplets.
     * @param X Presynaptic indices.
     * @param Xn Postsynaptic indices.
     * @param W Connection weights.
     * @param name Key used to reference the built matrix.
     * @param num_pre Number of presynaptic neurons.
     * @param num_post Number of postsynaptic neurons.
     * @param stream CUDA stream for preprocessing.
     */
    void buildCSR(const thrust::device_vector<int>& X,
        const thrust::device_vector<int>& Xn,
        const thrust::device_vector<float>& W,
        const std::string& name,
        int num_pre, int num_post,
        cudaStream_t stream = 0)
    {
        if (!initialized_) throw std::runtime_error("SparsePropagator not initialized.");

        CsrMatrix csr;
        csr.nrows = num_post;
        csr.ncols = num_pre;
        csr.nnz = static_cast<int>(W.size());

        auto pol = thrust::cuda::par.on(stream);

        // Sort by Post (Row)
        thrust::device_vector<int> sortedX = X;
        thrust::device_vector<int> sortedXn = Xn;
        thrust::device_vector<float> sortedW = W;

        auto value_iter = thrust::make_zip_iterator(thrust::make_tuple(sortedX.begin(), sortedW.begin()));
        thrust::sort_by_key(pol, sortedXn.begin(), sortedXn.end(), value_iter);

        csr.rowPtr.resize(num_post + 1);
        csr.rowPtr[0] = 0;

        thrust::upper_bound(pol,
            sortedXn.begin(), sortedXn.end(),
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_post),
            csr.rowPtr.begin() + 1
        );

        csr.colInd = std::move(sortedX);
        csr.values = std::move(sortedW);

        // Create Descriptor
        cusparseCreateCsr(&csr.matA, csr.nrows, csr.ncols, csr.nnz,
            (void*)thrust::raw_pointer_cast(csr.rowPtr.data()),
            (void*)thrust::raw_pointer_cast(csr.colInd.data()),
            (void*)thrust::raw_pointer_cast(csr.values.data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // Compute helper sums for Electrical math
        computeWeightSums(csr, num_pre, num_post);

        mats_.emplace(name, std::move(csr));
    }


    template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void propagate(genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const std::string& name,
        PropagationMode mode = PropagationMode::CHEMICAL_LINEAR_FORWARD,
        InputBehavior behavior = InputBehavior::INPUT_OVERRIDE,
        cudaStream_t stream = 0)
    {
        if (mats_.count(name) == 0) throw std::runtime_error("CSR not found");
        auto& csr = mats_.at(name);
        cusparseSetStream(handle_, stream);

        // --- PRE -> POST (Forward) ---
        // Used in ALL modes
        {
            auto* preStatePtr  = thrust::raw_pointer_cast(preLayer.template state_vec<stateVar>().data());
            auto* postInputPtr = thrust::raw_pointer_cast(postLayer.input().data());

            // Prepare Dense Vecs
            if (!csr.vecX) cusparseCreateDnVec(&csr.vecX, preLayer.size(), preStatePtr, CUDA_R_32F);
            else           cusparseDnVecSetValues(csr.vecX, preStatePtr);

            if (!csr.vecY) cusparseCreateDnVec(&csr.vecY, postLayer.size(), postInputPtr, CUDA_R_32F);
            else           cusparseDnVecSetValues(csr.vecY, postInputPtr);

            // Buffer logic
            float alpha = 1.0f;
            float beta  = (behavior == InputBehavior::INPUT_ADD) ? 1.0f : 0.0f;

            if (!csr.dBuffer) {
                cusparseSpMV_bufferSize(handle_, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, csr.matA, csr.vecX, &beta, csr.vecY,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &csr.bufferSize);
                cudaMalloc(&csr.dBuffer, csr.bufferSize);
            }

            // A * V_pre
            cusparseSpMV(handle_, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, csr.matA, csr.vecX, &beta, csr.vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, csr.dBuffer);

            // If Electrical: Subtract V_post * RowSum
            if (mode == PropagationMode::ELECTRICAL_LINEAR_FORWARD || mode == PropagationMode::ELECTRICAL_LINEAR_BIDIRECTIONAL) {
                auto zip_v_sum = thrust::make_zip_iterator(thrust::make_tuple(
                    postLayer.template state_vec<stateVar>().begin(),
                    csr.rowSums.begin()
                ));

                // Transform: Input[i] = Input[i] - (V_post[i] * RowSum[i])
                thrust::transform(thrust::cuda::par.on(stream),
                    postLayer.input().begin(), postLayer.input().end(), // Input source (already has A*V_pre)
                    zip_v_sum,                                          // Zip (V_post, RowSum)
                    postLayer.input().begin(),                          // Output dest
                    ElectricalLeakFunctor()
                );
            }
        }

        // --- POST -> PRE (Backward / Bidirectional) ---
        if (mode == PropagationMode::CHEMICAL_LINEAR_BIDIRECTIONAL || mode == PropagationMode::ELECTRICAL_LINEAR_BIDIRECTIONAL)
        {
            auto* postStatePtr = thrust::raw_pointer_cast(postLayer.template state_vec<stateVar>().data());
            auto* preInputPtr  = thrust::raw_pointer_cast(preLayer.input().data()); // NOTE: PreLayer must be writable

            if (!csr.vecX_rev) cusparseCreateDnVec(&csr.vecX_rev, postLayer.size(), postStatePtr, CUDA_R_32F);
            else               cusparseDnVecSetValues(csr.vecX_rev, postStatePtr);

            if (!csr.vecY_rev) cusparseCreateDnVec(&csr.vecY_rev, preLayer.size(), preInputPtr, CUDA_R_32F);
            else               cusparseDnVecSetValues(csr.vecY_rev, preInputPtr);

            // NOTE: Bidirectional always ADDS to the pre-synaptic input to allow multiple connections
            // Respect `behavior` for consistency. If OVERRIDE, we assume this is the first reverse op.
            float alpha = 1.0f;
            float beta = (behavior == InputBehavior::INPUT_ADD) ? 1.0f : 0.0f;

            // A^T * V_post
            cusparseSpMV(handle_, 
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha, csr.matA, csr.vecX_rev, &beta, csr.vecY_rev,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, csr.dBuffer);

            // If Electrical: Subtract V_pre * ColSum
            if (mode == PropagationMode::ELECTRICAL_LINEAR_BIDIRECTIONAL) {
                auto zip_v_sum = thrust::make_zip_iterator(thrust::make_tuple(
                    preLayer.template state_vec<stateVar>().begin(),
                    csr.colSums.begin()
                ));

                thrust::transform(thrust::cuda::par.on(stream),
                    preLayer.input().begin(), preLayer.input().end(),
                    zip_v_sum,
                    preLayer.input().begin(),
                    ElectricalLeakFunctor()
                );
            }
        }
    }
};
