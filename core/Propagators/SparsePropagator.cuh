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
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include <cuda_runtime.h>

struct RowCounterFunctor {
    int* rowPtr;
    __host__ __device__
        RowCounterFunctor(int* _rowPtr) : rowPtr(_rowPtr) {}

    __device__ void operator()(int r) const {
        atomicAdd(&rowPtr[r + 1], 1);
    }
};

/**
 * @brief Defines how SpMV results are merged into the destination input buffer.
 */
enum class InputBehavior : uint8_t {
    INPUT_OVERRIDE = 0,
    INPUT_ADD = 1
};

// Forward decl of your layer template
template <typename StateTypes>
class genericLayer;

/**
 * @brief RAII wrapper for cuSPARSE SpMV propagation with prebuilt CSR matrices.
 */
class SparsePropagator {
private:
    struct CsrMatrix {
        cusparseSpMatDescr_t matA = nullptr;
        thrust::device_vector<int> rowPtr;
        thrust::device_vector<int> colInd;
        thrust::device_vector<float> values;
        void* dBuffer = nullptr;
        size_t bufferSize = 0;
        int nrows = 0, ncols = 0, nnz = 0;

        CsrMatrix() = default;
        CsrMatrix(const CsrMatrix&) = delete;
        CsrMatrix& operator=(const CsrMatrix&) = delete;

        CsrMatrix(CsrMatrix&& other) noexcept
            : matA(other.matA),
            rowPtr(std::move(other.rowPtr)),
            colInd(std::move(other.colInd)),
            values(std::move(other.values)),
            dBuffer(other.dBuffer),
            bufferSize(other.bufferSize),
            nrows(other.nrows),
            ncols(other.ncols),
            nnz(other.nnz)
        {
            other.matA = nullptr;
            other.dBuffer = nullptr;
        }

        CsrMatrix& operator=(CsrMatrix&& other) noexcept {
            if (this != &other) {
                if (matA) cusparseDestroySpMat(matA);
                if (dBuffer) cudaFree(dBuffer);

                matA = other.matA;
                rowPtr = std::move(other.rowPtr);
                colInd = std::move(other.colInd);
                values = std::move(other.values);
                dBuffer = other.dBuffer;
                bufferSize = other.bufferSize;
                nrows = other.nrows;
                ncols = other.ncols;
                nnz = other.nnz;

                other.matA = nullptr;
                other.dBuffer = nullptr;
            }
            return *this;
        }

        ~CsrMatrix() {
            if (matA) cusparseDestroySpMat(matA);
            if (dBuffer) cudaFree(dBuffer);
        }
    };

    cusparseHandle_t handle_ = nullptr;
    std::unordered_map<std::string, CsrMatrix> mats_;
    bool initialized_ = false;

public:
    SparsePropagator() = default;
    ~SparsePropagator() { destroy(); }

    /**
     * @brief Initializes the cuSPARSE handle if it has not been created.
     */
    void init() {
        if (initialized_) return;
        cusparseCreate(&handle_);
        initialized_ = true;
    }

    /**
     * @brief Releases allocated matrices and destroys the cuSPARSE handle.
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
     * @brief Build CSR from (Xn, X, W) triplets.
     */
    /**
     * @brief Build CSR from (Xn, X, W) triplets.
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
        if (X.size() != Xn.size() || X.size() != W.size())
            throw std::length_error("X, Xn, W must have the same size.");

        const size_t nnz = W.size();
        CsrMatrix csr;
        csr.nrows = num_post;
        csr.ncols = num_pre;
        csr.nnz = static_cast<int>(nnz);

        auto pol = thrust::cuda::par.on(stream); 
        // --- sort by post (row) index ---
        thrust::device_vector<int> perm(nnz);
        thrust::sequence(pol, perm.begin(), perm.end());

        thrust::device_vector<int> Xn_tmp = Xn;
        thrust::sort_by_key(pol, Xn_tmp.begin(), Xn_tmp.end(), perm.begin());

        thrust::device_vector<int> Xn_sorted(nnz), X_sorted(nnz);
        thrust::device_vector<float> W_sorted(nnz);
        thrust::gather(pol, perm.begin(), perm.end(), Xn_tmp.begin(), Xn_sorted.begin());
        thrust::gather(pol, perm.begin(), perm.end(), X.begin(), X_sorted.begin());
        thrust::gather(pol, perm.begin(), perm.end(), W.begin(), W_sorted.begin());

        // --- compute rowPtr ---
        csr.rowPtr.resize(num_post + 1);
        thrust::fill(pol, csr.rowPtr.begin(), csr.rowPtr.end(), 0);

        thrust::for_each(
            pol,
            Xn_sorted.begin(), Xn_sorted.end(),
            RowCounterFunctor(thrust::raw_pointer_cast(csr.rowPtr.data())));

        thrust::inclusive_scan(pol, csr.rowPtr.begin(), csr.rowPtr.end(), csr.rowPtr.begin());

        // store columns and values
        csr.colInd = std::move(X_sorted);
        csr.values = std::move(W_sorted);

        // create matrix descriptor
        cusparseStatus_t stat = cusparseCreateCsr(&csr.matA,
            csr.nrows, csr.ncols, csr.nnz,
            (void*)thrust::raw_pointer_cast(csr.rowPtr.data()),
            (void*)thrust::raw_pointer_cast(csr.colInd.data()),
            (void*)thrust::raw_pointer_cast(csr.values.data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        if (stat != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("cusparseCreateCsr failed: " + std::to_string(stat));

        mats_.emplace(name, std::move(csr));
#ifdef _VERBOSE_DEBUG
        std::cout << "building " << name << " CsrMatrix at " << &csr
            << " handle " << csr.matA << "\n";
#endif
    }

    /**
     * @brief Propagate preLayer state -> postLayer input using a named CSR.
     */
    /**
     * @brief Propagate preLayer state -> postLayer input using a named CSR.
     * @tparam stateVar Index of the presynaptic state variable to propagate.
     * @tparam PreStateTypes Tuple type for the presynaptic layer.
     * @tparam PostStateTypes Tuple type for the postsynaptic layer.
     * @param preLayer Source layer.
     * @param postLayer Destination layer.
     * @param name Key of the CSR matrix to use.
     * @param behavior Input accumulation behavior.
     * @param stream CUDA stream for the SpMV call.
     */
    template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void propagate(genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const std::string& name,
        InputBehavior behavior = InputBehavior::INPUT_OVERRIDE,
        cudaStream_t stream = 0)
    {
        if (!initialized_) throw std::runtime_error("SparsePropagator not initialized.");

        if (mats_.count(name) == 0)
            throw std::runtime_error("CSR '" + name + "' not found.");

        auto& csr = mats_.at(name);

#ifdef _VERBOSE_DEBUG
        std::cout << "using " << name << " CsrMatrix at " << &csr
          << " handle " << csr.matA << "\n";
#endif

        auto* prePtr = thrust::raw_pointer_cast(preLayer.template state_vec<stateVar>().data());
        auto* postPtr = thrust::raw_pointer_cast(postLayer.input().data());

        cusparseSetStream(handle_, stream);

        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateDnVec(&vecX, preLayer.size(), prePtr, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, postLayer.size(), postPtr, CUDA_R_32F);

        float alpha = 1.0f;
        float beta = (behavior == InputBehavior::INPUT_ADD) ? 1.0f : 0.0f;

        // Allocate buffer lazily
        if (!csr.dBuffer) {
            cusparseSpMV_bufferSize(
                handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, csr.matA, vecX, &beta, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &csr.bufferSize);
            cudaMalloc(&csr.dBuffer, csr.bufferSize);
        }

        cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, csr.matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, csr.dBuffer);

        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    }
};
