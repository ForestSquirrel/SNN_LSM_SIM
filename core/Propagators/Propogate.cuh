#pragma once
#include "../core/genericLayer.cuh"
#include "../core/networkBuilder.cuh"
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace PropagateStatic {

    /**
     * Determines how computed inputs are combined with existing buffers.
     * @ingroup deprecated
     */
    enum class InputBehavior : uint8_t {
        INPUT_OVERRIDE = 0,   // replace post.input() with computed values
        INPUT_ADD = 1         // add computed values to post.input()
    };

    constexpr InputBehavior INPUT_OVERRIDE = InputBehavior::INPUT_OVERRIDE;
    constexpr InputBehavior INPUT_ADD = InputBehavior::INPUT_ADD;

    // ==========================================================
    // ================= SPARSE PROPAGATION ======================
    // ==========================================================
    /**
     * Sparse propagation using explicit edge lists.
     * @tparam stateVar Index of the presynaptic state variable to propagate.
     * @tparam PreStateTypes Tuple type for the presynaptic layer.
     * @tparam PostStateTypes Tuple type for the postsynaptic layer.
     * @param preLayer Source layer.
     * @param postLayer Destination layer.
     * @param X Presynaptic indices.
     * @param Xn Postsynaptic indices.
     * @param W Connection weights.
     * @param behavior Input accumulation behavior.
     * @param stream CUDA stream for execution.
     * 
     * @ingroup deprecated
     */
    template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void sparse_p(
        genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const thrust::device_vector<int>& X,    // presynaptic indices
        const thrust::device_vector<int>& Xn,   // postsynaptic indices
        const thrust::device_vector<float>& W,  // weights
        InputBehavior behavior = INPUT_OVERRIDE,
        cudaStream_t stream = 0
    ) {
        if (X.size() != Xn.size() || X.size() != W.size())
            throw std::length_error("X, Xn, W must have the same size.");

        const size_t Nconn = X.size();
        auto pol = thrust::cuda::par.on(stream);

        thrust::device_vector<int>   Xn_sorted = Xn;     // copy to sort
        thrust::device_vector<float> values(Nconn);

        // 1. Gather presynaptic values and apply weights
        thrust::gather(
            pol,
            X.begin(), X.end(),
            preLayer.template state_vec<stateVar>().cbegin(),
            values.begin()
        );
        thrust::transform(
            pol,
            values.begin(), values.end(),
            W.begin(),
            values.begin(),
            thrust::multiplies<float>()
        );

        // 2. Sort by postsynaptic index
        thrust::sort_by_key(
            pol,
            Xn_sorted.begin(), Xn_sorted.end(),
            values.begin()
        );

        // 3. Reduce-by-key: sum contributions for each post index
        thrust::device_vector<int>   unique_post(Xn_sorted.size());
        thrust::device_vector<float> reduced_vals(Xn_sorted.size());
        auto end_pair = thrust::reduce_by_key(
            pol,
            Xn_sorted.begin(), Xn_sorted.end(),
            values.begin(),
            unique_post.begin(),
            reduced_vals.begin(),
            thrust::equal_to<int>(),
            thrust::plus<float>()
        );
        const size_t Nunique = end_pair.first - unique_post.begin();

        // 4. Scatter reduced sums to a temp buffer
        thrust::device_vector<float> temp_input(postLayer.size(), 0.0f);
        thrust::scatter(
            pol,
            reduced_vals.begin(), reduced_vals.begin() + Nunique,
            unique_post.begin(),
            temp_input.begin()
        );

        // 5. Combine with post layer input
        if (behavior == InputBehavior::INPUT_ADD) {
            thrust::transform(
                pol,
                postLayer.input().begin(), postLayer.input().end(),
                temp_input.begin(),
                postLayer.input().begin(),
                thrust::plus<float>()
            );
        }
        else {
            thrust::copy(
                pol,
                temp_input.begin(), temp_input.end(),
                postLayer.input().begin()
            );
        }
    }

    // ==========================================================
    // ================= FORWARD PROPAGATION =====================
    // ==========================================================
    /**
     * One-to-one propagation (diagonal weight vector).
     * @tparam stateVar Index of the presynaptic state variable to propagate.
     * @tparam PreStateTypes Tuple type for the presynaptic layer.
     * @tparam PostStateTypes Tuple type for the postsynaptic layer.
     * @param preLayer Source layer.
     * @param postLayer Destination layer.
     * @param W Weight vector aligned with neuron indices.
     * @param behavior Input accumulation behavior.
     * @param stream CUDA stream for execution.
     * 
     * @ingroup deprecated
     */
    template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void forward_p(
        genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const thrust::device_vector<float>& W,
        InputBehavior behavior = INPUT_OVERRIDE,
        cudaStream_t stream = 0
    ) {
        const size_t N = preLayer.size();
        if (postLayer.size() != N || W.size() != N)
            throw std::length_error("Layer or weight size mismatch in forward_p().");
        
        auto pol = thrust::cuda::par.on(stream);
        thrust::device_vector<float> temp_input(N, 0.0f);

        // Weighted copy: input_i = pre[i] * W[i]
        thrust::transform(
            pol,
            preLayer.template state_vec<stateVar>().cbegin(),
            preLayer.template state_vec<stateVar>().cend(),
            W.begin(),
            temp_input.begin(),
            thrust::multiplies<float>()
        );

        if (behavior == INPUT_ADD) {
            thrust::transform(
                pol,
                postLayer.input().begin(), postLayer.input().end(),
                temp_input.begin(),
                postLayer.input().begin(),
                thrust::plus<float>()
            );
        }
        else {
            thrust::copy(pol,
                temp_input.begin(), temp_input.end(),
                postLayer.input().begin());
        }
    }


    // ==========================================================
    // ================= DENSE PROPAGATION =======================
    // ==========================================================
    /**
     * Dense matrix propagation using cuBLAS sgemv.
     * @tparam stateVar Index of the presynaptic state variable to propagate.
     * @tparam PreStateTypes Tuple type for the presynaptic layer.
     * @tparam PostStateTypes Tuple type for the postsynaptic layer.
     * @param preLayer Source layer.
     * @param postLayer Destination layer.
     * @param W_flat Column-major weight matrix (N_post x N_pre).
     * @param handle cuBLAS handle.
     * @param behavior Input accumulation behavior.
     * @param stream CUDA stream for execution.
     * 
     * @ingroup deprecated
     */
    template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void dense_p(
        genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const thrust::device_vector<float>& W_flat, // column-major (N_post × N_pre)
        cublasHandle_t handle,
        InputBehavior behavior = INPUT_OVERRIDE,
        cudaStream_t stream = 0
    ) {
        const size_t N_pre = preLayer.size();
        const size_t N_post = postLayer.size();
        if (W_flat.size() != N_post * N_pre)
            throw std::length_error("W_flat size != N_post * N_pre in dense_p().");

        const float alpha = 1.0f;
        const float beta = 0.0f; // always write to temp buffer first

        thrust::device_vector<float> temp_input(N_post, 0.0f);

        const float* d_W = thrust::raw_pointer_cast(W_flat.data());
        const float* d_x = thrust::raw_pointer_cast(preLayer.template state_vec<stateVar>().data());
        float* d_y = thrust::raw_pointer_cast(temp_input.data());

        cublasSetStream(handle, stream);
        cublasStatus_t stat = cublasSgemv(
            handle,
            CUBLAS_OP_N,               // y = W * x
            static_cast<int>(N_post),  // rows
            static_cast<int>(N_pre),   // cols
            &alpha,
            d_W,
            static_cast<int>(N_post),  // leading dimension
            d_x,
            1,
            &beta,
            d_y,
            1
        );
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS sgemv failed in dense_p().");

        auto pol = thrust::cuda::par.on(stream);
        if (behavior == INPUT_ADD) {
            thrust::transform(
                pol,
                postLayer.input().begin(), postLayer.input().end(),
                temp_input.begin(),
                postLayer.input().begin(),
                thrust::plus<float>()
            );
        }
        else {
            thrust::copy(pol,
                temp_input.begin(), temp_input.end(),
                postLayer.input().begin());
        }
    }

    // ==========================================================
    // ================= SPARSE PROPAGATION CUSPARSE=======================
    // ==========================================================
    /*template<size_t stateVar, typename PreStateTypes, typename PostStateTypes>
    void sparse_p(
        genericLayer<PreStateTypes>& preLayer,
        genericLayer<PostStateTypes>& postLayer,
        const networkBuilder& net,
        cusparseHandle_t handle,
        InputBehavior behavior = INPUT_OVERRIDE,
        cudaStream_t stream = 0)
    {
        cusparseSetStream(handle, stream);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* d_x = thrust::raw_pointer_cast(preLayer.template state_vec<stateVar>().data());
        thrust::device_vector<float> temp_input(net.N_post, 0.0f);
        float* d_y = thrust::raw_pointer_cast(temp_input.data());

        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateDnVec(&vecX, net.N_pre, (void*)d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, net.N_post, (void*)d_y, CUDA_R_32F);

        size_t bufferSize = 0;
        void* dBuffer = nullptr;
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, net.matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        cudaMallocAsync(&dBuffer, bufferSize, stream);

        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, net.matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

        cudaFreeAsync(dBuffer, stream);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);

        auto pol = thrust::cuda::par.on(stream);
        if (behavior == InputBehavior::INPUT_ADD) {
            thrust::transform(
                pol,
                postLayer.input().begin(), postLayer.input().end(),
                temp_input.begin(),
                postLayer.input().begin(),
                thrust::plus<float>()
            );
        }
        else {
            thrust::copy(
                pol,
                temp_input.begin(), temp_input.end(),
                postLayer.input().begin()
            );
        }
    }
    */
}
