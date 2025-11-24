#pragma once
#include "../core/Solver.cuh"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <tuple>
#include <vector>
#include <type_traits>
#include <numeric>
#include <stdexcept>
#include <cuda_runtime.h>

template<typename StateTypes> // e.g. thrust::tuple<float,float,float>
class genericLayer {
public:
    using state_tuple_type = StateTypes;
    static constexpr size_t N_VECTORS = thrust::tuple_size<StateTypes>::value;

private:
    size_t NNeurons_;
    thrust::device_vector<float> Input_;

    // tuple of device_vectors matching each state variable type
    template<size_t... I>
    static auto make_state_tuple(size_t N, std::index_sequence<I...>) {
        return std::make_tuple(
            thrust::device_vector<typename thrust::tuple_element<I, StateTypes>::type>(N, 0)
            ...
        );
    }

    decltype(make_state_tuple(0, std::make_index_sequence<N_VECTORS>{})) state_;

public:
    explicit genericLayer(size_t N)
        : NNeurons_(N)
        , Input_(N, 0.0f)
        , state_(make_state_tuple(N, std::make_index_sequence<N_VECTORS>{}))
    {}

    genericLayer(const genericLayer&) = delete;
    genericLayer& operator=(const genericLayer&) = delete;
    genericLayer(genericLayer&&) noexcept = default;
    genericLayer& operator=(genericLayer&&) noexcept = default;
    ~genericLayer() = default;

    // --- basic info ---
    size_t size() const noexcept { return NNeurons_; }

    // --- state vector access ---
    template<size_t I>
    auto& state_vec() noexcept { return std::get<I>(state_); }

    template<size_t I>
    const auto& state_vec() const noexcept { return std::get<I>(state_); }

    // --- zip creators ---
    auto make_zip_begin() {
        return thrust::make_zip_iterator(make_begin_tuple(std::make_index_sequence<N_VECTORS>{}));
    }
    auto make_zip_end() {
        return thrust::make_zip_iterator(make_end_tuple(std::make_index_sequence<N_VECTORS>{}));
    }
    auto make_zip_begin() const {
        return thrust::make_zip_iterator(make_cbegin_tuple(std::make_index_sequence<N_VECTORS>{}));
    }
    auto make_zip_end() const {
        return thrust::make_zip_iterator(make_cend_tuple(std::make_index_sequence<N_VECTORS>{}));
    }

    // --- full zip (state + input) ---
    auto full_zip_begin() {
        return thrust::make_zip_iterator(
            thrust::make_tuple(make_zip_begin(), Input_.begin()));
    }
    auto full_zip_end() {
        return thrust::make_zip_iterator(
            thrust::make_tuple(make_zip_end(), Input_.end()));
    }

    template<typename RHS_Functor>
    void step(const RHS_Functor& rhs, float t, float dt, cudaStream_t stream = 0) {
        auto pol = thrust::cuda::par.on(stream);
        thrust::for_each(pol,
            this->full_zip_begin(),
            this->full_zip_end(),
            RK4_Step_Functor<RHS_Functor, StateTypes>(t, dt, rhs));
    }

    thrust::device_vector<float>& input() noexcept { return Input_; }
    const thrust::device_vector<float>& input() const noexcept { return Input_; }

private:
    // --- tuple builders for iterator zips ---
    template<size_t... I>
    auto make_begin_tuple(std::index_sequence<I...>) {
        return thrust::make_tuple(std::get<I>(state_).begin()...);
    }

    template<size_t... I>
    auto make_end_tuple(std::index_sequence<I...>) {
        return thrust::make_tuple(std::get<I>(state_).end()...);
    }

    template<size_t... I>
    auto make_cbegin_tuple(std::index_sequence<I...>) const {
        return thrust::make_tuple(std::get<I>(state_).cbegin()...);
    }

    template<size_t... I>
    auto make_cend_tuple(std::index_sequence<I...>) const {
        return thrust::make_tuple(std::get<I>(state_).cend()...);
    }
};