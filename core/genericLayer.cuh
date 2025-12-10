#pragma once
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
#include "../core/Solvers/SolverCommon.cuh"
/**
 * Generic layer wrapper holding neuron states and inputs.
 * @tparam StateTypes Thrust tuple describing the per-neuron state variables.
 *
 * @ingroup network
 */
template<typename StateTypes> // e.g. thrust::tuple<float,float,float>
class genericLayer {
public:
    using state_tuple_type = StateTypes;
    static constexpr size_t N_VECTORS = thrust::tuple_size<StateTypes>::value;

private:
    size_t NNeurons_;
    thrust::device_vector<float> Input_;

    // tuple of device_vectors matching each state variable type
    /**
     * Helper to construct the tuple of state vectors on the device.
     *
     * For each element type in StateTypes, creates a corresponding
     * thrust::device_vector of size @p N initialized to zero.
     *
     * @tparam I Index pack over the elements of StateTypes.
     * @param N Number of elements in each state vector.
     * @return Tuple of device vectors matching StateTypes.
     */
    template<size_t... I>
    static auto make_state_tuple(size_t N, std::index_sequence<I...>) {
        return std::make_tuple(
            thrust::device_vector<typename thrust::tuple_element<I, StateTypes>::type>(N, 0)
            ...
        );
    }

    decltype(make_state_tuple(0, std::make_index_sequence<N_VECTORS>{})) state_;

public:
    /**
     * Constructs a layer with the given neuron count.
     * @param N Number of neurons/states to allocate.
     */
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
    /**
     * Returns the number of neurons in the layer.
     */
    size_t size() const noexcept { return NNeurons_; }

    // --- state vector access ---
    /**
     * Access a specific state vector by index.
     *
     * @tparam I Index of the state component in the StateTypes tuple.
     * @return Reference to the corresponding device vector.
     */
    template<size_t I>
    auto& state_vec() noexcept { return std::get<I>(state_); }

    /**
     * Access a specific state vector by index (const overload).
     *
     * @tparam I Index of the state component in the StateTypes tuple.
     * @return Const reference to the corresponding device vector.
     */
    template<size_t I>
    const auto& state_vec() const noexcept { return std::get<I>(state_); }

    // --- zip creators ---
    /**
     * Create a zip iterator over all state vectors (mutable).
     *
     * Each element of the iterator is a thrust::tuple of references to
     * the components of the neuron state.
     *
     * @return Iterator to the beginning of the zipped state range.
     */
    auto make_zip_begin() {
        return thrust::make_zip_iterator(make_begin_tuple(std::make_index_sequence<N_VECTORS>{}));
    }

    /**
     * Create a zip iterator over all state vectors (mutable) - end.
     *
     * @return Iterator to the end of the zipped state range.
     */
    auto make_zip_end() {
        return thrust::make_zip_iterator(make_end_tuple(std::make_index_sequence<N_VECTORS>{}));
    }

    /**
     * Create a zip iterator over all state vectors (const).
     *
     * @return Const iterator to the beginning of the zipped state range.
     */
    auto make_zip_begin() const {
        return thrust::make_zip_iterator(make_cbegin_tuple(std::make_index_sequence<N_VECTORS>{}));
    }

    /**
     * Create a zip iterator over all state vectors (const) - end.
     *
     * @return Const iterator to the end of the zipped state range.
     */
    auto make_zip_end() const {
        return thrust::make_zip_iterator(make_cend_tuple(std::make_index_sequence<N_VECTORS>{}));
    }

    // --- full zip (state + input) ---
    /**
     * Create a zip iterator over state and input (mutable).
     *
     * The iterator dereferences to a tuple of
     * (state tuple iterator, input value).
     *
     * @return Iterator to the beginning of the zipped state+input range.
     */
    auto full_zip_begin() {
        return thrust::make_zip_iterator(
            thrust::make_tuple(make_zip_begin(), Input_.begin()));
    }

    /**
     * Create a zip iterator over state and input (mutable) - end.
     *
     * @return Iterator to the end of the zipped state+input range.
     */
    auto full_zip_end() {
        return thrust::make_zip_iterator(
            thrust::make_tuple(make_zip_end(), Input_.end()));
    }

    /**
     * Advance the layer state by one integration step using RK4.
     *
     * Applies the provided RHS functor on all neurons in parallel using
     * Thrust on the given CUDA stream.
     *
     * @tparam RHS_Functor Functor type providing the right-hand side of the ODE.
     *                     It must be callable from device code.
     *                     @see Solvers
     * @param rhs Right-hand side functor.
     * @param t   Current time.
     * @param dt  Time step.
     * @param stream CUDA stream to execute the operation on (default 0).
     */
    template<typename RHS_Functor>
    void step(const RHS_Functor& rhs, float t, float dt, cudaStream_t stream = 0) {
        auto pol = thrust::cuda::par.on(stream);
        thrust::for_each(pol,
            this->full_zip_begin(),
            this->full_zip_end(),
            Solver_Step_Functor<RHS_Functor, StateTypes>(t, dt, rhs));
    }

    /**
     * Advance the layer state using per-neuron physical parameters.
     *
     * @tparam RHS_Functor The type of the functor stored in the collection.
     * @param rhs_collection Device vector of RHS functors (must be size NNeurons_).
     * @param t Current time.
     * @param dt Time step.
     * @param stream CUDA stream.
     */
    template<typename RHS_Functor>
    void step_heterogeneous(const thrust::device_vector<RHS_Functor>& rhs_collection,
        float t, float dt, cudaStream_t stream = 0) {

        if (rhs_collection.size() != NNeurons_) {
            // throw err
            return;
        }

        auto pol = thrust::cuda::par.on(stream);

        auto state_input_iter = this->full_zip_begin();
        auto grand_zip_begin = thrust::make_zip_iterator(
            thrust::make_tuple(state_input_iter, rhs_collection.begin())
        );

        auto grand_zip_end = thrust::make_zip_iterator(
            thrust::make_tuple(this->full_zip_end(), rhs_collection.end())
        );

        thrust::for_each(pol,
            grand_zip_begin,
            grand_zip_end,
            Heterogeneous_Solver_Step_Functor<StateTypes>(t, dt)
        );
    }

    /**
     * Access the input buffer for modification.
     * @return Reference to the device vector storing input currents.
     */
    thrust::device_vector<float>& input() noexcept { return Input_; }

    /**
     * Access the input buffer (read-only).
     * @return Const reference to the device vector storing input currents.
     */
    const thrust::device_vector<float>& input() const noexcept { return Input_; }

private:
    // --- tuple builders for iterator zips ---
    /**
     * Build a tuple of begin iterators for all state vectors.
     *
     * @tparam I Index pack over the elements of StateTypes.
     * @return Thrust tuple of begin iterators.
     */
    template<size_t... I>
    auto make_begin_tuple(std::index_sequence<I...>) {
        return thrust::make_tuple(std::get<I>(state_).begin()...);
    }

    /**
     * Build a tuple of end iterators for all state vectors.
     *
     * @tparam I Index pack over the elements of StateTypes.
     * @return Thrust tuple of end iterators.
     */
    template<size_t... I>
    auto make_end_tuple(std::index_sequence<I...>) {
        return thrust::make_tuple(std::get<I>(state_).end()...);
    }

    /**
     * Build a tuple of const begin iterators for all state vectors.
     *
     * @tparam I Index pack over the elements of StateTypes.
     * @return Thrust tuple of const begin iterators.
     */
    template<size_t... I>
    auto make_cbegin_tuple(std::index_sequence<I...>) const {
        return thrust::make_tuple(std::get<I>(state_).cbegin()...);
    }

    /**
     * Build a tuple of const end iterators for all state vectors.
     *
     * @tparam I Index pack over the elements of StateTypes.
     * @return Thrust tuple of const end iterators.
     */
    template<size_t... I>
    auto make_cend_tuple(std::index_sequence<I...>) const {
        return thrust::make_tuple(std::get<I>(state_).cend()...);
    }
};
