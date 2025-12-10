#pragma once
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include "solver_utils.cuh"

/**
 * Component-wise Heun update for a single k-step.
 * @tparam I Current tuple index (compile-time recursion).
 * @tparam State Tuple type storing state variables.
 * @param next_state Output state receiving the update.
 * @param current_state State at the beginning of the step.
 * @param k_step Derivative estimate for the current stage.
 * @param factor Scaling factor applied to k_step.
 *
 * @ingroup heun
 */
template <size_t I, typename State>
__host__ __device__ inline void apply_k_step_heun(State& next_state,
    const State& current_state,
    const State& k_step,
    float factor) {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    if constexpr (I < M) {
        thrust::get<I>(next_state) =
            thrust::get<I>(current_state) + thrust::get<I>(k_step) * factor;
        apply_k_step_heun<I + 1>(next_state, current_state, k_step, factor);
    }
}

/**
 * Component-wise update for the Heun method final step.
 * Computes: current_state = current_state + factor * (k1 + k2)
 *
 * @tparam I Current tuple index (compile-time recursion).
 * @tparam State Tuple type storing state variables.
 * @param current_state State being updated (y_n becomes y_{n+1}).
 * @param k1 Derivative estimate from the predictor (f(t_n, y_n)).
 * @param k2 Derivative estimate from the corrector (f(t_n+h, y_tilde)).
 * @param factor Scaling factor, typically dt / 2.0f.
 * 
 * @ingroup heun
 */
template <size_t I, typename State>
__host__ __device__ inline void apply_heun_final_step(State& current_state,
    const State& k1,
    const State& k2,
    float factor) {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    if constexpr (I < M) {
        thrust::get<I>(current_state) += factor *
            (thrust::get<I>(k1) + thrust::get<I>(k2));

        apply_heun_final_step<I + 1>(current_state, k1, k2, factor);
    }
}

/**
 * Functor executing a single Heun (Predictor-Corrector) integration step.
 * @tparam RHS Right-hand-side functor computing derivatives.
 * @tparam State Tuple type representing the neuron state.
 */
template<typename RHS, typename State>
struct Heun_Step_Functor {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    const float dt, t;
    RHS rhs;

    __host__ __device__
        Heun_Step_Functor(float t, float dt, RHS rhs) : t(t), dt(dt), rhs(rhs) {}

    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple t_zip) const {
        State state_n = thrust::get<0>(t_zip);
        float I = thrust::get<1>(t_zip);

        State k1{}, k2{}, predictor{};

        rhs(state_n, k1, I, t);
        apply_k_step_heun<0>(predictor, state_n, k1, dt);
        rhs(predictor, k2, I, t + dt);
        apply_heun_final_step<0>(state_n, k1, k2, dt / 2.0f);

        solver_utils::try_reset<RHS, State>(rhs, state_n);

        thrust::get<0>(t_zip) = state_n;
    }
};

/**
 * Functor executing a single Heun step where the RHS model is
 * provided per-thread via the iterator (Heterogeneous/Hetero-parameter).
 *
 * @tparam State Tuple type representing the neuron state.
 */
template<typename State>
struct Heterogeneous_Heun_Step_Functor {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    const float dt, t;

    __host__ __device__
        Heterogeneous_Heun_Step_Functor(float t, float dt) : t(t), dt(dt) {}

    template<typename Tuple>
    __host__ __device__
        void operator()(Tuple grand_zip) const {
        // 1. Unpack the structure: ((State, Input), RHS_Functor)
        auto state_input_pair = thrust::get<0>(grand_zip);
        auto specific_rhs = thrust::get<1>(grand_zip); // The per-neuron parameters

        State state_n = thrust::get<0>(state_input_pair);
        float I = thrust::get<1>(state_input_pair);

        State k1{}, k2{}, predictor{};

        specific_rhs(state_n, k1, I, t);
        apply_k_step_heun<0>(predictor, state_n, k1, dt);
        specific_rhs(predictor, k2, I, t + dt);
        apply_heun_final_step<0>(state_n, k1, k2, dt / 2.0f);

        solver_utils::try_reset(specific_rhs, state_n);

        thrust::get<0>(state_input_pair) = state_n;
    }
};