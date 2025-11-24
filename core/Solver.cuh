#pragma once
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

/**
 * @brief Component-wise RK4 update for a single k-step.
 * @tparam I Current tuple index (compile-time recursion).
 * @tparam State Tuple type storing state variables.
 * @param next_state Output state receiving the update.
 * @param current_state State at the beginning of the step.
 * @param k_step Derivative estimate for the current stage.
 * @param factor Scaling factor applied to k_step.
 */
template <size_t I, typename State>
__host__ __device__ inline void apply_k_step(State& next_state,
                                             const State& current_state,
                                             const State& k_step,
                                             float factor) {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    if constexpr (I < M) {
        thrust::get<I>(next_state) =
            thrust::get<I>(current_state) + thrust::get<I>(k_step) * factor;
        apply_k_step<I + 1>(next_state, current_state, k_step, factor);
    }
}

template <size_t I, typename State>
__host__ __device__ inline void apply_last_step(State& current_state,
                                                const State& k1, const State& k2,
                                                const State& k3, const State& k4,
                                                float factor) {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    if constexpr (I < M) {
        thrust::get<I>(current_state) += factor *
            (thrust::get<I>(k1) + 2.0f * thrust::get<I>(k2) +
             2.0f * thrust::get<I>(k3) + thrust::get<I>(k4));
        apply_last_step<I + 1>(current_state, k1, k2, k3, k4, factor);
    }
}

/**
 * @brief Functor executing a single Runge–Kutta 4 integration step.
 * @tparam RHS Right-hand-side functor computing derivatives.
 * @tparam State Tuple type representing the neuron state.
 */
template<typename RHS, typename State>
struct RK4_Step_Functor {
    static constexpr size_t M = thrust::tuple_size<State>::value;
    const float dt, t;
    RHS rhs;

    /**
     * @brief Constructs the functor with simulation parameters.
     * @param t Current simulation time.
     * @param dt Time step size.
     * @param rhs Right-hand-side functor instance.
     */
    RK4_Step_Functor(float t, float dt, RHS rhs) : t(t), dt(dt), rhs(rhs) {}

    template<typename Tuple>
    __host__ __device__
    /**
     * @brief Applies RK4 integration to a zipped state/input tuple.
     * @param t_zip Tuple containing the current state and input value.
     */
    void operator()(Tuple t_zip) const {
        State state_n = thrust::get<0>(t_zip);
        float I = thrust::get<1>(t_zip);

        State k1{}, k2{}, k3{}, k4{}, temp{};

        rhs(state_n, k1, I, t);
        apply_k_step<0>(temp, state_n, k1, dt * 0.5f);
        rhs(temp, k2, I, t + 0.5f*dt);
        apply_k_step<0>(temp, state_n, k2, dt * 0.5f);
        rhs(temp, k3, I, t + 0.5f*dt);
        apply_k_step<0>(temp, state_n, k3, dt);
        rhs(temp, k4, I, t + dt);
        apply_last_step<0>(state_n, k1, k2, k3, k4, dt / 6.0f);

        thrust::get<0>(t_zip) = state_n;
    }
};
