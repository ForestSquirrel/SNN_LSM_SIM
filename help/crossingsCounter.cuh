#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <stdexcept>
#include <iostream>

#include "../core/genericLayer.cuh"

// Forward decl
template<typename StateTypes> class genericLayer;

/**
 * @brief Defines the type of threshold crossing to count.
 */
enum class CounterBehavior : uint8_t {
    BELOW_THR = 0,   // Transition from value < thr to value >= thr
    ABOVE_THR = 1,   // Transition from value >= thr to value < thr
    BIDIRECTIONAL = 2 // Counts both BELOW_THR and ABOVE_THR transitions
};

template <typename T>
struct crossing_functor {
    const T threshold_;
    const CounterBehavior behavior_;

    __host__ __device__
        crossing_functor(T threshold, CounterBehavior behavior)
        : threshold_(threshold), behavior_(behavior) {}

    __host__ __device__
        int operator()(const thrust::tuple<T, T>& states) const {
        T prev = thrust::get<0>(states); // State at step t-1
        T curr = thrust::get<1>(states); // State at step t

        if (behavior_ == CounterBehavior::BELOW_THR) {
            // Check for transition from sub-threshold to super-threshold (e.g., a spike/crossing)
            if (prev < threshold_ && curr >= threshold_) {
                return 1;
            }
        }
        else if (behavior_ == CounterBehavior::ABOVE_THR) {
            // Check for transition from super-threshold to sub-threshold (e.g., returning to rest)
            if (prev >= threshold_ && curr < threshold_) {
                return 1;
            }
        }
        else if (behavior_ == CounterBehavior::BIDIRECTIONAL) {
            // Check for crossing in either direction
            // (prev < thr && curr >= thr) OR (prev >= thr && curr < thr)
            if ((prev < threshold_ && curr >= threshold_) ||
                (prev >= threshold_ && curr < threshold_)) {
                return 1;
            }
        }
        return 0;
    }
};

template<typename Layer, size_t I>
class crossingsCounter {
public:
    using state_value_type = typename thrust::tuple_element<I, typename Layer::state_tuple_type>::type;
    
private:
    Layer& layer_;
    const state_value_type threshold_;
    const CounterBehavior behavior_;
    const size_t N_; 

    thrust::device_vector<state_value_type> d_prev_state_;
    thrust::device_vector<int> d_crossings_; 

    bool initialized_ = false;

public:
    const thrust::device_vector<int>& crossings = d_crossings_;

    /**
     * @brief Constructor for the crossingsCounter.
     * @param layer The reference to the simulation layer.
     * @param threshold The value a state must cross to be counted.
     * @param behavior The type of crossing to count (BELOW_THR, ABOVE_THR, BIDIRECTIONAL).
     */
    crossingsCounter(Layer& layer, state_value_type threshold, CounterBehavior behavior = CounterBehavior::BELOW_THR)
        : layer_(layer), threshold_(threshold), behavior_(behavior), N_(layer.size())
    {
        try {
            d_prev_state_.resize(N_);
            d_crossings_.resize(N_, 0);

            thrust::copy(layer_.template state_vec<I>().begin(),
                layer_.template state_vec<I>().end(),
                d_prev_state_.begin());

            initialized_ = true;
        }
        catch (const thrust::system::system_error& e) {
            std::cerr << "Thrust error in crossingsCounter constructor: " << e.what() << std::endl;
            initialized_ = false;
        }
    }

    ~crossingsCounter() {
        destroy();
    }

    /**
     * @brief Performs the crossing count for the current step.
     * @param step The current simulation step number.
     */
    void count(int step) {
        if (!initialized_) return;

        if (step > 0) {
            auto d_current_state_begin = layer_.template state_vec<I>().begin();

            auto zip_iter = thrust::make_zip_iterator(
                thrust::make_tuple(d_prev_state_.begin(), d_current_state_begin)
            );

            thrust::device_vector<int> d_step_crossings(N_);

            thrust::transform(zip_iter,
                zip_iter + N_,
                d_step_crossings.begin(),
                crossing_functor<state_value_type>(threshold_, behavior_)); 

            thrust::transform(d_crossings_.begin(),
                d_crossings_.end(),
                d_step_crossings.begin(),
                d_crossings_.begin(),
                thrust::plus<int>());
        }

        thrust::copy(layer_.template state_vec<I>().begin(),
            layer_.template state_vec<I>().end(),
            d_prev_state_.begin());
    }

    /**
     * @brief Resets the crossing counts to zero.
     */
    void reset() {
        if (!initialized_) return;
        thrust::fill(d_crossings_.begin(), d_crossings_.end(), 0);
    }

    /**
     * @brief Cleans up device memory.
     */
    void destroy() {
        if (initialized_) {
            d_prev_state_.clear();
            d_crossings_.clear();
            initialized_ = false;
        }
    }
};