#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/shuffle.h>
#include <random>
#include <algorithm>

/**
 * Defines the selection mode for the random sampling.
 *
 * Selection strategy for mapping inputs to reservoir neurons.
 * 
 * @ingroup lsm_utils
 */
enum class SelectionMode {
    /** The same element from the indices vector can be selected multiple times (sampling with replacement). */
    REPEATING,
    /** Each element from the indices vector can be selected only once (sampling without replacement). */
    NON_REPEATING
};

struct RandomIntGenerator {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist;

    __host__ __device__
        RandomIntGenerator(unsigned int seed, int min_val, int max_val)
        : rng(seed), dist(min_val, max_val) {}

    __host__ __device__
        int operator()(int) {
        return dist(rng);
    }
};

// ---------------------------------------------------

/**
 * Maps input channels to reservoir neurons using random selection.
 * @param iN Number of input connections to generate.
 * @param indices_h Host vector of candidate neuron indices.
 * @param mode SelectionMode specifying replacement behavior.
 * @param X_d Output presynaptic indices (0..iN-1).
 * @param Xn_d Output postsynaptic indices sampled from indices_h.
 * @return true on success, false when constraints are violated.
 * 
 * @ingroup lsm_utils
 */
bool mapInputToLSM(
    int iN,
    const std::vector<int>& indices_h,
    SelectionMode mode,
    thrust::device_vector<int>& X_d,
    thrust::device_vector<int>& Xn_d)
{
    const int N_indices = indices_h.size();

    // Check for impossible scenario: NON_REPEATING selection requires iN <= indices.size()
    if (mode == SelectionMode::NON_REPEATING && iN > N_indices) {
        std::cerr << "Error: Cannot perform NON_REPEATING selection (iN=" << iN
            << ") when the source vector size is smaller (indices.size()=" << N_indices << ")." << std::endl;
        return false;
    }

    X_d.resize(iN);
    thrust::sequence(X_d.begin(), X_d.end());

    Xn_d.resize(iN);

    thrust::device_vector<int> indices_d = indices_h;

    const unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();

    if (mode == SelectionMode::REPEATING) {

        thrust::device_vector<int> random_indices_d(iN);

        RandomIntGenerator generator(seed, 0, N_indices - 1);

        thrust::transform(
            thrust::make_counting_iterator(0), // Start from 0
            thrust::make_counting_iterator(iN), // End at iN
            random_indices_d.begin(),
            generator // The functor
        );

        thrust::gather(
            random_indices_d.begin(),
            random_indices_d.end(),
            indices_d.begin(), // The map (source)
            Xn_d.begin()       // The result
        );

    }
    else {
        thrust::shuffle(indices_d.begin(), indices_d.end(), thrust::default_random_engine(seed));

        thrust::copy(
            indices_d.begin(),
            indices_d.begin() + iN,
            Xn_d.begin()
        );
    }

    return true;
}