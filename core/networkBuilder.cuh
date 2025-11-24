#pragma once
/**
 * @file networkBuilder.cuh
 * @brief Generates sparse connectivity for reservoir networks and exposes
 *        resulting matrices and index lists.
 */
#define __restrict
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define NOMINMAX

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

/**
 * @brief Constructs randomized excitatory/inhibitory connectivity patterns and
 *        associated metadata for use on host or device.
 */
class networkBuilder {
public:
    std::vector<int> X, Xn;
    std::vector<float> T, W, E;
    Eigen::MatrixXf R;
    std::vector<int> eIndices;  // indices of excitatory neurons
    std::vector<int> iIndices;  // indices of inhibitory neurons

    // UC: flattened 3D pattern, dims = {x, y, z}
    networkBuilder(
        const Eigen::Vector3i& resSize = Eigen::Vector3i(3, 3, 5),
        const Eigen::Matrix2f& w = (Eigen::Matrix2f() << 3, 6, -2, -2).finished(),
        float r0 = 2.0f,
        const Eigen::Matrix2f& k0 = (Eigen::Matrix2f() << 0.45f, 0.3f, 0.6f, 0.15f).finished(),
        float f_inhibit = 0.2f,
        float tau = 1e-3f,
        bool show = false,  // ignored
        const std::vector<int>& UC = {},
        const Eigen::Vector3i& UC_dims = Eigen::Vector3i(0, 0, 0),
        bool _normalize = true
    ) {
        build(resSize, w, r0, k0, f_inhibit, tau, show, UC, UC_dims, _normalize);
    }

private:
    /**
     * @brief Populate connectivity matrices and neuron labels based on provided
     *        lattice dimensions and model parameters.
     */
    void build(
        const Eigen::Vector3i& resSize,
        const Eigen::Matrix2f& w,
        float r0,
        const Eigen::Matrix2f& k0,
        float f_inhibit,
        float tau,
        bool /*show*/,
        const std::vector<int>& UC,
        const Eigen::Vector3i& UC_dims,
        bool _normalize
    ) {
        const int N = resSize.prod();
        R.resize(N, 3);

        // === 1. Coordinates ===
        int idx = 0;
        for (int i = 0; i < resSize(0); ++i)
            for (int j = 0; j < resSize(1); ++j)
                for (int k = 0; k < resSize(2); ++k)
                    R.row(idx++) << (i + 1), (j + 1), (k + 1);

        // === 2. E/I assignment ===
        std::default_random_engine rng(std::random_device{}());
        std::uniform_real_distribution<float> rand01(0.0f, 1.0f);
        E.resize(N);

        if (UC.empty() || UC_dims(0) == 0) {
            for (int i = 0; i < N; ++i)
                E[i] = (rand01(rng) < f_inhibit) ? -1.0f : 1.0f;
        }
        else {
            int ux = UC_dims(0), uy = UC_dims(1), uz = UC_dims(2);
            for (int i = 0; i < resSize(0); ++i)
                for (int j = 0; j < resSize(1); ++j)
                    for (int k = 0; k < resSize(2); ++k)
                        E[i * resSize(1) * resSize(2) + j * resSize(2) + k] =
                        static_cast<float>(UC[(i % ux) * uy * uz + (j % uy) * uz + (k % uz)]);
        }

        // === 3. Distance matrix ===
        Eigen::MatrixXf D = (R.rowwise().squaredNorm().replicate(1, N)
            + R.rowwise().squaredNorm().transpose().replicate(N, 1)
            - 2.0f * (R * R.transpose())).cwiseMax(0.0f).cwiseSqrt();

        // === 4. Sort by E ===
        std::vector<int> sort_E(N), sort_back(N);
        std::iota(sort_E.begin(), sort_E.end(), 0);
        std::sort(sort_E.begin(), sort_E.end(), [&](int a, int b) { return E[a] < E[b]; });
        for (int i = 0; i < N; ++i) sort_back[sort_E[i]] = i;

        Eigen::MatrixXf D_sorted(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                D_sorted(i, j) = D(sort_E[i], sort_E[j]);
        D.swap(D_sorted);

        // === 5. Connection probability ===
        Eigen::MatrixXf ConnProb(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                bool iE = (E[sort_E[i]] > 0);
                bool jE = (E[sort_E[j]] > 0);
                float coeff = 0.0f;
                if (!iE && !jE) coeff = k0(1, 1);
                else if (!iE && jE) coeff = k0(1, 0);
                else if (iE && !jE) coeff = k0(0, 1);
                else coeff = k0(0, 0);
                ConnProb(i, j) = coeff * std::exp(-std::pow(D(i, j), 2) / (r0 * r0));
            }

        // === 6. Random pruning ===
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (ConnProb(i, j) < rand01(rng))
                    D(i, j) = 0.0f;

        // === 6.5 Remove reflected connections ===
        const bool REMOVE_REFLECTED_LOOPS = true;
        if (REMOVE_REFLECTED_LOOPS) {
            struct Pair { int r, c; };
            std::vector<Pair> loops;
            for (int r = 0; r < N; ++r)
                for (int c = r + 1; c < N; ++c)
                    if (D(r, c) > 0.0f && D(c, r) > 0.0f)
                        loops.push_back({ r, c });

            if (!loops.empty()) {
                std::shuffle(loops.begin(), loops.end(), rng);
                int half = (int)loops.size() / 2;
                for (int i = 0; i < half; ++i) D(loops[i].r, loops[i].c) = 0.0f;
                for (int i = half; i < (int)loops.size(); ++i) D(loops[i].c, loops[i].r) = 0.0f;
            }
        }

        // === 7. Reverse sort back ===
        Eigen::MatrixXf D_back(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                D_back(i, j) = D(sort_back[i], sort_back[j]);
        D.swap(D_back);

        // === 8. Build adjacency & weights ===
        Eigen::ArrayXXf Xmat = (D.array() > 0).cast<float>();
        Eigen::MatrixXf Wmat = Eigen::MatrixXf::Zero(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (Xmat(i, j) == 0.0f) continue;
                bool Ei = (E[i] > 0);
                bool Ej = (E[j] > 0);
                if (Ei && Ej) Wmat(i, j) = w(0, 0);
                else if (Ei && !Ej) Wmat(i, j) = w(0, 1);
                else if (!Ei && Ej) Wmat(i, j) = w(1, 0);
                else Wmat(i, j) = w(1, 1);
            }

        // === 9. Time delays ===
        Eigen::MatrixXf Tmat;
        if (tau != 0.0f)
            Tmat = tau * Xmat.matrix();
        else
            Tmat = D * 1e-3f;

        // === 10. Flatten to 1D vectors ===
        X.clear(); Xn.clear(); T.clear(); W.clear();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (Xmat(i, j) != 0.0f) {
                    X.push_back(i);
                    Xn.push_back(j);
                    W.push_back(Wmat(i, j));
                    T.push_back(Tmat(i, j));
                    //E.push_back(std::signbit(Wmat(i, j)) ? 1.0f : -1.0f);
                }
        if (_normalize)
            W = normalize_weights(X, Xn, W);
        compute_EI_indices();

        std::cout << "Reservoir with size [" << resSize.transpose()
            << "] created " << X.size() << " connections, "
            << eIndices.size() << " excitatory, "
            << iIndices.size() << " inhibitory neurons." << std::endl;
    }

    std::vector<float> normalize_weights(
        const std::vector<int>& X,
        const std::vector<int>& Xn,
        const std::vector<float>& W)
    {
        if (X.size() != Xn.size() || X.size() != W.size()) {
            throw std::runtime_error("Input vectors must have the same size.");
        }

        // Map to store: Destination Index (Xn_i) -> Number of Incoming Connections
        std::unordered_map<int, int> incoming_counts;

        // Step 1: Count the number of times each destination index (Xn_i) appears
        for (int destination_index : Xn) {
            incoming_counts[destination_index]++;
        }

        // Step 2: Create the new normalized weights vector
        std::vector<float> W_normalized;
        W_normalized.reserve(W.size());

        // Step 3: Iterate through the original weights and normalize each one
        for (size_t i = 0; i < W.size(); ++i) {
            int destination_index = Xn[i];
            double original_weight = W[i];

            // Get the count of incoming connections for this destination
            int count = incoming_counts.at(destination_index);

            // Calculate the normalized weight
            double normalized_weight = original_weight / count;

            W_normalized.push_back(normalized_weight);
        }

        return W_normalized;
    }

    void compute_EI_indices() {
        eIndices.clear();
        iIndices.clear();
        for (int i = 0; i < static_cast<int>(E.size()); ++i) {
            if (E[i] > 0)
                eIndices.push_back(i);
            else
                iIndices.push_back(i);
        }
    }
};
