/*

#define _USE_MATH_DEFINES
#pragma once

// Standard C++ and CUDA/Thrust includes
#include "core\Solver.cu"
#include "core\Neurons.cu"
#include "core\genericLayer.cu"
#include "core\networkBuilder.cpp"
#include "core\Propogate.cu"
#include "core\layerLogger.cu"
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cublas_v2.h>
#include <chrono> // 🌟 Include Chrono library 🌟
#include <iomanip> // For std::setw and std::setfill

// --- external synaptic input functor (Unchanged) ---
struct ComputeI {
    float t;
    __host__ __device__
        float operator()(const float f_i) const {
        float s = sinf(2.0f * M_PI * f_i * t);
        return (s > 0.0f) ? 1.0f : 0.0f;
    }
};

// Global timing function for cleaner output
void print_chrono_duration(const std::string& version,
    std::chrono::high_resolution_clock::duration duration) {

    // Convert total duration to milliseconds
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    long long total_ms = duration_cast<milliseconds>(duration).count();

    // Calculate seconds and remaining milliseconds for the "s:ms" format
    long long seconds = total_ms / 1000;
    long long milliseconds_s = total_ms % 1000;

    // Print the result with padded milliseconds
    std::cout << version << " Runtime: ";
    std::cout << seconds << ":" << std::setw(3) << std::setfill('0') << milliseconds_s;
    std::cout << " (s:ms)\n";
}

struct LinspaceFunctor {
    float f_min, step;
    __host__ __device__
        float operator()(int i) const {
        return f_min + i * step;
    }
};

int main() {
    using namespace neuronModels;
    using Clock = std::chrono::high_resolution_clock;

    const int N = 1000;
    const float dt = 0.001f;
    const float T_end = 10.0f;
    const int steps = static_cast<int>(T_end / dt);

    // frequency table and other global setup (unchanged)
    const float f_cpu[N] = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f,
                             0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f,
                             0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.5f
    };

    // thrust::device_vector<float> f(f_cpu, f_cpu + N);



    // --- frequency setup: linspace from 0.1 to 5.0 ---
thrust::device_vector<float> f(N);
{
    thrust::counting_iterator<int> idx_first(0);
    thrust::counting_iterator<int> idx_last(N);

    float f_min = 0.1f;
    float f_max = 5.0f;
    float step = (f_max - f_min) / static_cast<float>(N - 1);

    LinspaceFunctor func{ f_min, step };
    thrust::transform(idx_first, idx_last, f.begin(), func);
}

std::vector<int> UC = {};
Eigen::Vector3i UC_dims(0, 0, 0);

networkBuilder net(
    Eigen::Vector3i(10, 10, 10),
    (Eigen::Matrix2f() << 3, 6, -2, -2).finished(),
    2.0f,
    (Eigen::Matrix2f() << 0.45f, 0.3f, 0.6f, 0.15f).finished(), // k0
    0.2f,
    1e-3f,
    false,
    UC, UC_dims
);

thrust::device_vector<int> d_X = net.X;
thrust::device_vector<int> d_Xn = net.Xn;
thrust::device_vector<float> d_W = net.W;

std::cout << "Simulation Steps: " << steps << "\n";

std::chrono::high_resolution_clock::duration async_duration;
std::chrono::high_resolution_clock::duration sync_duration;

// -----------------------------------------------------------
// ------------------------ ASYNC VERSION ----------------------
// -----------------------------------------------------------

{
    genericLayer<FHN> layer(N);
    FHN_RHS rhs;
    thrust::device_vector<float> d_input(N, 0.0f);

    cudaStream_t streamSim, streamIO;
    cudaStreamCreate(&streamSim);
    cudaStreamCreate(&streamIO);

    // 🌟 Start Chrono Timer 🌟
    auto start = Clock::now();

    layerLogger<genericLayer<FHN>> logger(layer, streamSim, streamIO, "FHN_layer");
    auto polSim = thrust::cuda::par.on(streamSim);

    logger.start();
    for (int step = 0; step < steps; ++step) {
        float t = step * dt;

        thrust::transform(polSim, f.begin(), f.end(), d_input.begin(), ComputeI{ t });
        thrust::transform(polSim, layer.input().begin(), layer.input().end(),
            d_input.begin(), layer.input().begin(), thrust::plus<float>());

        layer.step(rhs, t, dt, streamSim);
        PropagateStatic::sparse_p<0, FHN, FHN>(layer, layer, d_X, d_Xn, d_W,
            PropagateStatic::INPUT_OVERRIDE, streamSim);

        logger.write(step);
    }
    logger.stop();

    // Wait for all GPU work (in both streams) to complete
    cudaDeviceSynchronize();

    // 🌟 Stop Chrono Timer and Calculate Duration 🌟
    auto end = Clock::now();
    async_duration = end - start;

    print_chrono_duration("ASYNC", async_duration);

    cudaStreamDestroy(streamSim);
    cudaStreamDestroy(streamIO);
}

// -----------------------------------------------------------
// ------------------------ SYNC VERSION -----------------------
// -----------------------------------------------------------

{
    genericLayer<FHN> layer(N);
    FHN_RHS rhs;
    thrust::device_vector<float> d_input(N, 0.0f);

    // host buffers
    std::vector<std::vector<float>> traj(2 * N, std::vector<float>(steps, 0.0f));
    std::vector<std::vector<float>> input_hist(N, std::vector<float>(steps, 0.0f));

    // 🌟 Start Chrono Timer 🌟
    auto start = Clock::now();

    for (int step = 0; step < steps; ++step) {
        float t = step * dt;

        thrust::transform(thrust::device, f.begin(), f.end(), d_input.begin(), ComputeI{ t });
        thrust::transform(thrust::device, layer.input().begin(), layer.input().end(),
            d_input.begin(), layer.input().begin(), thrust::plus<float>());

        layer.step(rhs, t, dt);
        PropagateStatic::sparse_p<0, FHN, FHN>(layer, layer, d_X, d_Xn, d_W,
            PropagateStatic::INPUT_OVERRIDE);

        // Copy to Host (synchronous action)
        std::vector<float> u_h(N), v_h(N), input_h(N);
        thrust::copy(layer.state_vec<0>().begin(), layer.state_vec<0>().end(), u_h.begin());
        thrust::copy(layer.state_vec<1>().begin(), layer.state_vec<1>().end(), v_h.begin());
        thrust::copy(layer.input().begin(), layer.input().end(), input_h.begin());

        for (int i = 0; i < N; ++i) {
            traj[2 * i][step] = u_h[i];
            traj[2 * i + 1][step] = v_h[i];
            input_hist[i][step] = input_h[i];
        }
    }

    // --- write to file ---
    std::ofstream fout_state("trajectories_sync.csv");
    fout_state.setf(std::ios::fixed);
    fout_state.precision(6);

    for (auto& row : traj) {
        for (size_t j = 0; j < row.size(); ++j) {
            fout_state << row[j];
            if (j + 1 < row.size()) fout_state << ',';
        }
        fout_state << '\n';
    }
    fout_state.close();

    std::ofstream fout_input("input_sync.csv");
    fout_input.setf(std::ios::fixed);
    fout_input.precision(6);

    for (auto& row : input_hist) {
        for (size_t j = 0; j < row.size(); ++j) {
            fout_input << row[j];
            if (j + 1 < row.size()) fout_input << ',';
        }
        fout_input << '\n';
    }
    fout_input.close();

    std::cout << "Wrote trajectories_sync.csv & input_sync.csv ("
        << 2 * N << " + " << N << " rows × " << steps << " columns)\n";

    // 🌟 Stop Chrono Timer and Calculate Duration 🌟
    auto end = Clock::now();
    sync_duration = end - start;

    print_chrono_duration("SYNC", sync_duration);
}
// -----------------------------------------------------------
// ---------------------- COMPARISON ---------------------------
// -----------------------------------------------------------
std::cout << "\n--- Comparison ---\n";
if (async_duration < sync_duration) {
    std::cout << "The ASYNC version was FASTER. 🚀\n";
}
else if (sync_duration < async_duration) {
    std::cout << "The SYNC version was FASTER. (This is unusual for GPU code!)\n";
}
else {
    std::cout << "Both versions had the same runtime.\n";
}

system("pause"); // Removed non-standard call
return 0;
}

*/