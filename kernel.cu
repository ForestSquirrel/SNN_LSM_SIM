#define _USE_MATH_DEFINES

// #define _LOAD 
// #define _LOGGERS
#define _SPIKE_OUT


#pragma once
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iomanip>
#include <thrust/fill.h>

#include "core\Solver.cuh"
#include "core\Neurons.cuh"
#include "core\genericLayer.cuh"
#include "core\networkBuilder.cuh"
//#include "core\Propogate.cuh"
#include "core\Propagators\SparsePropagator.cuh"
#include "core\layerLogger.cuh"
#include "core\layerLogger_sync.cuh"
#include "help\progresBar.h"
#include "help\mapInputToLSM.cuh"
#include "help\balance3.h"
#include "help\config_io.cuh"
#include "help\crossingsCounter.cuh"
#include"help\mnist_reader.h"

#pragma region Helpers
struct ComputeI {
    float t;
    float amp;
    __host__ __device__
        ComputeI(float _amp, float _t) : amp(_amp), t(_t) {}
    __host__ __device__
        float operator()(const float f_i) const {
        float s = sinf(2.0f * M_PI * f_i * t);
        return (s > 0) ? amp : 0.0f;
    }
};

struct LinspaceFunctor {
    float f_min, step;
    __host__ __device__
        LinspaceFunctor(float _min, float _step) : f_min(_min), step(_step) {}
    __host__ __device__
        float operator()(int i) const {
        return f_min + i * step;
    }
};

thrust::device_vector<float> linspace(float min, float max, int N) {
    thrust::device_vector<float> result(N);

    if (N == 1) {
        result[0] = min;
        return result;
    }
    thrust::counting_iterator<int> idx_first(0);
    thrust::counting_iterator<int> idx_last(N);

    float step = (max - min) / (N - 1);
    thrust::transform(idx_first, idx_last, result.begin(), LinspaceFunctor(min, step));

    return result;
};
#pragma endregion

using MNISTLoader = typename MNIST<float>;

int main() {

    MNISTLoader mnist(
        R"(D:\SNN\MNIST digits\train-images-idx3-ubyte\train-images-idx3-ubyte)", // Train 
        R"(D:\SNN\MNIST digits\train-labels-idx1-ubyte\train-labels-idx1-ubyte)", // Train Labels
        R"(D:\SNN\MNIST digits\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte)",   // Test
        R"(D:\SNN\MNIST digits\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte)"    // Test Labels
    );
    mnist.load(MNISTLoader::PartitionType::ALL);
    mnist.sort(MNISTLoader::SortingType::REPEATING);
    
    /*
    auto img = mnist.images().at(1);
    auto label = mnist.labels().at(1);
    std::cout << "Label: " << static_cast<int>(label) << "\n";

    for (int y = 0; y < MNISTLoader::IMAGE_DIM; ++y) {
        for (int x = 0; x < MNISTLoader::IMAGE_DIM; ++x) {
            float pixel = img[y * MNISTLoader::IMAGE_DIM + x];
            std::cout << (pixel > 128.0f ? '*' : ' ');
        }
        std::cout << '\n';
    }
    */

    using namespace neuronModels;

    const int N = mnist.PIXEL_COUNT * 10; 
    const int iN = mnist.PIXEL_COUNT;

    const float dt = 1e-2f;
    const float T_end = 15e+3f * dt;
    const int steps = static_cast<int>(T_end / dt);

    //auto f = linspace(0.08f, 0.08f, N);
    float G_syn = 8e-4f;
#ifndef _LOAD
    std::vector<int> UC = {};
    Eigen::Vector3i UC_dims(0, 0, 0);
    int sz1, sz2, sz3;
    std::tie(sz1, sz2, sz3) = balanced3(N);
    networkBuilder net(
        Eigen::Vector3i(sz1, sz2, sz3),
        (Eigen::Matrix2f() << 0.8f, -0.6f, -0.5f, 0.4f).finished(),
        3.5f,
        (Eigen::Matrix2f() << 0.2f, 0.2f, 0.25f, 0.0f).finished(), // k0
        0.2f,
        2.0f,
        false,
        UC, UC_dims
    );

#ifdef _VERBOSE_DEBUG
    std::cout << "LSM with " << N << " neurons, " << net.X.size() << "connections" << std::endl;
    for (auto idx = 0; idx < net.X.size(); ++idx) {
        std::cout << "Pre " << net.X[idx] << " --> Post " << net.Xn[idx] << " | W = " << net.W[idx] << std::endl;
    }
#endif

    thrust::device_vector<int> d_X = net.X;
    thrust::device_vector<int> d_Xn = net.Xn;
    thrust::device_vector<float> d_W = net.W;

    thrust::device_vector<int> inMap, lsmMap;
    mapInputToLSM(iN, net.eIndices, SelectionMode::NON_REPEATING, inMap, lsmMap) ?
        std::cout << "Maping input layer to lsm successfull" << std::endl :
        std::cerr << "Maping input layer to lsm failed" << std::endl;

    saveLSM("cfg", inMap, lsmMap, thrust::device_vector<int>(inMap.size(), 1.0f), d_X, d_Xn, d_W);
#endif 
#ifdef _LOAD
    thrust::device_vector<float> inMap, lsmMap, lsmW, d_X, d_Xn, d_W;
    loadLSM("cfg", inMap, lsmMap, lsmW, d_X, d_Xn, d_W);
#endif 
    std::cout << "Simulation Steps: " << steps << "\n";
        
    genericLayer<MemTunnerNeuron> layer(N);
    genericLayer<MemTunnerNeuron> inputLayer(iN);

    SparsePropagator sp;
    sp.init();
    sp.buildCSR(inMap, lsmMap, thrust::device_vector<int>(inMap.size(), 1.0f), "IToLSM", inputLayer.size(), layer.size());
    sp.buildCSR(d_X, d_Xn, d_W, "LSMToLSM", layer.size(), layer.size());

    MemTunnerNeuron_RHS rhs(-0.1f, 17e-3f, 13e-3f);
#pragma region Log
#ifdef _LOGGERS
    layerLogger_sync<genericLayer<MemTunnerNeuron>> logger(layer, "FHN_layer", Mode::ACCUMULATE_AND_FINALIZE);
    //layerLogger_sync<genericLayer<MemTunnerNeuron>> inputLogger(inputLayer, "input_layer", Mode::ACCUMULATE_AND_FINALIZE);
    logger.start();
    //inputLogger.start();
#endif  
#pragma endregion
    crossingsCounter<genericLayer<MemTunnerNeuron>, 0> cs(layer, 0.2f, CounterBehavior::BELOW_THR);

#pragma region OUTPUT
#ifdef _SPIKE_OUT
    std::ofstream csvFile("MNIST_FEATURES.csv");
    csvFile << std::fixed << std::setprecision(6);

    // Write header
    {
        size_t N = cs.crossings.size();
        for (size_t i = 0; i < N; ++i) {
            csvFile << "SpikeCounts_" << (i + 1);
            if (i != N - 1)
                csvFile << ",";
        }
        csvFile << ",True_Class\n";
    }
#endif // _SPIKE_OUT
#pragma endregion
    std::vector<float> image(iN);
    thrust::device_vector<float> f(iN);
    int totalImages = 1;
    for (int imgIdx = 0; imgIdx < totalImages; ++imgIdx) {
        image = mnist.images().at(imgIdx);
        thrust::copy(image.begin(), image.end(), f.begin());
        thrust::transform(f.begin(), f.end(), thrust::make_constant_iterator(2550.0f), f.begin(), thrust::divides<float>());

        // Reset LSM
        thrust::fill(inputLayer.state_vec<0>().begin(), inputLayer.state_vec<0>().end(), 0.0f);
        thrust::fill(inputLayer.state_vec<1>().begin(), inputLayer.state_vec<0>().end(), 0.0f);
        thrust::fill(inputLayer.input().begin(), inputLayer.input().end(), 0.0f);
        thrust::fill(layer.state_vec<0>().begin(), layer.state_vec<0>().end(), 0.0f);
        thrust::fill(layer.state_vec<1>().begin(), layer.state_vec<0>().end(), 0.0f);
        thrust::fill(layer.input().begin(), layer.input().end(), 0.0f);

        for (int step = 0; step < steps; ++step) {
            float t = step * dt;

            thrust::transform(f.begin(), f.end(), inputLayer.input().begin(), ComputeI(660.0f * 1e-7f, t));
            inputLayer.step(rhs, t, dt);
#pragma region Log
#ifdef _LOGGERS 
            //inputLogger.write(step);
#endif  
#pragma endregion
            sp.propagate<0, MemTunnerNeuron, MemTunnerNeuron>(inputLayer, layer, "IToLSM", InputBehavior::INPUT_ADD);
            thrust::transform(layer.input().begin(), layer.input().end(), thrust::make_constant_iterator(G_syn), layer.input().begin(), thrust::multiplies<float>());
#pragma region Log
#ifdef _LOGGERS
            logger.write(step);
#endif  
#pragma endregion
            layer.step(rhs, t, dt);
            sp.propagate<0, MemTunnerNeuron, MemTunnerNeuron>(layer, layer, "LSMToLSM", InputBehavior::INPUT_OVERRIDE);
            cs.count(step);
            print_nested_progress(imgIdx, totalImages, step, steps, 30, R"(Image processing)", R"(MNIST images)");
        }
#pragma region OUTPUT
#ifdef _SPIKE_OUT
        // === CSV WRITE BLOCK ===
        {
            thrust::host_vector<int> hCrossings = cs.crossings; // pull to host
            for (size_t i = 0; i < hCrossings.size(); ++i) {
                csvFile << hCrossings[i];
                if (i != hCrossings.size() - 1)
                    csvFile << ",";
            }
            csvFile << "," << static_cast<int>(mnist.labels().at(imgIdx)) << "\n";
        }
        // ========================  

#endif // _SPIKE_OUT
#pragma endregion
        cs.reset();
        print_nested_progress(imgIdx, totalImages, steps, steps, 30, R"(Image processing)", R"(MNIST images)");
    }
    print_nested_progress(totalImages, totalImages, steps, steps, 30, R"(Image processing)", R"(MNIST images)");
#pragma region Log    
#ifdef _LOGGERS
    logger.stop();
    //inputLogger.stop();
#endif
#pragma endregion
    std::cout << "Finished " << "\n";

    mnist.destroy();
    sp.destroy();
    cs.destroy();
    system("pause"); 
    return 0;
}