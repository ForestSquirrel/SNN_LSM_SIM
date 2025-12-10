#define _USE_MATH_DEFINES

// #define _LOAD 
//#define _LOGGERS
#define _SPIKE_OUT
#define SOLVER_TYPE_HEUN

#pragma once
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iomanip>
#include <thrust/fill.h>

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
#include "help\mnist_reader.h"

// --- Common Constants for the HR Model (Host & Device access) ---
struct HRConstants {
    const float a, b, c, d, s, x0;
    const float Iext_min, Iext_max;
    const float r_min, r_max;

    __host__ __device__
        HRConstants(float a, float b, float c, float d, float s, float x0,
            float Imin, float Imax, float rmin, float rmax) :
        a(a), b(b), c(c), d(d), s(s), x0(x0),
        Iext_min(Imin), Iext_max(Imax), r_min(rmin), r_max(rmax) {}
};

// --- Functor to Encode and Construct RHS Object ---
struct RHS_Builder_Functor {
    const HRConstants constants;

    __host__ __device__
        RHS_Builder_Functor(HRConstants consts) : constants(consts) {}

    // Input: The packed data value (e.g., from dataValues[k])
    // Output: A fully constructed HRNeuron_RHS functor
    __host__ __device__
        neuronModels::HRNeuron_RHS operator()(uint8_t dVal) const {

        // 1. Decode Parameters (Identical to your MATLAB logic)
        uint8_t i_int = dVal & 15;
        uint8_t r_int = dVal >> 4;

        float Iext = constants.Iext_min + ((float)i_int / 15.0f) * (constants.Iext_max - constants.Iext_min);

        float r = constants.r_min + ((float)r_int / 15.0f) * (constants.r_max - constants.r_min);

        return neuronModels::HRNeuron_RHS(
            constants.a, constants.b, constants.c, constants.d,
            r, constants.s, constants.x0
        );
    }
};

struct Iext_Extractor_Functor {
    const HRConstants constants;

    __host__ __device__
        Iext_Extractor_Functor(HRConstants consts) : constants(consts) {}

    __host__ __device__
        float operator()(uint8_t dVal) const {
        uint8_t i_int = dVal & 15;
        // Iext scaling
        return constants.Iext_min + ((float)i_int / 15.0f) * (constants.Iext_max - constants.Iext_min);
    }
};

using MNISTLoader = typename MNIST<uint8_t>;

int main() {

    MNISTLoader mnist(
        R"(D:\SNN\MNIST digits\train-images-idx3-ubyte\train-images-idx3-ubyte)", // Train 
        R"(D:\SNN\MNIST digits\train-labels-idx1-ubyte\train-labels-idx1-ubyte)", // Train Labels
        R"(D:\SNN\MNIST digits\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte)",   // Test
        R"(D:\SNN\MNIST digits\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte)"    // Test Labels
    );
    mnist.load(MNISTLoader::PartitionType::ALL);
    mnist.sort(MNISTLoader::SortingType::REPEATING);

    using namespace neuronModels;

    const int N = mnist.PIXEL_COUNT * 10; 
    const int iN = mnist.PIXEL_COUNT;

    const float dt = 1e-2f;
    const int steps = 15e+4;
    const int transientSteps = 1e+6;

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
    
    using inputLayerType = typename genericLayer<HRNeuron>;
    using LSMLayerType = typename genericLayer<IzhNeuron>;

    LSMLayerType layer(N);
    inputLayerType inputLayer(iN);

    SparsePropagator sp;
    sp.init();
    sp.buildCSR(inMap, lsmMap, thrust::device_vector<int>(inMap.size(), 30.0f), "IToLSM", inputLayer.size(), layer.size());
    sp.buildCSR(d_X, d_Xn, d_W, "LSMToLSM", layer.size(), layer.size());

    // ----- IZH -----
    const float aIZH = 0.05f;//0.02f; 0.1f
    const float bIZH = 0.2f;//0.2f;
    const float cIZH = -65.0f;
    const float dIZH = 2.0f;
    const float initV = -65.0f;
    const float initU = initV * bIZH;
    IzhNeuron_RHS rhs(aIZH, bIZH, cIZH, dIZH);

    // ----- HR -----
    const float aHR = 1.0f;
    const float bHR = 3.0f;
    const float cHR = 1.0f;
    const float dHR = 5.0f;
    const float sHR = 4.0f;

    const float Iext_range_min = 1.8f;
    const float Iext_range_max = 2.1f;
    const float r_range_min = 0.0013f;
    const float r_range_max = 0.007f;

    const float initX = -1.6f;
    const float initY = -12.0f;
    const float initZ = 0.0f;

    HRConstants host_consts(aHR, bHR, cHR, dHR, sHR, initX,
        Iext_range_min, Iext_range_max, r_range_min, r_range_max);
#pragma region Log
#ifdef _LOGGERS
    layerLogger_sync<layerType> logger(layer, "layer", Mode::ACCUMULATE_AND_FINALIZE);
    //layerLogger_sync<genericLayer<MemTunnerNeuron>> inputLogger(inputLayer, "input_layer", Mode::ACCUMULATE_AND_FINALIZE);
    logger.start();
    //inputLogger.start();
#endif  
#pragma endregion
    crossingsCounter<LSMLayerType, 0> cs(layer, 25.0f, CounterBehavior::BELOW_THR);
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
    std::vector<uint8_t> h_image(iN);
    thrust::device_vector<uint8_t> d_image(iN);
    thrust::device_vector<HRNeuron_RHS> rhs_collection(iN);
    thrust::device_vector<float> Iext_vector(iN);

    int totalImages = 1;
    for (int imgIdx = 0; imgIdx < totalImages; ++imgIdx) {
        h_image = mnist.images().at(imgIdx);

        thrust::copy(h_image.begin(), h_image.end(), d_image.begin());

        thrust::transform(d_image.begin(), d_image.end(), rhs_collection.begin(), RHS_Builder_Functor(host_consts));
        thrust::transform(d_image.begin(), d_image.end(), inputLayer.input().begin(), Iext_Extractor_Functor(host_consts));

        // Reset LSM
        thrust::fill(inputLayer.state_vec<0>().begin(), inputLayer.state_vec<0>().end(), initX);
        thrust::fill(inputLayer.state_vec<1>().begin(), inputLayer.state_vec<1>().end(), initY);
        thrust::fill(inputLayer.state_vec<2>().begin(), inputLayer.state_vec<2>().end(), initZ);

        thrust::fill(layer.state_vec<0>().begin(), layer.state_vec<0>().end(), initV);
        thrust::fill(layer.state_vec<1>().begin(), layer.state_vec<1>().end(), initU);
        thrust::fill(layer.input().begin(), layer.input().end(), 0.0f);

        for (int tt = 0; tt < transientSteps; ++tt) {
            inputLayer.step_heterogeneous(rhs_collection, (float)tt * dt, dt);
            print_nested_progress(imgIdx, totalImages, tt, steps + transientSteps, 30, R"(MNIST images)", R"(Single image)");
        }

        for (int step = 0; step < steps; ++step) {
            float t = step * dt;

            inputLayer.step_heterogeneous(rhs_collection, t, dt);

            sp.propagate<0, HRNeuron, IzhNeuron>(inputLayer, layer, "IToLSM", PropagationMode::CHEMICAL_LINEAR_FORWARD, InputBehavior::INPUT_ADD);
            thrust::transform(layer.input().begin(), layer.input().end(), thrust::make_constant_iterator(2.0f), layer.input().begin(), thrust::multiplies<float>());
#pragma region Log
#ifdef _LOGGERS
            logger.write(step);
#endif  
#pragma endregion
            layer.step(rhs, t, dt);
            sp.propagate<0, IzhNeuron, IzhNeuron>(layer, layer, "LSMToLSM", PropagationMode::ELECTRICAL_LINEAR_BIDIRECTIONAL, InputBehavior::INPUT_OVERRIDE);
            cs.count(step);
            print_nested_progress(imgIdx, totalImages, transientSteps + step, steps + transientSteps, 30, R"(MNIST images)", R"(Single image)");
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
        print_nested_progress(imgIdx, totalImages, steps + transientSteps, steps + transientSteps, 30, R"(MNIST images)", R"(Single image)");
    }
    print_nested_progress(totalImages, totalImages, steps + transientSteps, steps + transientSteps, 30, R"(MNIST images)", R"(Single image)");
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