#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include "../core/genericLayer.cuh"

// Forward declaration of genericLayer
template<typename StateTypes>
class genericLayer;

/**
 * @brief Asynchronous logger that records layer states and inputs to CSV files.
 * @tparam Layer Layer type exposing state_tuple_type and input buffers.
 */
template<typename Layer>
class layerLogger {
public:
    using tuple_type = Layer::state_tuple_type;
    static constexpr size_t N_VARS = thrust::tuple_size<tuple_type>::value;

private:
    Layer& layer_;
    cudaStream_t streamSim_;
    cudaStream_t streamIO_;
    std::string baseName_;

    size_t N_;                        // number of neurons
    std::ofstream fout_state_;
    std::ofstream fout_input_;

    // double-buffered host/device storage per variable (+input)
    std::vector<thrust::device_vector<float>> d_stage_[2]; // [buffer][var]
    std::vector<float*> h_stage_[2];                       // pinned host buffers
    cudaEvent_t ev_ready_[2];

    bool started_ = false;

public:
    /**
     * @brief Constructs the logger with simulation and IO streams.
     * @param layer Layer to observe.
     * @param streamSim CUDA stream used by the simulation.
     * @param streamIO CUDA stream used for IO operations.
     * @param name Base file name for generated CSVs.
     */
    layerLogger(Layer& layer,
        cudaStream_t streamSim,
        cudaStream_t streamIO,
        const std::string& name)
        : layer_(layer), streamSim_(streamSim), streamIO_(streamIO), baseName_(name)
    {
        N_ = layer_.size();

        for (int b = 0; b < 2; ++b) {
            ev_ready_[b] = nullptr;
            cudaEventCreateWithFlags(&ev_ready_[b], cudaEventDisableTiming);
            d_stage_[b].resize(N_VARS + 1); // +1 for input
            h_stage_[b].resize(N_VARS + 1, nullptr);
        }
    }

    ~layerLogger() {
        stop();
        for (int b = 0; b < 2; ++b) {
            for (float* p : h_stage_[b])
                if (p) cudaFreeHost(p);
            cudaEventDestroy(ev_ready_[b]);
        }
    }

    // ------------------------------------------------------------
    /**
     * @brief Allocates buffers and opens CSV outputs.
     */
    void start() {
        if (started_) return;
        started_ = true;

        // allocate buffers
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < N_VARS + 1; ++i) {
                d_stage_[b][i].resize(N_);
                cudaHostAlloc(&h_stage_[b][i], N_ * sizeof(float), cudaHostAllocDefault);
            }
        }

        // open CSV files
        fout_state_.open(baseName_ + "_trajectories.csv", std::ios::out | std::ios::trunc);
        fout_input_.open(baseName_ + "_input.csv", std::ios::out | std::ios::trunc);
        if (!fout_state_ || !fout_input_)
            throw std::runtime_error("Failed to open logger CSV files");
    }

    // ------------------------------------------------------------
    /**
     * @brief Schedules a logging operation for the given simulation step.
     * @param step Current simulation step index.
     */
    void write(int step) {
        if (!started_) return;
        const int cur = step & 1;
        const int prev = (step ^ 1) & 1;

        // 1. record event when simulation step is done
        cudaEventRecord(ev_ready_[cur], streamSim_);

        // 2. make IO stream wait for simulation completion
        cudaStreamWaitEvent(streamIO_, ev_ready_[cur], 0);

        // 3. async copy each state variable + input to device staging buffers
        copy_to_stage(cur);

        // 4. async copy staged data to host
        for (size_t i = 0; i < N_VARS + 1; ++i) {
            cudaMemcpyAsync(h_stage_[cur][i],
                thrust::raw_pointer_cast(d_stage_[cur][i].data()),
                N_ * sizeof(float),
                cudaMemcpyDeviceToHost,
                streamIO_);
        }

        // 5. when copy finishes, append to CSVs
        cudaLaunchHostFunc(streamIO_, [](void* userData) {
            auto* pack = static_cast<std::tuple<layerLogger*, int>*>(userData);
            auto* self = std::get<0>(*pack);
            int idx = std::get<1>(*pack);
            self->append_to_files(idx);
            delete pack;
            }, new std::tuple<layerLogger*, int>(this, cur));

        // 6. ensure previous buffers write completed before reuse
        if (step > 1)
            cudaEventSynchronize(ev_ready_[prev]);
    }

    // ------------------------------------------------------------
    /**
     * @brief Flushes and closes files, ensuring pending copies finish.
     */
    void stop() {
        if (!started_) return;
        cudaStreamSynchronize(streamIO_);
        fout_state_.flush(); fout_state_.close();
        fout_input_.flush(); fout_input_.close();
        started_ = false;
    }

private:
    // copy active layer data to staging buffers
    void copy_to_stage(int buf) {
        auto pol = thrust::cuda::par.on(streamSim_);

        // state variables
        copy_state<0>(buf, pol);
        // input
        thrust::copy(pol, layer_.input().begin(), layer_.input().end(), d_stage_[buf][N_VARS].begin());
    }

    template<size_t I, typename Policy>
    void copy_state(int buf, Policy pol) {
        if constexpr (I < N_VARS) {
            thrust::copy(pol,
                layer_.template state_vec<I>().begin(),
                layer_.template state_vec<I>().end(),
                d_stage_[buf][I].begin());
            copy_state<I + 1>(buf, pol);
        }
    }

    // Append current buffer data to CSV files
    void append_to_files(int buf) {
        fout_state_ << std::fixed;
        fout_input_ << std::fixed;
        fout_state_.precision(6);
        fout_input_.precision(6);

        // Each variable = one line per step (like time-major)
        for (size_t i = 0; i < N_VARS; ++i) {
            for (size_t n = 0; n < N_; ++n) {
                fout_state_ << h_stage_[buf][i][n];
                if (n + 1 < N_) fout_state_ << ',';
            }
            fout_state_ << '\n';
        }

        // Input
        for (size_t n = 0; n < N_; ++n) {
            fout_input_ << h_stage_[buf][N_VARS][n];
            if (n + 1 < N_) fout_input_ << ',';
        }
        fout_input_ << '\n';
    }
};
