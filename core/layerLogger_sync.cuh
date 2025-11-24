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

template<typename StateTypes> class genericLayer;

enum class Mode : uint8_t {
    ACCUMULATE_AND_FINALIZE = 0, // Mode 0: Store all in RAM, write in stop()
    IMMEDIATE_WRITE = 1          // Mode 1: Write to file on every write() call
};

constexpr Mode ACCUMULATE_AND_FINALIZE = Mode::ACCUMULATE_AND_FINALIZE;
constexpr Mode IMMEDIATE_WRITE = Mode::IMMEDIATE_WRITE;

template<typename Layer>
class layerLogger_sync {
public:
    using tuple_type = Layer::state_tuple_type;
    static constexpr size_t N_VARS = thrust::tuple_size<tuple_type>::value;

private:
    Layer& layer_;
    std::string baseName_;
    const Mode mode_;

    size_t N_;
    int steps_ = 0;

    std::ofstream fout_state_;
    std::ofstream fout_input_;

    std::vector<std::vector<float>> traj_;      // State (u/v) history
    std::vector<std::vector<float>> input_hist_;// Input history

    std::vector<thrust::host_vector<float>> h_temp_buffers_;

    bool started_ = false;

public:
    layerLogger_sync(Layer& layer, const std::string& name, Mode mode = ACCUMULATE_AND_FINALIZE)
        : layer_(layer), baseName_(name), mode_(mode)
    {
        N_ = layer_.size();

        if (mode_ == ACCUMULATE_AND_FINALIZE) {
            traj_.resize(N_VARS * N_); // N_VARS rows for each of N neurons
            input_hist_.resize(N_);
        }

        h_temp_buffers_.resize(N_VARS + 1); // +1 for input
        for (size_t i = 0; i < N_VARS + 1; ++i) {
            h_temp_buffers_[i].resize(N_);
        }
    }

    ~layerLogger_sync() {
        if (started_) stop();
    }

    void start() {
        if (started_) return;
        started_ = true;
        steps_ = 0;

        fout_state_.open(baseName_ + "_trajectories.csv", std::ios::out | std::ios::trunc);
        fout_input_.open(baseName_ + "_input.csv", std::ios::out | std::ios::trunc);
        if (!fout_state_ || !fout_input_)
            throw std::runtime_error("Failed to open logger CSV files");

        if (mode_ == ACCUMULATE_AND_FINALIZE) {
            for (auto& row : traj_) row.clear();
            for (auto& row : input_hist_) row.clear();
        }
    }

    void write(int step) {
        if (!started_) return;

        copy_data_to_temp_sync();

        if (mode_ == IMMEDIATE_WRITE) {
            write_current_to_files(fout_state_, fout_input_);
            fout_state_.flush(); 
            fout_input_.flush();
        }
        else {
            append_to_history();
        }
        steps_++;
    }

    void stop() {
        if (!started_) return;

        if (mode_ == ACCUMULATE_AND_FINALIZE) {
            write_history_to_files(fout_state_, fout_input_);
            std::cout << "Logger (Accumulate Mode) wrote " << N_VARS * N_ + N_ << " rows × "
                << steps_ << " columns.\n";
        }

        fout_state_.close();
        fout_input_.close();

        started_ = false;
    }

private:

    void copy_data_to_temp_sync() {
        copy_state_recursive<0>();

        thrust::copy(layer_.input().begin(), layer_.input().end(), h_temp_buffers_[N_VARS].begin());
    }

    template<size_t I>
    void copy_state_recursive() {
        if constexpr (I < N_VARS) {
            thrust::copy(layer_.template state_vec<I>().begin(),
                layer_.template state_vec<I>().end(),
                h_temp_buffers_[I].begin());
            copy_state_recursive<I + 1>();
        }
    }

    void append_to_history() {
        // State variables (u and v)
        for (size_t i = 0; i < N_VARS; ++i) {
            for (size_t n = 0; n < N_; ++n) {
                traj_[i * N_ + n].push_back(h_temp_buffers_[i][n]);
            }
        }
        for (size_t n = 0; n < N_; ++n) {
            input_hist_[n].push_back(h_temp_buffers_[N_VARS][n]);
        }
    }

    void write_current_to_files(std::ofstream& fout_state, std::ofstream& fout_input) {
        fout_state.setf(std::ios::fixed);
        fout_input.setf(std::ios::fixed);
        fout_state.precision(6);
        fout_input.precision(6);

        for (size_t i = 0; i < N_VARS; ++i) {
            for (size_t n = 0; n < N_; ++n) {
                fout_state << h_temp_buffers_[i][n];
                if (n + 1 < N_) fout_state << ',';
            }
            fout_state << '\n';
        }

        for (size_t n = 0; n < N_; ++n) {
            fout_input << h_temp_buffers_[N_VARS][n];
            if (n + 1 < N_) fout_input << ',';
        }
        fout_input << '\n';
    }

    void write_history_to_files(std::ofstream& fout_state, std::ofstream& fout_input) {
        fout_state.setf(std::ios::fixed);
        fout_input.setf(std::ios::fixed);
        fout_state.precision(6);
        fout_input.precision(6);

        for (auto& row : traj_) {
            for (size_t j = 0; j < row.size(); ++j) {
                fout_state << row[j];
                if (j + 1 < row.size()) fout_state << ',';
            }
            fout_state << '\n';
        }

        for (auto& row : input_hist_) {
            for (size_t j = 0; j < row.size(); ++j) {
                fout_input << row[j];
                if (j + 1 < row.size()) fout_input << ',';
            }
            fout_input << '\n';
        }
    }
};

