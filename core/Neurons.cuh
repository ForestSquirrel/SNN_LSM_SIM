#pragma once
#include <thrust/tuple.h>
#include <cmath>
#include <cuda_runtime.h>

/**
 * Collection of neuron model state representations and right-hand side functors.
 */
namespace neuronModels {
    template<typename... T>
    using StateTuple = thrust::tuple<T...>;

    /**
     * State tuple for the FitzHugh–Nagumo model (u, v).
     */
    using FHN = StateTuple<float, float>;

    /**
     * Right-hand side functor for the FitzHugh–Nagumo neuron dynamics.
     * @see FHN
     */
    struct FHN_RHS {
        __host__ __device__ inline
            void operator()(const FHN& state_in, FHN& ddt_out, float I_syn, float t) const {
            float u = thrust::get<0>(state_in);
            float v = thrust::get<1>(state_in);

            float dudt = 20.0f * (u * (u - 1.0f) * (1.0f - 10.0f * u) - v + I_syn);
            float dvdt = 20.0f * u;

            thrust::get<0>(ddt_out) = dudt;
            thrust::get<1>(ddt_out) = dvdt;
        }
    };

    // ----------- O R L O V S K I I ----------
    /**
     * State tuple for the memristor-based tunneling neuron (Vc, XSV).
     */
    using MemTunnerNeuron = StateTuple<float, float>;

    /**
     * Right-hand side functor for the memristor tunneling neuron model.
     * @see MemTunnerNeuron
     */
    struct MemTunnerNeuron_RHS {

        const float U1;
        const float U2;
        const float C1_scaled;

        __host__ __device__
            MemTunnerNeuron_RHS(float u1, float u2, float c1)
            : U1(u1), U2(u2), C1_scaled(c1 * 1e-6f) {}

        /**
         * Computes the diode/tunnel current contribution.
         * @param e Diode voltage.
         * @return Combined current through the device.
         */
        __host__ __device__ inline
            float GI403(float e) const {
            const float Is = 1.1E-7f;
            const float Vt = 1.0f / 17.0f;
            const float Vp = 0.039f;
            const float Ip = 6.2e-5f;
            const float Iv = 6e-6f;
            const float D = 20.0f;
            const float E = 0.09f;

            // Idiode
            auto Idiode = [&](float v) {
                return Is * (exp(v / Vt) - exp(-v / Vt));
            };
            // Itunnel
            auto Itunnel = [&](float v) {
                return Ip / Vp * v * exp(-(v - Vp) / Vp);
            };
            // Iex
            auto Iex = [&](float v) {
                return Iv * (atan(D * (v - E)) + atan(D * (v + E)));
            };

            return Idiode(e) + Itunnel(e) + Iex(e);
        }

        // AND_TS(V1, V2)
        /**
         * Calculates memristor branch current and state derivative.
         * @param V1 Membrane voltage.
         * @param V2 State variable representing conductance fraction.
         * @return Pair of memristor current and state change.
         */
        __host__ __device__ inline
            MemTunnerNeuron AND_TS(float V1, float V2) const {
            const float Ron = 2e3f;
            const float Roff = 1e6f;
            const float Von1 = 0.28f;
            const float Voff1 = 0.14f;
            const float Von2 = -0.12f;
            const float Voff2 = -0.02f;
            const float TAU = 0.0000001f;
            const float T = 0.5f;
            const float boltz = 1.380649e-23f;
            const float echarge = 1.602176634e-19f;

            const float K = -1.0f / (T * boltz / echarge);

            float Ix_term1_A = 1.0f / (1.0f + exp(K * (V1 - Von1) * (V1 - Von2)));
            float Ix_term1 = Ix_term1_A * (1.0f - V2);

            float Ix_term2_A = 1.0f - (1.0f / (1.0f + exp(K * (V1 - Voff2) * (V1 - Voff1))));
            float Ix_term2 = Ix_term2_A * V2;

            float Ix = (1.0f / TAU) * (Ix_term1 - Ix_term2);

            auto G = [&](float V) {
                return (V / Ron + (1.0f - V) / Roff);
            };
            float Imem = V1 * G(V2);

            return MemTunnerNeuron(Imem, Ix);
        }


        /**
         * Evaluates time derivatives for the memristor neuron.
         * @param state_in Current state tuple (Vc, XSV).
         * @param ddt_out Output derivatives.
         * @param I_syn Input synaptic current.
         * @param t Current simulation time (unused).
         */
        __host__ __device__ inline
            void operator()(const MemTunnerNeuron& state_in, MemTunnerNeuron& ddt_out, float I_syn, float t) const {
            float X_1 = thrust::get<0>(state_in); // Vc
            float X_2 = thrust::get<1>(state_in); // XSV


            float Vc = X_1;
            float XSV = X_2;

            float Iin = I_syn;// *1e-7f;

            float Vd = Vc + U1;
            float Vm = Vc + U2;

            float Id = GI403(Vd);
            MemTunnerNeuron Im_Ix = AND_TS(Vm, XSV);
            float Im = thrust::get<0>(Im_Ix); // Imem
            float Ix = thrust::get<1>(Im_Ix); // dXSV

            float dVc_unscaled = (Iin - Im - Id) / C1_scaled;
            float dXSV_unscaled = Ix;                

            float dX1 = 1e-5f * dVc_unscaled;
            float dX2 = 1e-5f * dXSV_unscaled;

            thrust::get<0>(ddt_out) = dX1;
            thrust::get<1>(ddt_out) = dX2;
        }
    };

}