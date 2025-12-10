#pragma once
#include <cuda_runtime.h>

namespace solver_utils {
    template <typename R, typename S>
    __host__ __device__
        auto apply_reset_if_exists(const R& r, S& s, int) -> decltype(r.reset(s), void()) {
        r.reset(s);
    }

    template <typename R, typename S>
    __host__ __device__
        void apply_reset_if_exists(const R&, S&, long) {
        // No-op
    }

    template <typename R, typename S>
    __host__ __device__
        void try_reset(const R& rhs, S& state) {
        apply_reset_if_exists(rhs, state, 0);
    }
}