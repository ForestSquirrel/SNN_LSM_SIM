#pragma once
/**
 * @file balance3.h
 * @brief Utility to factor an integer into a balanced 3D grid.
 */
#include <tuple>
#include <cmath>
#include <stdexcept>

/**
 * @brief Factor N into three integers whose product equals N and whose values
 *        are as balanced as possible.
 */
inline std::tuple<int, int, int> balanced3(int N) {
    if (N == 0) return { 0, 0, 0 };  // trivial edge case

    auto icbrt = [](int x) { return (int)std::cbrt((double)x); };
    auto isqrt = [](int x) { return (int)std::sqrt((double)x); };

    int a = 1;
    for (int t = icbrt(N); t >= 1; --t) {
        if (N % t == 0) { a = t; break; }
    }

    int M = N / a;

    int b = 1;
    for (int s = isqrt(M); s >= 1; --s) {
        if (M % s == 0) { b = s; break; }
    }

    int c = M / b;

    // sort ascending
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);

    // sanity check
    if (a * b * c != N) {
        throw std::runtime_error("balanced3() sanity check failed: a*b*c != N");
    }

    return std::make_tuple(a, b, c);
}
