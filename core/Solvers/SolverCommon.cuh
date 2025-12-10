#pragma once
#include <cuda_runtime.h>
#include "RK4_Solver.cuh"
#include "Heun_Solver.cuh"

#pragma once

#if defined(SOLVER_TYPE_HEUN)
#include "Solvers/Heun_Solver.cuh"

template<typename RHS, typename State>
using Solver_Step_Functor = Heun_Step_Functor<RHS, State>;

template<typename State>
using Heterogeneous_Solver_Step_Functor = Heterogeneous_Heun_Step_Functor<State>;

#elif defined(SOLVER_TYPE_RK4)
#include "Solvers/RK4_Solver.cuh"

template<typename RHS, typename State>
using Solver_Step_Functor = RK4_Step_Functor<RHS, State>;

template<typename State>
using Heterogeneous_Solver_Step_Functor = Heterogeneous_RK4_Step_Functor<State>;

#else

#include "Solvers/RK4_Solver.cuh"

template<typename RHS, typename State>
using Solver_Step_Functor = RK4_Step_Functor<RHS, State>;

template<typename State>
using Heterogeneous_Solver_Step_Functor = Heterogeneous_RK4_Step_Functor<State>;
#endif