---
generator: doxide
---


# Solvers

Collection of numerical solvers

## Types

| Name | Description |
| ---- | ----------- |
| [RK4_Step_Functor](RK4_Step_Functor.md) | Functor executing a single Runge–Kutta 4 integration step. |

## Functions

| Name | Description |
| ---- | ----------- |
| [apply_k_step](#apply_k_step) | Component-wise RK4 update for a single k-step. |

## Function Details

### apply_k_step<a name="apply_k_step"></a>
!!! function "template &lt;size_t I, typename State&gt; __host__ __device__ inline void apply_k_step(State&amp; next_state, const State&amp; current_state, const State&amp; k_step, float factor)"

    Component-wise RK4 update for a single k-step.
    
    :material-code-tags: `I`
    :    Current tuple index (compile-time recursion).
        
    :material-code-tags: `State`
    :    Tuple type storing state variables.
        
    :material-location-enter: `next_state`
    :    Output state receiving the update.
        
    :material-location-enter: `current_state`
    :    State at the beginning of the step.
        
    :material-location-enter: `k_step`
    :    Derivative estimate for the current stage.
        
    :material-location-enter: `factor`
    :    Scaling factor applied to k_step.
    

