---
generator: doxide
---


# RK4_Step_Functor

**template&lt;typename RHS, typename State&gt; struct RK4_Step_Functor**

Functor executing a single Runge–Kutta 4 integration step.

:material-code-tags: `RHS`
:    Right-hand-side functor computing derivatives.
    
:material-code-tags: `State`
:    Tuple type representing the neuron state.


## Operators

| Name | Description |
| ---- | ----------- |
| [operator()](#operator_u0028_u0029) | Applies RK4 integration to a zipped state/input tuple. |

## Functions

| Name | Description |
| ---- | ----------- |
| [RK4_Step_Functor](#RK4_Step_Functor) | Constructs the functor with simulation parameters. |

## Operator Details

### operator()<a name="operator_u0028_u0029"></a>

!!! function "template&lt;typename Tuple&gt; __host__ __device__ /&#42;&#42; &#42; Applies RK4 integration to a zipped state/input tuple. &#42; @param t_zip Tuple containing the current state and input value. &#42;/ void operator()(Tuple t_zip) const"

    Applies RK4 integration to a zipped state/input tuple.
        
    :material-location-enter: `t_zip`
    :    Tuple containing the current state and input value.
    

## Function Details

### RK4_Step_Functor<a name="RK4_Step_Functor"></a>
!!! function "RK4_Step_Functor(float t, float dt, RHS rhs)"

    Constructs the functor with simulation parameters.
    
    :material-location-enter: `t`
    :    Current simulation time.
        
    :material-location-enter: `dt`
    :    Time step size.
        
    :material-location-enter: `rhs`
    :    Right-hand-side functor instance.
    

