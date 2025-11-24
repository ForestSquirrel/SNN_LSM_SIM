---
generator: doxide
---


# MemTunnerNeuron_RHS

**struct MemTunnerNeuron_RHS**

Right-hand side functor for the memristor tunneling neuron model.

:material-eye-outline: **See**
:    MemTunnerNeuron


## Operators

| Name | Description |
| ---- | ----------- |
| [operator()](#operator_u0028_u0029) | Evaluates time derivatives for the memristor neuron. |

## Functions

| Name | Description |
| ---- | ----------- |
| [GI403](#GI403) | Computes the diode/tunnel current contribution. |
| [AND_TS](#AND_TS) | Calculates memristor branch current and state derivative. |

## Operator Details

### operator()<a name="operator_u0028_u0029"></a>

!!! function "__host__ __device__ inline void operator()(const MemTunnerNeuron&amp; state_in, MemTunnerNeuron&amp; ddt_out, float I_syn, float t) const"

    Evaluates time derivatives for the memristor neuron.
        
    :material-location-enter: `state_in`
    :    Current state tuple (Vc, XSV).
        
    :material-location-enter: `ddt_out`
    :    Output derivatives.
        
    :material-location-enter: `I_syn`
    :    Input synaptic current.
        
    :material-location-enter: `t`
    :    Current simulation time (unused).
    

## Function Details

### AND_TS<a name="AND_TS"></a>
!!! function "__host__ __device__ inline MemTunnerNeuron AND_TS(float V1, float V2) const"

    Calculates memristor branch current and state derivative.
        
    :material-location-enter: `V1`
    :    Membrane voltage.
        
    :material-location-enter: `V2`
    :    State variable representing conductance fraction.
        
    :material-keyboard-return: **Return**
    :    Pair of memristor current and state change.
    

### GI403<a name="GI403"></a>
!!! function "__host__ __device__ inline float GI403(float e) const"

    Computes the diode/tunnel current contribution.
    
    :material-location-enter: `e`
    :    Diode voltage.
        
    :material-keyboard-return: **Return**
    :    Combined current through the device.
    

