---
generator: doxide
---


# neuronModels

Collection of neuron model state representations and right-hand side functors.


## Types

| Name | Description |
| ---- | ----------- |
| [FHN_RHS](FHN_RHS.md) | Right-hand side functor for the FitzHugh–Nagumo neuron dynamics. |
| [MemTunnerNeuron_RHS](MemTunnerNeuron_RHS.md) | Right-hand side functor for the memristor tunneling neuron model. |

## Type Aliases

| Name | Description |
| ---- | ----------- |
| [FHN](#FHN) | State tuple for the FitzHugh–Nagumo model (u, v). |
| [MemTunnerNeuron](#MemTunnerNeuron) | State tuple for the memristor-based tunneling neuron (Vc, XSV). |

## Type Alias Details

### FHN<a name="FHN"></a>

!!! typedef "using FHN = StateTuple&lt;float, float&gt;"

    State tuple for the FitzHugh–Nagumo model (u, v).
    

### MemTunnerNeuron<a name="MemTunnerNeuron"></a>

!!! typedef "using MemTunnerNeuron = StateTuple&lt;float, float&gt;"

    State tuple for the memristor-based tunneling neuron (Vc, XSV).
    

