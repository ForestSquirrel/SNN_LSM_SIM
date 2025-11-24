---
generator: doxide
---


# LSM Utilities

Utilities specific to Liquid State Machines

## Types

| Name | Description |
| ---- | ----------- |
| [SelectionMode](SelectionMode.md) | Defines the selection mode for the random sampling. |
| [networkBuilder](networkBuilder.md) | Generates reservoir connectivity, weights, and metadata for the LSM. |

## Functions

| Name | Description |
| ---- | ----------- |
| [loadLSM](#loadLSM) | Loads six Thrust device vectors from a binary configuration file. |
| [mapInputToLSM](#mapInputToLSM) | Maps input channels to reservoir neurons using random selection. |
| [saveLSM](#saveLSM) | Saves six Thrust device vectors to a binary configuration file. |

## Function Details

### loadLSM<a name="loadLSM"></a>
!!! function "bool loadLSM( string fileName, thrust::device_vector&lt;float&gt;&amp; ILSM_Xpre, thrust::device_vector&lt;float&gt;&amp; ILSM_Xpost, thrust::device_vector&lt;float&gt;&amp; ILSM_W, thrust::device_vector&lt;float&gt;&amp; LSM_Xpre, thrust::device_vector&lt;float&gt;&amp; LSM_Xpost, thrust::device_vector&lt;float&gt;&amp; LSM_W )"

    Loads six Thrust device vectors from a binary configuration file.
    
    
    :material-location-enter: `fileName`
    :    The path to the file.
        
    :material-location-enter: `ILSM_Xpre`
    :    Output vector 1 (modified by reference).
        
    :material-location-enter: `ILSM_Xpost`
    :    Output vector 2 (modified by reference).
        
    :material-location-enter: `ILSM_W`
    :    Output vector 3 (modified by reference).
        
    :material-location-enter: `LSM_Xpre`
    :    Output vector 4 (modified by reference).
        
    :material-location-enter: `LSM_Xpost`
    :    Output vector 5 (modified by reference).
        
    :material-location-enter: `LSM_W`
    :    Output vector 6 (modified by reference).
        
    :material-keyboard-return: **Return**
    :    true if successful, false otherwise.
    

### mapInputToLSM<a name="mapInputToLSM"></a>
!!! function "bool mapInputToLSM( int iN, const std::vector&lt;int&gt;&amp; indices_h, SelectionMode mode, thrust::device_vector&lt;int&gt;&amp; X_d, thrust::device_vector&lt;int&gt;&amp; Xn_d)"

    Maps input channels to reservoir neurons using random selection.
    
    :material-location-enter: `iN`
    :    Number of input connections to generate.
        
    :material-location-enter: `indices_h`
    :    Host vector of candidate neuron indices.
        
    :material-location-enter: `mode`
    :    SelectionMode specifying replacement behavior.
        
    :material-location-enter: `X_d`
    :    Output presynaptic indices (0..iN-1).
        
    :material-location-enter: `Xn_d`
    :    Output postsynaptic indices sampled from indices_h.
        
    :material-keyboard-return: **Return**
    :    true on success, false when constraints are violated.
    

### saveLSM<a name="saveLSM"></a>
!!! function "bool saveLSM( string fileName, thrust::device_vector&lt;float&gt; ILSM_Xpre, thrust::device_vector&lt;float&gt; ILSM_Xpost, thrust::device_vector&lt;float&gt; ILSM_W, thrust::device_vector&lt;float&gt; LSM_Xpre, thrust::device_vector&lt;float&gt; LSM_Xpost, thrust::device_vector&lt;float&gt; LSM_W )"

    Saves six Thrust device vectors to a binary configuration file.
    
    
    :material-location-enter: `fileName`
    :    The path to the file.
        
    :material-location-enter: `ILSM_Xpre`
    :    Input vector 1.
        
    :material-location-enter: `ILSM_Xpost`
    :    Input vector 2.
        
    :material-location-enter: `ILSM_W`
    :    Input vector 3.
        
    :material-location-enter: `LSM_Xpre`
    :    Input vector 4.
        
    :material-location-enter: `LSM_Xpost`
    :    Input vector 5.
        
    :material-location-enter: `LSM_W`
    :    Input vector 6.
        
    :material-keyboard-return: **Return**
    :    true if successful, false otherwise.
    

