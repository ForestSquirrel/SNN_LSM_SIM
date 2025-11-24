---
generator: doxide
---


# Deprecated Propagators

Outdated signal propagation components

## Types

| Name | Description |
| ---- | ----------- |
| [InputBehavior](InputBehavior.md) | Determines how computed inputs are combined with existing buffers. |

## Functions

| Name | Description |
| ---- | ----------- |
| [dense_p](#dense_p) | Dense matrix propagation using cuBLAS sgemv. |
| [forward_p](#forward_p) | One-to-one propagation (diagonal weight vector). |
| [sparse_p](#sparse_p) | Sparse propagation using explicit edge lists. |

## Function Details

### dense_p<a name="dense_p"></a>
!!! function "template&lt;size_t stateVar, typename PreStateTypes, typename PostStateTypes&gt; void dense_p( genericLayer&lt;PreStateTypes&gt;&amp; preLayer, genericLayer&lt;PostStateTypes&gt;&amp; postLayer, const thrust::device_vector&lt;float&gt;&amp; W_flat, // column-major (N_post × N_pre) cublasHandle_t handle, InputBehavior behavior = INPUT_OVERRIDE, cudaStream_t stream = 0 )"

    Dense matrix propagation using cuBLAS sgemv.
    
    :material-code-tags: `stateVar`
    :    Index of the presynaptic state variable to propagate.
        
    :material-code-tags: `PreStateTypes`
    :    Tuple type for the presynaptic layer.
        
    :material-code-tags: `PostStateTypes`
    :    Tuple type for the postsynaptic layer.
        
    :material-location-enter: `preLayer`
    :    Source layer.
        
    :material-location-enter: `postLayer`
    :    Destination layer.
        
    :material-location-enter: `W_flat`
    :    Column-major weight matrix (N_post x N_pre).
        
    :material-location-enter: `handle`
    :    cuBLAS handle.
        
    :material-location-enter: `behavior`
    :    Input accumulation behavior.
        
    :material-location-enter: `stream`
    :    CUDA stream for execution.
    

### forward_p<a name="forward_p"></a>
!!! function "template&lt;size_t stateVar, typename PreStateTypes, typename PostStateTypes&gt; void forward_p( genericLayer&lt;PreStateTypes&gt;&amp; preLayer, genericLayer&lt;PostStateTypes&gt;&amp; postLayer, const thrust::device_vector&lt;float&gt;&amp; W, InputBehavior behavior = INPUT_OVERRIDE, cudaStream_t stream = 0 )"

    One-to-one propagation (diagonal weight vector).
    
    :material-code-tags: `stateVar`
    :    Index of the presynaptic state variable to propagate.
        
    :material-code-tags: `PreStateTypes`
    :    Tuple type for the presynaptic layer.
        
    :material-code-tags: `PostStateTypes`
    :    Tuple type for the postsynaptic layer.
        
    :material-location-enter: `preLayer`
    :    Source layer.
        
    :material-location-enter: `postLayer`
    :    Destination layer.
        
    :material-location-enter: `W`
    :    Weight vector aligned with neuron indices.
        
    :material-location-enter: `behavior`
    :    Input accumulation behavior.
        
    :material-location-enter: `stream`
    :    CUDA stream for execution.
    

### sparse_p<a name="sparse_p"></a>
!!! function "template&lt;size_t stateVar, typename PreStateTypes, typename PostStateTypes&gt; void sparse_p( genericLayer&lt;PreStateTypes&gt;&amp; preLayer, genericLayer&lt;PostStateTypes&gt;&amp; postLayer, const thrust::device_vector&lt;int&gt;&amp; X,    // presynaptic indices const thrust::device_vector&lt;int&gt;&amp; Xn,   // postsynaptic indices const thrust::device_vector&lt;float&gt;&amp; W,  // weights InputBehavior behavior = INPUT_OVERRIDE, cudaStream_t stream = 0 )"

    Sparse propagation using explicit edge lists.
    
    :material-code-tags: `stateVar`
    :    Index of the presynaptic state variable to propagate.
        
    :material-code-tags: `PreStateTypes`
    :    Tuple type for the presynaptic layer.
        
    :material-code-tags: `PostStateTypes`
    :    Tuple type for the postsynaptic layer.
        
    :material-location-enter: `preLayer`
    :    Source layer.
        
    :material-location-enter: `postLayer`
    :    Destination layer.
        
    :material-location-enter: `X`
    :    Presynaptic indices.
        
    :material-location-enter: `Xn`
    :    Postsynaptic indices.
        
    :material-location-enter: `W`
    :    Connection weights.
        
    :material-location-enter: `behavior`
    :    Input accumulation behavior.
        
    :material-location-enter: `stream`
    :    CUDA stream for execution.
    

