---
generator: doxide
---


# SparsePropagator

**class SparsePropagator**

RAII wrapper for cuSPARSE SpMV propagation with prebuilt CSR matrices.


## Functions

| Name | Description |
| ---- | ----------- |
| [init](#init) | Initializes the cuSPARSE handle if it has not been created. |
| [destroy](#destroy) | Releases allocated matrices and destroys the cuSPARSE handle. |
| [buildCSR](#buildCSR) | Build CSR from (Xn, X, W) triplets. |
| [propagate](#propagate) | Propagate preLayer state -> postLayer input using a named CSR. |

## Function Details

### buildCSR<a name="buildCSR"></a>
!!! function "void buildCSR(const thrust::device_vector&lt;int&gt;&amp; X, const thrust::device_vector&lt;int&gt;&amp; Xn, const thrust::device_vector&lt;float&gt;&amp; W, const std::string&amp; name, int num_pre, int num_post, cudaStream_t stream = 0)"

    Build CSR from (Xn, X, W) triplets.
    Build CSR from (Xn, X, W) triplets.
    
    :material-location-enter: `X`
    :    Presynaptic indices.
        
    :material-location-enter: `Xn`
    :    Postsynaptic indices.
        
    :material-location-enter: `W`
    :    Connection weights.
        
    :material-location-enter: `name`
    :    Key used to reference the built matrix.
        
    :material-location-enter: `num_pre`
    :    Number of presynaptic neurons.
        
    :material-location-enter: `num_post`
    :    Number of postsynaptic neurons.
        
    :material-location-enter: `stream`
    :    CUDA stream for preprocessing.
    

### destroy<a name="destroy"></a>
!!! function "void destroy()"

    Releases allocated matrices and destroys the cuSPARSE handle.
    

### init<a name="init"></a>
!!! function "void init()"

    Initializes the cuSPARSE handle if it has not been created.
    

### propagate<a name="propagate"></a>
!!! function "template&lt;size_t stateVar, typename PreStateTypes, typename PostStateTypes&gt; void propagate(genericLayer&lt;PreStateTypes&gt;&amp; preLayer, genericLayer&lt;PostStateTypes&gt;&amp; postLayer, const std::string&amp; name, InputBehavior behavior = InputBehavior::INPUT_OVERRIDE, cudaStream_t stream = 0)"

    Propagate preLayer state -> postLayer input using a named CSR.
    Propagate preLayer state -> postLayer input using a named CSR.
        
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
        
    :material-location-enter: `name`
    :    Key of the CSR matrix to use.
        
    :material-location-enter: `behavior`
    :    Input accumulation behavior.
        
    :material-location-enter: `stream`
    :    CUDA stream for the SpMV call.
    

