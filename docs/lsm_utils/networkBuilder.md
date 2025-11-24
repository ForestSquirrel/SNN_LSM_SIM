---
generator: doxide
---


# networkBuilder

**class networkBuilder**

Generates reservoir connectivity, weights, and metadata for the LSM.


## Functions

| Name | Description |
| ---- | ----------- |
| [networkBuilder](#networkBuilder) | Constructs a network with randomized excitatory/inhibitory distribution and geometry. |
| [build](#build) | Builds connectivity, weights, and delays based on geometry. |
| [normalize_weights](#normalize_weights) | Normalizes weights by the number of incoming connections per target neuron. |
| [compute_EI_indices](#compute_EI_indices) | Populates lists of excitatory and inhibitory neuron indices. |

## Function Details

### build<a name="build"></a>
!!! function "void build( const Eigen::Vector3i&amp; resSize, const Eigen::Matrix2f&amp; w, float r0, const Eigen::Matrix2f&amp; k0, float f_inhibit, float tau, bool /&#42;show&#42;/, const std::vector&lt;int&gt;&amp; UC, const Eigen::Vector3i&amp; UC_dims, bool _normalize )"

    Builds connectivity, weights, and delays based on geometry.
        
    :material-location-enter: `resSize`
    :    Reservoir dimensions.
        
    :material-location-enter: `w`
    :    Weight matrix for E/I combinations.
        
    :material-location-enter: `r0`
    :    Spatial decay radius.
        
    :material-location-enter: `k0`
    :    Connection probability coefficients.
        
    :material-location-enter: `f_inhibit`
    :    Fraction of inhibitory neurons.
        
    :material-location-enter: `tau`
    :    Synaptic delay scaling.
        
    :material-location-enter: `show`
    :    Unused flag kept for compatibility.
        
    :material-location-enter: `UC`
    :    Optional user-defined pattern.
        
    :material-location-enter: `UC_dims`
    :    Dimensions of the user pattern.
        
    :material-location-enter: `_normalize`
    :    Whether to normalize outgoing weights per neuron.
    

### compute_EI_indices<a name="compute_EI_indices"></a>
!!! function "void compute_EI_indices()"

    Populates lists of excitatory and inhibitory neuron indices.
    

### networkBuilder<a name="networkBuilder"></a>
!!! function "networkBuilder( const Eigen::Vector3i&amp; resSize = Eigen::Vector3i(3, 3, 5), const Eigen::Matrix2f&amp; w = (Eigen::Matrix2f() &lt;&lt; 3, 6, -2, -2).finished(), float r0 = 2.0f, const Eigen::Matrix2f&amp; k0 = (Eigen::Matrix2f() &lt;&lt; 0.45f, 0.3f, 0.6f, 0.15f).finished(), float f_inhibit = 0.2f, float tau = 1e-3f, bool show = false,  // ignored const std::vector&lt;int&gt;&amp; UC = {}, const Eigen::Vector3i&amp; UC_dims = Eigen::Vector3i(0, 0, 0), bool _normalize = true )"

    Constructs a network with randomized excitatory/inhibitory distribution and geometry.
    
    :material-location-enter: `resSize`
    :    Reservoir dimensions.
        
    :material-location-enter: `w`
    :    Weight matrix for E/I combinations.
        
    :material-location-enter: `r0`
    :    Spatial decay radius.
        
    :material-location-enter: `k0`
    :    Connection probability coefficients.
        
    :material-location-enter: `f_inhibit`
    :    Fraction of inhibitory neurons.
        
    :material-location-enter: `tau`
    :    Synaptic delay scaling.
        
    :material-location-enter: `show`
    :    Deprecated flag (ignored).
        
    :material-location-enter: `UC`
    :    Optional user-defined pattern.
        
    :material-location-enter: `UC_dims`
    :    Dimensions of the user pattern.
        
    :material-location-enter: `_normalize`
    :    Whether to normalize outgoing weights per neuron.
    

### normalize_weights<a name="normalize_weights"></a>
!!! function "std::vector&lt;float&gt; normalize_weights( const std::vector&lt;int&gt;&amp; X, const std::vector&lt;int&gt;&amp; Xn, const std::vector&lt;float&gt;&amp; W)"

    Normalizes weights by the number of incoming connections per target neuron.
        
    :material-location-enter: `X`
    :    Source indices for each connection.
        
    :material-location-enter: `Xn`
    :    Destination indices for each connection.
        
    :material-location-enter: `W`
    :    Unnormalized weights.
        
    :material-keyboard-return: **Return**
    :    Weight vector scaled per destination neuron.
    

