---
generator: doxide
---


# genericLayer

**template&lt;typename StateTypes&gt; // e.g. thrust::tuple&lt;float,float,float&gt; class genericLayer**

Generic layer wrapper holding neuron states and inputs.

:material-code-tags: `StateTypes`
:    Thrust tuple describing the per-neuron state variables.


## Functions

| Name | Description |
| ---- | ----------- |
| [make_state_tuple](#make_state_tuple) | Helper to construct the tuple of state vectors on the device. |
| [genericLayer](#genericLayer) | Constructs a layer with the given neuron count. |
| [size](#size) | Returns the number of neurons in the layer. |
| [state_vec](#state_vec) | Access a specific state vector by index. |
| [state_vec](#state_vec) | Access a specific state vector by index (const overload). |
| [make_zip_begin](#make_zip_begin) | Create a zip iterator over all state vectors (mutable). |
| [make_zip_end](#make_zip_end) | Create a zip iterator over all state vectors (mutable) - end. |
| [make_zip_begin](#make_zip_begin) | Create a zip iterator over all state vectors (const). |
| [make_zip_end](#make_zip_end) | Create a zip iterator over all state vectors (const) - end. |
| [full_zip_begin](#full_zip_begin) | Create a zip iterator over state and input (mutable). |
| [full_zip_end](#full_zip_end) | Create a zip iterator over state and input (mutable) - end. |
| [step](#step) | Advance the layer state by one integration step using RK4. |
| [input](#input) | Access the input buffer for modification. |
| [input](#input) | Access the input buffer (read-only). |
| [make_begin_tuple](#make_begin_tuple) | Build a tuple of begin iterators for all state vectors. |
| [make_end_tuple](#make_end_tuple) | Build a tuple of end iterators for all state vectors. |
| [make_cbegin_tuple](#make_cbegin_tuple) | Build a tuple of const begin iterators for all state vectors. |
| [make_cend_tuple](#make_cend_tuple) | Build a tuple of const end iterators for all state vectors. |

## Function Details

### full_zip_begin<a name="full_zip_begin"></a>
!!! function "auto full_zip_begin()"

    Create a zip iterator over state and input (mutable).
    
    The iterator dereferences to a tuple of
    (state tuple iterator, input value).
    
    
    :material-keyboard-return: **Return**
    :    Iterator to the beginning of the zipped state+input range.
    

### full_zip_end<a name="full_zip_end"></a>
!!! function "auto full_zip_end()"

    Create a zip iterator over state and input (mutable) - end.
    
    
    :material-keyboard-return: **Return**
    :    Iterator to the end of the zipped state+input range.
    

### genericLayer<a name="genericLayer"></a>
!!! function "explicit genericLayer(size_t N)"

    Constructs a layer with the given neuron count.
        
    :material-location-enter: `N`
    :    Number of neurons/states to allocate.
    

### input<a name="input"></a>
!!! function "thrust::device_vector&lt;float&gt;&amp; input() noexcept"

    Access the input buffer for modification.
        
    :material-keyboard-return: **Return**
    :    Reference to the device vector storing input currents.
    

!!! function "const thrust::device_vector&lt;float&gt;&amp; input() const noexcept"

    Access the input buffer (read-only).
        
    :material-keyboard-return: **Return**
    :    Const reference to the device vector storing input currents.
    

### make_begin_tuple<a name="make_begin_tuple"></a>
!!! function "template&lt;size_t... I&gt; auto make_begin_tuple(std::index_sequence&lt;I...&gt;)"

    Build a tuple of begin iterators for all state vectors.
    
    
    :material-code-tags: `I`
    :    Index pack over the elements of StateTypes.
        
    :material-keyboard-return: **Return**
    :    Thrust tuple of begin iterators.
    

### make_cbegin_tuple<a name="make_cbegin_tuple"></a>
!!! function "template&lt;size_t... I&gt; auto make_cbegin_tuple(std::index_sequence&lt;I...&gt;) const"

    Build a tuple of const begin iterators for all state vectors.
    
    
    :material-code-tags: `I`
    :    Index pack over the elements of StateTypes.
        
    :material-keyboard-return: **Return**
    :    Thrust tuple of const begin iterators.
    

### make_cend_tuple<a name="make_cend_tuple"></a>
!!! function "template&lt;size_t... I&gt; auto make_cend_tuple(std::index_sequence&lt;I...&gt;) const"

    Build a tuple of const end iterators for all state vectors.
    
    
    :material-code-tags: `I`
    :    Index pack over the elements of StateTypes.
        
    :material-keyboard-return: **Return**
    :    Thrust tuple of const end iterators.
    

### make_end_tuple<a name="make_end_tuple"></a>
!!! function "template&lt;size_t... I&gt; auto make_end_tuple(std::index_sequence&lt;I...&gt;)"

    Build a tuple of end iterators for all state vectors.
    
    
    :material-code-tags: `I`
    :    Index pack over the elements of StateTypes.
        
    :material-keyboard-return: **Return**
    :    Thrust tuple of end iterators.
    

### make_state_tuple<a name="make_state_tuple"></a>
!!! function "template&lt;size_t... I&gt; static auto make_state_tuple(size_t N, std::index_sequence&lt;I...&gt;)"

    Helper to construct the tuple of state vectors on the device.
    
    For each element type in StateTypes, creates a corresponding
    thrust::device_vector of size `N` initialized to zero.
    
    
    :material-code-tags: `I`
    :    Index pack over the elements of StateTypes.
        
    :material-location-enter: `N`
    :    Number of elements in each state vector.
        
    :material-keyboard-return: **Return**
    :    Tuple of device vectors matching StateTypes.
    

### make_zip_begin<a name="make_zip_begin"></a>
!!! function "auto make_zip_begin()"

    Create a zip iterator over all state vectors (mutable).
    
    Each element of the iterator is a thrust::tuple of references to
    the components of the neuron state.
    
    
    :material-keyboard-return: **Return**
    :    Iterator to the beginning of the zipped state range.
    

!!! function "auto make_zip_begin() const"

    Create a zip iterator over all state vectors (const).
    
    
    :material-keyboard-return: **Return**
    :    Const iterator to the beginning of the zipped state range.
    

### make_zip_end<a name="make_zip_end"></a>
!!! function "auto make_zip_end()"

    Create a zip iterator over all state vectors (mutable) - end.
    
    
    :material-keyboard-return: **Return**
    :    Iterator to the end of the zipped state range.
    

!!! function "auto make_zip_end() const"

    Create a zip iterator over all state vectors (const) - end.
    
    
    :material-keyboard-return: **Return**
    :    Const iterator to the end of the zipped state range.
    

### size<a name="size"></a>
!!! function "size_t size() const noexcept"

    Returns the number of neurons in the layer.
    

### state_vec<a name="state_vec"></a>
!!! function "template&lt;size_t I&gt; auto&amp; state_vec() noexcept"

    Access a specific state vector by index.
    
    
    :material-code-tags: `I`
    :    Index of the state component in the StateTypes tuple.
        
    :material-keyboard-return: **Return**
    :    Reference to the corresponding device vector.
    

!!! function "template&lt;size_t I&gt; const auto&amp; state_vec() const noexcept"

    Access a specific state vector by index (const overload).
    
    
    :material-code-tags: `I`
    :    Index of the state component in the StateTypes tuple.
        
    :material-keyboard-return: **Return**
    :    Const reference to the corresponding device vector.
    

### step<a name="step"></a>
!!! function "template&lt;typename RHS_Functor&gt; void step(const RHS_Functor&amp; rhs, float t, float dt, cudaStream_t stream = 0)"

    Advance the layer state by one integration step using RK4.
    
    Applies the provided RHS functor on all neurons in parallel using
    Thrust on the given CUDA stream.
    
    
    :material-code-tags: `RHS_Functor`
    :    Functor type providing the right-hand side of the ODE.
                            It must be callable from device code.
                            
    :material-eye-outline: **See**
    :    Solvers
        
    :material-location-enter: `rhs`
    :    Right-hand side functor.
        
    :material-location-enter: `t`
    :      Current time.
        
    :material-location-enter: `dt`
    :     Time step.
        
    :material-location-enter: `stream`
    :    CUDA stream to execute the operation on (default 0).
    

