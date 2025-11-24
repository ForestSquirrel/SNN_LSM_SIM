---
generator: doxide
---


# Utilities

Helper functions and utilities

## Functions

| Name | Description |
| ---- | ----------- |
| [balanced3](#balanced3) | Factorizes an integer into a balanced 3D lattice (a <= b <= c). |
| [load_device_vector_from_file](#load_device_vector_from_file) | Reads a device vector from a binary stream. |
| [print_device_vector](#print_device_vector) | Prints basic information about a device vector. |
| [print_nested_progress](#print_nested_progress) | Renders two-level nested progress bars in-place. |
| [print_progress](#print_progress) | Renders a single-line progress bar. |
| [save_device_vector_to_file](#save_device_vector_to_file) | Writes a device vector to a binary stream. |

## Function Details

### balanced3<a name="balanced3"></a>
!!! function "inline std::tuple&lt;int, int, int&gt; balanced3(int N)"

    Factorizes an integer into a balanced 3D lattice (a <= b <= c).
    
    :material-location-enter: `N`
    :    Number of elements to distribute.
        
    :material-keyboard-return: **Return**
    :    Tuple of three factors whose product equals N.
    

### load_device_vector_from_file<a name="load_device_vector_from_file"></a>
!!! function "bool load_device_vector_from_file(std::fstream&amp; fs, thrust::device_vector&lt;float&gt;&amp; vec)"

    Reads a device vector from a binary stream.
    
    :material-location-enter: `fs`
    :    Input file stream.
        
    :material-location-enter: `vec`
    :    Destination vector populated from disk.
        
    :material-keyboard-return: **Return**
    :    true on success.
    

### print_device_vector<a name="print_device_vector"></a>
!!! function "void print_device_vector(const thrust::device_vector&lt;float&gt;&amp; vec, const string&amp; name)"

    Prints basic information about a device vector.
    
    :material-location-enter: `vec`
    :    Vector to inspect.
        
    :material-location-enter: `name`
    :    Label used in the output.
    

### print_nested_progress<a name="print_nested_progress"></a>
!!! function "inline void print_nested_progress( int i_current, int i_total, int j_current, int j_total, int width = 30, const std::string&amp; outer_text = &quot;Overall&quot;, const std::string&amp; inner_text = &quot;Batch&quot; )"

    Renders two-level nested progress bars in-place.
    
    :material-location-enter: `i_current`
    :    Outer loop current iteration.
        
    :material-location-enter: `i_total`
    :    Outer loop total iterations.
        
    :material-location-enter: `j_current`
    :    Inner loop current iteration.
        
    :material-location-enter: `j_total`
    :    Inner loop total iterations.
        
    :material-location-enter: `width`
    :    Width of each bar in characters.
        
    :material-location-enter: `outer_text`
    :    Label for the outer bar.
        
    :material-location-enter: `inner_text`
    :    Label for the inner bar.
    

### print_progress<a name="print_progress"></a>
!!! function "inline void print_progress(int current, int total, int width = 50)"

    Renders a single-line progress bar.
    
    :material-location-enter: `current`
    :    Current progress count.
        
    :material-location-enter: `total`
    :    Total count representing 100%.
        
    :material-location-enter: `width`
    :    Width of the bar in characters.
    

### save_device_vector_to_file<a name="save_device_vector_to_file"></a>
!!! function "bool save_device_vector_to_file(std::fstream&amp; fs, const thrust::device_vector&lt;float&gt;&amp; vec)"

    Writes a device vector to a binary stream.
    
    :material-location-enter: `fs`
    :    Output file stream.
        
    :material-location-enter: `vec`
    :    Vector to serialize.
        
    :material-keyboard-return: **Return**
    :    true on success.
    

