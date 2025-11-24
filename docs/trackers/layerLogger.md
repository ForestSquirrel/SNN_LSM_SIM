---
generator: doxide
---


# layerLogger

**template&lt;typename Layer&gt; class layerLogger**

Asynchronous logger that records layer states and inputs to CSV files.

:material-code-tags: `Layer`
:    Layer type exposing state_tuple_type and input buffers.


## Functions

| Name | Description |
| ---- | ----------- |
| [layerLogger](#layerLogger) | Constructs the logger with simulation and IO streams. |
| [start](#start) | Allocates buffers and opens CSV outputs. |
| [write](#write) | Schedules a logging operation for the given simulation step. |
| [stop](#stop) | Flushes and closes files, ensuring pending copies finish. |

## Function Details

### layerLogger<a name="layerLogger"></a>
!!! function "layerLogger(Layer&amp; layer, cudaStream_t streamSim, cudaStream_t streamIO, const std::string&amp; name)"

    Constructs the logger with simulation and IO streams.
    
    :material-location-enter: `layer`
    :    Layer to observe.
        
    :material-location-enter: `streamSim`
    :    CUDA stream used by the simulation.
        
    :material-location-enter: `streamIO`
    :    CUDA stream used for IO operations.
        
    :material-location-enter: `name`
    :    Base file name for generated CSVs.
    

### start<a name="start"></a>
!!! function "void start()"

    Allocates buffers and opens CSV outputs.
    

### stop<a name="stop"></a>
!!! function "void stop()"

    Flushes and closes files, ensuring pending copies finish.
    

### write<a name="write"></a>
!!! function "void write(int step)"

    Schedules a logging operation for the given simulation step.
        
    :material-location-enter: `step`
    :    Current simulation step index.
    

