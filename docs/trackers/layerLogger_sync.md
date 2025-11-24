---
generator: doxide
---


# layerLogger_sync

**template&lt;typename Layer&gt; class layerLogger_sync**

Synchronous logger optionally buffering data before writing to CSV.

:material-code-tags: `Layer`
:    Layer type exposing state_tuple_type and input buffers.


## Functions

| Name | Description |
| ---- | ----------- |
| [layerLogger_sync](#layerLogger_sync) | Constructs the synchronous logger. |
| [start](#start) | Opens output streams and prepares internal buffers. |
| [write](#write) | Copies current layer state and writes or buffers it. |
| [stop](#stop) | Flushes buffered data and closes files. |

## Function Details

### layerLogger_sync<a name="layerLogger_sync"></a>
!!! function "layerLogger_sync(Layer&amp; layer, const std::string&amp; name, Mode mode = ACCUMULATE_AND_FINALIZE)"

    Constructs the synchronous logger.
    
    :material-location-enter: `layer`
    :    Layer to observe.
        
    :material-location-enter: `name`
    :    Base filename for CSV outputs.
        
    :material-location-enter: `mode`
    :    Logging mode controlling buffering behavior.
    

### start<a name="start"></a>
!!! function "void start()"

    Opens output streams and prepares internal buffers.
    

### stop<a name="stop"></a>
!!! function "void stop()"

    Flushes buffered data and closes files.
    

### write<a name="write"></a>
!!! function "void write(int step)"

    Copies current layer state and writes or buffers it.
        
    :material-location-enter: `step`
    :    Current simulation step index.
    

