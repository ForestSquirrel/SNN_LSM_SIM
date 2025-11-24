---
generator: doxide
---


# crossingsCounter

**template&lt;typename Layer, size_t I&gt; class crossingsCounter**

Counts threshold crossings for a specific state variable in a layer.

:material-code-tags: `Layer`
:    Layer type exposing state_tuple_type.
    
:material-code-tags: `I`
:    Index of the state variable to monitor.


## Functions

| Name | Description |
| ---- | ----------- |
| [crossingsCounter](#crossingsCounter) | Constructor for the crossingsCounter. |
| [count](#count) | Performs the crossing count for the current step. |
| [reset](#reset) | Resets the crossing counts to zero. |
| [destroy](#destroy) | Cleans up device memory. |

## Function Details

### count<a name="count"></a>
!!! function "void count(int step)"

    Performs the crossing count for the current step.
        
    :material-location-enter: `step`
    :    The current simulation step number.
    

### crossingsCounter<a name="crossingsCounter"></a>
!!! function "crossingsCounter(Layer&amp; layer, state_value_type threshold, CounterBehavior behavior = CounterBehavior::BELOW_THR)"

    Constructor for the crossingsCounter.
    
    :material-location-enter: `layer`
    :    The reference to the simulation layer.
        
    :material-location-enter: `threshold`
    :    The value a state must cross to be counted.
        
    :material-location-enter: `behavior`
    :    The type of crossing to count (BELOW_THR, ABOVE_THR, BIDIRECTIONAL).
    

### destroy<a name="destroy"></a>
!!! function "void destroy()"

    Cleans up device memory.
    

### reset<a name="reset"></a>
!!! function "void reset()"

    Resets the crossing counts to zero.
    

