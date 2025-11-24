---
generator: doxide
---


# Datasets

Dataset handling and processing

## Types

| Name | Description |
| ---- | ----------- |
| [MNIST](MNIST.md) | Minimal MNIST dataset loader supporting sorting and splitting utilities. |

## Functions

| Name | Description |
| ---- | ----------- |
| [swap_endian](#swap_endian) | Swaps byte order for 32-bit integers (big-endian <-> little-endian). |

## Function Details

### swap_endian<a name="swap_endian"></a>
!!! function "inline uint32_t swap_endian(uint32_t val)"

    Swaps byte order for 32-bit integers (big-endian <-> little-endian).
    
    :material-location-enter: `val`
    :    Value to swap.
        
    :material-keyboard-return: **Return**
    :    Reordered integer.
    

