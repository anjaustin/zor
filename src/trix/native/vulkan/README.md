# GILLIES Vulkan

GPU-accelerated shape execution via Vulkan compute.

## Performance

**19.4 billion XOR operations per second** on NVIDIA Thor GPU.

## Quick Start

```bash
# Build
make

# Run benchmark
./gillies_vulkan_bench

# Run with verification
./gillies_vulkan_shapes
```

## Files

| File | Description |
|------|-------------|
| `gillies_vulkan_shapes.c` | Full pipeline with correctness verification |
| `gillies_vulkan_bench.c` | Scale benchmark (64K to 16M elements) |
| `xor_shader.spvasm` | SPIR-V assembly for XOR shape |
| `xor_shader.spv` | Compiled SPIR-V binary |

## XOR Shape

```
out[i] = a[i] + b[i] - 2.0 * a[i] * b[i]
```

This polynomial implements fuzzy XOR and compiles to 4 GPU instructions:
- `OpFAdd` (a + b)
- `OpFMul` (a * b)
- `OpFMul` (2 * product)
- `OpFSub` (sum - twice_product)

## Requirements

- Vulkan 1.2+ capable GPU
- `libvulkan-dev`
- `spirv-tools` (for shader modification)

## See Also

- [Full Documentation](/workspace/ZOR/docs/GILLIES_VULKAN.md)
- [Frozen Shapes](/workspace/ZOR/src/trix/nn/frozen_shapes.py)
