# GILLIES Vulkan: GPU-Accelerated Shape Execution

**19 billion shape operations per second on NVIDIA Thor GPU**

## Overview

GILLIES (Geometric Instruction Language Layer In Every System) is the GPU execution engine for TriX shapes. When CUDA proved incompatible with Thor's pre-production driver, we bypassed it entirely and built a Vulkan compute pipeline.

The result: **19.4 billion XOR shape evaluations per second**.

## The Discovery

Thor GPU runs NVIDIA's Blackwell architecture with a development driver (580.00 "TempVersion"). Standard CUDA 12.8 fails with `CUDA_ERROR_NOT_SUPPORTED` on `cuInit()`.

But Vulkan works perfectly. Same GPU, different path to the silicon.

```
CUDA cuInit()           → CUDA_ERROR_NOT_SUPPORTED (broken)
Vulkan vkCreateInstance → VK_SUCCESS (works!)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GILLIES Vulkan                        │
├─────────────────────────────────────────────────────────────┤
│  SPIR-V Shader                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  XOR Shape: out[i] = a[i] + b[i] - 2.0 * a[i] * b[i] │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Compute Pipeline                                    │
│  • VkInstance → VkDevice → VkQueue                         │
│  • VkShaderModule (SPIR-V bytecode)                        │
│  • VkComputePipeline                                        │
│  • VkDescriptorSet (buffer bindings)                       │
│  • VkCommandBuffer → vkCmdDispatch                         │
├─────────────────────────────────────────────────────────────┤
│  Thor GPU (Blackwell)                                       │
│  • 2048 CUDA cores @ 1.1 GHz                               │
│  • Unified memory architecture                              │
│  • Vulkan 1.3 support                                      │
└─────────────────────────────────────────────────────────────┘
```

## Benchmark Results

Tested on NVIDIA Tegra Thor (Blackwell architecture):

| Elements | GPU (M ops/s) | CPU (M ops/s) | GPU Speedup |
|----------|---------------|---------------|-------------|
| 64K      | 1,509         | 1,057         | 1.4x        |
| 256K     | 5,598         | 1,150         | 4.9x        |
| 1M       | 12,217        | 1,261         | 9.7x        |
| 4M       | 17,247        | 1,266         | 13.6x       |
| **16M**  | **19,440**    | **1,245**     | **15.6x**   |

### Verification

Three consecutive runs at 16M elements:

```
Run 1: 19.6 B ops/sec (15.7x speedup)
Run 2: 19.1 B ops/sec (15.2x speedup)
Run 3: 19.4 B ops/sec (15.5x speedup)
```

All results verified correct against CPU reference implementation.

## The Full Stack

```
Layer              Performance      Speedup vs HSOS
─────────────────────────────────────────────────────
HSOS (Python)      3.2 M ops/sec    1x (baseline)
GILLIES (C)        900 M ops/sec    280x
GILLIES (Vulkan)   19,400 M ops/sec 6,000x
```

## Shape Mathematics

The XOR shape is a polynomial that implements fuzzy XOR:

```
XOR(a, b) = a + b - 2ab
```

Truth table verification:
| a | b | XOR(a,b) |
|---|---|----------|
| 0 | 0 | 0        |
| 0 | 1 | 1        |
| 1 | 0 | 1        |
| 1 | 1 | 0        |

This polynomial works identically whether executed by:
- Python interpreter (HSOS)
- C compiler (GILLIES)
- GPU shader (GILLIES Vulkan)

**The shape is substrate-independent.**

## SPIR-V Shader

The XOR shape compiles to SPIR-V assembly:

```spirv
; XOR Compute Shader
; out[i] = a[i] + b[i] - 2.0 * a[i] * b[i]

OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 256 1 1

; ... buffer bindings ...

%main = OpFunction %void None %funcType
%entry = OpLabel

  ; Get thread index
  %gid = OpLoad %uint %gl_GlobalInvocationID

  ; Load a[idx] and b[idx]
  %a = OpLoad %float %ptrA
  %b = OpLoad %float %ptrB

  ; Compute XOR: a + b - 2*a*b
  %sum   = OpFAdd %float %a %b      ; a + b
  %prod  = OpFMul %float %a %b      ; a * b
  %twice = OpFMul %float %2.0 %prod ; 2 * a * b
  %result = OpFSub %float %sum %twice ; (a+b) - 2ab

  ; Store result
  OpStore %ptrOut %result

OpReturn
OpFunctionEnd
```

## Building

### Prerequisites

```bash
# Vulkan SDK
apt-get install libvulkan-dev

# SPIR-V tools (for shader compilation)
apt-get install spirv-tools
```

### Compile

```bash
cd /tmp/cuda_gillies

# Compile SPIR-V shader
spirv-as xor_shader.spvasm -o xor_shader.spv
spirv-val xor_shader.spv  # Validate

# Build benchmark
gcc -O3 -o gillies_vulkan_bench gillies_vulkan_bench.c -lvulkan -lm

# Run
./gillies_vulkan_bench
```

## Files

```
/tmp/cuda_gillies/
├── gillies_vulkan_shapes.c   # Full pipeline with verification
├── gillies_vulkan_bench.c    # Scale benchmark
├── xor_shader.spvasm         # SPIR-V assembly source
├── xor_shader.spv            # Compiled SPIR-V binary
└── xor_shader.comp           # GLSL source (reference)
```

## Why This Matters

1. **CUDA Independence**: Thor's CUDA driver is broken, but we don't need it. Vulkan provides direct GPU access.

2. **Substrate Independence**: The XOR shape `a + b - 2ab` is pure mathematics. It computes identically on CPU, GPU, FPGA, or any other substrate.

3. **Performance**: 19 billion ops/sec proves that shape-based computation scales to GPU-class performance.

4. **TriX Architecture Validation**: Shapes compile naturally to parallel compute shaders. Each GPU thread evaluates one shape invocation independently.

## Future Work

- [ ] AND shape: `out = a * b`
- [ ] OR shape: `out = a + b - a*b`
- [ ] NOT shape: `out = 1 - a`
- [ ] Multi-shape pipelines
- [ ] Frozen network execution on GPU
- [ ] Dynamic shape dispatch

## References

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/specs/1.3/html/)
- [SPIR-V Specification](https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html)
- TriX Frozen Shapes: `/workspace/ZOR/src/trix/nn/frozen_shapes.py`

---

*"The shape doesn't care about the substrate."*

*GILLIES Vulkan - December 2025*
