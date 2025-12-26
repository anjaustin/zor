# GILLIES Vulkan Runtime Specification

**Automatic Memory Management for Shape-Based Computation**

Version: 0.1.0
Date: December 25, 2025

---

## Overview

The GILLIES Vulkan Runtime provides automatic GPU memory management for TriX Fabric topologies. Developers define computation as shapes and connections; the runtime handles buffer allocation, memory pooling, and lifetime management.

**Goal**: Make Vulkan compute as easy as Python, with C performance.

```
Developer writes:     Fabric (topology + shapes)
Runtime handles:      Memory allocation, pooling, synchronization
Result:               18B+ ops/sec without manual memory management
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│                                                                              │
│    fabric = load_fdl("network.fdl")                                         │
│    runtime = GilliesVulkanRuntime(fabric)                                   │
│    result = runtime.execute(inputs)                                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         GILLIES VULKAN RUNTIME                               │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Fabric Loader  │  │ Memory Planner  │  │    Execution Engine         │  │
│  │                 │  │                 │  │                             │  │
│  │ • Parse FDL     │  │ • Analyze DAG   │  │ • Compile pipelines        │  │
│  │ • Build DAG     │  │ • Compute sizes │  │ • Bind descriptors         │  │
│  │ • Validate      │  │ • Pool buffers  │  │ • Dispatch compute         │  │
│  │                 │  │ • Track lifetime│  │ • Synchronize              │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬───────────────┘  │
│           │                    │                         │                   │
│           ▼                    ▼                         ▼                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        MEMORY MANAGER                                 │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Buffer Pool │  │ Descriptor  │  │  Lifetime   │  │  Defrag     │  │   │
│  │  │             │  │    Pool     │  │  Tracker    │  │  Engine     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                              VULKAN API                                      │
│                                                                              │
│    vkAllocateMemory, vkCreateBuffer, vkCmdDispatch, ...                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Fabric Loader

Parses FDL and constructs the computation DAG.

```c
typedef struct {
    uint32_t processor_count;
    Processor* processors;
    uint32_t connection_count;
    Connection* connections;
    uint32_t export_count;
    uint32_t* export_indices;
} FabricDAG;

// Load from FDL file
FabricDAG* fabric_load(const char* fdl_path);

// Load from FDL string
FabricDAG* fabric_parse(const char* fdl_source);

// Validate DAG (no cycles, all shapes exist, etc.)
bool fabric_validate(FabricDAG* dag);
```

**Processor Structure**:
```c
typedef struct {
    char signature[64];       // Hierarchical address
    char shape_name[32];      // Shape type (xor, and, matmul, etc.)
    ShapeID shape_id;         // Compiled shape index
    uint32_t input_count;
    uint32_t output_count;
    BufferID* input_buffers;  // Assigned by Memory Planner
    BufferID* output_buffers;
} Processor;
```

**Connection Structure**:
```c
typedef struct {
    uint32_t source_processor;
    uint32_t source_output;
    uint32_t target_processor;
    uint32_t target_input;
} Connection;
```

---

### 2. Memory Planner

Analyzes the DAG to compute optimal memory layout.

#### 2.1 Buffer Size Computation

Each shape has known input/output sizes:

```c
typedef struct {
    const char* name;
    uint32_t input_count;
    uint32_t output_count;
    size_t input_sizes[MAX_INPUTS];   // Bytes per input
    size_t output_sizes[MAX_OUTPUTS]; // Bytes per output
} ShapeDescriptor;

// Built-in shapes
ShapeDescriptor SHAPES[] = {
    {"xor",        2, 1, {4, 4}, {4}},           // 2 floats in, 1 float out
    {"and",        2, 1, {4, 4}, {4}},
    {"matmul",     3, 1, {0, 0, 0}, {0}},        // Dynamic sizing
    {"full_adder", 3, 2, {4, 4, 4}, {4, 4}},
    // ...
};
```

#### 2.2 Lifetime Analysis

Determine when each buffer is live:

```c
typedef struct {
    BufferID buffer;
    uint32_t first_use;   // First processor that reads/writes
    uint32_t last_use;    // Last processor that reads
    bool is_export;       // Must persist until execution complete
} BufferLifetime;

// Compute lifetimes via topological traversal
BufferLifetime* compute_lifetimes(FabricDAG* dag);
```

#### 2.3 Buffer Pooling

Reuse memory for non-overlapping lifetimes:

```
Processor timeline:
  P0: [====]
  P1:       [====]
  P2:             [====]
  P3:       [==========]

Buffer assignment:
  B0: P0.out, P2.out  (lifetimes don't overlap)
  B1: P1.out          (overlaps with P3)
  B2: P3.out

Memory saved: 25% (4 logical buffers → 3 physical)
```

**Pooling Algorithm**:
```c
typedef struct {
    size_t size;
    VkBuffer buffer;
    VkDeviceMemory memory;
    bool in_use;
} PooledBuffer;

typedef struct {
    PooledBuffer* buffers;
    uint32_t buffer_count;
    size_t total_allocated;
    size_t peak_usage;
} BufferPool;

// Request a buffer (reuses if available)
BufferID pool_acquire(BufferPool* pool, size_t size);

// Release a buffer back to pool
void pool_release(BufferPool* pool, BufferID id);
```

#### 2.4 Memory Layout

Final memory plan:

```c
typedef struct {
    uint32_t physical_buffer_count;
    size_t total_memory;

    // Mapping: logical buffer → physical buffer + offset
    struct {
        uint32_t physical_index;
        size_t offset;
        size_t size;
    }* buffer_map;
} MemoryPlan;

MemoryPlan* plan_memory(FabricDAG* dag, BufferLifetime* lifetimes);
```

---

### 3. Execution Engine

Compiles shapes to SPIR-V and executes the DAG.

#### 3.1 Shape Compilation

Each shape compiles to a SPIR-V compute shader:

```c
typedef struct {
    ShapeID id;
    VkShaderModule shader;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descriptor_layout;
} CompiledShape;

// Pre-compiled shapes (built at runtime init)
CompiledShape* compile_shape(VkDevice device, ShapeDescriptor* desc);
```

**XOR Shape SPIR-V** (example):
```spirv
; xor.spvasm
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID

; Bindings
OpDecorate %input_a Binding 0
OpDecorate %input_b Binding 1
OpDecorate %output Binding 2

%main = OpFunction %void None %func_type
  %idx = OpLoad %uint %gl_GlobalInvocationID
  %a = OpAccessChain %ptr_float %input_a %idx
  %b = OpAccessChain %ptr_float %input_b %idx
  %va = OpLoad %float %a
  %vb = OpLoad %float %b

  ; XOR: a + b - 2ab
  %sum = OpFAdd %float %va %vb
  %prod = OpFMul %float %va %vb
  %two = OpConstant %float 2.0
  %twice = OpFMul %float %two %prod
  %result = OpFSub %float %sum %twice

  %out = OpAccessChain %ptr_float %output %idx
  OpStore %out %result
OpReturn
OpFunctionEnd
```

#### 3.2 Execution Schedule

Topologically sort processors, respecting dependencies:

```c
typedef struct {
    uint32_t* order;        // Processor indices in execution order
    uint32_t* barriers;     // Indices where barriers are needed
    uint32_t barrier_count;
} ExecutionSchedule;

ExecutionSchedule* schedule_execution(FabricDAG* dag);
```

**Barrier Insertion**:
```
P0 (xor) ──┐
           ├──► P2 (and) ──► P3 (or)
P1 (xor) ──┘

Schedule: [P0, P1, BARRIER, P2, BARRIER, P3]
```

#### 3.3 Command Buffer Recording

```c
void record_execution(
    VkCommandBuffer cmd,
    FabricDAG* dag,
    ExecutionSchedule* schedule,
    MemoryPlan* memory,
    CompiledShape* shapes
) {
    for (uint32_t i = 0; i < schedule->order_count; i++) {
        uint32_t proc_idx = schedule->order[i];
        Processor* proc = &dag->processors[proc_idx];
        CompiledShape* shape = &shapes[proc->shape_id];

        // Bind pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shape->pipeline);

        // Bind descriptors (input/output buffers)
        VkDescriptorSet desc = allocate_descriptor(proc, memory);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                shape->layout, 0, 1, &desc, 0, NULL);

        // Dispatch
        uint32_t workgroups = (proc->element_count + 255) / 256;
        vkCmdDispatch(cmd, workgroups, 1, 1);

        // Insert barrier if needed
        if (is_barrier_point(schedule, i)) {
            VkMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }
}
```

---

### 4. Memory Manager

Handles all Vulkan memory operations.

#### 4.1 Initialization

```c
typedef struct {
    VkDevice device;
    VkPhysicalDevice physical_device;

    // Memory type indices
    uint32_t device_local_index;
    uint32_t host_visible_index;

    // Pools
    BufferPool device_pool;     // GPU-only memory
    BufferPool staging_pool;    // CPU→GPU transfer
    BufferPool readback_pool;   // GPU→CPU transfer

    // Descriptor pool
    VkDescriptorPool descriptor_pool;
} MemoryManager;

MemoryManager* memory_manager_create(VkDevice device, VkPhysicalDevice phys);
```

#### 4.2 Buffer Allocation Strategy

```
Size Class        Strategy
──────────────────────────────────────
< 256 bytes       Sub-allocate from 4KB block
256B - 64KB       Sub-allocate from 1MB block
64KB - 16MB       Individual allocation, pooled
> 16MB            Individual allocation, dedicated
```

```c
typedef struct {
    size_t block_size;
    size_t used;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped;  // If host-visible
} MemoryBlock;

BufferID allocate_buffer(MemoryManager* mm, size_t size, BufferUsage usage);
```

#### 4.3 Transfer Operations

```c
// Upload data to GPU
void upload_buffer(MemoryManager* mm, BufferID dst, void* src, size_t size);

// Download data from GPU
void download_buffer(MemoryManager* mm, void* dst, BufferID src, size_t size);

// Async upload (returns fence)
VkFence upload_buffer_async(MemoryManager* mm, BufferID dst, void* src, size_t size);
```

---

## Public API

### Runtime Interface

```c
// Create runtime for a fabric
GilliesRuntime* gillies_create(const char* fdl_source);
GilliesRuntime* gillies_load(const char* fdl_path);

// Execute with inputs
typedef struct {
    const char* signature;
    float* data;
    size_t count;
} InputBinding;

typedef struct {
    const char* signature;
    float* data;
    size_t count;
} OutputBinding;

void gillies_execute(
    GilliesRuntime* rt,
    InputBinding* inputs,
    uint32_t input_count,
    OutputBinding* outputs,
    uint32_t output_count
);

// Async execution
GilliesFuture* gillies_execute_async(
    GilliesRuntime* rt,
    InputBinding* inputs,
    uint32_t input_count
);

void gillies_wait(GilliesFuture* future);
void gillies_get_outputs(GilliesFuture* future, OutputBinding* outputs, uint32_t count);

// Cleanup
void gillies_destroy(GilliesRuntime* rt);
```

### Example Usage

```c
#include "gillies_vulkan.h"

int main() {
    // Load fabric
    GilliesRuntime* rt = gillies_load("half_adder.fdl");

    // Prepare inputs
    float a[] = {1.0, 0.0, 1.0, 0.0};
    float b[] = {1.0, 1.0, 0.0, 0.0};

    InputBinding inputs[] = {
        {"/adder/xor", a, 4},
        {"/adder/and", b, 4},
    };

    // Prepare outputs
    float sum[4], carry[4];
    OutputBinding outputs[] = {
        {"/adder/xor", sum, 4},
        {"/adder/and", carry, 4},
    };

    // Execute
    gillies_execute(rt, inputs, 2, outputs, 2);

    // Results
    for (int i = 0; i < 4; i++) {
        printf("%g + %g = %g (carry %g)\n", a[i], b[i], sum[i], carry[i]);
    }

    gillies_destroy(rt);
    return 0;
}
```

---

## Memory Optimization Strategies

### 1. In-Place Operations

When input and output have same size and input isn't reused:

```
Before: A ──► [XOR] ──► B
        Buffers: A (4 bytes), B (4 bytes)

After:  A ──► [XOR] ──► A (in-place)
        Buffers: A (4 bytes)

Memory saved: 50%
```

### 2. Buffer Aliasing

Non-overlapping lifetimes share physical memory:

```c
typedef struct {
    BufferID logical;
    BufferID physical;
    size_t offset;
} BufferAlias;

// Compute optimal aliasing
BufferAlias* compute_aliases(BufferLifetime* lifetimes, uint32_t count);
```

### 3. Streaming Execution

For large fabrics, execute in chunks:

```
Full fabric: P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7

Chunk 1: P0 → P1 → P2 → P3  (execute, keep P3 output)
Chunk 2: P4 → P5 → P6 → P7  (execute with P3 as input)

Peak memory: 50% of full execution
```

### 4. Memory Defragmentation

Periodic compaction for long-running processes:

```c
typedef struct {
    float fragmentation_ratio;  // Free space / total allocated
    size_t largest_free_block;
    size_t total_free;
} FragmentationStats;

FragmentationStats get_fragmentation(MemoryManager* mm);

// Defragment if fragmentation > threshold
void defragment(MemoryManager* mm, float threshold);
```

---

## Performance Targets

Based on GILLIES Vulkan benchmarks (Thor GPU):

| Metric | Target | Measured |
|--------|--------|----------|
| Shape evaluation | 15B+ ops/sec | 18.8B ops/sec |
| Memory bandwidth | 200+ GB/sec | 228 GB/sec |
| Buffer allocation | < 1μs | TBD |
| Descriptor update | < 100ns | TBD |
| Execution overhead | < 5% | TBD |

---

## Error Handling

```c
typedef enum {
    GILLIES_SUCCESS = 0,
    GILLIES_ERROR_INVALID_FDL,
    GILLIES_ERROR_UNKNOWN_SHAPE,
    GILLIES_ERROR_CYCLE_DETECTED,
    GILLIES_ERROR_OUT_OF_MEMORY,
    GILLIES_ERROR_VULKAN_INIT_FAILED,
    GILLIES_ERROR_SHADER_COMPILATION,
    GILLIES_ERROR_EXECUTION_FAILED,
} GilliesError;

GilliesError gillies_get_error();
const char* gillies_error_string(GilliesError err);
```

---

## Future Extensions

### 1. Dynamic Shapes

Shapes with runtime-determined sizes (matmul, conv, etc.):

```c
typedef struct {
    ShapeID base_shape;
    uint32_t* dynamic_dims;
    uint32_t dim_count;
} DynamicShape;
```

### 2. Multi-GPU Support

Distribute fabric across multiple devices:

```c
typedef struct {
    uint32_t device_count;
    VkDevice* devices;
    // Partition strategy
    FabricPartition* partitions;
} MultiGPURuntime;
```

### 3. Sparse Execution

Only execute processors with changed inputs:

```c
typedef struct {
    uint64_t* dirty_mask;  // Bitfield of changed processors
} IncrementalState;

void gillies_execute_incremental(GilliesRuntime* rt, IncrementalState* state);
```

### 4. Automatic Differentiation

Compile reverse-mode AD from fabric topology:

```c
FabricDAG* compile_backward(FabricDAG* forward);
```

---

## File Format: FDL Binary

For fast loading, support binary FDL:

```
Header (16 bytes):
  Magic: "FDL\0" (4 bytes)
  Version: uint32
  Processor count: uint32
  Connection count: uint32

Processor table:
  [signature_offset, shape_id, input_count, output_count] × N

Connection table:
  [src_proc, src_out, dst_proc, dst_in] × M

String table:
  Null-terminated signatures
```

---

## References

- [Vulkan Memory Management](https://gpuopen.com/learn/vulkan-memory-management/)
- [VMA - Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [GILLIES Vulkan Implementation](../src/trix/native/vulkan/)
- [TriX Fabric Architecture](./EXECUTION_STACK.md)

---

*GILLIES Vulkan Runtime Specification - December 2025*
*"The shape doesn't care about the substrate. The runtime does."*
