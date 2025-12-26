# GILLIES Runtime: Design Document

**A Declarative GPU Compute Layer**

*Structure eliminates coordination.*

---

## 1. Design Principles

### 1.1 The Three Laws

1. **Topology is truth** — The DAG is the single source of all derived properties
2. **Declare, don't specify** — Developer says what, runtime derives how
3. **Invisible until needed** — Memory, barriers, descriptors hidden by default

### 1.2 Non-Goals

- Not a general Vulkan wrapper
- Not a graphics API
- Not a neural network framework
- Not trying to be everything

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              USER CODE                                      │
│                                                                             │
│    GilliesContext* ctx = gillies_init();                                   │
│    GilliesFabric* fab = gillies_compile(ctx, "network.fdl");               │
│    gillies_run(fab, inputs, outputs);                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           PUBLIC API (libgillies.h)                        │
│                                                                             │
│    gillies_init    gillies_compile    gillies_run    gillies_destroy       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              COMPILER                                       │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    PARSER    │  │   ANALYZER   │  │   PLANNER    │  │   EMITTER    │   │
│  │              │  │              │  │              │  │              │   │
│  │  FDL → AST   │─▶│  AST → DAG   │─▶│  DAG → Plan  │─▶│ Plan → VkCmd │   │
│  │              │  │  Validate    │  │  Buffers     │  │ Pipelines    │   │
│  │              │  │  Typecheck   │  │  Lifetimes   │  │ Descriptors  │   │
│  │              │  │              │  │  Schedule    │  │ Barriers     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              RUNTIME                                        │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    MEMORY    │  │  DESCRIPTOR  │  │   COMMAND    │  │    SYNC      │   │
│  │    SYSTEM    │  │    CACHE     │  │   RECORDER   │  │   MANAGER    │   │
│  │              │  │              │  │              │  │              │   │
│  │  Pools       │  │  Per-shape   │  │  Record once │  │  Fences      │   │
│  │  Aliasing    │  │  layouts     │  │  Submit many │  │  Semaphores  │   │
│  │  Streaming   │  │  Fast update │  │              │  │  Timeline    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           BACKEND: VULKAN                                   │
│                                                                             │
│    VkInstance  VkDevice  VkQueue  VkCommandBuffer  VkPipeline  VkBuffer    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Intermediate Representation

```c
// ═══════════════════════════════════════════════════════════════════════════
// COMPILER IR
// ═══════════════════════════════════════════════════════════════════════════

typedef uint32_t NodeID;
typedef uint32_t PortID;
typedef uint32_t BufferID;

// A port is an input or output slot on a node
typedef struct {
    NodeID      node;
    uint8_t     index;      // Which input/output (0, 1, 2...)
    uint8_t     is_output;  // 0 = input, 1 = output
    uint32_t    size;       // Bytes
} Port;

// A node is a processor in the fabric
typedef struct {
    NodeID      id;
    ShapeID     shape;
    char        signature[64];

    uint8_t     input_count;
    uint8_t     output_count;
    PortID      inputs[8];
    PortID      outputs[8];

    // Computed by analyzer
    uint32_t    depth;          // Topological depth
    uint32_t    schedule_order; // Execution order
} Node;

// An edge connects an output port to an input port
typedef struct {
    PortID      source;     // Output port
    PortID      target;     // Input port
} Edge;

// The complete DAG
typedef struct {
    char        name[64];

    Node*       nodes;
    uint32_t    node_count;

    Port*       ports;
    uint32_t    port_count;

    Edge*       edges;
    uint32_t    edge_count;

    NodeID*     exports;
    uint32_t    export_count;

    NodeID*     entries;
    uint32_t    entry_count;
} DAG;
```

### 3.2 Execution Plan

```c
// ═══════════════════════════════════════════════════════════════════════════
// EXECUTION PLAN (output of Planner)
// ═══════════════════════════════════════════════════════════════════════════

// Buffer allocation decision
typedef struct {
    BufferID    id;
    uint32_t    size;

    // Lifetime (node indices in schedule order)
    uint32_t    first_write;
    uint32_t    last_read;

    // Physical assignment (after aliasing)
    uint32_t    physical_buffer;
    uint32_t    offset;

    // Flags
    uint8_t     is_input;       // Fed from outside
    uint8_t     is_output;      // Read from outside
    uint8_t     is_aliased;     // Shares physical memory
} BufferPlan;

// A step in the execution schedule
typedef struct {
    NodeID      node;
    ShapeID     shape;

    BufferID*   input_buffers;
    BufferID*   output_buffers;

    uint8_t     needs_barrier_before;
    uint8_t     needs_barrier_after;
} ScheduleStep;

// Complete execution plan
typedef struct {
    // Memory plan
    BufferPlan* buffers;
    uint32_t    buffer_count;
    uint32_t    total_memory;
    uint32_t    physical_buffer_count;  // After aliasing

    // Execution schedule
    ScheduleStep* steps;
    uint32_t    step_count;

    // Barrier positions (indices into steps)
    uint32_t*   barriers;
    uint32_t    barrier_count;
} ExecutionPlan;
```

### 3.3 Runtime State

```c
// ═══════════════════════════════════════════════════════════════════════════
// RUNTIME STATE
// ═══════════════════════════════════════════════════════════════════════════

// Physical buffer on GPU
typedef struct {
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    uint32_t        size;
    void*           mapped;     // NULL if not host-visible
} PhysicalBuffer;

// Compiled shape (shader + pipeline)
typedef struct {
    ShapeID                 id;
    char                    name[32];
    VkShaderModule          shader;
    VkPipelineLayout        layout;
    VkPipeline              pipeline;
    VkDescriptorSetLayout   desc_layout;
    uint8_t                 input_count;
    uint8_t                 output_count;
} CompiledShape;

// Compiled fabric (ready to execute)
typedef struct {
    DAG*                dag;
    ExecutionPlan*      plan;

    // GPU resources
    PhysicalBuffer*     physical_buffers;
    uint32_t            physical_buffer_count;

    // Pre-recorded command buffer
    VkCommandBuffer     cmd;
    VkDescriptorPool    desc_pool;
    VkDescriptorSet*    desc_sets;  // One per step

    // Sync
    VkFence             fence;

    // Stats
    uint64_t            run_count;
    double              total_time_ms;
} GilliesFabric;

// Global context
typedef struct {
    VkInstance          instance;
    VkPhysicalDevice    physical_device;
    VkDevice            device;
    VkQueue             compute_queue;
    uint32_t            queue_family;
    VkCommandPool       cmd_pool;

    // Shape registry
    CompiledShape*      shapes;
    uint32_t            shape_count;

    // Memory properties
    uint32_t            device_local_type;
    uint32_t            host_visible_type;

    // Limits
    uint32_t            max_workgroup_size;
    uint32_t            max_buffer_size;
} GilliesContext;
```

---

## 4. Compiler Stages

### 4.1 Parser

**Input:** FDL source text
**Output:** AST

```c
typedef enum {
    AST_FABRIC,
    AST_PROCESSOR,
    AST_PROPERTY,
    AST_CONNECTION,
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    char        name[64];
    char        value[256];
    struct ASTNode* children;
    uint32_t    child_count;
    uint32_t    line;   // For error messages
} ASTNode;

// Parse FDL into AST
ASTNode* parse_fdl(const char* source, GilliesError* error);
```

**Grammar (simplified):**
```
fabric     := "fabric" STRING "{" (processor | export | entry)* "}"
processor  := "processor" IDENT "{" property* "}"
property   := IDENT ":" value
value      := STRING | IDENT | reference
reference  := IDENT "." IDENT
export     := "export:" IDENT
entry      := "entry:" IDENT
```

### 4.2 Analyzer

**Input:** AST
**Output:** DAG

```c
// Analyze AST, produce validated DAG
DAG* analyze(ASTNode* ast, GilliesContext* ctx, GilliesError* error);

// Analysis passes:
// 1. Resolve shape references (check shapes exist)
// 2. Build node table
// 3. Build port table (from shape signatures)
// 4. Resolve connections (name → port mapping)
// 5. Validate (no cycles, all inputs connected, types match)
// 6. Compute depths (topological analysis)
```

**Validation rules:**
- Every processor has a valid shape
- Every processor has a valid signature
- No cycles in the DAG
- All required inputs are connected
- Output types match input types on connections
- Signatures are unique within fabric

### 4.3 Planner

**Input:** DAG
**Output:** ExecutionPlan

```c
ExecutionPlan* plan(DAG* dag, GilliesContext* ctx);
```

**Planning passes:**

#### Pass 1: Buffer Sizing
```c
// For each port, compute buffer size from shape signature
for each port p in dag->ports:
    shape = get_shape(dag->nodes[p.node].shape)
    if p.is_output:
        p.size = shape.output_sizes[p.index]
    else:
        p.size = shape.input_sizes[p.index]
```

#### Pass 2: Lifetime Analysis
```c
// For each buffer, compute first write and last read
for each port p in dag->ports:
    if p.is_output:
        buf = new BufferPlan(p)
        buf.first_write = dag->nodes[p.node].schedule_order
        buf.last_read = max(schedule_order of all nodes reading this port)
```

#### Pass 3: Scheduling
```c
// Topological sort with depth-first traversal
// Group nodes at same depth for parallel execution
schedule = topological_sort(dag)
for i, node in enumerate(schedule):
    node.schedule_order = i
```

#### Pass 4: Barrier Placement
```c
// Insert barriers where a write must complete before a read
barriers = []
for each edge (src, dst) in dag->edges:
    if schedule_order(dst.node) > schedule_order(src.node) + 1:
        // Non-adjacent: need barrier
        barriers.add(schedule_order(src.node))
```

#### Pass 5: Buffer Aliasing
```c
// Graph coloring: non-overlapping lifetimes share memory
// Greedy algorithm: assign to first physical buffer that's free

physical_buffers = []
for buf in buffers sorted by first_write:
    assigned = false
    for phys in physical_buffers:
        if phys.free_at <= buf.first_write and phys.size >= buf.size:
            buf.physical_buffer = phys.id
            buf.offset = 0  // Or pack tightly
            phys.free_at = buf.last_read
            assigned = true
            break
    if not assigned:
        phys = new PhysicalBuffer(buf.size)
        physical_buffers.add(phys)
        buf.physical_buffer = phys.id
```

### 4.4 Emitter

**Input:** ExecutionPlan + DAG
**Output:** GilliesFabric (GPU-ready)

```c
GilliesFabric* emit(DAG* dag, ExecutionPlan* plan, GilliesContext* ctx);
```

**Emission steps:**

1. **Allocate physical buffers**
```c
for each phys in plan->physical_buffers:
    VkBufferCreateInfo info = {
        .size = phys.size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    };
    vkCreateBuffer(ctx->device, &info, NULL, &phys.buffer);
    // Allocate and bind memory...
```

2. **Create descriptor sets**
```c
for each step in plan->steps:
    shape = ctx->shapes[step.shape];
    VkDescriptorSetAllocateInfo alloc = {
        .descriptorPool = fabric->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &shape.desc_layout,
    };
    vkAllocateDescriptorSets(ctx->device, &alloc, &fabric->desc_sets[i]);

    // Bind buffers to descriptor set
    for j in 0..step.input_count:
        buf = resolve_buffer(step.input_buffers[j], plan);
        write_descriptor(fabric->desc_sets[i], j, buf);
    for j in 0..step.output_count:
        buf = resolve_buffer(step.output_buffers[j], plan);
        write_descriptor(fabric->desc_sets[i], step.input_count + j, buf);
```

3. **Record command buffer**
```c
vkBeginCommandBuffer(fabric->cmd, &begin_info);

for i, step in enumerate(plan->steps):
    shape = ctx->shapes[step.shape];

    // Bind pipeline
    vkCmdBindPipeline(fabric->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      shape.pipeline);

    // Bind descriptors
    vkCmdBindDescriptorSets(fabric->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shape.layout, 0, 1, &fabric->desc_sets[i], 0, NULL);

    // Dispatch
    uint32_t workgroups = (element_count + 255) / 256;
    vkCmdDispatch(fabric->cmd, workgroups, 1, 1);

    // Barrier if needed
    if step.needs_barrier_after:
        VkMemoryBarrier barrier = {
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(fabric->cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);

vkEndCommandBuffer(fabric->cmd);
```

---

## 5. Runtime Execution

### 5.1 Input/Output Binding

```c
void gillies_bind_input(GilliesFabric* fab, const char* signature,
                        const float* data, uint32_t count);

void gillies_bind_output(GilliesFabric* fab, const char* signature,
                         float* data, uint32_t count);
```

**Binding process:**
1. Find node by signature
2. Find entry input buffer
3. Copy data to staging buffer
4. Record transfer command (staging → device)

### 5.2 Execution

```c
void gillies_run(GilliesFabric* fab) {
    // Reset fence
    vkResetFences(ctx->device, 1, &fab->fence);

    // Submit
    VkSubmitInfo submit = {
        .commandBufferCount = 1,
        .pCommandBuffers = &fab->cmd,
    };
    vkQueueSubmit(ctx->compute_queue, 1, &submit, fab->fence);

    // Wait
    vkWaitForFences(ctx->device, 1, &fab->fence, VK_TRUE, UINT64_MAX);
}
```

### 5.3 Async Execution

```c
GilliesFuture* gillies_run_async(GilliesFabric* fab) {
    GilliesFuture* future = alloc_future();
    future->fence = fab->fence;

    vkResetFences(ctx->device, 1, &fab->fence);
    vkQueueSubmit(ctx->compute_queue, 1, &submit, fab->fence);

    return future;
}

bool gillies_poll(GilliesFuture* future) {
    return vkGetFenceStatus(ctx->device, future->fence) == VK_SUCCESS;
}

void gillies_wait(GilliesFuture* future) {
    vkWaitForFences(ctx->device, 1, &future->fence, VK_TRUE, UINT64_MAX);
}
```

---

## 6. Shape System

### 6.1 Built-in Shapes

```c
// Shape descriptor
typedef struct {
    const char*     name;
    const uint32_t* spirv;
    uint32_t        spirv_size;
    uint8_t         input_count;
    uint8_t         output_count;
    uint32_t        input_sizes[4];
    uint32_t        output_sizes[4];
} ShapeDescriptor;

// Built-in shapes
static const ShapeDescriptor BUILTIN_SHAPES[] = {
    {"xor",         xor_spirv,       sizeof(xor_spirv),       2, 1, {4,4},     {4}    },
    {"and",         and_spirv,       sizeof(and_spirv),       2, 1, {4,4},     {4}    },
    {"or",          or_spirv,        sizeof(or_spirv),        2, 1, {4,4},     {4}    },
    {"not",         not_spirv,       sizeof(not_spirv),       1, 1, {4},       {4}    },
    {"identity",    identity_spirv,  sizeof(identity_spirv),  1, 1, {4},       {4}    },
    {"full_adder",  fulladd_spirv,   sizeof(fulladd_spirv),   3, 2, {4,4,4},   {4,4}  },
    {"half_adder",  halfadd_spirv,   sizeof(halfadd_spirv),   2, 2, {4,4},     {4,4}  },
    {"mux",         mux_spirv,       sizeof(mux_spirv),       3, 1, {4,4,4},   {4}    },
    {"compare",     compare_spirv,   sizeof(compare_spirv),   2, 3, {4,4},     {4,4,4}},
    // ...
};
```

### 6.2 Custom Shapes

```c
ShapeID gillies_register_shape(
    GilliesContext* ctx,
    const char* name,
    const uint32_t* spirv,
    uint32_t spirv_size,
    uint8_t input_count,
    uint8_t output_count,
    const uint32_t* input_sizes,
    const uint32_t* output_sizes
);
```

### 6.3 SPIR-V Template

```spirv
; shape_template.spvasm
; Replace: $NAME, $COMPUTE

OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 256 1 1

; Decorations
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %input_a DescriptorSet 0
OpDecorate %input_a Binding 0
OpDecorate %input_b DescriptorSet 0
OpDecorate %input_b Binding 1
OpDecorate %output DescriptorSet 0
OpDecorate %output Binding 2

; Types
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%ptr_buffer_float = OpTypePointer StorageBuffer %float

; Variables
%gl_GlobalInvocationID = OpVariable %ptr_input_v3uint Input

; Runtime arrays
%rta_float = OpTypeRuntimeArray %float
%struct_buf = OpTypeStruct %rta_float
%ptr_struct = OpTypePointer StorageBuffer %struct_buf
%input_a = OpVariable %ptr_struct StorageBuffer
%input_b = OpVariable %ptr_struct StorageBuffer
%output = OpVariable %ptr_struct StorageBuffer

; Constants
%uint_0 = OpConstant %uint 0
%float_2 = OpConstant %float 2.0

; Main
%main = OpFunction %void None %func
%entry = OpLabel

  ; Get global ID
  %gid_vec = OpLoad %v3uint %gl_GlobalInvocationID
  %gid = OpCompositeExtract %uint %gid_vec 0

  ; Load inputs
  %ptr_a = OpAccessChain %ptr_buffer_float %input_a %uint_0 %gid
  %ptr_b = OpAccessChain %ptr_buffer_float %input_b %uint_0 %gid
  %a = OpLoad %float %ptr_a
  %b = OpLoad %float %ptr_b

  ; === COMPUTE: XOR ===
  ; result = a + b - 2*a*b
  %sum = OpFAdd %float %a %b
  %prod = OpFMul %float %a %b
  %twice = OpFMul %float %float_2 %prod
  %result = OpFSub %float %sum %twice
  ; === END COMPUTE ===

  ; Store output
  %ptr_out = OpAccessChain %ptr_buffer_float %output %uint_0 %gid
  OpStore %ptr_out %result

OpReturn
OpFunctionEnd
```

---

## 7. Public API

### 7.1 Header (libgillies.h)

```c
#ifndef GILLIES_H
#define GILLIES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

typedef struct GilliesContext GilliesContext;
typedef struct GilliesFabric GilliesFabric;
typedef struct GilliesFuture GilliesFuture;
typedef uint32_t ShapeID;

typedef enum {
    GILLIES_OK = 0,
    GILLIES_ERROR_INIT_FAILED,
    GILLIES_ERROR_PARSE_FAILED,
    GILLIES_ERROR_VALIDATION_FAILED,
    GILLIES_ERROR_COMPILE_FAILED,
    GILLIES_ERROR_OUT_OF_MEMORY,
    GILLIES_ERROR_INVALID_ARGUMENT,
    GILLIES_ERROR_NOT_FOUND,
} GilliesResult;

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

// Initialize GILLIES (creates Vulkan context)
GilliesResult gillies_init(GilliesContext** ctx);

// Destroy context
void gillies_destroy(GilliesContext* ctx);

// Get error message
const char* gillies_error_string(GilliesResult result);

// ═══════════════════════════════════════════════════════════════════════════
// SHAPES
// ═══════════════════════════════════════════════════════════════════════════

// Register custom shape
GilliesResult gillies_register_shape(
    GilliesContext* ctx,
    const char* name,
    const void* spirv,
    uint32_t spirv_size,
    uint8_t input_count,
    uint8_t output_count,
    const uint32_t* input_sizes,
    const uint32_t* output_sizes,
    ShapeID* out_id
);

// ═══════════════════════════════════════════════════════════════════════════
// FABRIC
// ═══════════════════════════════════════════════════════════════════════════

// Compile fabric from FDL string
GilliesResult gillies_compile(
    GilliesContext* ctx,
    const char* fdl_source,
    GilliesFabric** out_fabric
);

// Compile fabric from file
GilliesResult gillies_compile_file(
    GilliesContext* ctx,
    const char* path,
    GilliesFabric** out_fabric
);

// Destroy fabric
void gillies_fabric_destroy(GilliesFabric* fabric);

// Get fabric stats
void gillies_fabric_stats(
    GilliesFabric* fabric,
    uint32_t* out_node_count,
    uint32_t* out_buffer_count,
    uint32_t* out_memory_bytes
);

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTION
// ═══════════════════════════════════════════════════════════════════════════

// Bind input data (copies to GPU)
GilliesResult gillies_set_input(
    GilliesFabric* fabric,
    const char* signature,
    const float* data,
    uint32_t count
);

// Bind output location (will copy from GPU after run)
GilliesResult gillies_set_output(
    GilliesFabric* fabric,
    const char* signature,
    float* data,
    uint32_t count
);

// Execute synchronously
GilliesResult gillies_run(GilliesFabric* fabric);

// Execute asynchronously
GilliesResult gillies_run_async(
    GilliesFabric* fabric,
    GilliesFuture** out_future
);

// Check if async execution is complete
bool gillies_poll(GilliesFuture* future);

// Wait for async execution
GilliesResult gillies_wait(GilliesFuture* future);

// Destroy future
void gillies_future_destroy(GilliesFuture* future);

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

// Get version
const char* gillies_version(void);

// Get GPU name
const char* gillies_gpu_name(GilliesContext* ctx);

// Benchmark fabric (returns ops/sec)
double gillies_benchmark(
    GilliesFabric* fabric,
    uint32_t iterations
);

#ifdef __cplusplus
}
#endif

#endif // GILLIES_H
```

---

## 8. Example Usage

### 8.1 Minimal Example

```c
#include "gillies.h"
#include <stdio.h>

int main() {
    GilliesContext* ctx;
    gillies_init(&ctx);

    const char* fdl =
        "fabric \"test\" {"
        "  processor gate {"
        "    shape: xor"
        "    signature: /0/0b00"
        "  }"
        "  export: gate"
        "}";

    GilliesFabric* fab;
    gillies_compile(ctx, fdl, &fab);

    float a[] = {1, 0, 1, 0};
    float b[] = {1, 1, 0, 0};
    float out[4];

    gillies_set_input(fab, "/0/0b00", a, 4);  // Sets 'a' input
    gillies_set_input(fab, "/0/0b00", b, 4);  // Sets 'b' input
    gillies_set_output(fab, "/0/0b00", out, 4);

    gillies_run(fab);

    for (int i = 0; i < 4; i++) {
        printf("XOR(%g, %g) = %g\n", a[i], b[i], out[i]);
    }

    gillies_fabric_destroy(fab);
    gillies_destroy(ctx);
    return 0;
}
```

### 8.2 Chain Example

```c
const char* fdl =
    "fabric \"chain\" {"
    "  processor a {"
    "    shape: xor"
    "    signature: /0/0b00"
    "  }"
    "  processor b {"
    "    shape: not"
    "    signature: /0/0b01"
    "    input.a: a.result"
    "  }"
    "  export: b"
    "}";

// XOR then NOT
// XOR(1,0) = 1, NOT(1) = 0
```

### 8.3 Async Example

```c
GilliesFuture* future;
gillies_run_async(fab, &future);

// Do other work...

gillies_wait(future);
gillies_future_destroy(future);
```

---

## 9. File Structure

```
gillies/
├── include/
│   └── gillies.h               # Public API
│
├── src/
│   ├── gillies.c               # API implementation
│   │
│   ├── compiler/
│   │   ├── parser.c            # FDL → AST
│   │   ├── analyzer.c          # AST → DAG
│   │   ├── planner.c           # DAG → Plan
│   │   └── emitter.c           # Plan → Vulkan
│   │
│   ├── runtime/
│   │   ├── memory.c            # Buffer management
│   │   ├── descriptors.c       # Descriptor allocation
│   │   ├── commands.c          # Command recording
│   │   └── sync.c              # Fences, semaphores
│   │
│   ├── shapes/
│   │   ├── registry.c          # Shape management
│   │   ├── builtins.c          # Built-in shapes
│   │   └── spirv/              # Pre-compiled SPIR-V
│   │       ├── xor.spv
│   │       ├── and.spv
│   │       ├── or.spv
│   │       └── ...
│   │
│   └── vulkan/
│       ├── context.c           # VkInstance, VkDevice
│       ├── memory.c            # VkBuffer, VkMemory
│       └── pipeline.c          # VkPipeline
│
├── test/
│   ├── test_parser.c
│   ├── test_analyzer.c
│   ├── test_planner.c
│   ├── test_execution.c
│   └── test_benchmark.c
│
├── examples/
│   ├── minimal.c
│   ├── chain.c
│   ├── half_adder.c
│   └── benchmark.c
│
├── CMakeLists.txt
└── README.md
```

---

## 10. Build System

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.16)
project(gillies VERSION 0.1.0 LANGUAGES C)

# Find Vulkan
find_package(Vulkan REQUIRED)

# SPIR-V compilation
find_program(SPIRV_AS spirv-as REQUIRED)

# Compile SPIR-V shaders
file(GLOB SPIRV_SOURCES "src/shapes/spirv/*.spvasm")
foreach(SPVASM ${SPIRV_SOURCES})
    get_filename_component(NAME ${SPVASM} NAME_WE)
    set(SPV "${CMAKE_BINARY_DIR}/spirv/${NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${SPIRV_AS} ${SPVASM} -o ${SPV}
        DEPENDS ${SPVASM}
    )
    list(APPEND SPIRV_OUTPUTS ${SPV})
endforeach()

add_custom_target(spirv ALL DEPENDS ${SPIRV_OUTPUTS})

# Library
add_library(gillies
    src/gillies.c
    src/compiler/parser.c
    src/compiler/analyzer.c
    src/compiler/planner.c
    src/compiler/emitter.c
    src/runtime/memory.c
    src/runtime/descriptors.c
    src/runtime/commands.c
    src/runtime/sync.c
    src/shapes/registry.c
    src/shapes/builtins.c
    src/vulkan/context.c
    src/vulkan/memory.c
    src/vulkan/pipeline.c
)

target_include_directories(gillies PUBLIC include)
target_link_libraries(gillies Vulkan::Vulkan)
add_dependencies(gillies spirv)

# Examples
add_executable(minimal examples/minimal.c)
target_link_libraries(minimal gillies)
```

---

## 11. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Compile time | < 10ms for 100 nodes | Interactive use |
| First run | < 50ms | Pipeline creation |
| Subsequent runs | < 1ms overhead | Pre-recorded commands |
| Throughput | 15B+ ops/sec | Match raw Vulkan |
| Memory overhead | < 10% | Pooling efficiency |

---

## 12. Future Work

1. **Metal backend** — Same API, different GPU layer
2. **WebGPU backend** — Browser deployment
3. **CPU fallback** — When no GPU available
4. **Dynamic sizing** — Runtime-determined buffer sizes
5. **Multi-GPU** — Partition fabrics across devices
6. **Streaming** — Execute fabrics larger than GPU memory
7. **Autodiff** — Compile backward pass from forward

---

*GILLIES Runtime Design — December 2025*
*"Structure eliminates coordination."*
