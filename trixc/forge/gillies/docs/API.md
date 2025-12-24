# GILLIES API Reference

Complete reference for the GILLIES C API.

---

## Header Files

```c
#include "gillies.h"           // Main API
#include "shapes_device.cuh"   // Shape implementations (included by gillies.h)
```

---

## Configuration Constants

```c
#define GILLIES_MAX_PORTS 256        // Maximum number of ports
#define GILLIES_MAX_INVOCATIONS 1024 // Maximum invocations per context
#define GILLIES_MAX_EVENTS 4096      // Maximum trace events
```

---

## Types

### gillies_shape_t

Enumeration of available shapes.

```c
typedef enum {
    GILLIES_SHAPE_XOR = 0,      // a + b - 2ab
    GILLIES_SHAPE_AND = 1,      // ab
    GILLIES_SHAPE_OR  = 2,      // a + b - ab
    GILLIES_SHAPE_NOT = 3,      // 1 - a
    GILLIES_SHAPE_NAND = 4,     // 1 - ab
    GILLIES_SHAPE_NOR = 5,      // 1 - a - b + ab
    GILLIES_SHAPE_XNOR = 6,     // 1 - a - b + 2ab
    GILLIES_SHAPE_ADD = 7,      // a + b
    GILLIES_SHAPE_SUB = 8,      // a - b
    GILLIES_SHAPE_MUL = 9,      // a * b
    GILLIES_SHAPE_IDENTITY = 10, // a (passthrough)
    GILLIES_NUM_SHAPES
} gillies_shape_t;
```

**Shape names array** (for debugging):
```c
const char* GILLIES_SHAPE_NAMES[];
// {"XOR", "AND", "OR", "NOT", "NAND", "NOR", "XNOR", "ADD", "SUB", "MUL", "IDENTITY"}
```

---

### gillies_invocation_t

A single shape invocation.

```c
typedef struct {
    uint8_t shape_id;       // Which shape (from gillies_shape_t)
    uint8_t num_inputs;     // Number of input ports used (1-2)
    uint8_t num_outputs;    // Number of output ports used (1)
    uint8_t flags;          // Reserved for future use

    uint16_t inputs[2];     // Port indices for inputs
    uint16_t outputs[1];    // Port indices for outputs
} gillies_invocation_t;
```

---

### gillies_ports_t

The port space.

```c
typedef struct {
    float data[GILLIES_MAX_PORTS];      // Port values
    uint8_t valid[GILLIES_MAX_PORTS];   // Validity flags
} gillies_ports_t;
```

---

### gillies_event_t

A trace event.

```c
typedef struct {
    uint32_t invocation_id;     // Which invocation
    uint8_t shape_id;           // Which shape was executed
    float input_a;              // Input value A
    float input_b;              // Input value B
    float output;               // Output value
    uint64_t timestamp_ns;      // Execution timestamp (nanoseconds)
} gillies_event_t;
```

---

### gillies_context_t

The complete execution context.

```c
typedef struct {
    gillies_ports_t ports;                              // Port space
    gillies_invocation_t invocations[GILLIES_MAX_INVOCATIONS];
    uint32_t num_invocations;

    uint8_t executed[GILLIES_MAX_INVOCATIONS];          // Execution flags
    uint32_t execution_order[GILLIES_MAX_INVOCATIONS];  // Resolved order
    uint32_t execution_count;

    gillies_event_t events[GILLIES_MAX_EVENTS];         // Trace events
    uint32_t num_events;
    bool tracing_enabled;

    uint64_t total_invocations;                         // Statistics
    uint64_t shape_counts[GILLIES_NUM_SHAPES];
} gillies_context_t;
```

---

## Context Management

### gillies_create

```c
gillies_context_t* gillies_create(void);
```

Create a new GILLIES context.

**Returns**: Pointer to a new context allocated in CUDA unified memory.

**Notes**:
- The context is accessible by both CPU and GPU
- Must be freed with `gillies_destroy()`

**Example**:
```c
gillies_context_t* ctx = gillies_create();
// ... use context ...
gillies_destroy(ctx);
```

---

### gillies_destroy

```c
void gillies_destroy(gillies_context_t* ctx);
```

Destroy a context and free all resources.

**Parameters**:
- `ctx`: Context to destroy (may be NULL)

---

### gillies_reset

```c
void gillies_reset(gillies_context_t* ctx);
```

Reset the context for a new computation.

**Effects**:
- Clears all invocations
- Clears execution state
- Clears trace events
- **Preserves port data** (call `gillies_clear_ports()` to clear ports)

---

### gillies_clear_ports

```c
void gillies_clear_ports(gillies_context_t* ctx);
```

Clear all ports.

**Effects**:
- Sets all port values to 0.0
- Marks all ports as invalid

---

## Port Operations

### gillies_set_port

```c
void gillies_set_port(gillies_context_t* ctx, uint16_t port, float value);
```

Set a port value.

**Parameters**:
- `ctx`: The context
- `port`: Port index (0 to GILLIES_MAX_PORTS-1)
- `value`: Value to set

**Notes**:
- Automatically marks the port as valid
- Out-of-range ports are silently ignored

**Example**:
```c
gillies_set_port(ctx, 0, 1.0f);  // Set port 0 to 1.0
gillies_set_port(ctx, 1, 0.5f);  // Set port 1 to 0.5
```

---

### gillies_get_port

```c
float gillies_get_port(gillies_context_t* ctx, uint16_t port);
```

Get a port value.

**Parameters**:
- `ctx`: The context
- `port`: Port index

**Returns**: The port value, or 0.0 if out of range.

---

### gillies_port_valid

```c
bool gillies_port_valid(gillies_context_t* ctx, uint16_t port);
```

Check if a port has been written.

**Parameters**:
- `ctx`: The context
- `port`: Port index

**Returns**: `true` if the port has been set or computed, `false` otherwise.

---

## Graph Building

### gillies_invoke

```c
int gillies_invoke(
    gillies_context_t* ctx,
    gillies_shape_t shape,
    uint16_t in0,
    uint16_t in1,
    uint16_t out0
);
```

Add a shape invocation to the graph.

**Parameters**:
- `ctx`: The context
- `shape`: Which shape to execute
- `in0`: First input port
- `in1`: Second input port (ignored for unary shapes like NOT)
- `out0`: Output port

**Returns**: Invocation index (0+), or -1 on error.

**Notes**:
- Invocations are stored but not executed until `gillies_execute()` is called
- For unary shapes (NOT, IDENTITY), `in1` is ignored but must be provided
- The same port can be used as input to multiple invocations
- An output port can overwrite a previous value

**Example**:
```c
// XOR(port[0], port[1]) -> port[2]
gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);

// NOT(port[2]) -> port[3]  (in1 ignored but required)
gillies_invoke(ctx, GILLIES_SHAPE_NOT, 2, 0, 3);
```

---

## Execution

### gillies_execute

```c
void gillies_execute(gillies_context_t* ctx);
```

Execute all invocations on the GPU.

**Effects**:
- Resolves dependencies (topological sort)
- Executes all invocations in valid order
- Writes results to output ports
- Updates statistics
- Records events if tracing is enabled

**Notes**:
- Uses CUDA unified memory
- Synchronizes after execution (blocking)
- Currently executes sequentially to respect dependencies

**Example**:
```c
gillies_set_port(ctx, 0, 1.0f);
gillies_set_port(ctx, 1, 0.0f);
gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);
gillies_execute(ctx);
float result = gillies_get_port(ctx, 2);  // 1.0
```

---

### gillies_execute_cpu

```c
void gillies_execute_cpu(gillies_context_t* ctx);
```

Execute all invocations on the CPU.

**Effects**: Same as `gillies_execute()` but uses CPU only.

**Use cases**:
- Comparison/validation (fungibility testing)
- Systems without GPU
- Debugging

**Example**:
```c
// Execute same graph on both substrates
gillies_execute(ctx_gpu);
gillies_execute_cpu(ctx_cpu);

// Compare results (should be identical)
assert(gillies_get_port(ctx_gpu, 2) == gillies_get_port(ctx_cpu, 2));
```

---

## Observability

### gillies_set_tracing

```c
void gillies_set_tracing(gillies_context_t* ctx, bool enabled);
```

Enable or disable execution tracing.

**Parameters**:
- `ctx`: The context
- `enabled`: `true` to enable, `false` to disable

**Notes**:
- When enabled, each invocation records an event
- Events include inputs, output, and timestamp
- Small performance overhead when enabled

---

### gillies_event_count

```c
uint32_t gillies_event_count(gillies_context_t* ctx);
```

Get the number of recorded events.

**Returns**: Number of events recorded since last reset.

---

### gillies_get_event

```c
const gillies_event_t* gillies_get_event(gillies_context_t* ctx, uint32_t index);
```

Get an event by index.

**Parameters**:
- `ctx`: The context
- `index`: Event index (0 to event_count-1)

**Returns**: Pointer to the event, or NULL if out of range.

**Example**:
```c
gillies_set_tracing(ctx, true);
gillies_execute(ctx);

for (uint32_t i = 0; i < gillies_event_count(ctx); i++) {
    const gillies_event_t* evt = gillies_get_event(ctx, i);
    printf("[%u] %s(%.2f, %.2f) = %.2f @ %lu ns\n",
           evt->invocation_id,
           GILLIES_SHAPE_NAMES[evt->shape_id],
           evt->input_a, evt->input_b, evt->output,
           evt->timestamp_ns);
}
```

---

## Debugging

### gillies_print_stats

```c
void gillies_print_stats(gillies_context_t* ctx);
```

Print execution statistics to stdout.

**Output**:
```
GILLIES Statistics:
  Total invocations: 1234
  Shape usage:
    XOR: 500
    AND: 400
    OR: 334
```

---

### gillies_print_graph

```c
void gillies_print_graph(gillies_context_t* ctx);
```

Print the invocation graph to stdout.

**Output**:
```
GILLIES Invocation Graph:
  [0] XOR(port[0], port[1]) -> port[2]
  [1] AND(port[2], port[3]) -> port[4]
  [2] NOT(port[4]) -> port[5]
```

---

## Shape Functions (Low-Level)

These are the underlying shape implementations, available for direct use.

### Float Shapes

```c
float gillies_xor_f32(float a, float b);   // a + b - 2ab
float gillies_and_f32(float a, float b);   // ab
float gillies_or_f32(float a, float b);    // a + b - ab
float gillies_not_f32(float a);            // 1 - a
float gillies_nand_f32(float a, float b);  // 1 - ab
float gillies_nor_f32(float a, float b);   // 1 - a - b + ab
float gillies_xnor_f32(float a, float b);  // 1 - a - b + 2ab
float gillies_add_f32(float a, float b);   // a + b
float gillies_sub_f32(float a, float b);   // a - b
float gillies_mul_f32(float a, float b);   // a * b
float gillies_identity_f32(float a);       // a
```

### Dispatch

```c
float gillies_dispatch_f32(uint8_t shape_id, float a, float b);
uint8_t gillies_dispatch_u8(uint8_t shape_id, uint8_t a, uint8_t b);
```

Route a shape ID to its implementation.

### Full Adder

```c
void gillies_full_adder_f32(float a, float b, float c, float* sum, float* carry);
void gillies_full_adder_u8(uint8_t a, uint8_t b, uint8_t c, uint8_t* sum, uint8_t* carry);
```

Compute a full adder (sum and carry) from three inputs.

---

## Complete Example

```c
#include <stdio.h>
#include "gillies.h"

int main() {
    // Create context
    gillies_context_t* ctx = gillies_create();

    // Build a full adder
    // Inputs: port[0]=a, port[1]=b, port[2]=cin
    // Outputs: port[10]=sum, port[11]=carry

    gillies_set_port(ctx, 0, 1.0f);  // a = 1
    gillies_set_port(ctx, 1, 1.0f);  // b = 1
    gillies_set_port(ctx, 2, 0.0f);  // cin = 0

    // sum = a XOR b XOR cin
    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 5);   // t = a XOR b
    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 5, 2, 10);  // sum = t XOR cin

    // carry = (a AND b) OR ((a XOR b) AND cin)
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 6);   // ab = a AND b
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 5, 2, 7);   // tc = t AND cin
    gillies_invoke(ctx, GILLIES_SHAPE_OR, 6, 7, 11);   // carry = ab OR tc

    // Execute
    gillies_execute(ctx);

    // Read results
    float sum = gillies_get_port(ctx, 10);
    float carry = gillies_get_port(ctx, 11);

    printf("1 + 1 + 0 = %d (carry %d)\n",
           (int)(sum + 0.5f), (int)(carry + 0.5f));
    // Output: 1 + 1 + 0 = 0 (carry 1)

    // Clean up
    gillies_destroy(ctx);
    return 0;
}
```

---

## Error Handling

GILLIES uses simple error handling:

- `gillies_create()` returns NULL on allocation failure
- `gillies_invoke()` returns -1 if max invocations reached
- Out-of-range port access is silently ignored (returns 0.0)
- CUDA errors cause program termination with error message

For production use, consider adding:
- Error callbacks
- Status return values
- Exception handling (C++)

---

*API version: 1.0*
*December 2025*
