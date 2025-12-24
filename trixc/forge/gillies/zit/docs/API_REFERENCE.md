# ZIT API Reference

Complete reference documentation for the ZIT (Zero-Instruction Topology) C API.

---

## Table of Contents

1. [Overview](#overview)
2. [Types](#types)
3. [Lifecycle Functions](#lifecycle-functions)
4. [Execution Functions](#execution-functions)
5. [Observation Functions](#observation-functions)
6. [Visualization Functions](#visualization-functions)
7. [Output Functions](#output-functions)
8. [Constants](#constants)
9. [Error Handling](#error-handling)
10. [Thread Safety](#thread-safety)
11. [Memory Management](#memory-management)

---

## Overview

ZIT provides a minimal C API for creating and running homeo-adaptive topological fabrics. The library is:

- **Header-only friendly**: Include `zit.h`, link `zit.c`
- **Zero external dependencies**: Only requires standard C library
- **Opaque handle design**: Implementation details are hidden
- **Callback-enabled**: Supports real-time observation

### Minimal Example

```c
#include "zit.h"

int main() {
    zit_fabric_t* f = zit_create(8);    // 8x8x8 = 512 nodes
    zit_run(f, 0);                       // Run to convergence
    zit_print(f);                        // Print results
    zit_destroy(f);                      // Clean up
    return 0;
}
```

---

## Types

### zit_fabric_t

```c
typedef struct zit_fabric zit_fabric_t;
```

Opaque handle to a ZIT fabric. All API functions operate on this handle.

**Notes:**
- Created by `zit_create()`
- Must be destroyed with `zit_destroy()`
- Not copyable (no copy function provided)

### zit_step_callback_t

```c
typedef void (*zit_step_callback_t)(zit_fabric_t* f, void* user_data);
```

Callback function type invoked after each simulation step.

**Parameters:**
- `f`: The fabric that just completed a step
- `user_data`: User-provided context pointer

**Example:**
```c
void my_callback(zit_fabric_t* f, void* data) {
    int* count = (int*)data;
    (*count)++;
    printf("Step %d: %d resonant\n", *count, zit_resonant(f));
}

int main() {
    int step_count = 0;
    zit_fabric_t* f = zit_create(8);
    zit_set_step_callback(f, my_callback, &step_count);
    zit_run(f, 100);
    zit_destroy(f);
}
```

---

## Lifecycle Functions

### zit_create

```c
zit_fabric_t* zit_create(int dim);
```

Create a new ZIT fabric with `dim³` nodes arranged in a 3D torus topology.

**Parameters:**
- `dim`: Dimension of the cubic lattice (nodes = dim × dim × dim)

**Returns:**
- Pointer to new fabric on success
- `NULL` on allocation failure

**Recommended dimensions:**
| dim | Nodes | Memory | Use Case |
|-----|-------|--------|----------|
| 4 | 64 | ~2 KB | Unit tests |
| 8 | 512 | ~15 KB | Demos, learning |
| 16 | 4,096 | ~120 KB | Experiments |
| 32 | 32,768 | ~1 MB | Research |

**Example:**
```c
zit_fabric_t* f = zit_create(8);
if (!f) {
    fprintf(stderr, "Failed to create fabric\n");
    return 1;
}
```

---

### zit_destroy

```c
void zit_destroy(zit_fabric_t* f);
```

Destroy a fabric and free all associated memory.

**Parameters:**
- `f`: Fabric to destroy (may be `NULL`)

**Notes:**
- Safe to call with `NULL`
- Sets no dangling pointer (caller must manage)
- Invalidates all prior observations

**Example:**
```c
zit_destroy(f);
f = NULL;  // Good practice
```

---

### zit_seed

```c
void zit_seed(zit_fabric_t* f, uint32_t seed);
```

Reset fabric to initial state with specified random seed.

**Parameters:**
- `f`: Target fabric
- `seed`: 32-bit seed for LFSR initialization

**Effects:**
- Resets all node states to deterministic initial values
- Resets all node resistances to 0
- Resets topology to regular 3D torus
- Resets cycle counter to 0
- Resets rewire counter to 0

**Special Seeds:**
| Seed | Name | Purpose |
|------|------|---------|
| 1122911624 | Second Star Constant | Reproducible paper results |
| 0 | (not recommended) | Produces degenerate LFSR |

**Example:**
```c
zit_seed(f, 1122911624);  // Reproducible
zit_seed(f, time(NULL));  // Random each run
```

---

## Execution Functions

### zit_step

```c
int zit_step(zit_fabric_t* f);
```

Execute one complete cycle of the simulation.

**Parameters:**
- `f`: Target fabric

**Returns:**
- Number of resonant nodes after this cycle

**One cycle consists of:**
1. Reset resonance flags for all nodes
2. Execute 6 sequential comparison phases (+X, -X, +Y, -Y, +Z, -Z)
3. Apply plasticity rules (resistance update, rewiring)
4. Count resonant nodes
5. Invoke step callback if set

**Performance:**
- O(n) where n = total nodes
- ~1ms for 512 nodes on modern CPU

**Example:**
```c
while (!zit_converged(f)) {
    int resonant = zit_step(f);
    printf("Cycle %d: %d/%d resonant\n",
           zit_cycle(f), resonant, zit_total(f));
}
```

---

### zit_run

```c
int zit_run(zit_fabric_t* f, int max_cycles);
```

Run simulation until convergence or cycle limit.

**Parameters:**
- `f`: Target fabric
- `max_cycles`: Maximum cycles to run (0 = no limit, defaults to 10000)

**Returns:**
- Number of cycles executed

**Notes:**
- Stops early if 100% resonance achieved
- Step callback invoked each cycle if set
- For indefinite run, use 0 (internally limited to 10000)

**Example:**
```c
int cycles = zit_run(f, 500);
if (zit_converged(f)) {
    printf("Converged in %d cycles\n", cycles);
} else {
    printf("Did not converge after %d cycles\n", cycles);
}
```

---

## Observation Functions

### zit_resonant

```c
int zit_resonant(zit_fabric_t* f);
```

Get count of resonant nodes (nodes that didn't change state this cycle).

**Returns:** Number of resonant nodes (0 to total)

---

### zit_total

```c
int zit_total(zit_fabric_t* f);
```

Get total number of nodes in the fabric.

**Returns:** dim³ (e.g., 512 for dim=8)

---

### zit_rewires

```c
int zit_rewires(zit_fabric_t* f);
```

Get cumulative count of rewiring attempts.

**Returns:** Total rewires since creation or last seed

**Notes:**
- Counts attempts, not successes
- Includes reverted rewires

---

### zit_converged

```c
bool zit_converged(zit_fabric_t* f);
```

Check if fabric has reached 100% resonance.

**Returns:** `true` if all nodes are resonant

---

### zit_cycle

```c
int zit_cycle(zit_fabric_t* f);
```

Get current cycle count.

**Returns:** Cycles executed since creation or last seed

---

### zit_dim

```c
int zit_dim(zit_fabric_t* f);
```

Get the dimension of the fabric.

**Returns:** Original dim parameter from `zit_create()`

---

## Visualization Functions

These functions provide per-node access for building visualizations.

### zit_node_state

```c
uint8_t zit_node_state(zit_fabric_t* f, int node_id);
```

Get current state value of a node.

**Parameters:**
- `f`: Target fabric
- `node_id`: Node index (0 to total-1)

**Returns:** State value (0-255), or 0 if invalid node_id

---

### zit_node_resistance

```c
uint8_t zit_node_resistance(zit_fabric_t* f, int node_id);
```

Get current resistance level of a node.

**Parameters:**
- `f`: Target fabric
- `node_id`: Node index (0 to total-1)

**Returns:** Resistance counter (0-255), or 0 if invalid node_id

**Interpretation:**
| Value | Meaning |
|-------|---------|
| 0 | No resistance (resonant) |
| 1-7 | Building resistance |
| 8+ | Would trigger rewiring |

---

### zit_node_resonant

```c
bool zit_node_resonant(zit_fabric_t* f, int node_id);
```

Check if a node is currently resonant.

**Returns:** `true` if node didn't change state this cycle

---

### zit_node_rewiring

```c
bool zit_node_rewiring(zit_fabric_t* f, int node_id);
```

Check if a node is currently in rewiring evaluation period.

**Returns:** `true` if node is evaluating a new neighbor

---

### zit_node_neighbor

```c
int zit_node_neighbor(zit_fabric_t* f, int node_id, int direction);
```

Get the neighbor index for a node in a given direction.

**Parameters:**
- `f`: Target fabric
- `node_id`: Node index (0 to total-1)
- `direction`: Direction index (0-5)

**Direction indices:**
| Index | Direction | Axis |
|-------|-----------|------|
| 0 | +X | East |
| 1 | -X | West |
| 2 | +Y | North |
| 3 | -Y | South |
| 4 | +Z | Up |
| 5 | -Z | Down |

**Returns:** Neighbor node index, or -1 if invalid parameters

---

### zit_node_coords

```c
void zit_node_coords(zit_fabric_t* f, int node_id, int* x, int* y, int* z);
```

Get 3D coordinates of a node.

**Parameters:**
- `f`: Target fabric
- `node_id`: Node index (0 to total-1)
- `x`, `y`, `z`: Output pointers (may be NULL to ignore)

**Coordinate system:**
- Origin at (0, 0, 0)
- Range: [0, dim-1] for each axis
- Index formula: `node_id = x + y*dim + z*dim*dim`

---

### zit_set_step_callback

```c
void zit_set_step_callback(zit_fabric_t* f, zit_step_callback_t cb, void* user_data);
```

Register a callback to be invoked after each step.

**Parameters:**
- `f`: Target fabric
- `cb`: Callback function (or NULL to disable)
- `user_data`: Pointer passed to callback

**Notes:**
- Only one callback at a time
- Callback runs synchronously during `zit_step()`
- Safe to query fabric state from within callback

---

## Output Functions

### zit_print

```c
void zit_print(zit_fabric_t* f);
```

Print summary to stdout.

**Output format:**
```
ZIT Fabric: 8x8x8 = 512 nodes
Cycle: 114
Resonant: 512/512 (100.0%)
Rewires: 1340
Status: CONVERGED
```

---

### zit_print_progress

```c
void zit_print_progress(zit_fabric_t* f);
```

Print single progress line to stdout.

**Output format:**
```
Cycle  114: 512/512 resonant, 1340 rewires
```

---

### zit_export

```c
int zit_export(zit_fabric_t* f, const char* path);
```

Export fabric topology to JSON file.

**Parameters:**
- `f`: Source fabric
- `path`: Output file path

**Returns:**
- 0 on success
- -1 on file error

**JSON format:**
```json
{
  "dim": 8,
  "total": 512,
  "cycle": 114,
  "rewires": 1340,
  "nodes": [
    {"neighbors": [1,7,8,504,64,448]},
    ...
  ]
}
```

---

## Constants

Defined in implementation (not exposed in header):

| Constant | Value | Meaning |
|----------|-------|---------|
| RESISTANCE_THRESHOLD | 8 | Cycles of non-resonance before rewiring |
| EVAL_PERIOD | 8 | Cycles to evaluate new neighbor |

---

## Error Handling

The ZIT API uses simple error handling:

- `zit_create()` returns NULL on allocation failure
- `zit_export()` returns -1 on file error
- Other functions assume valid input

**Defensive programming:**
```c
zit_fabric_t* f = zit_create(8);
if (!f) {
    // Handle allocation failure
}

if (zit_export(f, "/path/to/file.json") < 0) {
    // Handle file error
}
```

---

## Thread Safety

The ZIT library is **NOT thread-safe**.

- Do not share a fabric between threads without external synchronization
- Each thread should have its own fabric instance
- Callback functions run on the calling thread

**Safe pattern:**
```c
// Thread 1
zit_fabric_t* f1 = zit_create(8);
zit_run(f1, 0);

// Thread 2
zit_fabric_t* f2 = zit_create(8);
zit_run(f2, 0);
```

---

## Memory Management

### Allocation Sizes

| dim | Nodes | Approx. Memory |
|-----|-------|----------------|
| 4 | 64 | 2 KB |
| 8 | 512 | 15 KB |
| 16 | 4,096 | 120 KB |
| 32 | 32,768 | 1 MB |
| 64 | 262,144 | 8 MB |

### Memory Layout

Each fabric allocates:
- 1 × `zit_fabric` struct (~48 bytes)
- n × `zit_node_t` array (~30 bytes per node)
- n × `uint8_t` snapshot array (1 byte per node)

### Best Practices

1. Always call `zit_destroy()` when done
2. Set pointer to NULL after destroy
3. For repeated experiments, use `zit_seed()` instead of destroy/create

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0.0 | Initial release with core API |
| 1.1.0 | Added visualization functions |

---

*The topology IS the learned model.*
