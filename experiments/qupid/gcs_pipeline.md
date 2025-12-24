# Geometric Compute Shapes (GCS) Pipeline

## Overview

End-to-end pipeline for developing, verifying, and deploying frozen shape computations.

```
┌─────────────────────────────────────────────────────────────────┐
│                        GCS PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ DISCOVER│───▶│ COMPOSE │───▶│ VERIFY  │───▶│ COMPILE │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│        │              │              │              │           │
│        ▼              ▼              ▼              ▼           │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ Shape   │    │ Fabric  │    │ Proof   │    │ Routing │     │
│   │ Library │    │ Spec    │    │ Cert    │    │ Table   │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                     │           │
│                       ┌─────────────────────────────┘           │
│                       ▼                                         │
│                  ┌─────────┐    ┌─────────┐                     │
│                  │ EXECUTE │───▶│OPTIMIZE │──────┐              │
│                  └─────────┘    └─────────┘      │              │
│                       │              │           │              │
│                       ▼              ▼           │              │
│                  ┌─────────┐    ┌─────────┐      │              │
│                  │ Results │    │ Better  │──────┘              │
│                  │         │    │ Topology│   (iterate)         │
│                  └─────────┘    └─────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: DISCOVER (Shape Mining)

**Input:** Computational domain (e.g., "cryptography", "signal processing", "6502 CPU")

**Process:**
1. Identify primitive operations in the domain
2. Express each as a pure mathematical function
3. Categorize by input/output arity
4. Verify completeness (can all domain operations be expressed?)

**Output:** Shape Library

### Shape Definition Format

```yaml
shape:
  name: "XOR"
  arity: 2  # binary operation
  bits: 8   # operand width

  # Mathematical definition
  math: "a ⊕ b"

  # Equivalent forms
  algebraic: "a + b - 2ab"  # continuous form
  boolean: "(a ∧ ¬b) ∨ (¬a ∧ b)"  # logic form

  # CUDA implementation
  cuda: "return a ^ b;"

  # Properties
  properties:
    - commutative: true
    - associative: true
    - identity: 0
    - self_inverse: true  # a ⊕ a = 0
```

### Core Shape Library (Universal)

| Shape | Math | Properties |
|-------|------|------------|
| ADD | a + b | comm, assoc |
| SUB | a - b | - |
| AND | a ∧ b | comm, assoc, idem |
| OR | a ∨ b | comm, assoc, idem |
| XOR | a ⊕ b | comm, assoc, self-inv |
| NOT | ¬a | involutive |
| SHL | a << n | - |
| SHR | a >> n | - |
| ROL | rotate_left(a, n) | - |
| ROR | rotate_right(a, n) | - |
| MIN | min(a, b) | comm, assoc, idem |
| MAX | max(a, b) | comm, assoc, idem |
| MUL | a × b | comm, assoc |
| MUX | c ? a : b | - |
| PASS | a | identity |
| CONST | k | constant |

---

## Stage 2: COMPOSE (Fabric Design)

**Input:** Shape Library + Computation Goal

**Process:**
1. Define input/output signature
2. Select shapes for each node
3. Define routing (which outputs connect to which inputs)
4. Specify layer structure

**Output:** Fabric Specification

### Fabric Specification Format

```yaml
fabric:
  name: "8-element-sorter"
  inputs: 8
  outputs: 8
  layers: 3

  layer[0]:
    width: 8
    nodes:
      - {shape: MIN, inputs: [in.0, in.1]}
      - {shape: MAX, inputs: [in.0, in.1]}
      - {shape: MIN, inputs: [in.2, in.3]}
      - {shape: MAX, inputs: [in.2, in.3]}
      - {shape: MIN, inputs: [in.4, in.5]}
      - {shape: MAX, inputs: [in.4, in.5]}
      - {shape: MIN, inputs: [in.6, in.7]}
      - {shape: MAX, inputs: [in.6, in.7]}

  layer[1]:
    width: 8
    nodes:
      - {shape: MIN, inputs: [L0.0, L0.2]}
      - {shape: MAX, inputs: [L0.0, L0.2]}
      # ... etc
```

### Composition Methods

1. **Manual Design** - Engineer specifies topology
2. **Template Instantiation** - Use known patterns (butterfly, tree, systolic)
3. **Learned Routing** - Gradient descent on routing table (like our 6502 work)
4. **Synthesized** - SAT/SMT solver finds valid wiring
5. **Evolved** - Genetic algorithm searches topology space

---

## Stage 3: VERIFY (Correctness Proof)

**Input:** Fabric Specification + Expected Behavior

**Process:**
1. Shape-level: Verify each shape matches its mathematical definition
2. Type-level: Verify routing respects bit widths
3. Semantic-level: Verify fabric computes intended function
4. Property-level: Verify invariants (e.g., "output is sorted")

**Output:** Proof Certificate

### Verification Methods

```python
# Property-based testing
@given(arrays(uint8, 8))
def test_sorter_is_sorted(inputs):
    outputs = fabric.execute(inputs)
    assert all(outputs[i] <= outputs[i+1] for i in range(7))

@given(arrays(uint8, 8))
def test_sorter_is_permutation(inputs):
    outputs = fabric.execute(inputs)
    assert sorted(inputs) == sorted(outputs)
```

```smt
; SMT verification (Z3)
(declare-fun fabric ((_ BitVec 8) (_ BitVec 8) ... ) (_ BitVec 64))

; Sorter property: output is sorted
(assert (forall ((x0 (_ BitVec 8)) (x1 (_ BitVec 8)) ...)
  (let ((out (fabric x0 x1 ...)))
    (bvule (extract 7 0 out) (extract 15 8 out)))))

(check-sat)  ; Should be SAT
```

### Proof Certificate Format

```yaml
certificate:
  fabric: "8-element-sorter"

  shape_proofs:
    - shape: MIN
      method: "definition"
      status: "verified"
    - shape: MAX
      method: "definition"
      status: "verified"

  composition_proof:
    method: "smt"
    solver: "z3"
    time_ms: 127
    status: "verified"

  properties:
    - name: "output_sorted"
      status: "verified"
    - name: "is_permutation"
      status: "verified"
```

---

## Stage 4: COMPILE (Routing Generation)

**Input:** Verified Fabric Specification

**Process:**
1. Flatten fabric to routing tables
2. Generate CUDA kernel
3. Compile to PTX
4. Create execution wrapper

**Output:** Executable Routing Table + Binary

### Routing Table Format

```c
struct RoutingTable {
    uint8_t layer_count;
    uint8_t nodes_per_layer;

    // Per-node configuration
    struct NodeConfig {
        uint8_t shape_id;      // Which shape to execute
        uint8_t input_a;       // Source for first operand
        uint8_t input_b;       // Source for second operand
    } nodes[MAX_LAYERS][MAX_NODES];
};
```

### Compilation Pipeline

```
Fabric Spec (YAML)
       │
       ▼
┌─────────────────┐
│  GCS Compiler   │
│  (gcs-compile)  │
└────────┬────────┘
         │
    ┌────┴────┬─────────┐
    ▼         ▼         ▼
routing.bin  kernel.cu  kernel.ptx
    │         │         │
    └────┬────┴─────────┘
         ▼
   libfabric.so (executable)
```

---

## Stage 5: EXECUTE (Hardware Mapping)

**Input:** Compiled Fabric + Data

**Process:**
1. Load routing table to constant memory
2. Stream data through fabric
3. Collect results

**Output:** Computed Results

### Execution API

```cpp
#include <gcs/fabric.hpp>

int main() {
    // Load compiled fabric
    gcs::Fabric sorter = gcs::load("8-element-sorter.gcs");

    // Allocate buffers
    auto input = gcs::Buffer<uint8_t>(1024 * 1024, 8);  // 1M samples, 8 elements each
    auto output = gcs::Buffer<uint8_t>(1024 * 1024, 8);

    // Fill input
    input.randomize();

    // Execute
    sorter.execute(input, output);

    // Results in output buffer
    return 0;
}
```

### Performance Monitoring

```cpp
gcs::Profile profile = sorter.profile(input, output);

std::cout << "Samples/sec: " << profile.samples_per_sec << "\n";
std::cout << "Ops/sec: " << profile.ops_per_sec << "\n";
std::cout << "Latency: " << profile.latency_us << " μs\n";
std::cout << "Utilization: " << profile.sm_utilization << "%\n";
```

---

## Stage 6: OPTIMIZE (Topology Search)

**Input:** Working Fabric + Performance Target

**Process:**
1. Profile current fabric
2. Identify bottlenecks
3. Search for better topologies
4. Verify equivalence
5. Iterate

**Output:** Optimized Fabric

### Optimization Strategies

1. **Layer Fusion** - Combine adjacent layers where possible
2. **Dead Node Elimination** - Remove unused shapes
3. **Topology Compression** - Find equivalent smaller structures
4. **Parallelism Expansion** - Add redundant paths for throughput
5. **Pipelining** - Insert registers for frequency

### Optimization Loop

```python
def optimize(fabric, target_ops_per_sec):
    current = fabric

    while current.profile().ops_per_sec < target_ops_per_sec:
        candidates = generate_mutations(current)
        candidates = [c for c in candidates if verify_equivalent(c, fabric)]
        candidates.sort(key=lambda c: c.profile().ops_per_sec, reverse=True)

        if candidates[0].profile().ops_per_sec > current.profile().ops_per_sec:
            current = candidates[0]
        else:
            break  # Local optimum

    return current
```

---

## Directory Structure

```
gcs/
├── shapes/                 # Shape definitions
│   ├── core.yaml          # Universal shapes
│   ├── crypto.yaml        # Crypto-specific shapes
│   └── signal.yaml        # Signal processing shapes
│
├── fabrics/               # Fabric specifications
│   ├── sorter_8.yaml
│   ├── sha256_round.yaml
│   └── fir_filter.yaml
│
├── proofs/                # Verification certificates
│   ├── sorter_8.cert
│   └── sha256_round.cert
│
├── compiled/              # Compiled outputs
│   ├── sorter_8.gcs
│   ├── sorter_8.ptx
│   └── libsorter_8.so
│
├── tools/
│   ├── gcs-discover       # Shape mining tool
│   ├── gcs-compose        # Fabric designer
│   ├── gcs-verify         # Verification tool
│   ├── gcs-compile        # Compiler
│   └── gcs-optimize       # Optimization tool
│
└── runtime/
    ├── libgcs.so          # Runtime library
    └── gcs.hpp            # C++ API
```

---

## Example: Building a SHA-256 Round

### 1. Discover Shapes

```bash
$ gcs-discover --domain crypto --algorithm sha256

Identified shapes:
  - ADD32: 32-bit modular addition
  - XOR32: 32-bit XOR
  - AND32: 32-bit AND
  - NOT32: 32-bit NOT
  - ROTR32: 32-bit rotate right
  - SHR32: 32-bit shift right

Completeness: VERIFIED (all SHA-256 operations expressible)
```

### 2. Compose Fabric

```bash
$ gcs-compose --spec sha256_round.yaml

Fabric: sha256_round
  Inputs: 8 (A, B, C, D, E, F, G, H) + 1 (K+W)
  Outputs: 8 (A', B', C', D', E', F', G', H')
  Layers: 4
  Shapes: 32

Generated: fabrics/sha256_round.yaml
```

### 3. Verify

```bash
$ gcs-verify sha256_round.yaml --against reference/sha256.py

Shape verification: 6/6 PASS
Composition verification: PASS
End-to-end test: 1000/1000 PASS
Properties:
  - Deterministic: VERIFIED
  - Constant-time: VERIFIED

Certificate: proofs/sha256_round.cert
```

### 4. Compile

```bash
$ gcs-compile sha256_round.yaml --target cuda

Generated:
  - compiled/sha256_round.gcs (routing table)
  - compiled/sha256_round.cu (CUDA kernel)
  - compiled/sha256_round.ptx (GPU binary)
  - compiled/libsha256_round.so (shared library)
```

### 5. Execute

```cpp
#include <gcs/fabric.hpp>

auto sha256_round = gcs::load("sha256_round.gcs");
sha256_round.execute(state, next_state);
```

### 6. Optimize

```bash
$ gcs-optimize sha256_round.gcs --target 10B-ops

Current: 6.2B ops/sec
Searching...
  Candidate 1: 6.8B ops/sec (layer fusion)
  Candidate 2: 7.4B ops/sec (parallelism expansion)
  Candidate 3: 9.1B ops/sec (combined)

Optimized: 9.1B ops/sec
Saved: compiled/sha256_round_optimized.gcs
```

---

## Integration with Hollywood Squares OS

The GCS pipeline produces fabrics. Hollywood Squares OS orchestrates them.

```
Hollywood Squares OS (Coordination)
            │
            │ "Run sha256_round on these 1M blocks"
            ▼
    ┌───────────────────┐
    │   GCS Runtime     │
    │                   │
    │  ┌─────────────┐  │
    │  │ sha256_round│  │ ◄── Compiled Fabric
    │  │   fabric    │  │
    │  └─────────────┘  │
    │                   │
    └─────────┬─────────┘
              │
              ▼
         GPU Hardware
       (85B ops/sec)
```

---

## Next Steps

1. **Implement gcs-discover** - Shape mining from domain analysis
2. **Implement gcs-compose** - Visual fabric designer
3. **Implement gcs-verify** - SMT-based verification
4. **Implement gcs-compile** - YAML → CUDA compiler
5. **Implement gcs-optimize** - Topology search
6. **Build runtime library** - Unified C++ API
7. **Create shape libraries** - Crypto, signal, ML, etc.
