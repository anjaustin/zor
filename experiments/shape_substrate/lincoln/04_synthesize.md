# Synthesis: Shape Substrate

The clean cut. Concrete specification for production.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHAPE SUBSTRATE STACK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TRAINING (Python/PyTorch)                                       │   │
│  │    - Polynomial shapes with gradients                            │   │
│  │    - Signature/binding site optimization                         │   │
│  │    - Tap pattern evolution (optional)                            │   │
│  │    - Output: trained parameters                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ export                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INTERMEDIATE REPRESENTATION                                     │   │
│  │    - Frozen tap patterns (bit masks)                             │   │
│  │    - Frozen signatures (ternary vectors)                         │   │
│  │    - Composition graph (how atoms form molecules)                │   │
│  │    - Format: JSON or binary blob                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ compile                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  EXECUTION (CUDA/Verilog)                                        │   │
│  │    - LFSR fabric instantiation                                   │   │
│  │    - Native XOR operations                                       │   │
│  │    - 35+ Tbits/sec throughput                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### Decision 1: Separate Training and Execution Completely

**Because:** The 1000x speed gap makes mixed execution pointless. Train fully, then export fully.

**Implementation:**
- Training module: `trix.substrate.train`
- Export module: `trix.substrate.export`
- Execution module: `trix.substrate.cuda` or `trix.substrate.verilog`

### Decision 2: Atoms Are the Unit of Training

**Because:** Molecules are compositions of atoms. Train atoms with specific properties, compose without retraining.

**Implementation:**
```python
# Define atom training objectives
atom_a = train_atom(objective="mixing", steps=1000)
atom_b = train_atom(objective="diffusion", steps=1000)

# Compose into molecule (no training)
molecule = compose([atom_a, atom_b], mode="serial")
```

### Decision 3: Three Composition Modes

**Because:** Serial, parallel, and hybrid cover all useful patterns.

**Implementation:**
```python
# Serial: A → B → C (output chains)
serial = compose([a, b, c], mode="serial")

# Parallel: A ⊕ B ⊕ C (outputs XOR'd)
parallel = compose([a, b, c], mode="parallel")

# Hybrid: (A→B) ⊕ (C→D)
hybrid = compose([
    compose([a, b], mode="serial"),
    compose([c, d], mode="serial")
], mode="parallel")
```

### Decision 4: Proteins Are Molecules + Routing

**Because:** The protein abstraction adds binding/selection to composition.

**Implementation:**
```python
class Protein:
    def __init__(self, molecule, binding_site):
        self.molecule = molecule
        self.binding_site = binding_site  # Learned signature

    def affinity(self, input):
        return dot(input, self.binding_site)

    def process(self, input):
        if self.affinity(input) > threshold:
            return self.molecule.execute(input)
        return None
```

### Decision 5: Keep TriX and Substrate as Parallel Interfaces

**Because:** Different audiences prefer different metaphors.

**Implementation:**
- `trix.nn.HierarchicalTriXFFN` - ML audience, "tiles"
- `trix.substrate.ProteinCell` - Bio-curious, "proteins"
- Same underlying math, different APIs

---

## Implementation Spec

### Module: `trix.substrate.atoms`

```python
class Atom:
    """A single 512-bit LFSR with trained tap pattern."""

    def __init__(self, tap_mask: int, signature: Tensor):
        self.tap_mask = tap_mask      # 64-bit mask for feedback
        self.signature = signature    # Binding site (d_model vector)

    def step(self, state: Tensor, inject: int = 0) -> Tensor:
        """One LFSR step with optional injection."""
        ...

    def execute(self, state: Tensor, steps: int) -> Tensor:
        """Run N steps, return final state."""
        ...

    @classmethod
    def train(cls, objective: str, d_model: int, steps: int) -> 'Atom':
        """Train an atom for a specific objective."""
        ...
```

### Module: `trix.substrate.molecules`

```python
class Molecule:
    """Composition of atoms."""

    def __init__(self, atoms: List[Atom], mode: str):
        self.atoms = atoms
        self.mode = mode  # "serial", "parallel", "hybrid"

    def execute(self, state: Tensor, steps: int) -> Tensor:
        """Execute the molecular computation."""
        if self.mode == "serial":
            for atom in self.atoms:
                state = atom.execute(state, steps)
            return state
        elif self.mode == "parallel":
            outputs = [atom.execute(state, steps) for atom in self.atoms]
            return reduce(xor, outputs)
        ...
```

### Module: `trix.substrate.proteins`

```python
class Protein:
    """Molecule with binding site for routing."""

    def __init__(self, molecule: Molecule, binding_site: Tensor):
        self.molecule = molecule
        self.binding_site = binding_site

    def affinity(self, input: Tensor) -> float:
        return torch.dot(input.flatten(), self.binding_site.flatten())

    def fold(self, state: Tensor, steps: int) -> Tensor:
        """Execute if bound."""
        return self.molecule.execute(state, steps)


class Cell:
    """Collection of proteins with competitive binding."""

    def __init__(self, proteins: List[Protein]):
        self.proteins = proteins

    def process(self, input: Tensor, state: Tensor, steps: int) -> Tensor:
        affinities = [p.affinity(input) for p in self.proteins]
        winner = self.proteins[argmax(affinities)]
        return winner.fold(state, steps)
```

### Module: `trix.substrate.export`

```python
def export_cuda(cell: Cell, output_dir: str):
    """Export cell to CUDA kernel."""
    ...

def export_verilog(cell: Cell, output_dir: str):
    """Export cell to Verilog for FPGA/ASIC."""
    ...

def export_onnx(cell: Cell, output_path: str):
    """Export cell to ONNX (polynomial form)."""
    ...
```

---

## Success Criteria

### Phase 1: Core Library (Current)
- [x] Polynomial shapes work (XOR, AND, OR, etc.)
- [x] LFSR benchmark runs (35 Tbits/sec)
- [x] Composition works (serial, parallel, hybrid)
- [x] Protein demo works (binding, folding, output)
- [x] Onboarding tutorials complete (10 files)
- [x] Lincoln Manifold documentation complete

### Phase 2: Training Integration
- [ ] Atom training with gradient descent
- [ ] Signature optimization for routing
- [ ] End-to-end: train → export → execute
- [ ] Benchmark on real task

### Phase 3: Production Hardening
- [ ] CUDA kernel optimization
- [ ] Memory-mapped execution
- [ ] Multi-GPU support
- [ ] Verilog export for FPGA

### Phase 4: Ecosystem
- [ ] Integration with TriX training pipeline
- [ ] Docker container for reproducibility
- [ ] Documentation for external users
- [ ] Example applications

---

## Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `src/trix/substrate/__init__.py` | Module entry point | High |
| `src/trix/substrate/atoms.py` | Atom class | High |
| `src/trix/substrate/molecules.py` | Molecule composition | High |
| `src/trix/substrate/proteins.py` | Protein routing | High |
| `src/trix/substrate/train.py` | Training utilities | Medium |
| `src/trix/substrate/export.py` | Export to CUDA/Verilog | Medium |
| `tests/test_substrate.py` | Test suite | High |

---

## Timeline (No Dates, Just Sequence)

1. **Implement core classes** (Atom, Molecule, Protein, Cell)
2. **Add training utilities** (tap pattern optimization, signature learning)
3. **Create export pipeline** (Python → CUDA → executable)
4. **Build end-to-end demo** (train → export → benchmark)
5. **Harden for production** (memory management, error handling)
6. **Document for users** (API docs, tutorials, examples)

---

## The One-Liner

**Train shapes in Python, run shapes on silicon, get 1000x speedup.**

---

## Appendix: Verification Commands

```bash
# Run onboarding
cd experiments/shape_substrate
python 01_wait_what.py
python 09_put_it_together.py
python 10_protein_version.py

# Run CUDA benchmarks
./molecular_shapes
./protein_compute

# Verify Lincoln Manifold
ls lincoln/
# Should show: 01_raw.md, 02_nodes.md, 03_reflect.md, 04_synthesize.md
```

---

*Synthesis complete. The wood cuts itself.*
