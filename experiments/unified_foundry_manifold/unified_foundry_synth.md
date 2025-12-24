# SYNTHESIS: Unified Neural-Geometric Deterministic Systems Foundry

The clean cut. Concrete, actionable, implementable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FOUNDRY                                  │
│                     (trix.forge.foundry)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ AtomRegistry                                             │    │
│  │ ├── register(name, truth_fn) → ShapeTerms               │    │
│  │ ├── get(name) → ShapeTerms                              │    │
│  │ └── list() → [names]                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ CompositionEngine                                        │    │
│  │ ├── seq(a, b) → ShapeTerms                              │    │
│  │ ├── par(a, b) → ShapeTerms                              │    │
│  │ ├── sel(shapes) → CompositeShape                        │    │
│  │ └── rep(shape, n) → ShapeTerms                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SignatureDeriver                                         │    │
│  │ ├── from_terms(shape) → Tensor[sig_dim]                 │    │
│  │ └── from_behavior(shape, samples) → Tensor[sig_dim]     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ System (the built artifact)                              │    │
│  │ ├── execute(a, b, op) → int                             │    │
│  │ ├── execute_batch(a[], b[], op) → int[]                 │    │
│  │ ├── validate(exhaustive=True) → ValidationResult        │    │
│  │ ├── export_cuda(path)                                   │    │
│  │ ├── export_verilog(path)                                │    │
│  │ └── summary() → str                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Classes

### 1. Foundry

The main entry point.

```python
@dataclass
class Foundry:
    """Unified Neural-Geometric Deterministic Systems Foundry."""

    bits: int = 8
    atoms: Dict[str, ShapeTerms] = field(default_factory=dict)
    composites: Dict[str, CompositeShape] = field(default_factory=dict)

    def atom(self, name: str, truth_fn: Callable) -> "Foundry":
        """Register an atomic shape from truth function."""
        # Generate ShapeTerms from truth function
        shape = generate_from_truth(truth_fn, self.bits)
        self.atoms[name] = shape
        return self

    def compose(self, name: str, composition: Composition) -> "Foundry":
        """Register a composite shape."""
        # Evaluate composition to get ShapeTerms
        shape = composition.evaluate(self.atoms)
        self.composites[name] = shape
        return self

    def build(self) -> System:
        """Build executable system from registered shapes."""
        all_shapes = {**self.atoms, **self.composites}
        return System(shapes=all_shapes, bits=self.bits)
```

### 2. Composition Operators

```python
class Composition(ABC):
    """Abstract composition operator."""
    @abstractmethod
    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        pass

class Seq(Composition):
    """Sequential composition: output of A feeds input of B."""
    def __init__(self, a: str, b: str):
        self.a, self.b = a, b

    def evaluate(self, registry):
        a_terms = registry[self.a]
        b_terms = registry[self.b]
        return compose_sequential(a_terms, b_terms)

class Par(Composition):
    """Parallel composition: same input, concatenated outputs."""
    def __init__(self, *shapes: str):
        self.shapes = shapes

    def evaluate(self, registry):
        terms = [registry[s] for s in self.shapes]
        return compose_parallel(terms)

class Sel(Composition):
    """Selection composition: opcode selects shape."""
    def __init__(self, *shapes: str):
        self.shapes = shapes

    def evaluate(self, registry):
        terms = [registry[s] for s in self.shapes]
        return CompositeShape(terms)  # Special case: not a single ShapeTerms

class Rep(Composition):
    """Repetition: chain shape N times."""
    def __init__(self, shape: str, n: int):
        self.shape, self.n = shape, n

    def evaluate(self, registry):
        base = registry[self.shape]
        result = base
        for _ in range(self.n - 1):
            result = compose_sequential(result, base)
        return result

# Convenience functions
def seq(a, b): return Seq(a, b)
def par(*s): return Par(*s)
def sel(*s): return Sel(*s)
def rep(s, n): return Rep(s, n)
```

### 3. System (Built Artifact)

```python
class System:
    """An executable deterministic system."""

    def __init__(self, shapes: Dict[str, ShapeTerms], bits: int):
        self.shapes = shapes
        self.bits = bits
        self.backend = get_backend(bits=bits)
        self._signatures = self._derive_signatures()

    def _derive_signatures(self) -> Dict[str, Tensor]:
        """Derive ternary signatures from shape terms."""
        return {name: signature_from_terms(shape)
                for name, shape in self.shapes.items()}

    def execute(self, a: int, b: int, op: str) -> int:
        """Execute operation."""
        return self.backend.execute(op, a, b, bits=self.bits)

    def execute_batch(self, a: List[int], b: List[int], op: str) -> List[int]:
        """Batched execution."""
        return self.backend.execute_batch(op, a, b, bits=self.bits)

    def validate(self, exhaustive: bool = True) -> ValidationResult:
        """Validate all shapes against truth functions."""
        results = {}
        for name, shape in self.shapes.items():
            if exhaustive and self.bits <= 8:
                result = validate_exhaustive(shape, self.bits)
            else:
                result = validate_statistical(shape, self.bits)
            results[name] = result
        return ValidationResult(results)

    def export_cuda(self, path: str):
        """Export to CUDA."""
        export_cuda(self.shapes, path)

    def export_verilog(self, path: str):
        """Export to Verilog."""
        export_verilog(self.shapes, path)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"System: {len(self.shapes)} shapes, {self.bits}-bit",
            f"Shapes: {list(self.shapes.keys())}",
            f"Backend: {self.backend.info().name}",
        ]
        return "\n".join(lines)
```

---

## Key Algorithms

### generate_from_truth(fn, bits) → ShapeTerms

```python
def generate_from_truth(fn: Callable, bits: int) -> ShapeTerms:
    """Generate ShapeTerms from truth function."""

    # Determine arity (unary or binary)
    import inspect
    sig = inspect.signature(fn)
    arity = len(sig.parameters)

    if arity == 1:
        # Unary: use existing generator pattern
        return generate_unary_shape(fn, bits)
    elif arity == 2:
        # Binary: use existing generator pattern
        return generate_binary_shape(fn, bits)
    elif arity == 3:
        # Ternary (for full adder): extend pattern
        return generate_ternary_shape(fn, bits)
    else:
        raise ValueError(f"Unsupported arity: {arity}")
```

### signature_from_terms(shape) → Tensor

```python
def signature_from_terms(shape: ShapeTerms, sig_dim: int = 64) -> Tensor:
    """Derive ternary signature from term structure."""

    # Build a feature vector from term properties
    features = []

    for bit_terms in shape.bit_terms:
        for term in bit_terms.terms:
            # Encode term: (coefficient, variables)
            # Coefficient in {-2, -1, 1, 2} → normalized
            coef_feature = term.coefficient / 2.0

            # Variables as bit vector
            var_features = [1 if i in term.variables else 0
                           for i in range(shape.input_bits)]

            features.extend([coef_feature] + var_features)

    # Project to signature dimension
    feature_vec = torch.tensor(features, dtype=torch.float32)

    # Random projection (deterministic from shape name)
    torch.manual_seed(hash(shape.name) % (2**31))
    projection = torch.randn(len(feature_vec), sig_dim)
    projection = projection / projection.norm(dim=0, keepdim=True)

    # Project and ternarize
    signature = (feature_vec @ projection).sign()
    signature[signature == 0] = 1  # No zeros in signature

    return signature
```

### compose_sequential(a, b) → ShapeTerms

```python
def compose_sequential(a: ShapeTerms, b: ShapeTerms) -> ShapeTerms:
    """Compose shapes sequentially: b(a(x))."""

    # This requires substituting a's output bits into b's input bits
    # Complex but well-defined algebraically

    # For now, use behavioral composition (compute truth table of composition)
    def composed_truth(x):
        a_result = evaluate_shape(a, x)
        b_result = evaluate_shape(b, a_result)
        return b_result

    return generate_from_truth(composed_truth, a.output_bits)
```

---

## File Structure

```
trix/forge/
├── __init__.py          # Exports Foundry, System, operators
├── foundry.py           # NEW: Foundry class
├── composition.py       # NEW: Composition operators
├── system.py            # NEW: System class
├── signature.py         # NEW: signature_from_terms
├── term.py              # EXISTING: ShapeTerms (unchanged)
├── backend.py           # EXISTING: Backend layer (unchanged)
├── verilog.py           # EXISTING: Verilog export (unchanged)
├── cuda.py              # EXISTING: CUDA export (unchanged)
├── hardware.py          # EXISTING: Estimation (unchanged)
└── chip.py              # EXISTING: Chip DSL (becomes thin wrapper)
```

---

## Usage Examples

### Example 1: Simple ALU

```python
from trix.forge import Foundry, sel

foundry = Foundry(bits=8)

# Atoms
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.atom("or",  lambda a, b: a | b)
foundry.atom("add", lambda a, b: (a + b) & 0xFF)

# Build
alu = foundry.build()

# Validate (100% required)
result = alu.validate(exhaustive=True)
assert result.all_passed()

# Execute
print(alu.execute(42, 13, "xor"))  # 39

# Export
alu.export_cuda("output/alu_cuda/")
alu.export_verilog("output/alu_verilog/")
```

### Example 2: Ripple Carry Adder

```python
from trix.forge import Foundry, rep

foundry = Foundry(bits=1)  # 1-bit atoms

# Full adder atom (3 inputs: a, b, carry_in)
foundry.atom("full_adder", lambda a, b, c: (
    a ^ b ^ c,                           # sum
    (a & b) | (c & (a ^ b))              # carry
))

# 8-bit ripple adder
foundry.compose("ripple_add_8", rep("full_adder", 8))

adder = foundry.build()
result = adder.validate(exhaustive=True)
print(adder.execute(42, 13, "ripple_add_8"))  # 55
```

### Example 3: 6502 ALU

```python
from trix.forge import Foundry, sel

foundry = Foundry(bits=8)

# 6502 operations
foundry.atom("ADC", lambda a, b: (a + b) & 0xFF)
foundry.atom("SBC", lambda a, b: (a + (b ^ 0xFF) + 1) & 0xFF)
foundry.atom("AND", lambda a, b: a & b)
foundry.atom("ORA", lambda a, b: a | b)
foundry.atom("EOR", lambda a, b: a ^ b)
foundry.atom("ASL", lambda a, _: (a << 1) & 0xFF)
foundry.atom("LSR", lambda a, _: a >> 1)
foundry.atom("INC", lambda a, _: (a + 1) & 0xFF)
foundry.atom("DEC", lambda a, _: (a - 1) & 0xFF)

alu_6502 = foundry.build()
result = alu_6502.validate(exhaustive=True)
print(f"6502 ALU: {len(alu_6502.shapes)} operations, validation: {result.all_passed()}")
```

---

## Success Criteria

- [ ] `Foundry` class with `atom()`, `compose()`, `build()`
- [ ] Composition operators: `seq`, `par`, `sel`, `rep`
- [ ] `System` class with `execute()`, `validate()`, `export_*()`
- [ ] `signature_from_terms()` - deterministic signature derivation
- [ ] Tests for all new components
- [ ] Examples run and produce correct output
- [ ] Existing tests still pass (no regression)

---

## What This Enables

1. **Single interface** for truth table → hardware
2. **Composition algebra** for building systems from atoms
3. **Deterministic signatures** from polynomial structure
4. **100/100 guarantee** via exhaustive validation
5. **Multi-target export** (CUDA, Verilog, future: ASIC)

---

## Implementation Order

1. `foundry.py` - Foundry class (minimal, no composition)
2. `system.py` - System class
3. Tests for Foundry + System
4. `composition.py` - Composition operators
5. `signature.py` - Term-based signature derivation
6. Full test suite
7. Examples and documentation

Estimated: ~500 lines of new code, building on existing infrastructure.

---

*The wood cuts itself.*
