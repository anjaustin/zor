# Geocadesia Usage Guide

*A practical guide to the Kingdom of Shapes*

```
"It's all in the reflexes."
```

---

## Installation

Geocadesia is part of TRIX Forge. From the shapes directory:

```python
# Add to Python path
import sys
sys.path.insert(0, '/path/to/trixc/forge/shapes')

# Import the library
from geocadesia import XOR, ReLU, FullAdder, catalog
```

---

## Quick Start

### Using Individual Shapes

```python
from geocadesia import XOR, AND, OR, NOT
from geocadesia import ReLU, Sigmoid, GELU, Softmax
from geocadesia import HalfAdder, FullAdder

# Logic gates (continuous)
xor = XOR()
print(xor(0.5, 0.7))      # 0.8 (disagreement)
print(xor(1.0, 1.0))      # 0.0 (same → cancel)

# Logic gates (discrete)
print(XOR.discrete(1, 0)) # 1

# Activations
relu = ReLU()
print(relu(-5))           # 0.0
print(relu(5))            # 5.0

sigmoid = Sigmoid()
print(sigmoid(0))         # 0.5

# Compounds
adder = FullAdder()
sum_bit, carry = adder(1, 1, 0)
print(f"1 + 1 = {int(carry)}{int(sum_bit)}")  # "10" in binary
```

### Using the Catalog

```python
from geocadesia import catalog

# List all shapes
print(catalog)  # <Geocadesia Catalog: 26 shapes across 7 kingdoms>

# Get statistics
print(catalog.stats())
# {'total': 26, 'logic': 7, 'arithmetic': 6, ...}

# List shapes in a kingdom
for shape in catalog.list_kingdom("logic"):
    print(f"{shape.name}: {shape.formula}")

# Find shapes by criteria
frozen_binary = catalog.find(frozen=True, arity="binary")
for shape in frozen_binary:
    print(shape.name)

# Get detailed info on a shape
xor_shape = catalog.get("xor")
print(xor_shape.info())
```

---

## The Seven Kingdoms

### Logic Kingdom

Boolean operations — the foundation of computation.

```python
from geocadesia import XOR, AND, OR, NOT, NAND, NOR, XNOR

# All gates work with continuous values in [0, 1]
# and return exact results for discrete {0, 1}

# Truth table verification
for a in [0, 1]:
    for b in [0, 1]:
        print(f"XOR({a}, {b}) = {XOR()(a, b)}")
```

**Key insight**: These are *differentiable* logic gates. They compute exact boolean logic at binary inputs, but also work for continuous "probability-like" inputs.

### Arithmetic Kingdom

Numeric operations and binary arithmetic.

```python
from geocadesia import Add, Sub, Mul, Neg
from geocadesia import HalfAdder, FullAdder

# Elemental operations
print(Add()(3, 4))     # 7
print(Mul()(0.5, 0.5)) # 0.25

# Binary arithmetic
ha = HalfAdder()
sum_bit, carry = ha(1, 1)
print(f"1 + 1: sum={sum_bit}, carry={carry}")

fa = FullAdder()
sum_bit, carry = fa(1, 1, 1)
print(f"1 + 1 + 1: sum={sum_bit}, carry={carry}")
```

### Activation Kingdom

Nonlinearities for neural networks.

```python
from geocadesia import ReLU, Sigmoid, Tanh, GELU, Swish, Softmax, LeakyReLU

# The workhorses
relu = ReLU()
sigmoid = Sigmoid()
gelu = GELU()

# Softmax for probabilities
softmax = Softmax()
probs = softmax([2.0, 1.0, 0.1])
print(probs)  # [0.659, 0.242, 0.099]
print(sum(probs))  # 1.0

# Leaky ReLU with custom alpha
leaky = LeakyReLU(alpha=0.1)
print(leaky(-1.0))  # -0.1
```

### Normalization Kingdom

Stabilize training with statistical transforms.

```python
from geocadesia import LayerNorm, RMSNorm

# Layer normalization
ln = LayerNorm(eps=1e-5)
normalized = ln([1.0, 2.0, 3.0, 4.0])
print(normalized)  # Zero mean, unit variance

# RMS normalization (simpler)
rms = RMSNorm()
normalized = rms([3.0, 4.0])
print(normalized)  # Unit RMS
```

### Pooling Kingdom

Reduce dimensions by summarizing.

```python
from geocadesia import MaxPool, AvgPool, SumPool, MinPool

data = [0.2, 0.8, 0.5, 0.3]

print(MaxPool()(data))  # 0.8 (strongest)
print(AvgPool()(data))  # 0.45 (mean)
print(SumPool()(data))  # 1.8 (total)
print(MinPool()(data))  # 0.2 (weakest)
```

---

## Querying the Catalog

The catalog is your guide to Geocadesia.

### List by Kingdom

```python
from geocadesia import catalog

# All kingdoms
print(catalog.kingdoms())
# ['logic', 'arithmetic', 'activation', 'normalization', 'linear', 'attention', 'pooling']

# Shapes in a specific kingdom
logic_shapes = catalog.list_kingdom("logic")
for s in logic_shapes:
    print(f"  {s.name}")
```

### Find by Properties

```python
# All frozen shapes
frozen = catalog.find(frozen=True)

# All binary operations
binary = catalog.find(arity="binary")

# Frozen binary shapes in the logic kingdom
logic_binary = catalog.find(
    kingdom="logic",
    frozen=True,
    arity="binary"
)

# Compound shapes only
compounds = catalog.find(shape_type="compound")
```

### Get Shape Details

```python
# Get a specific shape
xor = catalog.get("xor")

# Access its properties
print(xor.name)        # "xor"
print(xor.kingdom)     # Kingdom.LOGIC
print(xor.formula)     # "a ⊕ b = a + b - 2ab"
print(xor.definition)  # "Exclusive OR..."
print(xor.see_also)    # ["and", "or", "xnor"]

# Pretty print full info
print(xor.info())

# Use it as a function
result = xor(0.5, 0.5)
```

---

## Building with Shapes

### Composition Example: 4-bit Adder

```python
from geocadesia import FullAdder

def ripple_carry_add(a: list, b: list) -> tuple:
    """Add two 4-bit numbers using full adders."""
    assert len(a) == len(b) == 4

    fa = FullAdder()
    result = []
    carry = 0

    # LSB to MSB
    for i in range(4):
        sum_bit, carry = fa.discrete(a[i], b[i], carry)
        result.append(sum_bit)

    return result, carry

# 5 + 3 = 8
a = [1, 0, 1, 0]  # 5 in binary (LSB first)
b = [1, 1, 0, 0]  # 3 in binary
sum_bits, overflow = ripple_carry_add(a, b)
print(sum_bits, overflow)  # [0, 0, 0, 1], 0 → 8
```

### Composition Example: Soft Logic

```python
from geocadesia import XOR, AND, OR

# Fuzzy/probabilistic logic
def fuzzy_majority(a, b, c):
    """Returns 1 if at least 2 of 3 inputs are 1."""
    return OR()(
        OR()(AND()(a, b), AND()(b, c)),
        AND()(a, c)
    )

# With probabilities
print(fuzzy_majority(0.9, 0.9, 0.1))  # High (two are high)
print(fuzzy_majority(0.1, 0.1, 0.9))  # Low (only one is high)
```

---

## Shape Anatomy

Every shape in Geocadesia has:

| Field | Description |
|-------|-------------|
| `name` | Lowercase identifier |
| `kingdom` | Which kingdom (logic, activation, etc.) |
| `shape_type` | Elemental or compound |
| `arity` | Unary, binary, ternary, or n-ary |
| `frozen` | Frozen, parameterized, or partial |
| `formula` | Mathematical definition |
| `definition` | Prose explanation |
| `fn` | The actual computation |
| `built_from` | Components (for compounds) |
| `see_also` | Related shapes |
| `examples` | Input → output pairs |

Access any of these on a shape object:

```python
shape = catalog.get("full_adder")
print(shape.built_from)  # ['xor', 'and', 'or']
print(shape.arity)       # Arity.TERNARY
```

---

## Philosophy

Geocadesia embodies three principles:

1. **Shapes are functions**: They compute. They're not just descriptions.

2. **The catalog enables discovery**: Find what you need. Learn what exists.

3. **Documentation IS implementation**: The scrolls and the code are unified.

---

## Contributing

To add a shape to Geocadesia:

1. **Implement** in the appropriate kingdom file (`logic.py`, `activation.py`, etc.)
2. **Register** with the `@shape` decorator
3. **Document** with a markdown file in `elements/` or `compounds/`
4. **Test** that it works: `from geocadesia import YourShape`

Every shape needs:
- Mathematical formula
- Prose definition
- Working implementation
- Examples
- Relationships to other shapes

---

*"A library enables composition by making shapes findable, comparable, and combinable."*

*Welcome to Geocadesia.*
