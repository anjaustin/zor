# Taxonomy of Shapes

*How Geocadesia organizes computational geometry*

---

## Two Fundamental Distinctions

### 1. Elemental vs Compound

**Elemental shapes** are atomic — they cannot be decomposed into simpler shapes within Geocadesia. They are the alphabet.

**Compound shapes** are molecular — they are built from elements (and possibly other compounds). They are words.

**Note:** "Elemental" is defined by *conceptual utility*, not mathematical irreducibility. ReLU could be decomposed into `max(0, x)`, but practitioners think in terms of "ReLU" — so ReLU is elemental.

### 2. Frozen vs Parameterized

**Frozen shapes** have no learnable parameters. Their behavior is fixed by their definition. XOR is XOR forever.

**Parameterized shapes** include weights that are learned during training. A linear layer's behavior depends on its weights.

**Partial** shapes have some fixed behavior and some learnable components. PReLU has a learnable negative slope.

---

## The Seven Kingdoms

### Kingdom: Logic

*The foundation of discrete computation*

| Property | Value |
|----------|-------|
| Domain | Boolean (0/1) or continuous [0,1] |
| Frozen | Always |
| Differentiable | No (discrete), Yes (continuous relaxation) |
| Character | Exact, foundational, combinatorial |

**Elements:**
- XOR — exclusive or
- AND — logical conjunction
- OR — logical disjunction
- NOT — logical negation
- NAND — NOT AND (universal gate)
- NOR — NOT OR (universal gate)
- XNOR — equivalence

**Compounds:**
- half_adder — XOR + AND
- full_adder — compound adder with carry

**Philosophy:** Logic gates are the primordial shapes. All digital computation reduces to them. In neural contexts, we use differentiable relaxations.

---

### Kingdom: Arithmetic

*Numeric computation without learning*

| Property | Value |
|----------|-------|
| Domain | Real numbers |
| Frozen | Always |
| Differentiable | Mostly (except discontinuities) |
| Character | Frozen, composable, numeric |

**Elements:**
- add — binary addition
- sub — binary subtraction
- mul — binary multiplication
- div — binary division
- neg — unary negation
- abs — absolute value
- sign — sign function
- sqrt — square root
- exp — exponential
- log — natural logarithm

**Compounds:**
- half_adder — produces sum and carry
- full_adder — adds with carry-in
- ripple_carry — N-bit addition

**Philosophy:** Arithmetic shapes perform fixed numeric operations. They're the building blocks of both classical and neural computation.

---

### Kingdom: Activation

*Nonlinearities that enable learning*

| Property | Value |
|----------|-------|
| Domain | Real numbers |
| Frozen | Mostly (PReLU is partial) |
| Differentiable | Yes (some piecewise) |
| Character | Nonlinear, differentiable, pointwise |

**Elements:**
- ReLU — max(0, x), the workhorse
- sigmoid — logistic curve, outputs (0,1)
- tanh — hyperbolic tangent, outputs (-1,1)
- GELU — Gaussian Error Linear Unit
- swish — x·sigmoid(x), self-gated
- softmax — probability distribution
- leaky_relu — allows negative slope
- elu — exponential linear unit
- selu — scaled ELU with self-normalization
- softplus — smooth ReLU approximation

**Compounds:**
- PReLU — ReLU with learnable negative slope (partial)
- mish — x·tanh(softplus(x))

**Philosophy:** Activations break linearity. Without them, deep networks collapse to a single linear transformation. They're what make depth meaningful.

---

### Kingdom: Normalization

*Statistical transforms for training stability*

| Property | Value |
|----------|-------|
| Domain | Tensors |
| Frozen | Mixed (statistics frozen, affine optional) |
| Differentiable | Yes |
| Character | Statistical, stabilizing |

**Elements:**
- mean — compute mean
- variance — compute variance
- standardize — (x - μ) / σ

**Compounds:**
- layer_norm — normalize across features (frozen stats)
- layer_norm_affine — with learnable scale/shift (partial)
- rms_norm — normalize by RMS only
- batch_norm — normalize across batch (requires running stats)

**Philosophy:** Normalization shapes stabilize training by controlling activation magnitudes. They exist because raw activations tend toward pathology.

---

### Kingdom: Linear

*Matrix operations with learnable weights*

| Property | Value |
|----------|-------|
| Domain | Tensors |
| Frozen | No (requires learned weights) |
| Differentiable | Yes |
| Character | Linear, parameterized, foundational |

**Elements:**
- matmul — matrix multiplication (frozen operation)
- matvec — matrix-vector product (frozen operation)

**Compounds:**
- linear — Wx + b (parameterized)
- conv1d — 1D convolution (parameterized)
- conv2d — 2D convolution (parameterized)
- embedding — lookup table (parameterized)

**Philosophy:** Linear shapes do the heavy lifting of neural networks. The weights are where learning happens. The shapes themselves are frozen operations on those weights.

---

### Kingdom: Attention

*Routing mechanisms that select and combine*

| Property | Value |
|----------|-------|
| Domain | Sequences of vectors |
| Frozen | No (Q, K, V projections learned) |
| Differentiable | Yes |
| Character | Complex, routing, parameterized |

**Elements:**
- dot_product — basic similarity
- scale — divide by sqrt(d)
- softmax_row — row-wise softmax

**Compounds:**
- scaled_dot_product — core attention operation
- multi_head — parallel attention heads
- cross_attention — attend across sequences
- self_attention — attend within sequence

**Philosophy:** Attention shapes route information dynamically. Unlike convolution (fixed spatial patterns) or recurrence (fixed temporal patterns), attention learns *what* to look at.

---

### Kingdom: Pooling

*Reduction operations that summarize*

| Property | Value |
|----------|-------|
| Domain | Tensors |
| Frozen | Always |
| Differentiable | Yes (max is piecewise) |
| Character | Frozen, reducing, summarizing |

**Elements:**
- max_pool — take maximum
- avg_pool — take average
- sum_pool — take sum
- min_pool — take minimum

**Compounds:**
- global_max_pool — max across spatial dims
- global_avg_pool — mean across spatial dims
- adaptive_pool — pool to target size

**Philosophy:** Pooling shapes reduce dimensionality while (hopefully) preserving important information. They trade resolution for robustness.

---

## Cross-Kingdom Relationships

Some shapes span kingdoms or combine them:

```
Logic + Arithmetic → half_adder, full_adder
Activation + Linear → gated linear units
Normalization + Activation → SELU (self-normalizing)
Attention + Linear → transformer layer
```

The kingdoms are not silos — they're regions of a continuous space.

---

## The Frozen Boundary

The most important distinction in Geocadesia:

| Frozen | Meaning | Examples |
|--------|---------|----------|
| **Yes** | No learnable parameters. Behavior is definition. | XOR, ReLU, max_pool |
| **No** | Requires learned weights. Behavior depends on training. | linear, conv2d, attention |
| **Partial** | Core frozen, some learnable components. | PReLU, layer_norm_affine |

**The unique contribution of TRIX Forge is frozen shapes.** We compute with geometry, not gradients. Geocadesia documents both, but the frozen shapes are the heart.

---

## Discovering New Shapes

When you encounter a new computational pattern, ask:

1. **Is it atomic or molecular?** Can it be built from existing shapes?
2. **Which kingdom?** What domain does it operate on?
3. **Frozen or parameterized?** Does it have learnable weights?
4. **What arity?** Unary, binary, n-ary?

If it's genuinely new and useful, it belongs in Geocadesia.

*"A library enables composition by making shapes findable, comparable, and combinable."*
