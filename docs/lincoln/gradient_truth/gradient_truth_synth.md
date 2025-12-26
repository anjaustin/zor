# Synthesis: Gradient Truth — A Training Paradigm Beyond STE

## The Principle

**Gradients flow only where there is genuine uncertainty. Structure stands apart as truth.**

This is not a trick to approximate discreteness. It is a recognition that neural network training has been conflating two ontologically distinct categories:

| Category | Nature | Method | Examples |
|----------|--------|--------|----------|
| **Structure** | Exists, is discovered | Derivation, evolution, distillation | XOR polynomial, tap patterns, converged topologies |
| **Navigation** | Uncertain, is learned | Gradient descent | Routing weights, scales, confidences |

STE tries to learn structure with navigation tools. Gradient Truth respects the boundary.

---

## Architecture: The Three-Layer Decomposition

Every computation in this paradigm decomposes into:

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: ROUTING                              │
│                                                                  │
│   Continuous, learned, receives gradients.                      │
│   Maps input to shape selection.                                │
│   Implementation: softmax over signatures, learned projection   │
│                                                                  │
│   Parameters: O(d_model × num_shapes) - fully differentiable    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: SHAPE BANK                           │
│                                                                  │
│   Discrete, frozen, no gradients.                               │
│   Library of computational primitives.                          │
│   Discovered via: derivation, evolution, or distillation        │
│                                                                  │
│   Parameters: 0 (frozen) or evolved (non-gradient)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: MAGNITUDE                            │
│                                                                  │
│   Continuous, learned, receives gradients.                      │
│   Scales shape output to match required magnitude.              │
│   Implementation: learnable scale per shape or per output dim   │
│                                                                  │
│   Parameters: O(num_shapes) or O(d_output) - fully differentiable│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Training Protocol: Four Phases

### Phase 0: Shape Genesis (Pre-training, no gradients)

**Objective:** Build the shape library.

**Methods (choose appropriate for domain):**

| Method | When to Use | How |
|--------|-------------|-----|
| **Derivation** | Mathematical domains (logic, arithmetic) | Write down the polynomials. XOR = a+b-2ab. AND = ab. |
| **Evolution** | Structure-searchable domains (circuits, LFSRs) | Genetic algorithm on topology, fitness = correctness |
| **Enumeration** | Small discrete spaces | Try all, keep valid |
| **Distillation** | Data-rich domains, no theory | Train conventional net with STE → observe converged patterns → freeze as shapes |

**Output:** Frozen shape bank S = {s₁, s₂, ..., sₙ}

**Key principle:** This phase uses NO gradients on structure. Evolution uses fitness, not loss. Distillation observes, then freezes.

### Phase 1: Routing Warmup (Gradient training begins)

**Objective:** Learn to route inputs to appropriate shapes.

**What's trained:** Routing layer only (Layer 1)
**What's frozen:** Shape bank (Layer 2), Magnitude scales initialized to 1.0

**Loss:** Task loss (cross-entropy, MSE, etc.) flows back through selected shape to routing weights.

**Duration:** Until routing stabilizes (routing entropy decreases, shape utilization balances)

**Key insight:** Gradients flow cleanly because routing is continuous. No STE needed.

### Phase 2: Magnitude Calibration (Full gradient training)

**Objective:** Fine-tune output scales for each shape.

**What's trained:** Routing layer (Layer 1) + Magnitude layer (Layer 3)
**What's frozen:** Shape bank (Layer 2)

**Loss:** Full task loss.

**Duration:** Until convergence.

**This is standard gradient descent.** All trainable parameters are continuous. The discrete shapes pass gradients through to the routing decision without needing STE.

### Phase 3: Deployment (Inference)

**All shapes frozen.** Routing can optionally be compiled to lookup tables if input signatures are discrete.

**The result:** A network where:
- Discrete structure is mathematically exact
- Continuous decisions are gradient-optimized
- No STE was used in the final architecture

---

## Shape Composition: The Hierarchy

Simple shapes compose into complex computations:

```
LEVEL 0: Atomic Shapes
  xor(a,b) = a + b - 2ab
  and(a,b) = ab
  or(a,b) = a + b - ab
  not(a) = 1 - a

LEVEL 1: Molecular Shapes (compositions of atoms)
  full_adder(a, b, cin) = {
    sum: xor(xor(a, b), cin)
    cout: or(and(a, b), and(xor(a, b), cin))
  }

LEVEL 2: Organelle Shapes (compositions of molecules)
  ripple_add_8(a[8], b[8]) = chain(full_adder, 8)

LEVEL 3: Cellular Shapes (full operations)
  alu_op(a, b, opcode) = route(opcode) → {add, sub, xor, and, or, ...}
```

**The routing layer selects at each level.** Opcode routes to operation, operation routes to bit-level shapes.

---

## Gradient Flow Analysis

Why this works without STE:

```
Forward:
  input → routing (continuous) → shape selection (discrete choice) → 
  shape execution (frozen, exact) → magnitude (continuous) → output

Backward:
  ∂L/∂output → ∂L/∂magnitude (direct) → ∂L/∂shape_output (passthrough) →
  ∂L/∂routing (the gradient of "which shape was selected")
```

**The key:** The routing decision is made continuously (soft attention over shapes or Gumbel-softmax). The discrete shape execution doesn't need gradients because it's not parameterized.

It's like attention: you don't backprop through the attended values, you backprop through the attention weights.

---

## Implementation Spec

### GradientTruthFFN

```python
class GradientTruthFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        shape_bank: FrozenShapeBank,  # Pre-built, frozen
        routing_temp: float = 1.0,
    ):
        self.d_model = d_model
        self.shape_bank = shape_bank  # No gradients
        self.num_shapes = len(shape_bank)
        
        # Layer 1: Routing (continuous, learned)
        self.router = nn.Linear(d_model, self.num_shapes)
        
        # Layer 3: Magnitude (continuous, learned)
        self.scales = nn.Parameter(torch.ones(self.num_shapes))
        
        self.routing_temp = routing_temp
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Routing: soft selection over shapes
        logits = self.router(x) / self.routing_temp
        weights = F.softmax(logits, dim=-1)  # [batch, num_shapes]
        
        # Execute all shapes (frozen, no grad needed)
        with torch.no_grad():
            shape_outputs = self.shape_bank(x)  # [batch, num_shapes, d_out]
        
        # But we DO need grad through the weighted combination
        shape_outputs = shape_outputs.detach()  # Ensure no shape grads
        
        # Apply learned scales
        scaled = shape_outputs * self.scales  # broadcast
        
        # Weighted combination (gradients flow through weights)
        output = (weights.unsqueeze(-1) * scaled).sum(dim=1)
        
        return output
```

### FrozenShapeBank

```python
class FrozenShapeBank(nn.Module):
    """Library of frozen computational shapes."""
    
    def __init__(self, shapes: List[Callable]):
        super().__init__()
        self.shapes = shapes
        
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all shapes, return stacked outputs."""
        outputs = [shape(x) for shape in self.shapes]
        return torch.stack(outputs, dim=1)
    
    def __len__(self):
        return len(self.shapes)
```

---

## Comparison: STE vs Gradient Truth

| Aspect | STE Approach | Gradient Truth |
|--------|--------------|----------------|
| Discrete weights | Learned with fake gradients | Discovered, then frozen |
| Gradient correctness | Wrong (identity through step) | Correct (only through continuous params) |
| Training stability | Can be unstable | Stable (standard optim) |
| Interpretability | Opaque (what did it learn?) | Clear (which shape was selected?) |
| Deployment | Quantize, hope it works | Already exact, shapes are truth |
| Shape discovery | Implicit (emerges during training) | Explicit (Phase 0) |

---

## When to Use Which

| Situation | Recommendation |
|-----------|----------------|
| Known mathematical domain (logic, arithmetic) | Gradient Truth with derived shapes |
| Searchable structure space (circuits, patterns) | Gradient Truth with evolved shapes |
| Unknown domain, have data | Distillation: STE → freeze → retrain routing |
| Need maximum flexibility | Hybrid: continuous early layers, GT later layers |
| Prototype/exploration | STE is fine, convert to GT when structure stabilizes |

---

## Success Criteria

- [ ] Shape bank executes with 0 learned parameters
- [ ] Routing layer converges with standard optimizers (no STE)
- [ ] Magnitude scales calibrate properly
- [ ] Task performance matches or exceeds STE baseline
- [ ] Gradients are mathematically correct throughout
- [ ] Inference is exact (no quantization gap)

---

## The One-Sentence Summary

**Discover structure (no gradients), learn routing (gradients), scale magnitude (gradients) — and the discrete/continuous boundary is never violated.**

---

## Appendix: Connection to TriX Components

| Gradient Truth Component | TriX Equivalent | Location |
|--------------------------|-----------------|----------|
| FrozenShapeBank | FrozenShapeBank, Frozen6502Net | nn/frozen.py |
| Routing layer | HierarchicalTriXFFN routing | nn/hierarchical.py |
| Magnitude scales | VGem's output_scale, tile scales | nn/hierarchical.py |
| Distillation | XOR superposition compression | nn/xor_superposition.py |
| Shape composition | Atoms → Molecules → Proteins | docs/SHAPE_SUBSTRATE.md |
| Manifold interpretation | Mesa 11 UAT | docs/MESA11_UAT.md |

**TriX already has the pieces. Gradient Truth names the principle that unifies them.**
