# TriX Quick Start Guide

A hands-on tutorial for researchers getting started with TriX.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Basic familiarity with transformer architectures

## Installation

```bash
git clone https://github.com/your-org/trix.git
cd trix
pip install -e .
```

Verify:
```bash
python -c "from trix import HierarchicalTriXFFN; print('Ready!')"
```

---

## Tutorial 1: Your First TriX Layer

### Step 1: Create a Simple FFN

```python
import torch
from trix import HierarchicalTriXFFN

# Configuration
d_model = 128
num_tiles = 16
batch_size = 4
seq_len = 32

# Create the FFN
ffn = HierarchicalTriXFFN(
    d_model=d_model,
    num_tiles=num_tiles,
    tiles_per_cluster=4,
)

print(f"Parameters: {sum(p.numel() for p in ffn.parameters()):,}")
```

### Step 2: Forward Pass

```python
# Random input
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, routing_info, aux_losses = ffn(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Aux loss:     {aux_losses['total_aux'].item():.4f}")
```

### Step 3: Examine Routing

```python
# Which tiles were selected?
tile_indices = routing_info['global_indices']
print(f"Tile indices shape: {tile_indices.shape}")
print(f"Unique tiles used:  {tile_indices.unique().tolist()}")

# Routing distribution
tile_counts = torch.bincount(tile_indices.flatten(), minlength=num_tiles)
print(f"Tile usage: {tile_counts.tolist()}")
```

---

## Tutorial 2: Training Loop

### Setup

```python
import torch
import torch.nn as nn
from trix import HierarchicalTriXFFN

# Simple model: embedding + TriX FFN + output
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_tiles):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ffn = HierarchicalTriXFFN(d_model, num_tiles)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embed(x)
        h, routing_info, aux_losses = self.ffn(h)
        logits = self.output(h)
        return logits, aux_losses

model = SimpleModel(vocab_size=1000, d_model=128, num_tiles=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### Training Step

```python
def train_step(model, batch, targets, optimizer):
    optimizer.zero_grad()
    
    # Forward
    logits, aux_losses = model(batch)
    
    # Task loss
    task_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    
    # Total loss includes auxiliary losses for balanced routing
    total_loss = task_loss + aux_losses['total_aux']
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return {
        'task_loss': task_loss.item(),
        'aux_loss': aux_losses['total_aux'].item(),
        'total_loss': total_loss.item(),
    }
```

### Training Loop

```python
# Dummy data
for epoch in range(10):
    batch = torch.randint(0, 1000, (32, 64))
    targets = torch.randint(0, 1000, (32, 64))
    
    metrics = train_step(model, batch, targets, optimizer)
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}")
```

---

## Tutorial 3: Inspecting Signatures

### View Tile Signatures

```python
from trix import HierarchicalTriXFFN

ffn = HierarchicalTriXFFN(d_model=64, num_tiles=8, tiles_per_cluster=4)

# Get all signatures
signatures = ffn.get_signatures()
print(f"Signatures shape: {signatures.shape}")  # (8, 64)

# Each signature is ternary
print(f"Unique values: {signatures.unique().tolist()}")  # [-1, 0, 1]
```

### Visualize Signature Diversity

```python
import matplotlib.pyplot as plt

# Compute pairwise distances
def signature_distance(s1, s2):
    return (s1 != s2).float().sum()

n_tiles = signatures.shape[0]
distances = torch.zeros(n_tiles, n_tiles)
for i in range(n_tiles):
    for j in range(n_tiles):
        distances[i, j] = signature_distance(signatures[i], signatures[j])

plt.imshow(distances, cmap='viridis')
plt.colorbar(label='Hamming Distance')
plt.title('Signature Diversity Matrix')
plt.xlabel('Tile')
plt.ylabel('Tile')
plt.savefig('signature_diversity.png')
```

---

## Tutorial 4: SparseLookup Architecture

### Concept

SparseLookup takes routing further: the routing decision **is** the computation.

```python
from trix import SparseLookupFFN

ffn = SparseLookupFFN(
    d_model=128,
    num_tiles=64,
    spline_knots=8,  # Magnitude modulation
)

x = torch.randn(4, 32, 128)
output, routing_info, aux_losses = ffn(x)

# Check parameter count
params = sum(p.numel() for p in ffn.parameters())
print(f"SparseLookup params: {params:,}")
```

### How It Works

1. **Routing** selects a direction (tile signature)
2. **Spline** modulates magnitude based on input
3. **No matrix multiply** in the hot path

---

## Tutorial 5: Temporal Routing

### For Sequential Tasks

```python
from trix.nn import TemporalTileLayer

temporal = TemporalTileLayer(
    d_model=64,
    d_state=16,  # State dimension
    num_tiles=8,
)

# Process a sequence
x = torch.randn(4, 100, 64)  # (batch, seq, d_model)
output, final_state, routing_infos = temporal.forward_sequence(x)

print(f"Output shape: {output.shape}")
print(f"Final state shape: {final_state.shape}")
print(f"Routing decisions: {len(routing_infos)}")
```

### Stateful Processing

```python
# Process step by step with explicit state
state = temporal.init_state(batch_size=4)

for t in range(10):
    x_t = torch.randn(4, 64)  # Single timestep
    out_t, state, info = temporal(x_t, state)
    print(f"Step {t}: tile={info['tile_indices'][0].item()}")
```

---

## Tutorial 6: Production Inference

### Gradient Truth Mode (Default)

Both `HierarchicalTriXFFN` and `SparseLookupFFN` now use Gradient Truth by default:

```python
from trix import HierarchicalTriXFFN, SparseLookupFFN

# Path A: Ternary MatMul - frozen weights, learned scales
ffn_a = HierarchicalTriXFFN(d_model=128, num_tiles=32)

# Path B: MatMul-Free - frozen directions, learned scales  
ffn_b = SparseLookupFFN(d_model=128, num_tiles=32)

# Both train the same way
x = torch.randn(2, 16, 128)
output, routing_info, aux = ffn_a(x)
loss = output.sum() + aux['total_aux']
loss.backward()

# Weights are already ternary (buffers) - no quantization needed for inference
print(f"Ternary weights: {ffn_a.tiles[0].up_weight.unique()}")  # tensor([-1., 1.])
```

### Compiled Dispatch (Legacy)

> **Note**: CompiledDispatch uses deprecated `SparseLookupFFNv2`.
> For new projects, use `SparseLookupFFN` with Gradient Truth.

```python
from trix.nn import SparseLookupFFNv2, CompiledDispatch

# Legacy approach (deprecated)
ffn = SparseLookupFFNv2(d_model=128, num_tiles=32)
compiled = CompiledDispatch(ffn)
```

---

## Tutorial 7: Weight Packing

### Memory Compression

```python
from trix import pack_weights, unpack_weights

# Get weights from a trained model
ffn = HierarchicalTriXFFN(d_model=256, num_tiles=16)

# Pack all weights
packed = ffn.pack_weights()

# Compare sizes
original_size = sum(p.numel() * 4 for p in ffn.parameters())  # FP32
packed_size = sum(p.numel() for p in packed.values())

print(f"Original: {original_size:,} bytes")
print(f"Packed:   {packed_size:,} bytes")
print(f"Compression: {original_size / packed_size:.1f}x")
```

### Save and Load

```python
# Save packed weights
torch.save(packed, 'model_packed.pt')

# Load and unpack
loaded = torch.load('model_packed.pt')
ffn.unpack_weights(loaded)
```

---

## Common Patterns

### Pattern 1: Replace FFN in Existing Model

```python
# Before
class OldBlock(nn.Module):
    def __init__(self, d_model):
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

# After
class NewBlock(nn.Module):
    def __init__(self, d_model):
        self.ffn = HierarchicalTriXFFN(d_model, num_tiles=64)
    
    def forward(self, x):
        out, _, aux = self.ffn(x)
        return out, aux  # Return aux for training
```

### Pattern 2: Auxiliary Loss Weighting

```python
# Early training: high aux weight for routing stability
# Late training: low aux weight for task focus

def get_aux_weight(epoch, total_epochs):
    # Linear decay from 0.1 to 0.001
    return 0.1 * (1 - epoch / total_epochs) + 0.001

# In training loop
aux_weight = get_aux_weight(epoch, total_epochs)
loss = task_loss + aux_weight * aux_losses['total_aux']
```

### Pattern 3: Monitoring Routing Health

```python
def check_routing_health(ffn, dataloader):
    tile_counts = torch.zeros(ffn.num_tiles)
    
    for batch in dataloader:
        _, routing_info, _ = ffn(batch)
        indices = routing_info['global_indices'].flatten()
        tile_counts += torch.bincount(indices, minlength=ffn.num_tiles)
    
    # Check for dead tiles
    dead_tiles = (tile_counts == 0).sum().item()
    
    # Check for dominant tiles
    max_share = tile_counts.max() / tile_counts.sum()
    
    print(f"Dead tiles: {dead_tiles}/{ffn.num_tiles}")
    print(f"Max tile share: {max_share:.1%}")
    
    return dead_tiles == 0 and max_share < 0.5
```

---

## Tutorial 8: Deterministic Computation (Forge)

For exact, verifiable computation with hardware export.

### Create a Foundry

```python
from trix.forge import Foundry

# Create foundry
foundry = Foundry(bits=8)

# Define shapes from truth tables
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.atom("or",  lambda a, b: a | b)

# Build executable system
system = foundry.build()

# Execute
result = system.execute(42, 13, "xor")
print(f"42 XOR 13 = {result}")  # 39
```

### Validate 100%

```python
# Exhaustive validation (65,536 test cases)
validation = system.validate(exhaustive=True)
assert validation.all_passed()
print("All shapes validated 100%")
```

### Export to Hardware

```python
# Generate CUDA kernels
system.export_cuda("output/cuda/")

# Generate Verilog RTL
system.export_verilog("output/verilog/")
```

### Run the Demo

```bash
python experiments/the_way_demo.py
```

---

## Tutorial 9: Frozen 6502 Emulator

A complete MOS 6502 CPU on frozen shapes.

### Run Tests

```bash
python experiments/frozen_emulator/frozen_6502.py --test
# Output: Tests: 48 passed, 0 failed
```

### Execute 6502 Code

```python
import sys
sys.path.insert(0, 'experiments/frozen_emulator')
from frozen_6502 import CPU6502

cpu = CPU6502()

# LDA #$42, STA $00, BRK
program = bytes([0xA9, 0x42, 0x85, 0x00, 0x00])
cpu.load_binary(program, 0x0600)
cpu.run(0x0600)

print(f"A = ${cpu.a:02X}")  # A = $42
```

---

## Tutorial 10: CUDA Benchmarks

High-performance XOR operations on GPU.

### Run Benchmarks

```bash
# Frozen fabric (requires NVIDIA GPU)
./experiments/qupid/frozen
# Output: 35.58 Tbits/sec

# XOR performance
./experiments/qupid/xor_perf
# Output: 1.12 trillion XORs/sec
```

---

## Tutorial 11: Anchored Dual-Mode FFN

Anchor-informed routing for chips AND modal-models from one substrate.

### Concept

Frozen anchors partition the input space. Learned routing searches within partitions.

```python
from trix.nn import AnchoredDualModeFFN, get_temperature_schedule

ffn = AnchoredDualModeFFN(
    d_model=256,
    num_anchors=16,    # Partition count
    num_tiles=64,      # Execution variants
    temperature=1.0,   # Partition sharpness
)

x = torch.randn(4, 32, 256)
output, info = ffn(x)

# Inspect routing
print(f"Anchor probs shape: {info['anchor_probs'].shape}")
print(f"Selected tiles: {info['tile_idx'].unique().tolist()}")
```

### Training with Temperature Annealing

```python
optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)

for step in range(1000):
    # Anneal temperature: warm -> cold
    temp = get_temperature_schedule(step, 1000, start_temp=2.0, end_temp=0.1)
    ffn.set_temperature(temp)

    optimizer.zero_grad()
    output, info = ffn(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Inference uses hard selection automatically
ffn.eval()
output, info = ffn(x)
```

### Two Deployment Modes

| Mode | Use | Behavior |
|------|-----|----------|
| **Chips** | `.eval()` | Anchors + tiles only, deterministic |
| **Modal-models** | `.train()` or `.eval()` | Full routing, constrained search |

See [ANCHORED_DUAL_MODE.md](ANCHORED_DUAL_MODE.md) for the full theory.

---

## The Unified View

```
                        THE WAY
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    PROVIDENCE          FOUNDRY           FABRIC
    (Tutorial 1-7)    (Tutorial 8)      (Tutorial 10)
         │                 │                 │
   Ternary routing    ShapeTerms IR       CUDA/Verilog
   Tile signatures    Composition         35 Tbits/sec
   Soft gradients     Hard validation     Silicon speed
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                    SHAPE = COMPUTE
```

Train with gradients. Execute with geometry. **This is The Way.**

---

## Next Steps

1. **[THE_WAY.md](THE_WAY.md)** - The unified philosophy
2. **[Architecture Guide](ARCHITECTURE.md)** - Deep dive into system design
3. **[ANCHORED_DUAL_MODE.md](ANCHORED_DUAL_MODE.md)** - Dual-mode architecture (chips + modal-models)
4. **[GRADIENT_TRUTH.md](GRADIENT_TRUTH.md)** - Training beyond STE
5. **[XORPU_COMPLETE.md](XORPU_COMPLETE.md)** - Deterministic computation details
6. **[Theory](THEORY.md)** - Mathematical foundations
7. **[API Reference](API.md)** - Complete API documentation
8. **[Benchmarks](BENCHMARKS.md)** - Reproduce our results
