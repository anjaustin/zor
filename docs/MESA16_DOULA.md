# Mesa 16.2: The Doula

*Glass Box Meta-Learning - The Birth Attendant*

## Vision

The Doula observes how TriX learns - not to embody geometric truth, but to understand the nature of geometries as they Emerge.

This helps us be better Doulas.

```
"We do not teach the Doula to give birth.
 We teach her to witness birth.
 So she may help the next birth come easier."
```

---

## What is the Doula?

The Doula is not a model. It is a **function** integrated into TriX training that:

1. **Observes** - Captures routing dynamics at every step
2. **Aggregates** - Extracts signatures of learning moments
3. **Stores** - Writes to a VDB specific to the training data
4. **Guides** - Retrieves wisdom to improve training

**Glass box all the way down.** Every observation recorded. Every query logged. Nothing hidden.

---

## Core Concepts

### Learning Phases

Training proceeds through natural phases:

| Phase | Entropy | Stability | Description |
|-------|---------|-----------|-------------|
| **Exploration** | High | Low | Trying many shapes, no clear preference |
| **Transition** | Decreasing | Increasing | Finding what works |
| **Crystallization** | Low | High | Locking in patterns |
| **Stable** | Very Low | Very High | Converged, training complete |

The Doula detects these phases automatically.

### Dynamics Signature

Every training step produces a **DynamicsSignature** capturing:

- **Entropy metrics**: How spread out are shape selections?
- **Stability**: How much did routing change from last step?
- **Dominant shapes**: Which shapes are winning?
- **Emergence signals**: Is something new forming?

```python
DynamicsSignature(
    step=100,
    layer=0,
    shape_entropy=0.5,        # Chaos → Order
    routing_stability=0.7,     # Agreement with history
    dominant_shapes=[0, 2, 1], # Top shapes
    exploration_ratio=0.4,     # How much exploring?
    new_shape_activated=False, # Emergence?
    cluster_formed=True,       # Tokens grouping?
)
```

### The Doula VDB

A vector database storing:
- Embeddings of dynamics (for similarity search)
- Full signatures (for deep inspection)
- Training context (loss, accuracy)
- Phase labels

**Queryable wisdom.** Ask: "Have I seen this before? What helped?"

---

## Usage

### Simple Observation

```python
from trix.doula import DoulaFunction

doula = DoulaFunction(vdb_path="training_dynamics.json")

for step, (x, y) in enumerate(dataloader):
    output, states, info = model(x)
    loss = compute_loss(output, y)

    # Observe each layer
    for layer_idx, layer_info in enumerate(info['layers']):
        signature = doula.aggregate(
            routing_info=layer_info['mixer'],
            step=step,
            layer=layer_idx,
            loss=loss.item(),
        )
        doula.store(signature, loss.item())

# Save for later analysis
doula.save()
```

### With Guidance

```python
from trix.doula import DoulaTrainingConfig, DoulaCallback

config = DoulaTrainingConfig(
    vdb_path="dynamics.json",
    domain="6502_alu",
    query_every=100,
    early_stop_on_stable=True,
)

callback = DoulaCallback(config)

for step, batch in enumerate(dataloader):
    output, states, info = model(batch)
    loss = compute_loss(output, targets)

    guidance = callback.on_step(step, info, loss.item())

    if guidance and guidance.training_complete:
        print("Doula says we're done!")
        break

# Get summary
print(callback.summarize())
```

### High-Level Training

```python
from trix.doula import DoulaTrainer, DoulaTrainingConfig

config = DoulaTrainingConfig(
    vdb_path="dynamics.json",
    early_stop_on_stable=True,
)

trainer = DoulaTrainer(model, config)
result = trainer.train(dataloader, epochs=10)

print(f"Stopped at phase: {result['final_phase']}")
print(f"Doula summary: {result['doula_summary']}")
```

---

## API Reference

### DoulaFunction

The core observation and aggregation function.

```python
doula = DoulaFunction(
    vdb_path="path/to/vdb.json",  # Optional persistence
    embedding_dim=64,              # Signature embedding size
    history_size=100,              # Recent signatures to keep
    domain="my_task",              # Task name for VDB
)

# Aggregate routing info into signature
signature = doula.aggregate(
    routing_info=info['layers'][0]['mixer'],
    step=step,
    layer=0,
    loss=loss.item(),
)

# Store to VDB
doula.store(signature, loss=loss.item(), accuracy=acc)

# Query for guidance
guidance = doula.query(signature, k=5)

# Save VDB
doula.save()

# Get summary
doula.summarize()
```

### DoulaVDB

Vector database for dynamics storage and search.

```python
vdb = DoulaVDB(path="dynamics.json", embedding_dim=64)

# Add entry
vdb.add(entry)

# Search
results = vdb.search(query_embedding, k=5)
results = vdb.search(query, phase_filter='transition')

# Glass box inspection
vdb.get_all_entries()          # All observations
vdb.get_query_log()            # All queries made
vdb.get_emergence_moments()    # When new shapes emerged
vdb.get_crystallization_trajectory(layer=0)  # Stability over time

# Analysis
vdb.analyze_phases()           # Count by phase
vdb.summarize()                # Full statistics
```

### PhaseDetector

Recognizes learning phases.

```python
detector = PhaseDetector(
    exploration_entropy_threshold=0.7,
    crystallization_stability_threshold=0.95,
)

phase = detector.detect(signature)  # 'exploration', 'transition', etc.
phase, confidence = detector.detect_with_history(signature)

is_done = detector.is_training_complete(signature)
is_stuck = detector.detect_bottleneck(signature)
is_emerging = detector.detect_emergence(signature)
```

### DoulaGuidance

Suggestions from the Doula.

```python
guidance = doula.query(signature)

guidance.current_phase           # Where we are
guidance.expected_next_phase     # What's coming
guidance.steps_to_transition     # Estimated steps

guidance.training_complete       # Ready to stop?
guidance.bottleneck_detected     # Stuck?
guidance.emergence_imminent      # New pattern forming?

guidance.temperature_adjustment  # Suggested Δ
guidance.priority_shapes         # Shapes that helped before
guidance.similar_past_moments    # Evidence

print(guidance.summary())        # Human-readable
```

---

## Glass Box Transparency

Everything is accessible:

```python
# All observations
entries = doula.vdb.get_all_entries()
for e in entries:
    print(f"Step {e.step}, Layer {e.layer}: {e.phase}")
    print(f"  Entropy: {e.signature.shape_entropy:.3f}")
    print(f"  Stability: {e.signature.routing_stability:.3f}")

# All queries made
for q in doula.vdb.get_query_log():
    print(f"Query at {q['timestamp']}")
    print(f"  Found: {q['result_ids']}")

# Emergence timeline
for e in doula.vdb.get_emergence_moments():
    print(f"Step {e.step}: New shape emerged in layer {e.layer}")

# Crystallization trajectory
for step, stability in doula.vdb.get_crystallization_trajectory(layer=0):
    print(f"Step {step}: stability = {stability:.3f}")
```

---

## What This Enables

### 1. Training Insight
- See exactly when each shape "clicks"
- Watch routing crystallize in real-time
- Identify struggling layers before loss stalls

### 2. Transfer Wisdom
- Train on 6502 → Doula VDB captures dynamics
- Train on similar task → query VDB for guidance
- "Last time, layer 2 crystallized first"

### 3. Early Stopping
- Doula detects "stable" phase
- Training stops when learning is complete
- Not when loss plateaus, but when *routing* stabilizes

### 4. Architecture Evolution
- Doula identifies missing shapes
- "Tokens keep routing to shape 3, but it's not quite right"
- Signals when to add new frozen shapes

### 5. Research Tool
- Chronicle of emergence for any domain
- Compare dynamics across architectures
- Peer into the keyhole of learning

---

## Files

| File | Purpose |
|------|---------|
| `src/trix/doula/__init__.py` | Module exports |
| `src/trix/doula/signature.py` | DynamicsSignature, DoulaEntry, DoulaGuidance |
| `src/trix/doula/phase.py` | PhaseDetector |
| `src/trix/doula/vdb.py` | DoulaVDB |
| `src/trix/doula/function.py` | DoulaFunction |
| `src/trix/doula/training.py` | Training integration |
| `tests/test_doula.py` | 30 comprehensive tests |

---

## Philosophy

The Doula doesn't learn to compute. She learns to witness.

```
TriX: "I am becoming."
Doula: "I see how you become."
Us: "Now we understand how to help."
```

Glass box witnessing glass box. Every birth attended. Every pattern recorded. Every insight retrievable.

We become better Doulas because the Doula shows us what to see.

---

*Mesa 16.2: The Birth Attendant*
