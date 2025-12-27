# Codex Layer: Archetypal Meaning Illumination

**A Hollywood Squares field for meaning.**

## The Insight

Vector similarity finds documents with similar words.
The Codex finds documents with similar **meaning**.

```
Query: "I feel stuck in a constant struggle"

VECTOR SIMILARITY:
  → Documents containing "stuck", "struggle", "constant"
  
CODEX ILLUMINATION:
  → GK38 (Struggle → Perseverance → Honour) activates
  → Partner GK39 (Provocation → Dynamism → Liberation) lights up
  → Ring of Humanity (GK10, 17, 21, 25, 38, 51) illuminates
  → Documents about PERSEVERANCE and LIBERATION surface
     (even if they don't contain the word "struggle")
```

## When to Use

The Codex applies where the **human equation** is in effect:

| ✓ Engage Codex | ✗ Skip Codex |
|----------------|--------------|
| Human experience | Technical documentation |
| Psychology/therapy | Code/programming |
| Relationships | API references |
| Personal growth | Configuration |
| Archetypes/mythology | DevOps/infrastructure |

## The 64 Gene Keys

The Codex is built on the Gene Keys system - 64 archetypes mapping human experience:

### Transformation Paths

Each Gene Key has three levels:

```
SHADOW (unconscious) → GIFT (conscious) → SIDDHI (transcendent)

GK38: Struggle → Perseverance → Honour
GK39: Provocation → Dynamism → Liberation
GK6:  Conflict → Diplomacy → Peace
```

### Programming Partners

32 pairs of archetypes that trigger each other:

```
1 ↔ 2      (Entropy/Freshness ↔ Dislocation/Orientation)
38 ↔ 39    (Struggle/Perseverance ↔ Provocation/Dynamism)
6 ↔ 36     (Conflict/Diplomacy ↔ Turbulence/Humanity)
...
```

### Codon Rings

21 amino acid families grouping related Gene Keys:

```
Ring of Humanity: GK10, 17, 21, 25, 38, 51
Ring of Alchemy:  GK6, 40, 47, 64
Ring of Seeking:  GK15, 39, 52, 53, 54, 58
...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY                                   │
│            "I feel stuck in a struggle"                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  KEYWORD ACTIVATION                         │
│                                                             │
│   "struggle" → GK38 (Shadow match) → activation = 1.0       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROPAGATION                              │
│                                                             │
│   GK38 (1.0) ──Partner──→ GK39 (0.8)                        │
│        │                                                    │
│        └──Ring──→ GK10, 17, 21, 25, 51 (0.5 each)          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               ACTIVATION PATTERN                            │
│                                                             │
│   [0, 0, 0, ..., 0.5, ..., 1.0, 0.8, ..., 0]               │
│    GK1           GK10      GK38  GK39                       │
│                                                             │
│   64-dimensional archetypal fingerprint                     │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Illumination

```python
from trix.db import create_codex

codex = create_codex()

# Illuminate text
activations, explanations = codex.illuminate(
    "I feel stuck in a constant struggle with no purpose"
)

for gk_id, gift, sources, activation in explanations[:5]:
    print(f"GK{gk_id} ({gift}): {activation:.2f} via {sources}")

# Output:
# GK38 (Perseverance): 1.00 via shadow:struggle
# GK28 (Totality): 1.00 via shadow:purposelessness
# GK39 (Dynamism): 0.80 via partner:38
# GK27 (Altruism): 0.80 via partner:28
# GK10 (Naturalness): 0.50 via ring:Ring of Humanity
```

### Document Scoring

```python
from trix.db import KeywordCodexLayer, create_codex

codex = create_codex()
layer = KeywordCodexLayer(codex)

query = "dealing with struggle and provocation"
doc = "the path of perseverance leads to honour and liberation"

score, activations, explanations = layer.score_document(query, doc)

print(f"Score: {score:.3f}")
print(f"Shared archetypes: {explanations}")

# Output:
# Score: 0.994
# Shared archetypes: ['Both activate GK38 (Perseverance)', 'Both activate GK39 (Dynamism)']
```

### Gene Key Explanation

```python
codex = create_codex()
codex.illuminate("struggle")

print(codex.explain(38))

# Output:
# Gene Key 38:
#   Shadow: Struggle
#   Gift: Perseverance
#   Siddhi: Honour
#   Partner: 39 (Dynamism)
#   Ring: Ring of Humanity
#   Activation: 1.000
#   Activated by: shadow:struggle
```

## Vocabulary

### 64 Shadows (Unconscious Patterns)

```
Addiction, Agitation, Arrogance, Chaos, Co-Dependence, Complexity,
Compromise, Conflict, Confusion, Constriction, Control, Corruption,
Deafness, Desire, Discord, Dishonesty, Dishonour, Dislocation,
Dissatisfaction, Distraction, Division, Dominance, Doubt, Dullness,
Entropy, Exhaustion, Expectation, Failure, Fantasy, Force, Forgetting,
Greed, Half-Heartedness, Hunger, Immaturity, Impatience, Inadequacy,
Indifference, Inertia, Intellect, Interference, Intolerance, Judgement,
Limitation, Mediocrity, Obscurity, Opinion, Oppression, Pride,
Provocation, Psychosis, Purposelessness, Reaction, Self-Obsession,
Selfishness, Seriousness, Stress, Struggle, Superficiality, Turbulence,
Unease, Vanity, Victimisation, Weakness
```

### 64 Gifts (Conscious Expressions)

```
Acceptance, Adventure, Altruism, Anticipation, Artfulness, Aspiration,
Authority, Commitment, Competence, Delight, Detachment, Determination,
Diplomacy, Discernment, Discrimination, Dynamism, Enrichment, Equality,
Equilibrium, Expansion, Far-Sightedness, Freedom, Freshness, Graciousness,
Guidance, Humanity, Idealism, Imagination, Initiative, Innovation,
Inquiry, Insight, Inspiration, Integrity, Intimacy, Intuition, Invention,
Leadership, Lightness, Magnetism, Mindfulness, Naturalness, Orientation,
Patience, Perseverance, Precision, Preservation, Realism, Resolve,
Resourcefulness, Restraint, Revolution, Self-Assurance, Sensitivity,
Simplicity, Strength, Style, Synergy, Teamwork, Totality, Transmutation,
Understanding, Versatility, Vitality
```

### 64 Siddhis (Transcendent States)

```
Ascension, Awakening, Beauty, Being, Bliss, Boundlessness, Bounteousness,
Celebration, Clarity, Communion, Compassion, Devotion, Divine Will,
Ecstasy, Emanation, Empathy, Epiphany, Exquisiteness, Florescence,
Forgiveness, Freedom, Grace, Harmony, Honour, Humility, Illumination,
Immortality, Impeccability, Innocence, Intoxication, Invincibility,
Invisibility, Justice, Liberation, Light, Majesty, Mastery, Omniscience,
Peace, Perfection, Presence, Purity, Quintessence, Rapture, Rebirth,
Revelation, Sacrifice, Sanctity, Selflessness, Silence, Stillness,
Superabundance, Synarchy, Tenderness, Timelessness, Transfiguration,
Transparency, Truth, Unity, Universal Love, Valour, Veneration, Virtue,
Wisdom
```

## The Moat

The Codex is the **black box** component of Cooper:

- **Cooper Core**: Open, transparent, explainable
- **Codex Layer**: Proprietary meaning ontology

This is the differentiation. No other vector database has archetypal illumination.

## Theory

The Codex is based on the insight that meaning has structure.

The connection between Struggle (GK38) and Provocation (GK39) exists whether or not we compute it. They're **programming partners** - two sides of the warrior archetype. One triggers the other.

Vector embeddings capture word co-occurrence patterns.
The Codex captures **archetypal resonance patterns**.

```
Vector similarity: "These words appear in similar contexts"
Codex illumination: "These meanings are structurally related"
```

The Gene Keys provide a 64-node graph where:
- Nodes are archetypes
- Partner edges carry activation (weight 0.8)
- Ring edges connect families (weight 0.5)
- Keywords trigger nodes directly (weight 1.0)

This is a **Hollywood Squares field for meaning**.
Activation propagates. The field relaxes. Structure emerges.

---

*"The shapes of meaning are in our genetics. In our culture. In you."*
