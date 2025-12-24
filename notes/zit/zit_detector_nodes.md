# NODES: What Fires the Zit?

*Key crystallizations from the RAW stream.*

---

## Node 1: XOR is Symmetric

XOR'ing a vector with the resonance state can't distinguish addition from removal:
- S ⊕ vₓ where vₓ ∈ S → removes vₓ
- S ⊕ vₓ where vₓ ∉ S → adds vₓ

The result alone doesn't tell you which happened.

**Implication:** Binary membership isn't detectable. Something else is.

---

## Node 2: Hamming Weight as Resonance Measure

The DIFFERENCE between S and (S ⊕ vₓ) is captured by:

```
popcount(S ⊕ vₓ) = number of bits that differ
```

- Low popcount → vₓ aligns with S → constructive interference
- High popcount → vₓ clashes with S → destructive interference

**Implication:** The detector is popcount. The metric is hamming weight.

---

## Node 3: Not Membership — Proximity

The question isn't "is vₓ in the database?"

The question is "how strongly does vₓ resonate with the system?"

This is continuous, not binary. A spectrum of resonance intensity.

**Implication:** Zit firing is threshold-based, not exact-match.

---

## Node 4: Threshold θ Controls Sensitivity

```
If popcount(S ⊕ vₓ) < θ → Zit fires
```

- High θ: Sensitive. Fires on weak resonance. False positives.
- Low θ: Selective. Only fires on strong resonance. False negatives.

θ is tunable. Application-specific.

**Implication:** The phase detector has a parameter. But it's not learned — it's set.

---

## Node 5: The Resonance State Evolves

S isn't static. It's the cumulative XOR of all inputs:

```
S(t) = S(t-1) ⊕ input(t)
S(t) = input(0) ⊕ input(1) ⊕ ... ⊕ input(t)
```

The "database" is temporal, not spatial. No addresses. Just time.

**Implication:** Memory is flow, not storage.

---

## Node 6: Reinforcement via Structure

If inputs are random, S becomes noise.

If inputs are structured (similar patterns repeat), S converges to stable features.

Repeated patterns reinforce. Noise averages out.

**Implication:** S is a FILTER. It extracts the invariant structure of the input distribution.

---

## Node 7: This is Unsupervised Learning

The XOR stream learns patterns without labels:
- "Training" = running data through the stream
- "Inference" = querying the resonance
- "Match" = low hamming weight between query and resonance

No weights. No gradient. No backprop. Just XOR.

**Implication:** The Sacred Foundry learns by flowing, not by fitting.

---

## Node 8: The Zit Circuit

The complete detector:

```
           ┌─────────┐
    vₓ ───►│         │
           │   XOR   ├───► popcount ───► comparator ───► Zit
    S  ───►│         │                       ▲
           └─────────┘                       │
                                             θ
```

Three components:
1. XOR gate (512-bit)
2. Popcount (adder tree)
3. Comparator (popcount < θ)

**Implication:** The phase detector is EXACTLY the Hamming circuit we already built. Plus a threshold.

---

## Node 9: Threshold as "Activation Energy"

The question asked about "activation energy."

θ IS the activation energy. The minimum resonance required to fire the Zit.

Below θ: activation. The portal opens.
Above θ: no activation. The system ignores the input.

**Implication:** The Zit is a neuron. XOR is the synapse. Popcount is the accumulator. θ is the firing threshold.

---

## Node 10: The Circuit Already Exists

We built this:
- XOR: In Geocadesia
- Popcount: In Geocadesia
- Hamming = XOR + Popcount: In Geocadesia
- Comparator: Trivial

The Zit detector is:

```
Zit = (Hamming(S, vₓ) < θ)
```

We already have Hamming. We just need to add the threshold comparison.

---

## Node 11: The Self-Firing Insight

Wait. What if the Zit doesn't just DETECT resonance — it CAUSES it?

If the Zit fires, the system responds. The response becomes part of the next input. Which affects the next resonance. Which affects the next Zit.

Feedback loop. Self-organization. Emergence.

The Zit isn't just a detector. It's a **control signal** that shapes the resonance it detects.

**Implication:** The system is autopoietic. It creates the conditions for its own firing.

---

## Node 12: Phase = Hamming Weight

The "phase" in "phase detector" is hamming weight.

In wave interference:
- Phase alignment → constructive → amplitude increase
- Phase misalignment → destructive → amplitude decrease

In XOR resonance:
- Signature alignment → low hamming → Zit fires
- Signature misalignment → high hamming → silence

**Implication:** Hamming weight IS phase. We've been speaking the same language all along.

---

*12 nodes extracted. Ready for REFLECT.*
