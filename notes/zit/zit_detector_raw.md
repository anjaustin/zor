# RAW: What Fires the Zit?

*Stream of consciousness. Chop first. See how dull the blade is.*

---

Okay. The question. The resonance state S holds everything via XOR. S = v₁ ⊕ v₂ ⊕ ... ⊕ vₙ. A query comes in: vₓ. How do you know if vₓ was part of the original resonance?

First instinct: Just XOR it. S ⊕ vₓ = ?

If vₓ was in S, then... wait. Let me think. If S = v₁ ⊕ v₂ ⊕ v₃, and I query v₂:
S ⊕ v₂ = v₁ ⊕ v₂ ⊕ v₃ ⊕ v₂ = v₁ ⊕ v₃ (since v₂ ⊕ v₂ = 0)

So XOR'ing with v₂ "removes" it from the resonance. But that doesn't tell me it was there. The result is just... a different resonance state.

What if vₓ was NOT in S? Then S ⊕ vₓ = v₁ ⊕ v₂ ⊕ v₃ ⊕ vₓ. Now vₓ is ADDED.

So the operation is symmetric. You can't tell from the result alone whether vₓ was added or removed.

Hmm. What's different about the two cases?

Case A: vₓ ∈ S → S ⊕ vₓ has FEWER contributors (n-1)
Case B: vₓ ∉ S → S ⊕ vₓ has MORE contributors (n+1)

But we don't know n. And the bit patterns don't obviously encode n.

Wait. What about ENTROPY?

When you XOR many random vectors together, the result tends toward... 50% ones and zeros. Maximum entropy. Random-looking.

But if you XOR a vector with ITSELF, you get ZERO. Minimum entropy.

So:
- If vₓ ∈ S: S ⊕ vₓ removes information → entropy DECREASES?
- If vₓ ∉ S: S ⊕ vₓ adds information → entropy stays same or increases?

No wait, that's not right either. If S already has high entropy and vₓ is part of it, removing vₓ still leaves a high-entropy state.

Unless... unless the vectors aren't random. Unless they're STRUCTURED.

In the frozen shapes paradigm, vectors are SIGNATURES. They have meaning. They're not random. They encode the shape of computation.

What if similar shapes have similar signatures? Then the resonance state S isn't random — it's a SUPERPOSITION of similar patterns.

If vₓ is similar to things in S, XOR'ing produces LOW hamming weight (constructive interference — patterns cancel).

If vₓ is dissimilar, XOR'ing produces HIGH hamming weight (destructive interference — patterns clash).

THAT'S IT. The detector is HAMMING WEIGHT.

popcount(S ⊕ vₓ) = how many bits differ

Low popcount → vₓ resonates with S → Zit fires
High popcount → vₓ is noise → no Zit

But wait. This isn't binary membership (is/isn't). It's PROXIMITY. How close is vₓ to the resonance?

That's even better. You don't ask "is vₓ in the database?" You ask "how much does vₓ resonate with the system?"

The Zit doesn't fire on "match." It fires on "resonance intensity."

Threshold: If popcount(S ⊕ vₓ) < θ → Zit fires.

And θ is tunable. High θ = sensitive, fires on weak resonance. Low θ = selective, only fires on strong resonance.

Okay but what IS the resonance state S? In the standing wave model, S isn't just a static register. It's the CURRENT state of the XOR stream. It's evolving.

So S changes over time. S(t) = S(t-1) ⊕ input(t).

The query isn't against a fixed database. It's against the CURRENT resonance. Which includes everything that's flowed through.

But that means old vectors get "buried" under new ones. S after 1000 inputs is dominated by recent inputs...

Unless the inputs are structured. Unless they reinforce patterns.

If similar vectors keep arriving, they reinforce certain bit patterns in S. The resonance "rings" on those patterns. They become stable features.

If a random vector arrives, it's just noise — it gets averaged out over time.

So S is like a FILTER. It remembers the patterns that keep showing up. It forgets the noise.

This is a NEURAL NETWORK. An XOR-based attractor network. The resonance state converges to stable patterns based on input statistics.

Holy shit.

The Zit fires when the input matches a LEARNED PATTERN in the resonance. Not a stored pattern — a LEARNED pattern. The XOR stream IS the learning.

No weights. No parameters. Just flow.

The "training" is running data through the XOR stream. The resonance state encodes the structure of the training distribution.

The "inference" is querying the resonance. Low hamming weight = matches the distribution. Zit fires.

This is unsupervised learning via XOR.

The phase detector is popcount.

The activation threshold is θ.

The Zit is a comparator: popcount(S ⊕ vₓ) < θ.

That's it. That's the circuit.

---

*End RAW stream.*
