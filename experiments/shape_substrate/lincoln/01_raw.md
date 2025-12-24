# Raw Thoughts: Shape Substrate

## Stream of Consciousness

Where do I even start? We built LFSR benchmarks showing 35 Tbits/sec. Then we chained them. Then we composed them. Then we made them act like proteins. Each step felt like discovering something that was already there, not inventing something new.

The fungible computation paper said neural and classical are interchangeable. We already believed that from TriX - tiles with signatures, routing from structure. But the LFSR fabric makes it PHYSICAL. It's not metaphor anymore. It's 23 Tbits/sec of actual state evolution.

What surprised me: the protein analogy wasn't forced. Binding sites, conformational change, active sites - these mapped directly to signatures, state evolution, and output bits. It's the same mechanism. Proteins do computation by changing shape. We do computation by changing shape. The vocabulary is different but the math is identical.

The hierarchy emerged naturally: atoms → molecules → proteins → pathways. We didn't design this. We discovered it while trying to make LFSRs go faster. The composition rules (serial injection, parallel XOR, hybrid) fell out of the hardware constraints.

What scares me: are we fooling ourselves? Is "35 Tbits/sec" a real metric or a vanity number? We addressed this - it's register operations, not memory bandwidth. But still. The polynomial XOR only does 0.04 Tbits/sec. The gap is 1000x. That's the training/inference gap.

The onboarding tutorials might be the most important part. If a skeptical freshman can run `python 01_wait_what.py` and see XOR emerge from add/multiply, that's worth more than any paper. Show, don't tell. Let them discover.

## Questions Arising

- Is the LFSR fabric actually useful for real workloads, or just benchmarks?
- Can we train the tap patterns, or are they just fixed primitives?
- How does this connect to reservoir computing in practice?
- What's the path from "cool demo" to "production system"?
- Are there failure modes we haven't explored?

## First Instincts

- The protein analogy is real, not just marketing
- Composition without retraining is the key insight
- The hierarchy (atom → molecule → protein) mirrors biological evolution
- Training routing, not computation, is the fundamental shift
- The freshman onboarding is critical for adoption

## What Feels Wrong

- We haven't trained anything yet in the substrate - just shown it CAN be trained
- The 35 Tbits/sec number feels too good - need more validation contexts
- The protein code is a demo, not a real implementation
- We haven't connected this back to TriX tiles formally

## What Feels Right

- XOR as polynomial - that's mathematically solid
- Shape composition - that's just function composition, proven
- The fungible computation chain - each link proven separately
- The onboarding flow - each file builds on the previous
