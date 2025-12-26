# Raw Thoughts: Gradient Truth — Beyond the STE Lie

## Stream of Consciousness

So we have this problem. Neural networks work because gradients flow. Discrete things have no gradients. The STE is a hack — we pretend the step function is the identity during backprop. It works. It's ugly. The gradient is literally wrong. We're lying to the optimizer.

But wait. TriX already has pieces of something better. The frozen shapes. The routing. The scales. Mesa 15 showed 76 params beating 5896 params with 100% accuracy vs 94.6%. That's not a small win — that's a different paradigm.

What if discreteness isn't the enemy? What if we've been framing the problem wrong?

The problem isn't "how do we get gradients through discrete things." The problem is "why are we trying to learn discrete things with gradients in the first place?"

Consider: the XOR operation. It's a mathematical truth. XOR(a,b) = a + b - 2ab. That's not learned. That's discovered. You don't train a network to "learn" that 1 XOR 1 = 0. It's a fact of the universe.

But we've been treating ternary weights like they're continuous things that got discretized. What if they're discrete things that we're wrongly trying to continuify?

The shapes in TriX are frozen because they represent mathematical truths. The 16 shapes for 6502 emulation aren't learned — they're the geometry of computation itself.

So where does learning happen? ROUTING. Which shape to use for which input. That's a continuous decision — or at least, it can be made continuous.

Wait. Let me think about this differently.

In traditional NNs: learn weights → weights define function → function maps input to output
In frozen shapes: shapes ARE functions (fixed) → learn routing → routing selects function → selected function maps input to output

The second one separates what from which.

WHAT to compute: discrete, frozen, exact (the shapes)
WHICH one to use: continuous, learned, approximate (the routing)

This feels like it might be important. The discrete part doesn't need gradients because it's not learned. The learned part doesn't need discretization because it's naturally continuous.

But... where do shapes come from if we don't learn them?

Options:
1. Mathematical derivation (XOR polynomial is a theorem)
2. Enumeration (for small spaces, just try them all)
3. Evolution (genetic algorithms on structure, validation on function)
4. Observation (measure what the trained network converged to, then freeze it)

Option 4 is interesting. We could still use STE to discover shapes, but only once. Then freeze forever. The STE becomes a shape-discovery tool, not a runtime training method.

Actually, the LFSR experiments in TriX do something like this. The tap patterns are evolved, not gradient-trained. They're validated by whether they produce the right outputs.

Hmm. I'm also thinking about the "scales" approach. VGem's fix with learnable output_scale. The ternary structure is the skeleton, the continuous scales are the muscle. Skeleton doesn't change (discrete), muscle does (continuous).

This is like... factorization? Separate the computation into:
- Structure (discrete, frozen or evolved)
- Magnitude (continuous, gradient-trained)

Matrix factorization does this: W = UV where U is one thing and V is another. What if we did: W = S * T where S is continuous scale and T is ternary structure?

Then gradients flow through S (no STE needed), and T is found some other way.

But the structure T still needs to come from somewhere. And for expressive power, you probably need many different T's and learn which one to use...

...which brings us back to routing.

I think I'm circling around something. Let me try to state it:

THE INSIGHT (maybe): Discrete structure + continuous routing + continuous magnitude = full expressiveness without STE.

The discrete structure is the WHAT.
The continuous routing is the WHICH.
The continuous magnitude is the HOW MUCH.

All learning happens in continuous space. Discreteness is embraced, not fought.

## Questions Arising

- Is this actually more expressive than STE-trained networks?
- Where does the initial discrete structure library come from?
- How do you know you have enough shapes in your library?
- What about domains where we don't know the "right" shapes a priori?
- Is there a principled way to discover shapes without gradients?
- How does this connect to the manifold view in Mesa 11?
- What's the relationship between shapes and Voronoi cells?
- Can shapes be composed? (Molecules from atoms?)
- Is there a universal set of primitive shapes that can build anything?

## First Instincts

- This feels right. Cleaner than STE.
- The separation of concerns (structure vs routing vs magnitude) is elegant.
- There's probably a connection to basis functions and splines.
- Evolution/enumeration for shape discovery feels underexplored.
- The "atoms → molecules → proteins" hierarchy in TriX already has this structure.
- Manifold interpretation: shapes are fixed points, learning is navigation.

## Fears and Doubts

- Am I just describing lookup tables with extra steps?
- The shape library might explode combinatorially for complex domains.
- This might not work for tasks without obvious discrete primitives.
- I might be pattern-matching to what I want to see.
- The STE works well in practice — is "elegance" worth the engineering cost?

## What's probably wrong with my first instinct

- Overconfidence that "shapes" exist for all computational primitives
- Underestimating the importance of gradient-based shape discovery
- Maybe STE isn't a lie — maybe it's a valid relaxation?
- The boundary between "structure" and "routing" might be blurrier than I think
