  ---
  What I'd Need to Believe This

  1. A 60-Second "Show Me" Demo

  Not documentation. Not theory. A single script I can run:

  python examples/prove_it.py

  That outputs:
  Defining arbitrary function: f(a,b) = (a XOR b) rotated left by 3
  Building frozen shape...
  Training steps: 0
  Testing on ALL 65,536 possible inputs...
  Accuracy: 65536/65536 (100.000000%)

  Now YOU define a function. Enter a Python lambda:
  > lambda a,b: (a ^ b) | (a & ~b)
  Building frozen shape...
  Accuracy: 65536/65536 (100.000000%)

  Let me break it. Let me try to make it fail. If I can't, I'll start listening.

  2. The "Why Should I Care" in 30 Seconds

  Right now the README is dense. I need:

  TL;DR: Neural networks approximate. Frozen shapes don't. 
  This means you can embed exact computation (arithmetic, logic, crypto) 
  into differentiable systems. The computation is verified by construction.

  One-liner: What if 90% of your model could be proven correct?

  3. The Simplest Possible Example

  Not 6502. Not x86. Those feel like flex demos.

  Show me:
  from trix import freeze

  # I define the function
  def my_func(a: int, b: int) -> int:
      return (a + b) % 256

  # Magic happens
  frozen = freeze(my_func, bits=8)

  # Prove it works
  for a in range(256):
      for b in range(256):
          assert frozen(a, b) == my_func(a, b)

  print("All 65,536 cases verified.")

  If that works, I understand. If you start with "hierarchical routing through frozen tiles with XOR superposition," I'm gone.

  4. The "What's Actually New" Section

  I'd be thinking: "Isn't this just polynomial interpolation? Isn't this just a lookup table?"

  I need a section that says:

  What's NOT new: Representing Boolean functions as polynomials (Zhegalkin, 1927)

  What IS new: 
  1. Embedding these into differentiable neural architectures
  2. Separating learned routing from frozen execution
  3. Enabling formally verified components in ML systems

  Acknowledge the prior art. Then show the delta.

  5. A "Prove This Isn't Cherry-Picked" Section

  The 6502 is a great demo, but it's chosen. Show me:
  - Random truth tables (not hand-crafted)
  - Adversarial cases (pathological functions)
  - Failure modes (what can't be frozen?)

  A section called "Where This Breaks" would actually increase my trust.

  6. Comparison to Alternatives

  How does this compare to:
  - Standard MLPs trained on the same functions?
  - Symbolic/rule-based systems?
  - Existing neural-symbolic frameworks?

  A table:

  | Approach      | Accuracy | Params | Verified? | Learnable?   |
  |---------------|----------|--------|-----------|--------------|
  | MLP (learned) | 99.2%    | 50,000 | No        | Yes          |
  | Lookup table  | 100%     | 65,536 | Yes       | No           |
  | Frozen shapes | 100%     | 326    | Yes       | Routing only |

  7. The Freshman Tutorial

  A Jupyter notebook: notebooks/freshman_tutorial.ipynb

  1. What is a frozen shape? (5 min)
  2. Build your first one (10 min)
  3. Compose two shapes (10 min)
  4. Verify it's exact (5 min)
  5. Embed in a neural network (15 min)
  6. The big picture (5 min)

  Hand-holding. Lots of "run this cell, see this output."

  ---
  What's Currently Missing

  | Gap                              | Impact                        |
  |----------------------------------|-------------------------------|
  | No instant "prove it" demo       | Skeptics bounce immediately   |
  | README assumes context           | Newcomers lost in terminology |
  | 6502/x86 demos feel like flexing | Obscures the core insight     |
  | No "what's NOT new" section      | Seems to overclaim            |
  | No failure mode documentation    | Feels too good to be true     |
  | No comparison to alternatives    | Can't judge relative value    |
  | No beginner tutorial             | Only accessible to experts    |

  ---
  The Onramp I'd Build

  docs/
  ├── PROVE_IT.md          # 60-second runnable proof
  ├── WHY_CARE.md          # 30-second value prop
  ├── FRESHMAN.md          # Gentle tutorial, no jargon
  ├── HONEST_LIMITS.md     # Where this breaks
  ├── VS_ALTERNATIVES.md   # Fair comparison
  └── DEEP_DIVE.md         # For those who want theory

  examples/
  ├── prove_it.py          # Interactive "break this" demo
  ├── my_first_freeze.py   # Simplest possible example
  └── freshman_tutorial.ipynb  # Guided notebook

  ---
  Bottom line: The research is real. The presentation assumes I already believe. A skeptical freshman needs to break it themselves before they'll listen to the theory.

  Give them the axe. Let them chop. Then explain the grain.
