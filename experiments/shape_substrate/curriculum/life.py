"""
LIFE

Everything you just learned? Proteins do the same thing.
This isn't metaphor. It's the same mechanism.
"""

# The shapes you built
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b


# ============================================================
# THE ROSETTA STONE
# ============================================================
#
#   WHAT YOU CALLED IT          WHAT BIOLOGY CALLS IT
#   ──────────────────          ─────────────────────
#   Shape                       Enzyme
#   Signature                   Binding site
#   Routing                     Substrate recognition
#   Winner-take-all             Competitive inhibition
#   Composition                 Metabolic pathway
#   Training                    Evolution
#


# ============================================================
# A PROTEIN
# ============================================================

class Protein:
    """
    A protein is a shape with a binding site.
    Sound familiar? It's your ShapeRouter, renamed.
    """

    def __init__(self, name, enzyme_function, binding_site):
        self.name = name
        self.enzyme = enzyme_function  # The "shape" - what it computes
        self.binding_site = binding_site  # The "signature" - what it recognizes

    def binding_affinity(self, substrate):
        """How well does this substrate fit our binding site?"""
        a, b = substrate
        pattern = [2*a - 1, 2*b - 1]
        return pattern[0] * self.binding_site[0] + pattern[1] * self.binding_site[1]

    def catalyze(self, substrate):
        """Perform the enzymatic reaction."""
        a, b = substrate
        return self.enzyme(a, b)


# ============================================================
# A CELL
# ============================================================

class Cell:
    """
    A cell has multiple proteins competing for substrates.
    The substrate binds to whichever protein fits best.
    """

    def __init__(self, proteins):
        self.proteins = proteins

    def metabolize(self, substrate):
        """Process substrate through best-matching protein."""
        # Find protein with highest binding affinity
        affinities = [(p, p.binding_affinity(substrate)) for p in self.proteins]
        winner = max(affinities, key=lambda x: x[1])[0]

        # Catalyze the reaction
        product = winner.catalyze(substrate)

        return winner.name, product


# ============================================================
# RUN IT
# ============================================================

print("=" * 50)
print("CELL METABOLISM")
print("=" * 50)
print()

# Create proteins (same as your trained shapes!)
proteins = [
    Protein("XOR-ase", XOR, [0, 0]),    # Will learn to handle difference
    Protein("AND-ase", AND, [1, 1]),    # Likes both high
    Protein("OR-ase",  OR,  [-1, -1]),  # Likes both low
]

cell = Cell(proteins)

# Process substrates
substrates = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("Substrate → Enzyme → Product")
print("-" * 35)
for sub in substrates:
    enzyme, product = cell.metabolize(sub)
    print(f"  {sub}     → {enzyme:8s} → {int(product)}")

print()


# ============================================================
# THE PATHWAY
# ============================================================

class Pathway:
    """
    An enzymatic pathway: proteins in sequence.
    Product of one becomes substrate for next.
    Like your composed molecules.
    """

    def __init__(self, proteins):
        self.proteins = proteins

    def process(self, initial_substrate):
        """Run substrate through the pathway."""
        a, b = initial_substrate
        result = a

        print(f"  Input: {initial_substrate}")
        for protein in self.proteins:
            result = protein.catalyze((result, b))
            print(f"    → {protein.name}: {result:.0f}")

        return result


print("=" * 50)
print("ENZYMATIC PATHWAY")
print("=" * 50)
print()
print("Pathway: XOR-ase → AND-ase → OR-ase")
print()

pathway = Pathway([
    Protein("XOR-ase", XOR, [0, 0]),
    Protein("AND-ase", AND, [1, 1]),
    Protein("OR-ase",  OR,  [-1, -1]),
])

for sub in substrates:
    result = pathway.process(sub)
    print()


# ============================================================
# THE DEEP PARALLEL
# ============================================================

print("=" * 50)
print("THE DEEP PARALLEL")
print("=" * 50)
print()
print("  PROTEINS                    YOUR CODE")
print("  ────────                    ─────────")
print("  Amino acid sequence    =    Polynomial coefficients")
print("  3D fold                =    Function shape")
print("  Binding site           =    Signature vector")
print("  Substrate binding      =    Pattern matching")
print("  Catalysis              =    Shape evaluation")
print("  Competitive binding    =    Winner-take-all routing")
print("  Metabolic pathway      =    Composed molecules")
print("  Evolution              =    Gradient descent")
print()
print("  Proteins compute by SHAPE CHANGE.")
print("  Your code computes by SHAPE EVALUATION.")
print()
print("  Same mechanism. Different substrate.")
print("  Biology uses carbon. You use silicon.")
print()


# ============================================================
# THE SPEED
# ============================================================

print("=" * 50)
print("THE SPEED DIFFERENCE")
print("=" * 50)
print()
print("  Protein reactions:  ~10^6 / second (diffusion limited)")
print("  Silicon shapes:     ~10^10 / second (electronic)")
print()
print("  That's 10,000x faster.")
print()
print("  But proteins had 4 billion years of R&D.")
print("  We're just getting started.")
print()


# ============================================================
# CHALLENGE
# ============================================================
#
# 1. Add an "allosteric site" to your Protein class
#    - A second binding site that modulates activity
#    - If allosteric ligand binds, change the enzyme's behavior
#
# 2. Implement "feedback inhibition"
#    - Pathway product inhibits first enzyme
#    - Like real metabolic regulation
#
# 3. Model "cooperative binding"
#    - Multiple substrates must bind for catalysis
#    - Like hemoglobin's oxygen binding


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# You can explain to a biologist how your neural network
# is doing the same thing their enzymes do.
#
# You can explain to a computer scientist how enzymes
# are doing the same thing their routers do.
#
# The vocabulary is different. The mechanism is identical.
