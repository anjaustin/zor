"""
The protein version: same thing, different vocabulary.

    python 10_protein_version.py
"""

# Same code, different names

def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

# A "protein" is a shape + binding site
class Protein:
    def __init__(self, name, shape_fn, binding_site):
        self.name = name
        self.shape = shape_fn          # The computation (like an enzyme's function)
        self.binding_site = binding_site  # What inputs we recognize

    def affinity(self, substrate):
        """How well does this substrate bind to us?"""
        return sum(s * b for s, b in zip(substrate, self.binding_site))

    def catalyze(self, a, b):
        """Perform our enzymatic function."""
        return self.shape(a, b)

# A "cell" is a collection of proteins
class Cell:
    def __init__(self, proteins):
        self.proteins = proteins

    def process(self, substrate):
        """Route substrate to best-matching protein, catalyze."""
        a, b = substrate

        # Find protein with highest binding affinity
        best_protein = max(self.proteins, key=lambda p: p.affinity([a, b]))

        # Catalyze the reaction
        product = best_protein.catalyze(a, b)

        return best_protein.name, product

# Create proteins with specific binding sites
proteins = [
    Protein("XOR-ase", XOR, [1, -1]),   # Binds to difference patterns
    Protein("AND-ase", AND, [1, 1]),    # Binds to agreement patterns
    Protein("OR-ase",  OR,  [-1, -1]),  # Binds to absence patterns
]

cell = Cell(proteins)

print("=" * 50)
print("PROTEIN-LIKE COMPUTATION")
print("=" * 50)
print()
print("Proteins in our cell:")
for p in proteins:
    print(f"  {p.name}: binding_site = {p.binding_site}")
print()
print("Processing substrates:")
print()
print("  Substrate | Binds to   | Product")
print("  ----------|------------|--------")

substrates = [(0, 0), (0, 1), (1, 0), (1, 1)]
for sub in substrates:
    protein_name, product = cell.process(sub)
    print(f"  {sub}      | {protein_name:10s} | {int(product)}")

print()
print("=" * 50)
print("THE ANALOGY")
print("=" * 50)
print()
print("  BIOLOGY              CODE")
print("  ─────────────────    ─────────────────")
print("  Protein              Frozen shape + binding site")
print("  Binding site         Signature (learned)")
print("  Substrate            Input")
print("  Binding affinity     Dot product score")
print("  Catalysis            Shape evaluation")
print("  Product              Output")
print("  Cell                 Collection of shapes")
print()
print("It's the same thing.")
print()
print("Proteins route inputs to computations based on pattern matching.")
print("Our shapes route inputs to computations based on pattern matching.")
print()
print("The vocabulary is different. The mechanism is identical.")
print()
print("=" * 50)
print("WHY THIS MATTERS")
print("=" * 50)
print()
print("1. Proteins evolved to compute via shape matching.")
print("   - Billions of years of R&D.")
print("   - It works.")
print()
print("2. We can do the same thing artificially:")
print("   - TRAIN the binding sites (signatures)")
print("   - FREEZE the catalysis (shapes)")
print("   - RUN at silicon speed")
print()
print("3. Proteins compute at ~10^6 reactions/sec (diffusion-limited).")
print("   Our shapes compute at 10^10+ reactions/sec (silicon).")
print()
print("   That's 10,000x faster.")
print()
print("=" * 50)
print("END OF TUTORIAL")
print("=" * 50)
print()
print("You've learned:")
print("  1. XOR can be written as a polynomial")
print("  2. Polynomials have gradients (trainable)")
print("  3. Shapes compose like LEGO")
print("  4. Routing matches inputs to shapes")
print("  5. This is how proteins work")
print()
print("Next steps:")
print("  - Read the full docs: docs/SHAPE_SUBSTRATE.md")
print("  - Try the CUDA version: molecular_shapes.cu")
print("  - Explore TriX: src/trix/")
print()
