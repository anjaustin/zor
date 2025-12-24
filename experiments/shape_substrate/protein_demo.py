"""
Protein-like Computation with Frozen Shapes

The simplest model: TriX tiles ARE proteins.
- Atoms = polynomial shapes (XOR, AND, OR)
- Binding = signature matching (dot product)
- Folding = shape execution (polynomial evaluation)
- Output = result

This is what TriX already does, just reframed.
"""

import torch
import torch.nn.functional as F

# =============================================================================
# ATOMIC SHAPES (frozen polynomials)
# =============================================================================

def shape_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR: a + b - 2ab"""
    return a + b - 2 * a * b

def shape_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """AND: ab"""
    return a * b

def shape_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """OR: a + b - ab"""
    return a + b - a * b

def shape_nand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """NAND: 1 - ab"""
    return 1 - a * b

def shape_xnor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XNOR: 1 - a - b + 2ab"""
    return 1 - a - b + 2 * a * b

# =============================================================================
# PROTEIN: A shape with a binding site
# =============================================================================

class ProteinShape:
    """
    A protein is a shape + binding site.

    - binding_site: pattern this protein recognizes
    - shape_fn: computation to perform when bound
    - name: human-readable identifier
    """

    def __init__(self, name: str, binding_site: torch.Tensor, shape_fn):
        self.name = name
        self.binding_site = binding_site  # What inputs we "like"
        self.shape_fn = shape_fn          # What we compute

    def binding_affinity(self, input_pattern: torch.Tensor) -> float:
        """How well does input match our binding site? (dot product)"""
        return torch.dot(input_pattern.flatten(), self.binding_site.flatten()).item()

    def fold(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Execute our shape (the conformational output)"""
        return self.shape_fn(a, b)


# =============================================================================
# MOLECULAR PATHWAY: Chain of proteins
# =============================================================================

class Pathway:
    """
    An enzymatic pathway: proteins in sequence.
    Each protein's output becomes context for the next.
    """

    def __init__(self, proteins: list):
        self.proteins = proteins

    def process(self, substrate_a: torch.Tensor, substrate_b: torch.Tensor,
                verbose: bool = False) -> torch.Tensor:
        """Process substrate through the pathway."""
        result = substrate_a

        for i, protein in enumerate(self.proteins):
            # Compute binding affinity (pad input to match binding site)
            input_pattern = torch.cat([result.flatten(), substrate_b.flatten()])
            binding_size = len(protein.binding_site)
            if len(input_pattern) < binding_size:
                input_pattern = F.pad(input_pattern, (0, binding_size - len(input_pattern)))
            affinity = protein.binding_affinity(input_pattern[:binding_size])

            if verbose:
                print(f"  Step {i}: {protein.name}")
                print(f"    Binding affinity: {affinity:.3f}")

            # Fold (execute shape)
            result = protein.fold(result, substrate_b)

            if verbose:
                print(f"    Output: {result.item():.3f}")

        return result


# =============================================================================
# CELL: Multiple proteins competing for substrate
# =============================================================================

class Cell:
    """
    A cell with multiple proteins.
    Input binds to best-matching protein (winner-take-all).
    This IS TriX tile routing, just reframed.
    """

    def __init__(self, proteins: list):
        self.proteins = proteins

    def process(self, a: torch.Tensor, b: torch.Tensor,
                verbose: bool = False) -> tuple:
        """
        Route input to best-matching protein, execute its shape.

        Returns: (output, winning_protein_name, affinities)
        """
        # Create input pattern for binding
        input_pattern = torch.cat([a.flatten(), b.flatten()])

        # Compute binding affinity for each protein
        affinities = []
        for protein in self.proteins:
            # Pad or truncate to match binding site size
            pattern = input_pattern[:len(protein.binding_site)]
            if len(pattern) < len(protein.binding_site):
                pattern = F.pad(pattern, (0, len(protein.binding_site) - len(pattern)))
            affinity = protein.binding_affinity(pattern)
            affinities.append(affinity)

        if verbose:
            print("  Binding affinities:")
            for protein, aff in zip(self.proteins, affinities):
                print(f"    {protein.name}: {aff:.3f}")

        # Winner-take-all (conformational selection)
        winner_idx = torch.tensor(affinities).argmax().item()
        winner = self.proteins[winner_idx]

        if verbose:
            print(f"  Winner: {winner.name}")

        # Execute winning protein's shape
        output = winner.fold(a, b)

        return output, winner.name, affinities


# =============================================================================
# DEMO
# =============================================================================

def main():
    print("=" * 60)
    print("PROTEIN-LIKE COMPUTATION WITH FROZEN SHAPES")
    print("=" * 60)
    print()

    # Define binding sites (these would be learned in practice)
    # For demo: different patterns for different operations
    bind_xor = torch.tensor([1.0, -1.0, -1.0, 1.0])   # Likes difference
    bind_and = torch.tensor([1.0, 1.0, 1.0, 1.0])     # Likes agreement
    bind_or = torch.tensor([1.0, 1.0, -1.0, -1.0])    # Likes presence
    bind_nand = torch.tensor([-1.0, -1.0, 1.0, 1.0])  # Likes absence

    # Create proteins
    proteins = [
        ProteinShape("XOR", bind_xor, shape_xor),
        ProteinShape("AND", bind_and, shape_and),
        ProteinShape("OR", bind_or, shape_or),
        ProteinShape("NAND", bind_nand, shape_nand),
    ]

    print("ATOMIC SHAPES (frozen polynomials):")
    print("-" * 60)
    for p in proteins:
        print(f"  {p.name}: binding_site = {p.binding_site.tolist()}")
    print()

    # Create cell
    cell = Cell(proteins)

    # Test inputs
    test_cases = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    print("CELL PROCESSING (winner-take-all routing):")
    print("-" * 60)

    for a_val, b_val in test_cases:
        a = torch.tensor([a_val])
        b = torch.tensor([b_val])

        print(f"\nInput: a={a_val}, b={b_val}")
        output, winner, affinities = cell.process(a, b, verbose=True)
        print(f"  Output: {output.item():.3f}")

    print()
    print("=" * 60)
    print("MOLECULAR PATHWAY (enzymatic cascade)")
    print("=" * 60)
    print()

    # Create a pathway: XOR → AND → OR
    # This computes: OR(AND(XOR(a, b), b), b) = complex function
    pathway = Pathway([
        ProteinShape("XOR", bind_xor, shape_xor),
        ProteinShape("AND", bind_and, shape_and),
        ProteinShape("OR", bind_or, shape_or),
    ])

    print("Pathway: XOR → AND → OR")
    print("-" * 60)

    for a_val, b_val in test_cases:
        a = torch.tensor([a_val])
        b = torch.tensor([b_val])

        print(f"\nSubstrate: a={a_val}, b={b_val}")
        result = pathway.process(a, b, verbose=True)
        print(f"  Final product: {result.item():.3f}")

    print()
    print("=" * 60)
    print("THE INSIGHT")
    print("=" * 60)
    print()
    print("  TriX tiles ARE proteins:")
    print("    - Tile weights → Amino acid sequence")
    print("    - Tile signature → Binding site")
    print("    - Winner-take-all → Conformational selection")
    print("    - Tile computation → Shape execution")
    print()
    print("  The routing IS the binding.")
    print("  The computation IS the folding.")
    print("  The output IS the active site.")
    print()
    print("  We already have protein-like computation.")
    print("  We just didn't call it that.")
    print()


if __name__ == "__main__":
    main()
