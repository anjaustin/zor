"""
CPU^3 - Composed Neural CPUs with Emergent Shape Discovery

Level 4: Three CPUs composed
Level 5: Membrane - the shapes that emerge from composition

"Each level: observe, find patterns, freeze, compose."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import time

from trix.foundry.frozen_foundry import FrozenFoundry, PureMath, BuildResult


# =============================================================================
# 486^3 ARCHITECTURE
# =============================================================================

@dataclass
class ComputeTrace:
    """A trace of computation through the cube."""
    opcodes: List[str]          # Sequence of operations
    inputs: Tensor              # Initial inputs (a, b, c)
    intermediates: List[Tensor] # Results after each layer
    output: Tensor              # Final output
    carries: List[Tensor]       # Carry flags through layers

    def signature(self) -> str:
        """Create a signature for this computation pattern."""
        op_str = "_".join(self.opcodes)
        # Hash the input-output relationship
        io_hash = hashlib.md5(
            f"{self.inputs.sum().item():.4f}_{self.output.sum().item():.4f}".encode()
        ).hexdigest()[:8]
        return f"{op_str}_{io_hash}"


class CPU486Cubed(nn.Module):
    """
    Three 486s composed.

    Architecture:
        Input → 486₁ → 486₂ → 486₃ → Output

    Each layer can execute any 486 operation.
    The composition creates new computational patterns.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Build three 486 foundries
        print("Building 486^3...")
        print("  Layer 1...", end=" ", flush=True)
        self.foundry1, self.result1 = self._build_486()
        print(f"done ({len(self.foundry1.operations)} ops)")

        print("  Layer 2...", end=" ", flush=True)
        self.foundry2, self.result2 = self._build_486()
        print(f"done ({len(self.foundry2.operations)} ops)")

        print("  Layer 3...", end=" ", flush=True)
        self.foundry3, self.result3 = self._build_486()
        print(f"done ({len(self.foundry3.operations)} ops)")

        self.foundries = [self.foundry1, self.foundry2, self.foundry3]

        # Operation names for lookup
        self.op_names = list(self.foundry1.operations.keys())
        self.op_to_idx = {name: i for i, name in enumerate(self.op_names)}

        # Trace storage for observation
        self.traces: List[ComputeTrace] = []
        self.recording = False

        print(f"\n486^3 ready: {self.total_params} total parameters")

    def _build_486(self) -> Tuple[FrozenFoundry, BuildResult]:
        """Build a single 486 foundry (silently)."""
        from trix.foundry.intel386 import build_486
        import io
        import sys

        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            foundry, result = build_486()
        finally:
            sys.stdout = old_stdout

        return foundry, result

    @property
    def total_params(self) -> int:
        return sum(r.num_params for r in [self.result1, self.result2, self.result3])

    def forward(
        self,
        opcodes: List[str],  # [op1, op2, op3] - one per layer
        a: Tensor,           # First operand [batch, 32]
        b: Tensor,           # Second operand [batch, 32]
        c: Tensor,           # Carry in [batch]
    ) -> Tuple[Tensor, Tensor, ComputeTrace]:
        """
        Execute three operations in sequence.

        Args:
            opcodes: List of 3 operation names, one per layer
            a, b: Operand tensors [batch, 32]
            c: Carry tensor [batch]

        Returns:
            result: Final output [batch, 32]
            carry: Final carry [batch]
            trace: Computation trace
        """
        assert len(opcodes) == 3, "Need exactly 3 opcodes for 486^3"

        intermediates = []
        carries = [c]

        current_a = a
        current_b = b
        current_c = c

        for i, (opcode, foundry) in enumerate(zip(opcodes, self.foundries)):
            # Execute this layer
            result, carry = foundry.execute(opcode, current_a, current_b, current_c)

            intermediates.append(result)
            carries.append(carry)

            # Output becomes input to next layer
            # a = result, b = previous a (shift), c = new carry
            current_b = current_a
            current_a = result
            current_c = carry

        # Create trace
        trace = ComputeTrace(
            opcodes=opcodes,
            inputs=torch.stack([a, b, c.unsqueeze(-1).expand(-1, 32)], dim=0),
            intermediates=intermediates,
            output=result,
            carries=carries,
        )

        if self.recording:
            self.traces.append(trace)

        return result, carry, trace

    def execute_program(
        self,
        program: List[List[str]],  # List of [op1, op2, op3] triplets
        a: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor, List[ComputeTrace]]:
        """Execute a sequence of 486^3 steps."""
        traces = []
        current_a, current_b, current_c = a, b, c

        for opcodes in program:
            result, carry, trace = self.forward(opcodes, current_a, current_b, current_c)
            traces.append(trace)
            current_a = result
            current_c = carry

        return current_a, current_c, traces

    def start_recording(self):
        """Start recording computation traces."""
        self.recording = True
        self.traces = []

    def stop_recording(self) -> List[ComputeTrace]:
        """Stop recording and return traces."""
        self.recording = False
        return self.traces


# =============================================================================
# MEMBRANE OBSERVER - LEVEL 5
# =============================================================================

@dataclass
class EmergentPattern:
    """A pattern discovered from composed computation."""
    name: str
    opcodes: Tuple[str, str, str]
    input_signature: str
    output_signature: str
    frequency: int
    examples: List[ComputeTrace] = field(default_factory=list)

    def truth_fn(self) -> Optional[Callable]:
        """
        Attempt to derive a truth function for this pattern.
        Returns None if pattern is not deterministic.
        """
        if len(self.examples) < 2:
            return None

        # Check if same inputs always give same outputs
        io_map = {}
        for ex in self.examples:
            in_key = ex.inputs.sum().item()
            out_val = ex.output.sum().item()
            if in_key in io_map and io_map[in_key] != out_val:
                return None  # Non-deterministic
            io_map[in_key] = out_val

        # Pattern is deterministic - could derive truth function
        return lambda a, b, c: self._interpolate(a, b, c)

    def _interpolate(self, a, b, c):
        """Interpolate output from examples (placeholder)."""
        # In practice, would use the frozen shapes from each layer
        return None


class MembraneObserver:
    """
    Observes 486^3 computation and discovers emergent shapes.

    The membrane is the boundary between:
    - Inside: The 486^3 machinery
    - Outside: The patterns it exposes

    "The cell doesn't show you its ribosomes. It shows you its receptors."
    """

    def __init__(self, cube: CPU486Cubed):
        self.cube = cube
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.pattern_examples: Dict[str, List[ComputeTrace]] = defaultdict(list)
        self.discovered_shapes: Dict[str, EmergentPattern] = {}

    def observe(self, traces: List[ComputeTrace]):
        """Observe a batch of computation traces."""
        for trace in traces:
            sig = self._pattern_signature(trace)
            self.pattern_counts[sig] += 1

            # Keep up to 10 examples per pattern
            if len(self.pattern_examples[sig]) < 10:
                self.pattern_examples[sig].append(trace)

    def _pattern_signature(self, trace: ComputeTrace) -> str:
        """Create a signature for a computation pattern."""
        return "_".join(trace.opcodes)

    def discover_shapes(self, min_frequency: int = 5) -> List[EmergentPattern]:
        """
        Discover emergent shapes from observed patterns.

        Shapes must:
        1. Appear frequently (min_frequency)
        2. Be deterministic (same input → same output)
        """
        discovered = []

        for sig, count in self.pattern_counts.items():
            if count < min_frequency:
                continue

            examples = self.pattern_examples[sig]
            opcodes = tuple(examples[0].opcodes)

            # Check determinism
            io_pairs = {}
            is_deterministic = True

            for ex in examples:
                # Create input key (simplified)
                in_key = (
                    ex.inputs[0].sum().item(),
                    ex.inputs[1].sum().item(),
                    ex.inputs[2].sum().item(),  # Carry bits summed
                )
                out_key = ex.output.sum().item()

                if in_key in io_pairs:
                    if abs(io_pairs[in_key] - out_key) > 1e-6:
                        is_deterministic = False
                        break
                io_pairs[in_key] = out_key

            if is_deterministic:
                pattern = EmergentPattern(
                    name=f"L5_{sig}",
                    opcodes=opcodes,
                    input_signature=f"({len(io_pairs)} unique inputs)",
                    output_signature=f"deterministic",
                    frequency=count,
                    examples=examples[:5],
                )
                discovered.append(pattern)
                self.discovered_shapes[pattern.name] = pattern

        return discovered

    def freeze_pattern(self, pattern: EmergentPattern) -> Callable:
        """
        Freeze an emergent pattern into a callable shape.

        The shape executes the 486^3 composition but presents
        a single-operation interface.
        """
        opcodes = list(pattern.opcodes)
        cube = self.cube

        def frozen_shape(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
            """Frozen Level 5 shape."""
            result, carry, _ = cube.forward(opcodes, a, b, c)
            return result, carry

        frozen_shape.__name__ = pattern.name
        frozen_shape.__doc__ = f"Frozen shape: {' → '.join(opcodes)}"

        return frozen_shape

    def summary(self) -> str:
        """Generate a summary of discovered patterns."""
        lines = [
            "=" * 60,
            "MEMBRANE OBSERVER - LEVEL 5",
            "=" * 60,
            f"Total patterns observed: {len(self.pattern_counts)}",
            f"Total traces: {sum(self.pattern_counts.values())}",
            "",
            "Top patterns:",
        ]

        sorted_patterns = sorted(
            self.pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for sig, count in sorted_patterns:
            lines.append(f"  {sig}: {count}x")

        if self.discovered_shapes:
            lines.extend([
                "",
                "Discovered shapes:",
            ])
            for name, pattern in self.discovered_shapes.items():
                lines.append(f"  {name}: {pattern.frequency}x, {pattern.output_signature}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# LEVEL 5 FOUNDRY - EMERGENT SHAPES
# =============================================================================

class Level5Foundry:
    """
    A foundry for Level 5 shapes.

    Takes frozen shapes from 486^3 compositions and creates
    a new shape library at the membrane level.
    """

    def __init__(self, cube: CPU486Cubed):
        self.cube = cube
        self.observer = MembraneObserver(cube)
        self.shapes: Dict[str, Callable] = {}
        self.shape_metadata: Dict[str, EmergentPattern] = {}

    def explore(
        self,
        n_samples: int = 1000,
        program_length: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Explore the 486^3 space to discover patterns.

        Runs random programs and observes the patterns.
        """
        torch.manual_seed(seed)

        op_names = self.cube.op_names
        n_ops = len(op_names)

        print(f"Exploring 486^3 space with {n_samples} samples...")
        self.cube.start_recording()

        for i in range(n_samples):
            # Random operands
            a = torch.randint(0, 2, (1, 32)).float()
            b = torch.randint(0, 2, (1, 32)).float()
            c = torch.randint(0, 2, (1,)).float()

            # Random program (triplet of ops)
            program = [[op_names[torch.randint(0, n_ops, (1,)).item()] for _ in range(3)]]

            # Execute
            self.cube.execute_program(program, a, b, c)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{n_samples} samples...")

        traces = self.cube.stop_recording()
        self.observer.observe(traces)

        print(f"  Recorded {len(traces)} traces")

    def discover(self, min_frequency: int = 3) -> List[EmergentPattern]:
        """Discover and freeze emergent shapes."""
        patterns = self.observer.discover_shapes(min_frequency)

        print(f"\nDiscovered {len(patterns)} emergent shapes:")
        for p in patterns:
            shape_fn = self.observer.freeze_pattern(p)
            self.shapes[p.name] = shape_fn
            self.shape_metadata[p.name] = p
            print(f"  {p.name}: {' → '.join(p.opcodes)} ({p.frequency}x)")

        return patterns

    def get_shape(self, name: str) -> Optional[Callable]:
        """Get a frozen Level 5 shape by name."""
        return self.shapes.get(name)

    def list_shapes(self) -> List[str]:
        """List all discovered shapes."""
        return list(self.shapes.keys())

    def summary(self) -> str:
        """Generate summary of Level 5 foundry."""
        lines = [
            "=" * 60,
            "LEVEL 5 FOUNDRY - MEMBRANE SHAPES",
            "=" * 60,
            f"Base: 486^3 ({self.cube.total_params} params)",
            f"Discovered shapes: {len(self.shapes)}",
            "",
        ]

        if self.shapes:
            lines.append("Frozen shapes:")
            for name, pattern in self.shape_metadata.items():
                ops = " → ".join(pattern.opcodes)
                lines.append(f"  {name}")
                lines.append(f"    Operations: {ops}")
                lines.append(f"    Frequency: {pattern.frequency}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# MAIN - BUILD AND EXPLORE
# =============================================================================

def build_486_cubed() -> Tuple[CPU486Cubed, Level5Foundry]:
    """Build 486^3 and create Level 5 foundry."""

    print("=" * 60)
    print("BUILDING 486^3 - LEVEL 4")
    print("=" * 60)
    print()

    cube = CPU486Cubed()

    print()
    print("=" * 60)
    print("CREATING LEVEL 5 FOUNDRY")
    print("=" * 60)
    print()

    foundry = Level5Foundry(cube)

    return cube, foundry


def explore_and_discover(
    foundry: Level5Foundry,
    n_samples: int = 1000,
    min_frequency: int = 3,
) -> List[EmergentPattern]:
    """Explore 486^3 space and discover emergent shapes."""

    print()
    print("=" * 60)
    print("EXPLORING 486^3 SPACE")
    print("=" * 60)
    print()

    foundry.explore(n_samples=n_samples)

    print()
    print("=" * 60)
    print("DISCOVERING EMERGENT SHAPES")
    print("=" * 60)

    patterns = foundry.discover(min_frequency=min_frequency)

    print()
    print(foundry.summary())

    return patterns


if __name__ == "__main__":
    # Build
    cube, foundry = build_486_cubed()

    # Explore
    patterns = explore_and_discover(foundry, n_samples=2000, min_frequency=2)

    # Test a discovered shape
    if patterns:
        print("\n" + "=" * 60)
        print("TESTING EMERGENT SHAPE")
        print("=" * 60)

        shape = foundry.get_shape(patterns[0].name)

        # Test input
        a = torch.randint(0, 2, (1, 32)).float()
        b = torch.randint(0, 2, (1, 32)).float()
        c = torch.tensor([0.0])

        result, carry = shape(a, b, c)

        print(f"\nShape: {patterns[0].name}")
        print(f"Operations: {' → '.join(patterns[0].opcodes)}")
        print(f"Input a: {a.sum().item():.0f} bits set")
        print(f"Input b: {b.sum().item():.0f} bits set")
        print(f"Output: {result.sum().item():.0f} bits set")
        print(f"Carry: {carry.item()}")

    print("\n" + "=" * 60)
    print("486^3 COMPLETE")
    print("\"Each level: observe, find patterns, freeze, compose.\"")
    print("=" * 60)
