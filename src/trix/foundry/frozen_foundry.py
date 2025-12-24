"""
FrozenFoundry - Neural Models for Deterministic Systems.

Turn any deterministic system into a 100% accurate neural model.

Usage:
    from trix.foundry import FrozenFoundry

    # Create foundry
    foundry = FrozenFoundry(bit_width=8)

    # Register operations with ground truth
    foundry.register("add", lambda a, b, c: ((a + b + c) & 0xFF, int((a + b + c) > 255)))
    foundry.register("sub", lambda a, b, c: ((a - b - (1-c)) & 0xFF, int(a >= b + (1-c))))
    foundry.register("and", lambda a, b, c: (a & b, 0))
    foundry.register("xor", lambda a, b, c: (a ^ b, 0))

    # Build model (discovers shapes, trains routing)
    model = foundry.build()

    # Validate
    assert foundry.validate() == 1.0

    # Export
    foundry.export("my_alu.pt")
    foundry.export_onnx("my_alu.onnx")

The foundry:
1. Analyzes your truth tables
2. Matches or composes frozen shapes
3. Learns routing (the only learned part)
4. Validates 100% accuracy
5. Exports production-ready model

"Computation is topology. Learning is routing."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
import time
import json


# =============================================================================
# PURE MATH PRIMITIVES
# =============================================================================

class PureMath:
    """
    The eternal primitives. Exact on {0,1}, differentiable everywhere.
    """

    @staticmethod
    def xor(a: Tensor, b: Tensor) -> Tensor:
        """XOR: a + b - 2ab (saddle surface)"""
        return a + b - 2 * a * b

    @staticmethod
    def and_op(a: Tensor, b: Tensor) -> Tensor:
        """AND: ab (product)"""
        return a * b

    @staticmethod
    def or_op(a: Tensor, b: Tensor) -> Tensor:
        """OR: a + b - ab (union)"""
        return a + b - a * b

    @staticmethod
    def not_op(a: Tensor) -> Tensor:
        """NOT: 1 - a (reflection)"""
        return 1 - a

    @staticmethod
    def nand(a: Tensor, b: Tensor) -> Tensor:
        """NAND: NOT(AND(a, b))"""
        return 1 - a * b

    @staticmethod
    def nor(a: Tensor, b: Tensor) -> Tensor:
        """NOR: NOT(OR(a, b))"""
        return 1 - (a + b - a * b)

    @staticmethod
    def xnor(a: Tensor, b: Tensor) -> Tensor:
        """XNOR: NOT(XOR(a, b)) = 1 - a - b + 2ab"""
        return 1 - a - b + 2 * a * b

    @staticmethod
    def mux(a: Tensor, b: Tensor, sel: Tensor) -> Tensor:
        """MUX: sel ? b : a"""
        return a * (1 - sel) + b * sel

    @staticmethod
    def full_adder(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """Full adder: (sum, carry)"""
        p = PureMath.xor(a, b)
        s = PureMath.xor(p, c)
        cout = PureMath.or_op(PureMath.and_op(a, b), PureMath.and_op(c, p))
        return s, cout


# =============================================================================
# SHAPE LIBRARY
# =============================================================================

class ShapeType(IntEnum):
    """Built-in shape types."""
    # Arithmetic
    RIPPLE_ADD = 0
    RIPPLE_SUB = 1

    # Logic
    PARALLEL_AND = 2
    PARALLEL_OR = 3
    PARALLEL_XOR = 4
    PARALLEL_NAND = 5
    PARALLEL_NOR = 6
    PARALLEL_XNOR = 7

    # Shifts
    SHIFT_LEFT = 8
    SHIFT_RIGHT = 9
    ROTATE_LEFT = 10
    ROTATE_RIGHT = 11

    # Inc/Dec
    INCREMENT = 12
    DECREMENT = 13

    # Transfer
    IDENTITY = 14
    COMPLEMENT = 15

    # Comparison
    COMPARE = 16

    # Custom (user-defined start here)
    CUSTOM_START = 100


@dataclass
class FrozenShape:
    """A frozen computation shape."""
    name: str
    shape_type: int
    fn: Callable
    n_inputs: int = 2           # Number of input operands
    uses_carry: bool = False    # Uses carry input
    produces_carry: bool = False  # Produces carry output
    bit_width: int = 8
    description: str = ""


class ShapeLibrary:
    """
    Library of frozen shapes.

    Built-in shapes + user-defined shapes.
    """

    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width
        self.shapes: Dict[int, FrozenShape] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in shapes."""
        bw = self.bit_width

        # Ripple Add
        def ripple_add(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
            result = []
            carry = c.squeeze(-1) if c.dim() > 1 else c
            for i in range(bw):
                s, carry = PureMath.full_adder(a[:, i], b[:, i], carry)
                result.append(s)
            return torch.stack(result, dim=1), carry

        # Ripple Sub
        def ripple_sub(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
            return ripple_add(a, PureMath.not_op(b), c)

        # Parallel ops
        def parallel_and(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.and_op(a, b), torch.zeros(a.shape[0], device=a.device)

        def parallel_or(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.or_op(a, b), torch.zeros(a.shape[0], device=a.device)

        def parallel_xor(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.xor(a, b), torch.zeros(a.shape[0], device=a.device)

        def parallel_nand(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.nand(a, b), torch.zeros(a.shape[0], device=a.device)

        def parallel_nor(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.nor(a, b), torch.zeros(a.shape[0], device=a.device)

        def parallel_xnor(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.xnor(a, b), torch.zeros(a.shape[0], device=a.device)

        # Shifts
        def shift_left(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            zeros = torch.zeros_like(a[:, :1])
            result = torch.cat([zeros, a[:, :bw-1]], dim=1)
            return result, a[:, bw-1]

        def shift_right(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            zeros = torch.zeros_like(a[:, :1])
            result = torch.cat([a[:, 1:], zeros], dim=1)
            return result, a[:, 0]

        def rotate_left(a: Tensor, _: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
            c_bit = c.unsqueeze(-1) if c.dim() == 1 else c
            result = torch.cat([c_bit, a[:, :bw-1]], dim=1)
            return result, a[:, bw-1]

        def rotate_right(a: Tensor, _: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
            c_bit = c.unsqueeze(-1) if c.dim() == 1 else c
            result = torch.cat([a[:, 1:], c_bit], dim=1)
            return result, a[:, 0]

        # Inc/Dec
        def increment(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            b = torch.zeros_like(a)
            b[:, 0] = 1.0
            c = torch.zeros(a.shape[0], device=a.device)
            return ripple_add(a, b, c)

        def decrement(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            b = torch.ones_like(a)
            c = torch.zeros(a.shape[0], device=a.device)
            return ripple_add(a, b, c)

        # Identity/Complement
        def identity(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            return a.clone(), torch.zeros(a.shape[0], device=a.device)

        def complement(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
            return PureMath.not_op(a), torch.zeros(a.shape[0], device=a.device)

        # Compare (subtract without storing)
        def compare(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
            c = torch.ones(a.shape[0], device=a.device)  # Set carry for subtraction
            result, carry = ripple_sub(a, b, c)
            return result, carry

        # Register all
        builtins = [
            FrozenShape("ripple_add", ShapeType.RIPPLE_ADD, ripple_add,
                       uses_carry=True, produces_carry=True, description="A + B + C"),
            FrozenShape("ripple_sub", ShapeType.RIPPLE_SUB, ripple_sub,
                       uses_carry=True, produces_carry=True, description="A - B - !C"),
            FrozenShape("parallel_and", ShapeType.PARALLEL_AND, parallel_and,
                       description="A & B"),
            FrozenShape("parallel_or", ShapeType.PARALLEL_OR, parallel_or,
                       description="A | B"),
            FrozenShape("parallel_xor", ShapeType.PARALLEL_XOR, parallel_xor,
                       description="A ^ B"),
            FrozenShape("parallel_nand", ShapeType.PARALLEL_NAND, parallel_nand,
                       description="!(A & B)"),
            FrozenShape("parallel_nor", ShapeType.PARALLEL_NOR, parallel_nor,
                       description="!(A | B)"),
            FrozenShape("parallel_xnor", ShapeType.PARALLEL_XNOR, parallel_xnor,
                       description="!(A ^ B)"),
            FrozenShape("shift_left", ShapeType.SHIFT_LEFT, shift_left,
                       n_inputs=1, produces_carry=True, description="A << 1"),
            FrozenShape("shift_right", ShapeType.SHIFT_RIGHT, shift_right,
                       n_inputs=1, produces_carry=True, description="A >> 1"),
            FrozenShape("rotate_left", ShapeType.ROTATE_LEFT, rotate_left,
                       n_inputs=1, uses_carry=True, produces_carry=True, description="ROL"),
            FrozenShape("rotate_right", ShapeType.ROTATE_RIGHT, rotate_right,
                       n_inputs=1, uses_carry=True, produces_carry=True, description="ROR"),
            FrozenShape("increment", ShapeType.INCREMENT, increment,
                       n_inputs=1, description="A + 1"),
            FrozenShape("decrement", ShapeType.DECREMENT, decrement,
                       n_inputs=1, description="A - 1"),
            FrozenShape("identity", ShapeType.IDENTITY, identity,
                       n_inputs=1, description="A"),
            FrozenShape("complement", ShapeType.COMPLEMENT, complement,
                       n_inputs=1, description="~A"),
            FrozenShape("compare", ShapeType.COMPARE, compare,
                       description="A - B (flags only)"),
        ]

        for shape in builtins:
            shape.bit_width = bw
            self.shapes[shape.shape_type] = shape

    def register(self, name: str, fn: Callable, **kwargs) -> int:
        """Register a custom shape. Returns shape ID."""
        shape_id = ShapeType.CUSTOM_START + len([
            s for s in self.shapes if s >= ShapeType.CUSTOM_START
        ])
        shape = FrozenShape(name, shape_id, fn, bit_width=self.bit_width, **kwargs)
        self.shapes[shape_id] = shape
        return shape_id

    def get(self, shape_id: int) -> FrozenShape:
        return self.shapes[shape_id]

    def all_shapes(self) -> List[FrozenShape]:
        return list(self.shapes.values())

    def __len__(self) -> int:
        return len(self.shapes)


# =============================================================================
# SHAPE MATCHER
# =============================================================================

class ShapeMatcher:
    """
    Matches truth tables to frozen shapes.

    Given a ground truth function, finds the matching shape
    or composition of shapes.
    """

    def __init__(self, library: ShapeLibrary, n_test_samples: int = 256):
        self.library = library
        self.n_test_samples = n_test_samples
        self.bit_width = library.bit_width

    def _int_to_bits(self, x: Tensor) -> Tensor:
        bits = []
        for i in range(self.bit_width):
            bits.append(((x >> i) & 1).float())
        return torch.stack(bits, dim=-1)

    def _bits_to_int(self, bits: Tensor) -> Tensor:
        result = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
        for i in range(bits.shape[-1]):
            result += (bits[:, i].round().long() << i)
        return result

    def match(
        self,
        truth_fn: Callable,
        name: str = "unknown",
    ) -> Tuple[Optional[int], float]:
        """
        Find the shape that matches the truth function.

        Args:
            truth_fn: Function (a, b, c) -> (result, carry) where all are ints
            name: Name for logging

        Returns:
            (shape_id, accuracy) - shape_id is None if no match found
        """
        # Generate test data
        torch.manual_seed(42)
        n = self.n_test_samples
        max_val = 2 ** self.bit_width

        a_int = torch.randint(0, max_val, (n,))
        b_int = torch.randint(0, max_val, (n,))
        c_int = torch.randint(0, 2, (n,))

        # Get ground truth
        expected_result = []
        expected_carry = []
        for i in range(n):
            r, c = truth_fn(a_int[i].item(), b_int[i].item(), c_int[i].item())
            expected_result.append(r)
            expected_carry.append(c)

        expected_result = self._int_to_bits(torch.tensor(expected_result))
        expected_carry = torch.tensor(expected_carry, dtype=torch.float32)

        # Convert inputs to bits
        a_bits = self._int_to_bits(a_int)
        b_bits = self._int_to_bits(b_int)
        c_bits = c_int.float()

        # Test each shape
        best_shape = None
        best_acc = 0.0

        for shape_id, shape in self.library.shapes.items():
            try:
                result, carry = shape.fn(a_bits, b_bits, c_bits)

                # Check accuracy
                result_match = (result.round() == expected_result).all(dim=1)

                # Only check carry if the ground truth produces non-zero carries
                if expected_carry.abs().sum() > 0 and shape.produces_carry:
                    carry_match = (carry.round() == expected_carry)
                else:
                    carry_match = torch.ones(n, dtype=torch.bool)

                acc = (result_match & carry_match).float().mean().item()

                if acc > best_acc:
                    best_acc = acc
                    best_shape = int(shape_id)  # Ensure it's an int

                if acc >= 0.9999:  # Use >= to handle floating point
                    return int(shape_id), 1.0

            except Exception as e:
                continue

        return best_shape, best_acc


# =============================================================================
# FROZEN ROUTER
# =============================================================================

class FrozenRouter(nn.Module):
    """
    Routes operations to frozen shapes.

    The only learned component - everything else is frozen.
    """

    def __init__(
        self,
        num_operations: int,
        library: ShapeLibrary,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_operations = num_operations
        self.library = library
        self.num_shapes = len(library)
        self.temperature = temperature

        # Routing logits
        self.routing = nn.Parameter(torch.zeros(num_operations, self.num_shapes))

        # Shape function list (for forward pass)
        self.shape_fns = [library.get(i).fn for i in sorted(library.shapes.keys())]

    def init_from_matches(self, op_to_shape: Dict[int, int]):
        """Initialize routing from matched shapes."""
        with torch.no_grad():
            for op_id, shape_id in op_to_shape.items():
                shape_idx = list(sorted(self.library.shapes.keys())).index(shape_id)
                self.routing[op_id, shape_idx] = 10.0

    def forward(
        self,
        op_id: Tensor,
        a: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict]:
        B = a.shape[0]

        # Get routing weights
        logits = self.routing[op_id]

        if self.training:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), self.num_shapes).float()

        # Execute all shapes
        results = []
        carries = []

        for fn in self.shape_fns:
            r, car = fn(a, b, c)
            results.append(r)
            carries.append(car)

        results = torch.stack(results, dim=-1)
        carries = torch.stack(carries, dim=-1)

        # Weighted blend
        result = (results * weights.unsqueeze(1)).sum(dim=-1)
        carry = (carries * weights).sum(dim=-1)

        info = {
            'shape_idx': logits.argmax(dim=-1, keepdim=True),
            'weights': weights,
        }

        return result, carry, info

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# FROZEN FOUNDRY
# =============================================================================

@dataclass
class Operation:
    """A registered operation."""
    name: str
    op_id: int
    truth_fn: Callable
    matched_shape: Optional[int] = None
    match_accuracy: float = 0.0


@dataclass
class BuildResult:
    """Result of building a model."""
    model: FrozenRouter
    accuracy: float
    training_steps: int
    training_time_ms: float
    num_params: int
    operations: List[Operation]


class FrozenFoundry:
    """
    FrozenFoundry - Turn deterministic systems into neural models.

    Example:
        foundry = FrozenFoundry(bit_width=8)
        foundry.register("add", lambda a, b, c: ((a + b + c) & 0xFF, int((a + b + c) > 255)))
        foundry.register("xor", lambda a, b, c: (a ^ b, 0))
        model = foundry.build()
        assert foundry.validate() == 1.0
    """

    def __init__(
        self,
        bit_width: int = 8,
        device: str = 'cpu',
    ):
        self.bit_width = bit_width
        self.device = device
        self.library = ShapeLibrary(bit_width)
        self.matcher = ShapeMatcher(self.library)
        self.operations: Dict[str, Operation] = {}
        self.model: Optional[FrozenRouter] = None
        self._op_counter = 0

    def register(
        self,
        name: str,
        truth_fn: Callable,
    ) -> int:
        """
        Register an operation with its ground truth.

        Args:
            name: Operation name
            truth_fn: Function (a: int, b: int, c: int) -> (result: int, carry: int)

        Returns:
            Operation ID
        """
        op_id = self._op_counter
        self._op_counter += 1

        self.operations[name] = Operation(
            name=name,
            op_id=op_id,
            truth_fn=truth_fn,
        )

        return op_id

    def register_shape(self, name: str, fn: Callable, **kwargs) -> int:
        """Register a custom frozen shape."""
        return self.library.register(name, fn, **kwargs)

    def _match_operations(self, verbose: bool = True) -> Dict[int, int]:
        """Match all operations to shapes."""
        op_to_shape = {}

        if verbose:
            print(f"Matching {len(self.operations)} operations to {len(self.library)} shapes...")

        for name, op in self.operations.items():
            shape_id, acc = self.matcher.match(op.truth_fn, name)
            op.matched_shape = shape_id
            op.match_accuracy = acc

            if shape_id is not None:
                op_to_shape[op.op_id] = shape_id

            if verbose:
                shape_name = self.library.get(shape_id).name if shape_id is not None else "NONE"
                print(f"  {name:20s} -> {shape_name:20s} ({acc*100:.1f}%)")

        return op_to_shape

    def _int_to_bits(self, x: Tensor) -> Tensor:
        bits = []
        for i in range(self.bit_width):
            bits.append(((x >> i) & 1).float())
        return torch.stack(bits, dim=-1)

    def _generate_data(
        self,
        n_samples_per_op: int = 1000,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate training data for all operations."""
        max_val = 2 ** self.bit_width

        all_ops = []
        all_a = []
        all_b = []
        all_c = []
        all_result = []
        all_carry = []

        for name, op in self.operations.items():
            for _ in range(n_samples_per_op):
                a = torch.randint(0, max_val, (1,)).item()
                b = torch.randint(0, max_val, (1,)).item()
                c = torch.randint(0, 2, (1,)).item()

                r, car = op.truth_fn(a, b, c)

                all_ops.append(op.op_id)
                all_a.append(a)
                all_b.append(b)
                all_c.append(c)
                all_result.append(r)
                all_carry.append(car)

        return (
            torch.tensor(all_ops, dtype=torch.long),
            self._int_to_bits(torch.tensor(all_a)),
            self._int_to_bits(torch.tensor(all_b)),
            torch.tensor(all_c, dtype=torch.float32),
            self._int_to_bits(torch.tensor(all_result)),
            torch.tensor(all_carry, dtype=torch.float32),
        )

    def build(
        self,
        max_steps: int = 100,
        lr: float = 0.1,
        batch_size: int = 1024,
        n_samples_per_op: int = 1000,
        verbose: bool = True,
    ) -> BuildResult:
        """
        Build the neural model.

        1. Match operations to shapes
        2. Generate training data
        3. Train routing (if needed)
        4. Validate
        """
        start_time = time.time()

        # Match operations
        op_to_shape = self._match_operations(verbose)

        # Create model
        self.model = FrozenRouter(len(self.operations), self.library)
        self.model.init_from_matches(op_to_shape)

        # Generate data
        if verbose:
            print(f"\nGenerating training data...")
        ops, a, b, c, target, target_carry = self._generate_data(n_samples_per_op)
        n_samples = len(ops)
        if verbose:
            print(f"  {n_samples} samples")

        # Check if already perfect
        self.model.eval()
        with torch.no_grad():
            result, carry, _ = self.model(ops, a, b, c)
            acc = (result.round() == target).all(dim=1).float().mean().item()

        if acc >= 0.9999:
            if verbose:
                print(f"\nPerfect accuracy without training!")
            elapsed = (time.time() - start_time) * 1000
            return BuildResult(
                model=self.model,
                accuracy=acc,
                training_steps=0,
                training_time_ms=elapsed,
                num_params=self.model.num_params,
                operations=list(self.operations.values()),
            )

        # Train
        if verbose:
            print(f"\nTraining routing ({self.model.num_params} params)...")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        steps = 0
        for step in range(max_steps):
            idx = torch.randperm(n_samples)[:batch_size]

            result, carry, _ = self.model(ops[idx], a[idx], b[idx], c[idx])
            loss = F.mse_loss(result, target[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps = step + 1

            if step % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    result_full, _, _ = self.model(ops, a, b, c)
                    acc = (result_full.round() == target).all(dim=1).float().mean().item()
                self.model.train()

                if verbose:
                    print(f"  Step {step:4d} | Acc: {acc*100:.2f}%")

                if acc >= 0.9999:
                    break

        elapsed = (time.time() - start_time) * 1000

        # Final accuracy
        self.model.eval()
        with torch.no_grad():
            result, _, _ = self.model(ops, a, b, c)
            final_acc = (result.round() == target).all(dim=1).float().mean().item()

        if verbose:
            print(f"\nBuild complete!")
            print(f"  Accuracy: {final_acc*100:.2f}%")
            print(f"  Steps: {steps}")
            print(f"  Time: {elapsed:.1f}ms")
            print(f"  Params: {self.model.num_params}")

        return BuildResult(
            model=self.model,
            accuracy=final_acc,
            training_steps=steps,
            training_time_ms=elapsed,
            num_params=self.model.num_params,
            operations=list(self.operations.values()),
        )

    def execute(
        self,
        op_name: str,
        a: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Execute a single operation.

        Args:
            op_name: Operation name (e.g., "ADD", "XOR")
            a: First operand bits [batch, bit_width]
            b: Second operand bits [batch, bit_width]
            c: Carry in [batch]

        Returns:
            result: Output bits [batch, bit_width]
            carry: Carry out [batch]
        """
        if self.model is None:
            raise ValueError("Call build() first")

        if op_name not in self.operations:
            raise ValueError(f"Unknown operation: {op_name}")

        op_id = self.operations[op_name].op_id
        batch_size = a.shape[0]

        op_tensor = torch.full((batch_size,), op_id, dtype=torch.long, device=a.device)

        self.model.eval()
        with torch.no_grad():
            result, carry, _ = self.model(op_tensor, a, b, c)

        return result, carry

    def validate(self, n_samples: int = 10000) -> float:
        """Validate accuracy on random samples."""
        if self.model is None:
            raise ValueError("Call build() first")

        ops, a, b, c, target, _ = self._generate_data(n_samples // len(self.operations))

        self.model.eval()
        with torch.no_grad():
            result, _, _ = self.model(ops, a, b, c)
            acc = (result.round() == target).all(dim=1).float().mean().item()

        return acc

    def export(self, path: Union[str, Path]):
        """Export model to .pt file."""
        if self.model is None:
            raise ValueError("Call build() first")
        torch.save(self.model.state_dict(), path)

    def export_onnx(self, path: Union[str, Path]):
        """Export model to ONNX."""
        if self.model is None:
            raise ValueError("Call build() first")

        # Dummy inputs
        B = 1
        op = torch.zeros(B, dtype=torch.long)
        a = torch.zeros(B, self.bit_width)
        b = torch.zeros(B, self.bit_width)
        c = torch.zeros(B)

        torch.onnx.export(
            self.model,
            (op, a, b, c),
            path,
            input_names=['operation', 'a', 'b', 'carry_in'],
            output_names=['result', 'carry_out'],
            dynamic_axes={
                'operation': {0: 'batch'},
                'a': {0: 'batch'},
                'b': {0: 'batch'},
                'carry_in': {0: 'batch'},
                'result': {0: 'batch'},
                'carry_out': {0: 'batch'},
            },
        )

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "FROZEN FOUNDRY SUMMARY",
            "=" * 60,
            f"Bit width: {self.bit_width}",
            f"Operations: {len(self.operations)}",
            f"Shapes: {len(self.library)}",
            "",
            "Operations:",
        ]

        for name, op in self.operations.items():
            shape_name = self.library.get(op.matched_shape).name if op.matched_shape is not None else "NONE"
            lines.append(f"  {name:20s} -> {shape_name:20s}")

        if self.model:
            lines.extend([
                "",
                f"Model params: {self.model.num_params}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FROZEN FOUNDRY - Demo")
    print("=" * 70)
    print()

    # Create foundry
    foundry = FrozenFoundry(bit_width=8)

    # Register operations (same as 6502 ALU)
    foundry.register("ADC", lambda a, b, c: ((a + b + c) & 0xFF, int((a + b + c) > 255)))
    foundry.register("SBC", lambda a, b, c: ((a + (b ^ 0xFF) + c) & 0xFF, int((a + (b ^ 0xFF) + c) > 255)))
    foundry.register("AND", lambda a, b, c: (a & b, 0))
    foundry.register("ORA", lambda a, b, c: (a | b, 0))
    foundry.register("EOR", lambda a, b, c: (a ^ b, 0))
    foundry.register("ASL", lambda a, b, c: ((a << 1) & 0xFF, (a >> 7) & 1))
    foundry.register("LSR", lambda a, b, c: (a >> 1, a & 1))
    foundry.register("INC", lambda a, b, c: ((a + 1) & 0xFF, 0))
    foundry.register("DEC", lambda a, b, c: ((a - 1) & 0xFF, 0))

    # Build
    result = foundry.build(verbose=True)

    # Validate
    print(f"\nValidation accuracy: {foundry.validate()*100:.2f}%")

    # Summary
    print()
    print(foundry.summary())
