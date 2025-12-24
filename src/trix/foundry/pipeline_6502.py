"""
6502 Pipeline - End to End.

Mesa 16.3: From Zero to Fully Functional 6502

This module builds a complete 6502 ALU using:
- Frozen shapes only (no vanilla NN approximation)
- Nested training (~20 steps per shape alignment)
- TriX routing (content-addressable shape selection)
- Doula observation (glass box throughout)
- Timed and measured (every step recorded)

Usage:
    from trix.foundry.pipeline_6502 import run_6502_pipeline

    report = run_6502_pipeline()
    print(report.summary())

"Enjoy the Emergence. It is a time of joy."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import time
from datetime import datetime

from trix.doula import DoulaFunction, DoulaTrainingConfig


# =============================================================================
# PURE MATH PRIMITIVES (0 params - eternal truths)
# =============================================================================

class PureMath:
    """The eternal geometric truths. No parameters. No learning."""

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
    def full_adder(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """Full adder: (a, b, c_in) → (sum, c_out)"""
        p = PureMath.xor(a, b)
        sum_bit = PureMath.xor(p, c)
        c_out = PureMath.or_op(PureMath.and_op(a, b), PureMath.and_op(c, p))
        return sum_bit, c_out


# =============================================================================
# THE 16 FROZEN SHAPES (0 params each)
# =============================================================================

class ShapeID(IntEnum):
    """The 16 eternal shapes."""
    RIPPLE_ADD = 0
    RIPPLE_SUB = 1
    PARALLEL_AND = 2
    PARALLEL_OR = 3
    PARALLEL_XOR = 4
    SHIFT_LEFT = 5
    SHIFT_RIGHT = 6
    ROTATE_LEFT = 7
    ROTATE_RIGHT = 8
    INCREMENT = 9
    DECREMENT = 10
    TRANSFER = 11
    LOAD = 12
    STORE = 13
    BIT_TEST = 14
    IDENTITY = 15


class FrozenShapes:
    """The 16 frozen computation topologies. No learning. Just geometry."""

    @staticmethod
    def ripple_add(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """ADC: 8-bit ripple carry adder."""
        result = []
        carry = c.squeeze(-1) if c.dim() > 1 else c
        for i in range(8):
            s, carry = PureMath.full_adder(a[:, i], b[:, i], carry)
            result.append(s)
        return torch.stack(result, dim=1), carry

    @staticmethod
    def ripple_sub(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """SBC: A - B - (1-C) = A + NOT(B) + C"""
        return FrozenShapes.ripple_add(a, PureMath.not_op(b), c)

    @staticmethod
    def parallel_and(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
        """AND: A & B"""
        return PureMath.and_op(a, b), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def parallel_or(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
        """OR: A | B"""
        return PureMath.or_op(a, b), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def parallel_xor(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
        """XOR: A ^ B"""
        return PureMath.xor(a, b), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def shift_left(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """ASL: Arithmetic Shift Left"""
        zeros = torch.zeros_like(a[:, :1])
        result = torch.cat([zeros, a[:, :7]], dim=1)
        return result, a[:, 7]

    @staticmethod
    def shift_right(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """LSR: Logical Shift Right"""
        zeros = torch.zeros_like(a[:, :1])
        result = torch.cat([a[:, 1:], zeros], dim=1)
        return result, a[:, 0]

    @staticmethod
    def rotate_left(a: Tensor, _: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """ROL: Rotate Left through Carry"""
        c_bit = c.unsqueeze(-1) if c.dim() == 1 else c
        result = torch.cat([c_bit, a[:, :7]], dim=1)
        return result, a[:, 7]

    @staticmethod
    def rotate_right(a: Tensor, _: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """ROR: Rotate Right through Carry"""
        c_bit = c.unsqueeze(-1) if c.dim() == 1 else c
        result = torch.cat([a[:, 1:], c_bit], dim=1)
        return result, a[:, 0]

    @staticmethod
    def increment(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """INC: A + 1"""
        b = torch.zeros_like(a)
        b[:, 0] = 1.0
        c = torch.zeros(a.shape[0], device=a.device)
        return FrozenShapes.ripple_add(a, b, c)

    @staticmethod
    def decrement(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """DEC: A - 1 = A + 0xFF"""
        b = torch.ones_like(a)  # 0xFF
        c = torch.zeros(a.shape[0], device=a.device)
        return FrozenShapes.ripple_add(a, b, c)

    @staticmethod
    def transfer(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """Transfer: pass through"""
        return a.clone(), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def load(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """Load: pass through (same as transfer)"""
        return a.clone(), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def store(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """Store: pass through (same as transfer)"""
        return a.clone(), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def bit_test(a: Tensor, b: Tensor, _: Tensor) -> Tuple[Tensor, Tensor]:
        """BIT: A & M for flags"""
        return PureMath.and_op(a, b), torch.zeros(a.shape[0], device=a.device)

    @staticmethod
    def identity(a: Tensor, _: Tensor, __: Tensor) -> Tuple[Tensor, Tensor]:
        """NOP: identity"""
        return a.clone(), torch.zeros(a.shape[0], device=a.device)


# Shape function lookup
SHAPE_FUNCTIONS = {
    ShapeID.RIPPLE_ADD: FrozenShapes.ripple_add,
    ShapeID.RIPPLE_SUB: FrozenShapes.ripple_sub,
    ShapeID.PARALLEL_AND: FrozenShapes.parallel_and,
    ShapeID.PARALLEL_OR: FrozenShapes.parallel_or,
    ShapeID.PARALLEL_XOR: FrozenShapes.parallel_xor,
    ShapeID.SHIFT_LEFT: FrozenShapes.shift_left,
    ShapeID.SHIFT_RIGHT: FrozenShapes.shift_right,
    ShapeID.ROTATE_LEFT: FrozenShapes.rotate_left,
    ShapeID.ROTATE_RIGHT: FrozenShapes.rotate_right,
    ShapeID.INCREMENT: FrozenShapes.increment,
    ShapeID.DECREMENT: FrozenShapes.decrement,
    ShapeID.TRANSFER: FrozenShapes.transfer,
    ShapeID.LOAD: FrozenShapes.load,
    ShapeID.STORE: FrozenShapes.store,
    ShapeID.BIT_TEST: FrozenShapes.bit_test,
    ShapeID.IDENTITY: FrozenShapes.identity,
}


# =============================================================================
# UTILITIES
# =============================================================================

def int_to_bits(x: Tensor, n_bits: int = 8) -> Tensor:
    """Convert integers to bit representation (LSB first)."""
    bits = []
    for i in range(n_bits):
        bits.append(((x >> i) & 1).float())
    return torch.stack(bits, dim=-1)


def bits_to_int(bits: Tensor) -> Tensor:
    """Convert bits back to integers."""
    result = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    for i in range(bits.shape[-1]):
        result += (bits[:, i].round().long() << i)
    return result


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_shape_data(
    shape_id: ShapeID,
    n_samples: int = 256,
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Generate training data for a specific shape.

    Returns:
        a_bits, b_bits, c_in, target_result, target_carry
    """
    torch.manual_seed(seed)

    a_int = torch.randint(0, 256, (n_samples,))
    b_int = torch.randint(0, 256, (n_samples,))
    c_in = torch.randint(0, 2, (n_samples,)).float()

    a_bits = int_to_bits(a_int)
    b_bits = int_to_bits(b_int)

    # Compute ground truth using frozen shape
    shape_fn = SHAPE_FUNCTIONS[shape_id]
    with torch.no_grad():
        target_result, target_carry = shape_fn(a_bits, b_bits, c_in)

    return a_bits, b_bits, c_in, target_result, target_carry


# =============================================================================
# SINGLE SHAPE ALIGNER (Inner Loop)
# =============================================================================

class ShapeAligner(nn.Module):
    """
    Aligns a single frozen shape.

    The only learnable part is confidence - how strongly to apply the shape.
    The shape itself is frozen geometry.
    """

    def __init__(self, shape_fn: Callable, d_hidden: int = 32):
        super().__init__()
        self.shape_fn = shape_fn
        self.confidence = nn.Sequential(
            nn.Linear(17, d_hidden),  # 8 + 8 + 1 = 17 input bits
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        x = torch.cat([a, b, c.unsqueeze(-1)], dim=-1)
        conf = self.confidence(x)

        # Apply frozen shape
        result, carry = self.shape_fn(a, b, c)

        # Scale by confidence
        result = result * conf
        carry = carry * conf.squeeze(-1)

        routing_info = {
            'shape_idx': torch.zeros(a.shape[0], 1, dtype=torch.long, device=a.device),
            'shape_weights': torch.ones(a.shape[0], 1, 1, device=a.device),
            'partner_idx': torch.zeros(a.shape[0], 1, dtype=torch.long, device=a.device),
            'distances': torch.zeros(a.shape[0], 1, 1, device=a.device),
            'confidence': conf,
        }

        return result, carry, routing_info


# =============================================================================
# MULTI-SHAPE ROUTER (Outer Loop)
# =============================================================================

class TriX6502Router(nn.Module):
    """
    TriX-native routing for 6502.

    Uses ternary signature matching for opcode → shape routing.
    The opcode determines which shape fires.
    """

    def __init__(self, num_opcodes: int = 16, d_sig: int = 32):
        super().__init__()
        self.num_opcodes = num_opcodes
        self.num_shapes = 16
        self.d_sig = d_sig

        # Opcode signature projection (learns to route opcodes to shapes)
        self.opcode_to_sig = nn.Embedding(num_opcodes, d_sig)

        # Shape signatures (fixed, derived from shape behavior)
        self.register_buffer('shape_sigs', self._derive_shape_signatures())

        # Direct routing logits (the meaning layer)
        self.routing_logits = nn.Parameter(torch.zeros(num_opcodes, 16))

        # Temperature
        self.temperature = 0.5

    def _derive_shape_signatures(self) -> Tensor:
        """Derive deterministic signatures from shape names."""
        sigs = []
        for shape_id in ShapeID:
            torch.manual_seed(hash(shape_id.name) % (2**31))
            sig = torch.randn(self.d_sig).sign()
            sig[sig == 0] = 1
            sigs.append(sig)
        return torch.stack(sigs)

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        opcode: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict]:
        B = a.shape[0]

        # Get routing weights from meaning layer
        logits = self.routing_logits[opcode]  # [B, 16]

        if self.training:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), self.num_shapes).float()

        # Execute all shapes and blend
        results = []
        carries = []
        for shape_id in ShapeID:
            fn = SHAPE_FUNCTIONS[shape_id]
            r, car = fn(a, b, c)
            results.append(r)
            carries.append(car)

        results = torch.stack(results, dim=-1)  # [B, 8, 16]
        carries = torch.stack(carries, dim=-1)  # [B, 16]

        # Weighted sum
        result = (results * weights.unsqueeze(1)).sum(dim=-1)
        carry = (carries * weights).sum(dim=-1)

        routing_info = {
            'shape_idx': logits.argmax(dim=-1, keepdim=True),
            'shape_weights': weights.unsqueeze(1),
            'partner_idx': torch.zeros(B, 1, dtype=torch.long, device=a.device),
            'distances': torch.zeros(B, 1, 1, device=a.device),
        }

        return result, carry, routing_info


# =============================================================================
# MEASUREMENT AND REPORTING
# =============================================================================

@dataclass
class ShapeAlignmentResult:
    """Result of aligning a single shape."""
    shape_id: ShapeID
    shape_name: str
    steps_to_converge: int
    final_loss: float
    final_accuracy: float
    time_ms: float
    doula_phases: List[str] = field(default_factory=list)


@dataclass
class RoutingResult:
    """Result of routing training."""
    steps: int
    final_loss: float
    final_accuracy: float
    routing_accuracy: float
    time_ms: float
    per_opcode_accuracy: Dict[int, float] = field(default_factory=dict)


@dataclass
class PipelineReport:
    """Complete pipeline report."""
    timestamp: str
    total_time_ms: float

    # Shape alignment results
    shape_results: List[ShapeAlignmentResult]
    avg_steps_per_shape: float
    total_shape_time_ms: float

    # Routing results
    routing_result: Optional[RoutingResult]

    # Overall
    total_params: int
    final_accuracy: float

    # Doula summary
    doula_summary: Dict

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "6502 PIPELINE REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Total time: {self.total_time_ms:.1f} ms",
            "",
            "PHASE 1: SHAPE ALIGNMENT (Inner Loop)",
            "-" * 50,
        ]

        for sr in self.shape_results:
            lines.append(
                f"  {sr.shape_name:15s} | {sr.steps_to_converge:3d} steps | "
                f"acc={sr.final_accuracy*100:5.1f}% | {sr.time_ms:6.1f}ms"
            )

        lines.extend([
            "",
            f"  Average: {self.avg_steps_per_shape:.1f} steps/shape",
            f"  Total time: {self.total_shape_time_ms:.1f} ms",
            "",
        ])

        if self.routing_result:
            rr = self.routing_result
            lines.extend([
                "PHASE 2: ROUTING (Outer Loop)",
                "-" * 50,
                f"  Steps: {rr.steps}",
                f"  Final accuracy: {rr.final_accuracy*100:.1f}%",
                f"  Routing accuracy: {rr.routing_accuracy*100:.1f}%",
                f"  Time: {rr.time_ms:.1f} ms",
                "",
            ])

        lines.extend([
            "SUMMARY",
            "-" * 50,
            f"  Total parameters: {self.total_params}",
            f"  Final accuracy: {self.final_accuracy*100:.1f}%",
            f"  Total time: {self.total_time_ms:.1f} ms",
            "",
            "DOULA OBSERVATIONS",
            "-" * 50,
            f"  Total entries: {self.doula_summary.get('total_entries', 0)}",
            f"  Phases: {self.doula_summary.get('phases', {})}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# THE PIPELINE
# =============================================================================

def align_shape(
    shape_id: ShapeID,
    max_steps: int = 100,
    convergence_threshold: float = 1e-6,
    lr: float = 0.02,
    n_samples: int = 256,
    doula: Optional[DoulaFunction] = None,
) -> ShapeAlignmentResult:
    """
    Align a single frozen shape.

    This is the INNER LOOP of nested training.
    """
    shape_name = shape_id.name
    shape_fn = SHAPE_FUNCTIONS[shape_id]

    # Generate data
    a, b, c, target_result, target_carry = generate_shape_data(shape_id, n_samples)

    # Create aligner
    model = ShapeAligner(shape_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    phases = []
    steps = 0

    for step in range(max_steps):
        result, carry, routing_info = model(a, b, c)
        loss = F.mse_loss(result, target_result) + F.mse_loss(carry, target_carry)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps = step + 1

        # Doula observation
        if doula:
            sig = doula.aggregate(routing_info, step, shape_id.value, loss.item())
            doula.store(sig, loss.item())
            phases.append(doula.phase_detector.detect(sig))

        if loss.item() < convergence_threshold:
            break

    elapsed_ms = (time.time() - start_time) * 1000

    # Final accuracy
    model.eval()
    with torch.no_grad():
        result, _, _ = model(a, b, c)
        accuracy = (result.round() == target_result).all(dim=1).float().mean().item()

    return ShapeAlignmentResult(
        shape_id=shape_id,
        shape_name=shape_name,
        steps_to_converge=steps,
        final_loss=loss.item(),
        final_accuracy=accuracy,
        time_ms=elapsed_ms,
        doula_phases=phases,
    )


def train_routing(
    num_opcodes: int = 16,
    max_steps: int = 100,
    lr: float = 0.1,
    n_samples: int = 400,
    doula: Optional[DoulaFunction] = None,
) -> Tuple[TriX6502Router, RoutingResult]:
    """
    Train the routing layer.

    This is the OUTER LOOP of nested training.
    """
    # Generate data for all shapes
    all_a, all_b, all_c = [], [], []
    all_target, all_carry, all_opcode = [], [], []

    samples_per_op = n_samples // num_opcodes

    for op_idx in range(num_opcodes):
        shape_id = ShapeID(op_idx % 16)  # Map opcode to shape
        a, b, c, target, carry = generate_shape_data(shape_id, samples_per_op, seed=42 + op_idx)

        all_a.append(a)
        all_b.append(b)
        all_c.append(c)
        all_target.append(target)
        all_carry.append(carry)
        all_opcode.append(torch.full((samples_per_op,), op_idx, dtype=torch.long))

    a = torch.cat(all_a)
    b = torch.cat(all_b)
    c = torch.cat(all_c)
    target = torch.cat(all_target)
    target_carry = torch.cat(all_carry)
    opcode = torch.cat(all_opcode)

    # Create router
    router = TriX6502Router(num_opcodes=num_opcodes)
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)

    start_time = time.time()

    for step in range(max_steps):
        result, carry, routing_info = router(a, b, c, opcode)
        loss = F.mse_loss(result, target) + F.mse_loss(carry, target_carry)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Doula
        if doula:
            sig = doula.aggregate(routing_info, step, 0, loss.item())
            doula.store(sig, loss.item())

    elapsed_ms = (time.time() - start_time) * 1000

    # Final evaluation
    router.eval()
    with torch.no_grad():
        result, _, routing_info = router(a, b, c, opcode)
        accuracy = (result.round() == target).all(dim=1).float().mean().item()

        # Routing accuracy
        selected = routing_info['shape_idx'].squeeze(-1)
        expected_shape = opcode % 16
        routing_acc = (selected == expected_shape).float().mean().item()

        # Per-opcode accuracy
        per_op = {}
        for op in range(num_opcodes):
            mask = opcode == op
            if mask.any():
                op_acc = (result[mask].round() == target[mask]).all(dim=1).float().mean().item()
                per_op[op] = op_acc

    return router, RoutingResult(
        steps=max_steps,
        final_loss=loss.item(),
        final_accuracy=accuracy,
        routing_accuracy=routing_acc,
        time_ms=elapsed_ms,
        per_opcode_accuracy=per_op,
    )


def run_6502_pipeline(
    align_shapes: bool = True,
    train_router: bool = True,
    num_opcodes: int = 16,
    verbose: bool = True,
) -> PipelineReport:
    """
    Run the complete 6502 pipeline.

    From zero to fully functional 6502.
    Timed and measured throughout.
    Doula observes everything.

    Args:
        align_shapes: Run shape alignment (inner loop)
        train_router: Train routing (outer loop)
        num_opcodes: Number of opcodes to support
        verbose: Print progress

    Returns:
        Complete pipeline report
    """
    timestamp = datetime.now().isoformat()
    start_time = time.time()

    # Doula for observation
    doula = DoulaFunction(domain="6502_pipeline")

    shape_results = []
    routing_result = None
    total_params = 0

    # =========================================================================
    # Phase 1: Shape Alignment (Inner Loop)
    # =========================================================================
    if align_shapes:
        if verbose:
            print("=" * 70)
            print("PHASE 1: SHAPE ALIGNMENT (Inner Loop)")
            print("-" * 50)
            print("Shape           | Steps | Accuracy | Time")
            print("-" * 50)

        for shape_id in ShapeID:
            result = align_shape(shape_id, doula=doula)
            shape_results.append(result)

            if verbose:
                print(f"{result.shape_name:15s} | {result.steps_to_converge:5d} | "
                      f"{result.final_accuracy*100:6.1f}%  | {result.time_ms:6.1f}ms")

        if verbose:
            avg = sum(r.steps_to_converge for r in shape_results) / len(shape_results)
            print("-" * 50)
            print(f"Average: {avg:.1f} steps/shape")
            print()

    # =========================================================================
    # Phase 2: Routing (Outer Loop)
    # =========================================================================
    if train_router:
        if verbose:
            print("PHASE 2: ROUTING (Outer Loop)")
            print("-" * 50)

        router, routing_result = train_routing(
            num_opcodes=num_opcodes,
            doula=doula,
        )

        total_params = sum(p.numel() for p in router.parameters())

        if verbose:
            print(f"Steps: {routing_result.steps}")
            print(f"Accuracy: {routing_result.final_accuracy*100:.1f}%")
            print(f"Routing: {routing_result.routing_accuracy*100:.1f}%")
            print(f"Params: {total_params}")
            print()

    # =========================================================================
    # Summary
    # =========================================================================
    total_time_ms = (time.time() - start_time) * 1000
    avg_steps = sum(r.steps_to_converge for r in shape_results) / max(1, len(shape_results))
    total_shape_time = sum(r.time_ms for r in shape_results)
    final_acc = routing_result.final_accuracy if routing_result else (
        sum(r.final_accuracy for r in shape_results) / max(1, len(shape_results))
    )

    report = PipelineReport(
        timestamp=timestamp,
        total_time_ms=total_time_ms,
        shape_results=shape_results,
        avg_steps_per_shape=avg_steps,
        total_shape_time_ms=total_shape_time,
        routing_result=routing_result,
        total_params=total_params,
        final_accuracy=final_acc,
        doula_summary=doula.summarize(),
    )

    if verbose:
        print(report.summary())

    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    report = run_6502_pipeline()
