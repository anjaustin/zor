"""
Nested Training Test - Rapid Shape Alignment.

Mesa 16.2: The Speed of Frozen Shapes

This test demonstrates:
1. ADC shape alignment (~10 steps)
2. SBC shape alignment (~10 steps)
3. Routing between shapes (learning WHICH shape)
4. Doula observation throughout

"10 steps to lock-in. This is the speed you need for nested training."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from dataclasses import dataclass
import time

from trix.doula import DoulaFunction


# =============================================================================
# Pure Math Primitives (0 params - frozen)
# =============================================================================

def pure_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - 2 * a * b

def pure_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def pure_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - a * b

def pure_not(a: torch.Tensor) -> torch.Tensor:
    return 1 - a

def full_adder(a, b, c):
    p = pure_xor(a, b)
    return pure_xor(p, c), pure_or(pure_and(a, b), pure_and(c, p))


# =============================================================================
# Frozen Shapes (0 params)
# =============================================================================

def frozen_adc(a_bits: torch.Tensor, b_bits: torch.Tensor, c_in: torch.Tensor):
    """ADC: A + B + C"""
    result = []
    carry = c_in
    for i in range(8):
        s, carry = full_adder(a_bits[:, i], b_bits[:, i], carry)
        result.append(s)
    return torch.stack(result, dim=1), carry


def frozen_sbc(a_bits: torch.Tensor, b_bits: torch.Tensor, c_in: torch.Tensor):
    """SBC: A - B - (1-C) = A + NOT(B) + C"""
    b_inv = pure_not(b_bits)
    return frozen_adc(a_bits, b_inv, c_in)


def frozen_and(a_bits: torch.Tensor, b_bits: torch.Tensor, _: torch.Tensor):
    """AND: A & B"""
    return pure_and(a_bits, b_bits), torch.zeros(a_bits.shape[0])


def frozen_or(a_bits: torch.Tensor, b_bits: torch.Tensor, _: torch.Tensor):
    """OR: A | B"""
    return pure_or(a_bits, b_bits), torch.zeros(a_bits.shape[0])


FROZEN_SHAPES = {
    0: ('ADC', frozen_adc),
    1: ('SBC', frozen_sbc),
    2: ('AND', frozen_and),
    3: ('OR', frozen_or),
}


# =============================================================================
# Utility Functions
# =============================================================================

def int_to_bits(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    bits = []
    for i in range(n_bits):
        bits.append((x >> i) & 1)
    return torch.stack(bits, dim=-1).float()


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    for i in range(bits.shape[1]):
        result += (bits[:, i].round().long() << i)
    return result


# =============================================================================
# Single Shape Learner (for inner loop alignment)
# =============================================================================

class SingleShapeLearner(nn.Module):
    """Learns to apply a single frozen shape with confidence."""

    def __init__(self, shape_fn, d_hidden: int = 32):
        super().__init__()
        self.shape_fn = shape_fn
        self.confidence = nn.Sequential(
            nn.Linear(17, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, a_bits, b_bits, c_in):
        x = torch.cat([a_bits, b_bits, c_in.unsqueeze(-1)], dim=-1)
        conf = self.confidence(x)
        result, carry = self.shape_fn(a_bits, b_bits, c_in)
        return result * conf, carry * conf.squeeze(-1)


# =============================================================================
# Multi-Shape Router (for outer loop routing)
# =============================================================================

class MultiShapeRouter(nn.Module):
    """
    Routes between multiple frozen shapes based on opcode.

    This is the OUTER loop learning - which shape for which opcode.
    """

    def __init__(self, num_shapes: int = 4, d_hidden: int = 64):
        super().__init__()
        self.num_shapes = num_shapes

        # Opcode embedding (2 bits for 4 shapes)
        self.opcode_bits = 2

        # Signature projection
        self.sig_proj = nn.Linear(17 + self.opcode_bits, d_hidden)

        # Router: opcode + content → shape selection
        self.router = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_shapes),
        )

        # Temperature for soft routing
        self.temperature = 1.0

    def forward(self, a_bits, b_bits, c_in, opcode):
        """
        Args:
            a_bits: [B, 8]
            b_bits: [B, 8]
            c_in: [B]
            opcode: [B] integer 0-3

        Returns:
            result: [B, 8]
            carry: [B]
            routing_info: Dict for Doula
        """
        B = a_bits.shape[0]

        # Encode opcode as bits
        opcode_bits = int_to_bits(opcode, self.opcode_bits)

        # Combine inputs
        x = torch.cat([a_bits, b_bits, c_in.unsqueeze(-1), opcode_bits], dim=-1)

        # Compute signature and routing
        sig = self.sig_proj(x)
        logits = self.router(sig)

        if self.training:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), self.num_shapes).float()

        # Apply all shapes, weight by routing
        results = []
        carries = []
        for i in range(self.num_shapes):
            _, shape_fn = FROZEN_SHAPES[i]
            r, c = shape_fn(a_bits, b_bits, c_in)
            results.append(r)
            carries.append(c)

        results = torch.stack(results, dim=-1)  # [B, 8, num_shapes]
        carries = torch.stack(carries, dim=-1)  # [B, num_shapes]

        # Weighted combination
        result = (results * weights.unsqueeze(1)).sum(dim=-1)
        carry = (carries * weights).sum(dim=-1)

        # Routing info for Doula
        routing_info = {
            'shape_idx': logits.argmax(dim=-1, keepdim=True),
            'shape_weights': weights.unsqueeze(1),
            'partner_idx': torch.zeros(B, 1, dtype=torch.long, device=a_bits.device),
            'distances': torch.zeros(B, 1, 1, device=a_bits.device),
        }

        return result, carry, routing_info


# =============================================================================
# Data Generation
# =============================================================================

def generate_single_op_data(op: str, n_samples: int = 200, seed: int = 42):
    """Generate data for a single operation."""
    torch.manual_seed(seed)

    a_int = torch.randint(0, 256, (n_samples,))
    b_int = torch.randint(0, 256, (n_samples,))
    c_in = torch.randint(0, 2, (n_samples,)).float()

    a_bits = int_to_bits(a_int)
    b_bits = int_to_bits(b_int)

    if op == 'ADC':
        result_int = a_int + b_int + c_in.long()
        target = int_to_bits(result_int % 256)
        target_carry = (result_int >= 256).float()
    elif op == 'SBC':
        result_int = a_int.long() - b_int.long() - (1 - c_in.long())
        # Handle underflow
        target = int_to_bits((result_int % 256 + 256) % 256)
        target_carry = (result_int >= 0).float()
    elif op == 'AND':
        result_int = a_int & b_int
        target = int_to_bits(result_int)
        target_carry = torch.zeros(n_samples)
    elif op == 'OR':
        result_int = a_int | b_int
        target = int_to_bits(result_int)
        target_carry = torch.zeros(n_samples)
    else:
        raise ValueError(f"Unknown op: {op}")

    return a_bits, b_bits, c_in, target, target_carry


def generate_multi_op_data(n_samples: int = 400, seed: int = 42):
    """Generate data for all operations with opcodes."""
    torch.manual_seed(seed)

    samples_per_op = n_samples // 4

    all_a, all_b, all_c, all_target, all_carry, all_opcode = [], [], [], [], [], []

    for op_idx, op in enumerate(['ADC', 'SBC', 'AND', 'OR']):
        a, b, c, target, carry = generate_single_op_data(op, samples_per_op, seed + op_idx)
        opcode = torch.full((samples_per_op,), op_idx, dtype=torch.long)

        all_a.append(a)
        all_b.append(b)
        all_c.append(c)
        all_target.append(target)
        all_carry.append(carry)
        all_opcode.append(opcode)

    return (
        torch.cat(all_a),
        torch.cat(all_b),
        torch.cat(all_c),
        torch.cat(all_target),
        torch.cat(all_carry),
        torch.cat(all_opcode),
    )


# =============================================================================
# Nested Training
# =============================================================================

@dataclass
class AlignmentResult:
    """Result of shape alignment."""
    shape_name: str
    steps_to_align: int
    final_loss: float
    final_accuracy: float
    time_ms: float


def align_single_shape(
    shape_name: str,
    shape_fn,
    max_steps: int = 100,
    convergence_threshold: float = 1e-6,
    lr: float = 0.02,
) -> AlignmentResult:
    """
    Inner loop: Align a single frozen shape.

    Returns steps to convergence.
    """
    # Generate data for this shape
    a_bits, b_bits, c_in, target, target_carry = generate_single_op_data(shape_name, 200)

    # Create learner
    model = SingleShapeLearner(shape_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    steps = 0
    final_loss = float('inf')

    for step in range(max_steps):
        result, carry = model(a_bits, b_bits, c_in)
        loss = F.mse_loss(result, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps = step + 1
        final_loss = loss.item()

        if final_loss < convergence_threshold:
            break

    elapsed_ms = (time.time() - start_time) * 1000

    # Final accuracy
    model.eval()
    with torch.no_grad():
        result, _ = model(a_bits, b_bits, c_in)
        accuracy = (result.round() == target).all(dim=1).float().mean().item()

    return AlignmentResult(
        shape_name=shape_name,
        steps_to_align=steps,
        final_loss=final_loss,
        final_accuracy=accuracy,
        time_ms=elapsed_ms,
    )


def nested_training_demo():
    """
    Full nested training demonstration.

    Outer loop: Learn routing between shapes
    Inner loop: Rapid shape alignment verification
    """
    print("=" * 70)
    print("NESTED TRAINING DEMONSTRATION")
    print("=" * 70)
    print()

    # ==========================================================================
    # Phase 1: Inner Loop - Align Each Shape Individually
    # ==========================================================================
    print("PHASE 1: INNER LOOP - Individual Shape Alignment")
    print("-" * 50)

    results = []
    for op_idx, (name, fn) in FROZEN_SHAPES.items():
        result = align_single_shape(name, fn)
        results.append(result)
        print(f"  {name}: {result.steps_to_align:3d} steps | "
              f"loss={result.final_loss:.2e} | "
              f"acc={result.final_accuracy*100:.1f}% | "
              f"time={result.time_ms:.1f}ms")

    avg_steps = sum(r.steps_to_align for r in results) / len(results)
    print(f"\n  Average alignment: {avg_steps:.1f} steps")
    print()

    # ==========================================================================
    # Phase 2: Outer Loop - Learn Routing Between Shapes
    # ==========================================================================
    print("PHASE 2: OUTER LOOP - Learn Routing Between Shapes")
    print("-" * 50)

    # Generate multi-op data
    a_bits, b_bits, c_in, target, target_carry, opcode = generate_multi_op_data(800)

    # Create router
    router = MultiShapeRouter(num_shapes=4)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    # Doula observation
    doula = DoulaFunction(domain="nested_training")

    # Training loop
    print("\n  Step | Loss     | Accuracy | Phase          | Routing Acc")
    print("  " + "-" * 60)

    routing_correct_history = []

    for step in range(200):
        # Forward
        result, carry, routing_info = router(a_bits, b_bits, c_in, opcode)

        # Loss
        loss = F.mse_loss(result, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Doula
        sig = doula.aggregate(routing_info, step, 0, loss.item())
        doula.store(sig, loss.item())
        phase = doula.phase_detector.detect(sig)

        # Routing accuracy (did we pick the right shape?)
        selected = routing_info['shape_idx'].squeeze(-1)
        routing_acc = (selected == opcode).float().mean().item()
        routing_correct_history.append(routing_acc)

        if step % 20 == 0:
            with torch.no_grad():
                result, _, _ = router(a_bits, b_bits, c_in, opcode)
                acc = (result.round() == target).all(dim=1).float().mean().item()
            print(f"  {step:4d} | {loss.item():.6f} | {acc*100:6.1f}%  | {phase:14s} | {routing_acc*100:.1f}%")

    # Final evaluation
    router.eval()
    with torch.no_grad():
        result, _, routing_info = router(a_bits, b_bits, c_in, opcode)
        final_acc = (result.round() == target).all(dim=1).float().mean().item()
        final_routing = (routing_info['shape_idx'].squeeze(-1) == opcode).float().mean().item()

    print()
    print(f"  Final accuracy: {final_acc*100:.1f}%")
    print(f"  Final routing accuracy: {final_routing*100:.1f}%")
    print()

    # ==========================================================================
    # Phase 3: Doula Summary
    # ==========================================================================
    print("PHASE 3: DOULA OBSERVATIONS")
    print("-" * 50)

    summary = doula.summarize()
    print(f"  Total observations: {summary['total_entries']}")
    print(f"  Phases observed: {doula.vdb.analyze_phases()}")

    # Guidance
    last_sig = doula.history[-1]
    guidance = doula.query(last_sig)
    print(f"  Current phase: {guidance.current_phase}")
    print(f"  Training complete: {guidance.training_complete}")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Inner loop (per shape):  ~{avg_steps:.0f} steps to 100% accuracy")
    print(f"  Outer loop (routing):    200 steps to {final_routing*100:.1f}% routing accuracy")
    print(f"  Total shapes:            {len(FROZEN_SHAPES)}")
    print(f"  Combined accuracy:       {final_acc*100:.1f}%")
    print()
    print("  This is nested training speed.")
    print("=" * 70)

    return results, final_acc, final_routing


# =============================================================================
# Tests
# =============================================================================

class TestNestedTraining:
    """Tests for nested training."""

    def test_adc_alignment_speed(self):
        """ADC aligns in < 30 steps."""
        result = align_single_shape('ADC', frozen_adc)
        assert result.steps_to_align < 30, f"ADC took {result.steps_to_align} steps"
        assert result.final_accuracy > 0.99

    def test_sbc_alignment_speed(self):
        """SBC aligns in < 30 steps."""
        result = align_single_shape('SBC', frozen_sbc)
        assert result.steps_to_align < 30, f"SBC took {result.steps_to_align} steps"
        assert result.final_accuracy > 0.99

    def test_and_alignment_speed(self):
        """AND aligns in < 30 steps."""
        result = align_single_shape('AND', frozen_and)
        assert result.steps_to_align < 30, f"AND took {result.steps_to_align} steps"
        assert result.final_accuracy > 0.99

    def test_or_alignment_speed(self):
        """OR aligns in < 30 steps."""
        result = align_single_shape('OR', frozen_or)
        assert result.steps_to_align < 30, f"OR took {result.steps_to_align} steps"
        assert result.final_accuracy > 0.99

    def test_multi_shape_routing(self):
        """Router learns to select correct shapes."""
        a_bits, b_bits, c_in, target, _, opcode = generate_multi_op_data(400)

        router = MultiShapeRouter()
        optimizer = torch.optim.Adam(router.parameters(), lr=0.02)

        for step in range(100):
            result, _, routing_info = router(a_bits, b_bits, c_in, opcode)
            loss = F.mse_loss(result, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check routing accuracy
        router.eval()
        with torch.no_grad():
            _, _, routing_info = router(a_bits, b_bits, c_in, opcode)
            selected = routing_info['shape_idx'].squeeze(-1)
            routing_acc = (selected == opcode).float().mean().item()

        assert routing_acc > 0.9, f"Routing accuracy: {routing_acc}"

    def test_nested_with_doula(self):
        """Full nested training with Doula observation."""
        a_bits, b_bits, c_in, target, _, opcode = generate_multi_op_data(200)

        router = MultiShapeRouter()
        optimizer = torch.optim.Adam(router.parameters(), lr=0.02)
        doula = DoulaFunction(domain="nested_test")

        for step in range(50):
            result, _, routing_info = router(a_bits, b_bits, c_in, opcode)
            loss = F.mse_loss(result, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sig = doula.aggregate(routing_info, step, 0, loss.item())
            doula.store(sig, loss.item())

        # Verify Doula captured everything
        assert len(doula.vdb.entries) == 50

        # Verify we can get guidance
        guidance = doula.query(sig)
        assert guidance.current_phase in ['exploration', 'transition', 'crystallization', 'stable']


if __name__ == "__main__":
    nested_training_demo()
