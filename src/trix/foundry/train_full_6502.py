"""
Full 6502 Training Pipeline - From Ground Truth to Complete CPU.

Mesa 16.5: Training the Sacred Foundry

Uses:
- Ground truth from flynnconceivable
- Frozen shapes from pipeline_6502
- Doula observation
- All operations: ADC, SBC, AND, ORA, EOR, ASL, LSR, ROL, ROR,
                  INC, DEC, TAX, TXA, LDA, etc.

"We have way more than we need."
"""

import sys
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from trix.foundry.pipeline_6502 import (
    PureMath, FrozenShapes, SHAPE_FUNCTIONS,
    int_to_bits, bits_to_int, ShapeID,
)
from trix.doula import DoulaFunction

# Import ground truth
from training.data import (
    adc_truth, sbc_truth,
    and_truth, ora_truth, eor_truth, bit_truth,
    asl_truth, lsr_truth, rol_truth, ror_truth,
    inc_truth, dec_truth,
    transfer_truth,
    generate_alu_data, generate_shift_data,
    generate_logic_data, generate_incdec_data,
    generate_transfer_data, generate_load_data,
    generate_branch_data, generate_flags_data,
)


# =============================================================================
# OPERATION DEFINITIONS
# =============================================================================

@dataclass
class OpDef:
    """Operation definition mapping to frozen shape."""
    name: str
    shape_id: ShapeID
    needs_b: bool = True       # Needs second operand
    needs_carry: bool = False  # Uses carry input
    updates_carry: bool = False  # Updates carry output
    category: str = "misc"


# All 6502 ALU operations mapped to frozen shapes
ALL_OPERATIONS = [
    # Arithmetic
    OpDef("ADC", ShapeID.RIPPLE_ADD, needs_b=True, needs_carry=True, updates_carry=True, category="alu"),
    OpDef("SBC", ShapeID.RIPPLE_SUB, needs_b=True, needs_carry=True, updates_carry=True, category="alu"),

    # Logic
    OpDef("AND", ShapeID.PARALLEL_AND, needs_b=True, category="logic"),
    OpDef("ORA", ShapeID.PARALLEL_OR, needs_b=True, category="logic"),
    OpDef("EOR", ShapeID.PARALLEL_XOR, needs_b=True, category="logic"),
    OpDef("BIT", ShapeID.BIT_TEST, needs_b=True, category="logic"),

    # Shifts (without carry)
    OpDef("ASL", ShapeID.SHIFT_LEFT, needs_b=False, updates_carry=True, category="shift"),
    OpDef("LSR", ShapeID.SHIFT_RIGHT, needs_b=False, updates_carry=True, category="shift"),

    # Rotates (with carry)
    OpDef("ROL", ShapeID.ROTATE_LEFT, needs_b=False, needs_carry=True, updates_carry=True, category="rotate"),
    OpDef("ROR", ShapeID.ROTATE_RIGHT, needs_b=False, needs_carry=True, updates_carry=True, category="rotate"),

    # Inc/Dec
    OpDef("INC", ShapeID.INCREMENT, needs_b=False, category="incdec"),
    OpDef("DEC", ShapeID.DECREMENT, needs_b=False, category="incdec"),
    OpDef("INX", ShapeID.INCREMENT, needs_b=False, category="incdec"),
    OpDef("DEX", ShapeID.DECREMENT, needs_b=False, category="incdec"),
    OpDef("INY", ShapeID.INCREMENT, needs_b=False, category="incdec"),
    OpDef("DEY", ShapeID.DECREMENT, needs_b=False, category="incdec"),

    # Transfers
    OpDef("TAX", ShapeID.TRANSFER, needs_b=False, category="transfer"),
    OpDef("TXA", ShapeID.TRANSFER, needs_b=False, category="transfer"),
    OpDef("TAY", ShapeID.TRANSFER, needs_b=False, category="transfer"),
    OpDef("TYA", ShapeID.TRANSFER, needs_b=False, category="transfer"),
    OpDef("TSX", ShapeID.TRANSFER, needs_b=False, category="transfer"),
    OpDef("TXS", ShapeID.TRANSFER, needs_b=False, category="transfer"),

    # Loads
    OpDef("LDA", ShapeID.LOAD, needs_b=False, category="load"),
    OpDef("LDX", ShapeID.LOAD, needs_b=False, category="load"),
    OpDef("LDY", ShapeID.LOAD, needs_b=False, category="load"),

    # Stores
    OpDef("STA", ShapeID.STORE, needs_b=False, category="store"),
    OpDef("STX", ShapeID.STORE, needs_b=False, category="store"),
    OpDef("STY", ShapeID.STORE, needs_b=False, category="store"),

    # NOP
    OpDef("NOP", ShapeID.IDENTITY, needs_b=False, category="nop"),
]

OP_TO_IDX = {op.name: i for i, op in enumerate(ALL_OPERATIONS)}
NUM_OPS = len(ALL_OPERATIONS)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_full_dataset(n_alu_samples: int = 50000) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Generate complete training dataset for all operations.

    Returns:
        opcode, a_bits, b_bits, c_in, target_result, target_carry
    """
    opcodes = []
    a_vals = []
    b_vals = []
    c_vals = []
    results = []
    carries = []

    print("Generating full 6502 dataset...")

    # ALU (ADC)
    for d in generate_alu_data(n_alu_samples):
        opcodes.append(OP_TO_IDX["ADC"])
        a_vals.append(d['a'])
        b_vals.append(d['operand'])
        c_vals.append(d['c_in'])
        results.append(d['result'])
        carries.append(d['c'])

    # Logic (AND, ORA, EOR)
    for d in generate_logic_data(n_alu_samples):
        op_name = d['op']
        if op_name in OP_TO_IDX:
            opcodes.append(OP_TO_IDX[op_name])
            a_vals.append(d['a'])
            b_vals.append(d['operand'])
            c_vals.append(0)
            results.append(d['result'])
            carries.append(0)

    # Shifts (ASL, LSR)
    for d in generate_shift_data():
        op_name = d['op']
        if op_name in OP_TO_IDX:
            c_in = d.get('c_in', 0)
            opcodes.append(OP_TO_IDX[op_name])
            a_vals.append(d['val'])
            b_vals.append(0)
            c_vals.append(c_in)
            results.append(d['result'])
            carries.append(d['c'])

    # Inc/Dec
    for d in generate_incdec_data():
        op_name = d['op']
        if op_name in OP_TO_IDX:
            opcodes.append(OP_TO_IDX[op_name])
            a_vals.append(d['val'])
            b_vals.append(0)
            c_vals.append(0)
            results.append(d['result'])
            carries.append(0)

    # Transfers
    for d in generate_transfer_data():
        op_name = d.get('op', 'TAX')
        if op_name in OP_TO_IDX:
            opcodes.append(OP_TO_IDX[op_name])
            a_vals.append(d['src'])
            b_vals.append(0)
            c_vals.append(0)
            results.append(d['dst'])
            carries.append(0)

    # Loads
    for d in generate_load_data():
        for op_name in ['LDA', 'LDX', 'LDY']:
            opcodes.append(OP_TO_IDX[op_name])
            a_vals.append(d['val'])
            b_vals.append(0)
            c_vals.append(0)
            results.append(d['val'])
            carries.append(0)

    print(f"  Total samples: {len(opcodes)}")

    # Convert to tensors
    opcode_t = torch.tensor(opcodes, dtype=torch.long)
    a_bits = int_to_bits(torch.tensor(a_vals))
    b_bits = int_to_bits(torch.tensor(b_vals))
    c_in = torch.tensor(c_vals, dtype=torch.float32)
    target_result = int_to_bits(torch.tensor(results))
    target_carry = torch.tensor(carries, dtype=torch.float32)

    return opcode_t, a_bits, b_bits, c_in, target_result, target_carry


# =============================================================================
# FROZEN 6502 ROUTER
# =============================================================================

class Frozen6502Router(nn.Module):
    """
    Routes opcodes to frozen shapes.

    The shapes are frozen (0 params).
    Only the routing is learned.
    """

    def __init__(self, num_ops: int = NUM_OPS, use_prior: bool = True):
        super().__init__()
        self.num_ops = num_ops
        self.num_shapes = 16

        # Routing logits: opcode -> shape
        self.routing = nn.Parameter(torch.zeros(num_ops, 16))
        self.temperature = 0.5

        # Initialize from known mappings
        self._init_from_spec(use_prior=use_prior)

    def _init_from_spec(self, use_prior: bool = True):
        """Initialize routing from operation definitions."""
        if use_prior:
            with torch.no_grad():
                for i, op in enumerate(ALL_OPERATIONS):
                    self.routing[i, op.shape_id] = 5.0  # Strong prior
        # else: random init (zeros)

    def forward(
        self,
        opcode: Tensor,
        a: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict]:
        B = a.shape[0]

        # Get routing weights
        logits = self.routing[opcode]

        if self.training:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), 16).float()

        # Execute all shapes
        results = []
        carries = []

        for shape_id in ShapeID:
            fn = SHAPE_FUNCTIONS[shape_id]
            r, car = fn(a, b, c)
            results.append(r)
            carries.append(car)

        results = torch.stack(results, dim=-1)  # [B, 8, 16]
        carries = torch.stack(carries, dim=-1)  # [B, 16]

        # Weighted blend
        result = (results * weights.unsqueeze(1)).sum(dim=-1)
        carry = (carries * weights).sum(dim=-1)

        info = {
            'shape_idx': logits.argmax(dim=-1, keepdim=True),
            'shape_weights': weights.unsqueeze(1),
            'partner_idx': torch.zeros(B, 1, dtype=torch.long, device=a.device),
            'distances': torch.zeros(B, 1, 1, device=a.device),
        }

        return result, carry, info

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingResult:
    """Training result."""
    steps: int
    final_loss: float
    final_accuracy: float
    routing_accuracy: float
    time_ms: float
    num_params: int
    per_category_accuracy: Dict[str, float]


def train_full_6502(
    max_steps: int = 500,
    lr: float = 0.05,
    batch_size: int = 2048,
    verbose: bool = True,
    use_prior: bool = True,
) -> Tuple[Frozen6502Router, TrainingResult]:
    """
    Train the full 6502 on all operations.
    """
    # Generate data
    opcode, a, b, c, target, target_carry = generate_full_dataset()
    n_samples = len(opcode)

    # Create model
    model = Frozen6502Router(use_prior=use_prior)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Doula
    doula = DoulaFunction(domain="6502_full")

    start_time = time.time()

    if verbose:
        print(f"\nTraining Full 6502 ({model.num_params} params)")
        print(f"  Samples: {n_samples}")
        print(f"  Operations: {NUM_OPS}")
        print("=" * 60)

    best_acc = 0.0

    for step in range(max_steps):
        # Mini-batch
        idx = torch.randperm(n_samples)[:batch_size]

        result, carry, info = model(opcode[idx], a[idx], b[idx], c[idx])

        # Loss
        loss = F.mse_loss(result, target[idx])
        if target_carry[idx].abs().sum() > 0:
            loss = loss + 0.1 * F.mse_loss(carry, target_carry[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        if step % 50 == 0 or step == max_steps - 1:
            model.eval()
            with torch.no_grad():
                result_full, _, info = model(opcode, a, b, c)
                acc = (result_full.round() == target).all(dim=1).float().mean().item()

                # Routing accuracy
                selected = info['shape_idx'].squeeze(-1)
                expected = torch.tensor([ALL_OPERATIONS[op].shape_id for op in opcode])
                routing_acc = (selected == expected).float().mean().item()
            model.train()

            if verbose and step % 100 == 0:
                print(f"  Step {step:4d} | Loss: {loss.item():.6f} | Acc: {acc*100:.2f}% | Routing: {routing_acc*100:.1f}%")

            if acc > best_acc:
                best_acc = acc

            # Doula
            sig = doula.aggregate(info, step, 0, loss.item())
            doula.store(sig, loss.item(), acc)

            # Early stop
            if acc >= 0.9999:
                if verbose:
                    print(f"  Converged at step {step}!")
                break

    elapsed_ms = (time.time() - start_time) * 1000

    # Final evaluation
    model.eval()
    with torch.no_grad():
        result_full, carry_full, info = model(opcode, a, b, c)
        final_acc = (result_full.round() == target).all(dim=1).float().mean().item()

        selected = info['shape_idx'].squeeze(-1)
        expected = torch.tensor([ALL_OPERATIONS[op].shape_id for op in opcode])
        routing_acc = (selected == expected).float().mean().item()

        # Per-category accuracy
        per_cat = {}
        for cat in set(op.category for op in ALL_OPERATIONS):
            cat_ops = [i for i, op in enumerate(ALL_OPERATIONS) if op.category == cat]
            mask = torch.isin(opcode, torch.tensor(cat_ops))
            if mask.any():
                cat_acc = (result_full[mask].round() == target[mask]).all(dim=1).float().mean().item()
                per_cat[cat] = cat_acc

    if verbose:
        print("=" * 60)
        print(f"FINAL RESULTS")
        print(f"  Accuracy: {final_acc*100:.2f}%")
        print(f"  Routing: {routing_acc*100:.1f}%")
        print(f"  Parameters: {model.num_params}")
        print(f"  Time: {elapsed_ms:.1f} ms")
        print(f"\nPer-category accuracy:")
        for cat, acc in sorted(per_cat.items()):
            print(f"    {cat:12s}: {acc*100:.1f}%")

    return model, TrainingResult(
        steps=step + 1,
        final_loss=loss.item(),
        final_accuracy=final_acc,
        routing_accuracy=routing_acc,
        time_ms=elapsed_ms,
        num_params=model.num_params,
        per_category_accuracy=per_cat,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FULL MOS 6502 TRAINING - FROZEN SHAPES")
    print("Mesa 16.5: The Sacred Foundry")
    print("=" * 70)

    # Experiment 1: With prior (should be instant)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: WITH PRIOR INITIALIZATION")
    print("=" * 70)
    model1, result1 = train_full_6502(max_steps=100, verbose=True, use_prior=True)

    # Experiment 2: From scratch (watch the emergence)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: FROM SCRATCH (No Prior)")
    print("=" * 70)
    model2, result2 = train_full_6502(max_steps=500, lr=0.1, verbose=True, use_prior=False)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  {'':20s} {'With Prior':>15s} {'From Scratch':>15s}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':20s} {result1.final_accuracy*100:14.2f}% {result2.final_accuracy*100:14.2f}%")
    print(f"  {'Steps':20s} {result1.steps:15d} {result2.steps:15d}")
    print(f"  {'Time (ms)':20s} {result1.time_ms:15.1f} {result2.time_ms:15.1f}")
    print(f"  {'Routing accuracy':20s} {result1.routing_accuracy*100:14.1f}% {result2.routing_accuracy*100:14.1f}%")
    print()
    print(f"  Total operations: {NUM_OPS}")
    print(f"  Frozen shapes: 16 (0 learnable params)")
    print(f"  Routing params: {result1.num_params}")
    print()
    print("\"Computation is topology. Learning is routing.\"")
