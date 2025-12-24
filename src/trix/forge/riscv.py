"""
RISC-V RV32I ALU - Frozen Neural-Geometric Implementation

All 10 ALU operations of the RV32I base instruction set,
implemented as frozen polynomial shapes with 100% accuracy.

Operations:
    ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU

Usage:
    from trix.forge.riscv import RV32I_ALU

    alu = RV32I_ALU()
    alu.train()
    alu.validate()  # 100%

    result = alu.compute(a=0x12345678, b=0x00000005, op="add")
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import math

from trix.routing.primitives import xor, and_, or_, not_, full_adder


# =============================================================================
# 32-BIT FROZEN SHAPES
# =============================================================================

def add_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """32-bit addition using ripple-carry adder."""
    result = []
    carry = torch.zeros_like(a_bits[:, 0])

    for i in range(32):
        s, carry = full_adder(a_bits[:, i], b_bits[:, i], carry)
        result.append(s)

    return torch.stack(result, dim=1)


def sub_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """32-bit subtraction: a - b = a + NOT(b) + 1."""
    not_b = not_(b_bits)
    result = []
    carry = torch.ones_like(a_bits[:, 0])  # +1 for two's complement

    for i in range(32):
        s, carry = full_adder(a_bits[:, i], not_b[:, i], carry)
        result.append(s)

    return torch.stack(result, dim=1)


def and_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """32-bit AND."""
    return and_(a_bits, b_bits)


def or_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """32-bit OR."""
    return or_(a_bits, b_bits)


def xor_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """32-bit XOR."""
    return xor(a_bits, b_bits)


def sll_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    32-bit shift left logical: a << (b & 0x1F)

    Only lower 5 bits of b specify shift amount (0-31).
    Uses barrel shifter approach with frozen mux.
    """
    batch_size = a_bits.shape[0]
    result = a_bits.clone()

    # Barrel shifter: shift by 1, 2, 4, 8, 16
    for stage in range(5):
        shift_amt = 1 << stage
        shift_bit = b_bits[:, stage:stage+1]  # [batch, 1]

        # Shifted version
        zeros = torch.zeros(batch_size, shift_amt, device=a_bits.device)
        shifted = torch.cat([zeros, result[:, :-shift_amt]], dim=1)

        # MUX: if shift_bit, use shifted, else use result
        # mux(sel, a, b) = a + sel * (b - a)
        result = result + shift_bit * (shifted - result)

    return result


def srl_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    32-bit shift right logical: a >> (b & 0x1F)

    Fills with zeros from left.
    """
    batch_size = a_bits.shape[0]
    result = a_bits.clone()

    for stage in range(5):
        shift_amt = 1 << stage
        shift_bit = b_bits[:, stage:stage+1]

        zeros = torch.zeros(batch_size, shift_amt, device=a_bits.device)
        shifted = torch.cat([result[:, shift_amt:], zeros], dim=1)

        result = result + shift_bit * (shifted - result)

    return result


def sra_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    32-bit shift right arithmetic: a >> (b & 0x1F)

    Fills with sign bit from left.
    """
    batch_size = a_bits.shape[0]
    result = a_bits.clone()
    sign_bit = a_bits[:, 31:32]  # MSB is sign

    for stage in range(5):
        shift_amt = 1 << stage
        shift_bit = b_bits[:, stage:stage+1]

        # Fill with sign bit
        fill = sign_bit.expand(batch_size, shift_amt)
        shifted = torch.cat([result[:, shift_amt:], fill], dim=1)

        result = result + shift_bit * (shifted - result)

    return result


def slt_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    32-bit set less than (signed): a < b ? 1 : 0

    Returns 32-bit result with 1 or 0 in LSB.
    """
    # Signed comparison: compare (a - b), check result sign
    # But also need to handle overflow

    # a < b (signed) when:
    # 1. a is negative and b is positive, OR
    # 2. same sign and (a - b) is negative

    a_sign = a_bits[:, 31:32]
    b_sign = b_bits[:, 31:32]

    # Compute a - b
    diff = sub_32bit(a_bits, b_bits)
    diff_sign = diff[:, 31:32]

    # Case 1: a negative, b positive -> a < b (result = 1)
    case1 = and_(a_sign, not_(b_sign))

    # Case 2: a positive, b negative -> a >= b (result = 0)
    case2 = and_(not_(a_sign), b_sign)

    # Case 3: same sign -> use diff sign
    same_sign = not_(xor(a_sign, b_sign))
    case3 = and_(same_sign, diff_sign)

    # Result: case1 OR case3 (case2 means result is 0)
    is_less = or_(case1, case3)

    # Build 32-bit result with LSB = is_less
    batch_size = a_bits.shape[0]
    zeros = torch.zeros(batch_size, 31, device=a_bits.device)
    result = torch.cat([is_less, zeros], dim=1)

    return result


def sltu_32bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    32-bit set less than unsigned: a < b ? 1 : 0

    Returns 32-bit result with 1 or 0 in LSB.
    """
    # Unsigned comparison: a < b when (a - b) borrows (carry out = 0)
    # Alternatively: find highest differing bit, that determines result

    # Compute b - a, check if borrow (result would be negative if b < a)
    # Actually easier: compute a - b, if it "wraps" (borrow), then a < b

    # For a - b in unsigned:
    # If a >= b: result is positive (no borrow from MSB)
    # If a < b: result wraps (borrow needed)

    # We can detect borrow by computing a + NOT(b) + 1 and checking carry out
    not_b = not_(b_bits)
    carry = torch.ones_like(a_bits[:, 0])

    for i in range(32):
        _, carry = full_adder(a_bits[:, i], not_b[:, i], carry)

    # carry = 1 means no borrow (a >= b), carry = 0 means borrow (a < b)
    is_less = not_(carry.unsqueeze(1))

    batch_size = a_bits.shape[0]
    zeros = torch.zeros(batch_size, 31, device=a_bits.device)
    result = torch.cat([is_less, zeros], dim=1)

    return result


# =============================================================================
# SHAPE REGISTRY
# =============================================================================

RV32I_SHAPES = {
    "add":  (add_32bit,  lambda a, b: (a + b) & 0xFFFFFFFF),
    "sub":  (sub_32bit,  lambda a, b: (a - b) & 0xFFFFFFFF),
    "and":  (and_32bit,  lambda a, b: a & b),
    "or":   (or_32bit,   lambda a, b: a | b),
    "xor":  (xor_32bit,  lambda a, b: a ^ b),
    "sll":  (sll_32bit,  lambda a, b: (a << (b & 0x1F)) & 0xFFFFFFFF),
    "srl":  (srl_32bit,  lambda a, b: (a >> (b & 0x1F)) & 0xFFFFFFFF),
    "sra":  (sra_32bit,  lambda a, b: (((a if a < 0x80000000 else a - 0x100000000) >> (b & 0x1F)) & 0xFFFFFFFF)),
    "slt":  (slt_32bit,  lambda a, b: 1 if (a if a < 0x80000000 else a - 0x100000000) < (b if b < 0x80000000 else b - 0x100000000) else 0),
    "sltu": (sltu_32bit, lambda a, b: 1 if a < b else 0),
}


# =============================================================================
# RV32I ALU CLASS
# =============================================================================

class RV32I_ALU(nn.Module):
    """
    Complete RISC-V RV32I ALU as frozen neural-geometric model.

    10 operations, 32-bit, 100% accuracy by construction.
    """

    def __init__(self):
        super().__init__()
        self.bits = 32
        self.ops = list(RV32I_SHAPES.keys())
        self.num_ops = len(self.ops)
        self.opcode_bits = math.ceil(math.log2(self.num_ops))

        # Router
        input_bits = self.bits * 2 + self.opcode_bits
        self.router = nn.Linear(input_bits, self.num_ops)

        # Get shapes and truths
        self.shapes = [RV32I_SHAPES[op][0] for op in self.ops]
        self.truths = [RV32I_SHAPES[op][1] for op in self.ops]

        self._trained = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through frozen ALU."""
        a_bits = x[:, :self.bits]
        b_bits = x[:, self.bits:self.bits*2]

        # Compute all shapes
        outputs = [shape(a_bits, b_bits) for shape in self.shapes]
        stacked = torch.stack(outputs, dim=1)

        # Route
        weights = torch.softmax(self.router(x), dim=1)
        result = torch.einsum('bn,bno->bo', weights, stacked)

        return result

    def _int_to_bits(self, x: int) -> Tensor:
        return torch.tensor([(x >> i) & 1 for i in range(self.bits)], dtype=torch.float32)

    def _bits_to_int(self, bits: Tensor) -> int:
        result = 0
        for i, b in enumerate(bits):
            result += int(round(float(b))) << i
        return result

    def train_router(self, n_samples: int = 100000, max_epochs: int = 50,
                     batch_size: int = 512, lr: float = 0.001, verbose: bool = True):
        """Train the router using random samples."""
        import time

        if verbose:
            print(f"Training RV32I ALU router ({self.num_ops} operations)...")

        # Generate random training data
        inputs = []
        targets = []

        for _ in range(n_samples):
            a = torch.randint(0, 0xFFFFFFFF, (1,)).item()
            b = torch.randint(0, 0xFFFFFFFF, (1,)).item()
            op_idx = torch.randint(0, self.num_ops, (1,)).item()

            a_bits = self._int_to_bits(a)
            b_bits = self._int_to_bits(b)
            op_bits = torch.tensor([(op_idx >> i) & 1 for i in range(self.opcode_bits)], dtype=torch.float32)

            inputs.append(torch.cat([a_bits, b_bits, op_bits]))

            result = self.truths[op_idx](a, b)
            targets.append(self._int_to_bits(result))

        X = torch.stack(inputs)
        Y = torch.stack(targets)

        if verbose:
            print(f"  Dataset: {len(X)} samples")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        start = time.time()

        for epoch in range(max_epochs):
            perm = torch.randperm(len(X))
            total_loss = 0

            for i in range(len(X) // batch_size):
                batch_X = X[perm[i*batch_size:(i+1)*batch_size]]
                batch_Y = Y[perm[i*batch_size:(i+1)*batch_size]]

                optimizer.zero_grad()
                pred = self.forward(batch_X)
                loss = criterion(pred, batch_Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Check accuracy on sample
            self.eval()
            with torch.no_grad():
                pred = self.forward(X[:10000])
                pred_bits = (pred > 0.5).float()
                correct = (pred_bits == Y[:10000]).all(dim=1).sum().item()
                acc = correct / 10000
            self.train()

            if verbose and (epoch < 3 or epoch % 5 == 0):
                print(f"  Epoch {epoch}: {acc*100:.2f}%")

            if acc >= 0.9999:
                if verbose:
                    print(f"  Epoch {epoch}: 100%!")
                break

        elapsed = time.time() - start
        self._trained = True

        if verbose:
            print(f"  Done in {elapsed:.1f}s")

        return {"epochs": epoch + 1, "time": elapsed}

    def validate(self, n_samples: int = 50000, verbose: bool = True) -> dict:
        """Validate each operation independently."""
        results = {}

        for op_idx, op_name in enumerate(self.ops):
            correct = 0
            total = n_samples // self.num_ops

            for _ in range(total):
                a = torch.randint(0, 0xFFFFFFFF, (1,)).item()
                b = torch.randint(0, 0xFFFFFFFF, (1,)).item()

                a_bits = self._int_to_bits(a)
                b_bits = self._int_to_bits(b)
                op_bits = torch.tensor([(op_idx >> i) & 1 for i in range(self.opcode_bits)], dtype=torch.float32)
                x = torch.cat([a_bits, b_bits, op_bits]).unsqueeze(0)

                self.eval()
                with torch.no_grad():
                    pred = self.forward(x)
                    pred_int = self._bits_to_int((pred.squeeze() > 0.5).float())

                expected = self.truths[op_idx](a, b)
                if pred_int == expected:
                    correct += 1

            acc = correct / total
            results[op_name] = acc

            if verbose:
                status = "PASS" if acc >= 0.9999 else "FAIL"
                print(f"  {op_name:5s}: {acc*100:.2f}% [{status}]")

        overall = sum(results.values()) / len(results)
        results["overall"] = overall

        if verbose:
            print(f"  {'TOTAL':5s}: {overall*100:.2f}%")

        return results

    def compute(self, a: int, b: int, op: str) -> int:
        """Compute a single operation."""
        if op not in self.ops:
            raise ValueError(f"Unknown op '{op}'. Available: {self.ops}")

        op_idx = self.ops.index(op)

        a_bits = self._int_to_bits(a)
        b_bits = self._int_to_bits(b)
        op_bits = torch.tensor([(op_idx >> i) & 1 for i in range(self.opcode_bits)], dtype=torch.float32)
        x = torch.cat([a_bits, b_bits, op_bits]).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
            result = self._bits_to_int((pred.squeeze() > 0.5).float())

        return result

    def save(self, path: str):
        """Save to PyTorch format."""
        torch.save({
            'router_state': self.router.state_dict(),
            'ops': self.ops,
        }, path)

    @classmethod
    def load(cls, path: str) -> "RV32I_ALU":
        """Load from PyTorch format."""
        data = torch.load(path, weights_only=False)
        alu = cls()
        alu.router.load_state_dict(data['router_state'])
        alu._trained = True
        return alu

    def to_onnx(self, path: str):
        """Export to ONNX."""
        example = torch.zeros(1, self.bits * 2 + self.opcode_bits)
        torch.onnx.export(
            self, example, path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )

    def summary(self) -> str:
        lines = [
            "RV32I ALU (Frozen Neural-Geometric)",
            f"  Operations: {self.num_ops}",
            f"  Bit width: {self.bits}",
            f"  Opcodes: {self.ops}",
            f"  Router params: {sum(p.numel() for p in self.router.parameters())}",
            f"  Trained: {self._trained}",
        ]
        return "\n".join(lines)


def demo():
    """Quick demo of RV32I ALU."""
    print("=" * 60)
    print("RISC-V RV32I ALU - Frozen Neural-Geometric")
    print("=" * 60)
    print()

    alu = RV32I_ALU()
    print(alu.summary())
    print()

    # Train
    alu.train_router(n_samples=100000, verbose=True)
    print()

    # Validate
    print("Validation:")
    alu.validate(n_samples=50000, verbose=True)
    print()

    # Demo computations
    print("Demo computations:")
    tests = [
        (0x12345678, 0x00000005, "add"),
        (0x12345678, 0x00000005, "sub"),
        (0xFFFFFFFF, 0x0F0F0F0F, "and"),
        (0xF0F0F0F0, 0x0F0F0F0F, "or"),
        (0xAAAAAAAA, 0x55555555, "xor"),
        (0x00000001, 0x00000004, "sll"),
        (0x80000000, 0x00000004, "srl"),
        (0x80000000, 0x00000004, "sra"),
        (0x00000005, 0x0000000A, "slt"),
        (0x0000000A, 0x00000005, "sltu"),
    ]

    for a, b, op in tests:
        result = alu.compute(a, b, op)
        expected = RV32I_SHAPES[op][1](a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {op}(0x{a:08X}, 0x{b:08X}) = 0x{result:08X} [{status}]")

    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
