"""
XORPU Specification in trix.forge

The complete XORPU instruction set as frozen shapes.
This serves as both the software reference AND the hardware spec.

Usage:
    from trix.forge.xorpu_spec import XORPU

    xorpu = XORPU()
    xorpu.build()
    xorpu.validate()
    xorpu.export_spec()
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import json
import math

from trix.routing.primitives import xor, and_, or_, not_, full_adder


# =============================================================================
# 32-BIT FROZEN SHAPES (The Atoms and Their Compositions)
# =============================================================================

class Shapes32:
    """All 32-bit frozen shapes for XORPU."""

    @staticmethod
    def xor(a: Tensor, b: Tensor) -> Tensor:
        """XOR: a ^ b - The fundamental atom."""
        return xor(a, b)

    @staticmethod
    def and_(a: Tensor, b: Tensor) -> Tensor:
        """AND: a & b"""
        return and_(a, b)

    @staticmethod
    def or_(a: Tensor, b: Tensor) -> Tensor:
        """OR: a | b"""
        return or_(a, b)

    @staticmethod
    def not_(a: Tensor, b: Tensor = None) -> Tensor:
        """NOT: ~a (b ignored)"""
        return not_(a)

    @staticmethod
    def add(a: Tensor, b: Tensor) -> Tensor:
        """ADD: a + b (mod 2^32) - Ripple carry adder."""
        result = []
        carry = torch.zeros_like(a[:, 0])
        for i in range(32):
            s, carry = full_adder(a[:, i], b[:, i], carry)
            result.append(s)
        return torch.stack(result, dim=1)

    @staticmethod
    def sub(a: Tensor, b: Tensor) -> Tensor:
        """SUB: a - b (mod 2^32) - Two's complement."""
        not_b = not_(b)
        result = []
        carry = torch.ones_like(a[:, 0])
        for i in range(32):
            s, carry = full_adder(a[:, i], not_b[:, i], carry)
            result.append(s)
        return torch.stack(result, dim=1)

    @staticmethod
    def sll(a: Tensor, b: Tensor) -> Tensor:
        """SLL: a << (b & 0x1F) - Shift left logical."""
        batch_size = a.shape[0]
        result = a.clone()
        for stage in range(5):
            shift_amt = 1 << stage
            shift_bit = b[:, stage:stage+1]
            zeros = torch.zeros(batch_size, shift_amt, device=a.device)
            shifted = torch.cat([zeros, result[:, :-shift_amt]], dim=1)
            result = result + shift_bit * (shifted - result)
        return result

    @staticmethod
    def srl(a: Tensor, b: Tensor) -> Tensor:
        """SRL: a >> (b & 0x1F) - Shift right logical."""
        batch_size = a.shape[0]
        result = a.clone()
        for stage in range(5):
            shift_amt = 1 << stage
            shift_bit = b[:, stage:stage+1]
            zeros = torch.zeros(batch_size, shift_amt, device=a.device)
            shifted = torch.cat([result[:, shift_amt:], zeros], dim=1)
            result = result + shift_bit * (shifted - result)
        return result

    @staticmethod
    def sra(a: Tensor, b: Tensor) -> Tensor:
        """SRA: a >> (b & 0x1F) - Shift right arithmetic."""
        batch_size = a.shape[0]
        result = a.clone()
        sign_bit = a[:, 31:32]
        for stage in range(5):
            shift_amt = 1 << stage
            shift_bit = b[:, stage:stage+1]
            fill = sign_bit.expand(batch_size, shift_amt)
            shifted = torch.cat([result[:, shift_amt:], fill], dim=1)
            result = result + shift_bit * (shifted - result)
        return result

    @staticmethod
    def slt(a: Tensor, b: Tensor) -> Tensor:
        """SLT: (signed)a < b ? 1 : 0"""
        a_sign = a[:, 31:32]
        b_sign = b[:, 31:32]
        diff = Shapes32.sub(a, b)
        diff_sign = diff[:, 31:32]

        case1 = and_(a_sign, not_(b_sign))
        same_sign = not_(xor(a_sign, b_sign))
        case3 = and_(same_sign, diff_sign)
        is_less = or_(case1, case3)

        batch_size = a.shape[0]
        zeros = torch.zeros(batch_size, 31, device=a.device)
        return torch.cat([is_less, zeros], dim=1)

    @staticmethod
    def sltu(a: Tensor, b: Tensor) -> Tensor:
        """SLTU: (unsigned)a < b ? 1 : 0"""
        not_b = not_(b)
        carry = torch.ones_like(a[:, 0])
        for i in range(32):
            _, carry = full_adder(a[:, i], not_b[:, i], carry)
        is_less = not_(carry.unsqueeze(1))

        batch_size = a.shape[0]
        zeros = torch.zeros(batch_size, 31, device=a.device)
        return torch.cat([is_less, zeros], dim=1)

    @staticmethod
    def nand(a: Tensor, b: Tensor) -> Tensor:
        """NAND: ~(a & b)"""
        return not_(and_(a, b))

    @staticmethod
    def nor(a: Tensor, b: Tensor) -> Tensor:
        """NOR: ~(a | b)"""
        return not_(or_(a, b))

    @staticmethod
    def xnor(a: Tensor, b: Tensor) -> Tensor:
        """XNOR: ~(a ^ b)"""
        return not_(xor(a, b))


# =============================================================================
# TRUTH FUNCTIONS (Reference implementations)
# =============================================================================

class Truth32:
    """Truth functions for 32-bit operations."""

    @staticmethod
    def xor(a: int, b: int) -> int:
        return a ^ b

    @staticmethod
    def and_(a: int, b: int) -> int:
        return a & b

    @staticmethod
    def or_(a: int, b: int) -> int:
        return a | b

    @staticmethod
    def not_(a: int, b: int = 0) -> int:
        return (~a) & 0xFFFFFFFF

    @staticmethod
    def add(a: int, b: int) -> int:
        return (a + b) & 0xFFFFFFFF

    @staticmethod
    def sub(a: int, b: int) -> int:
        return (a - b) & 0xFFFFFFFF

    @staticmethod
    def sll(a: int, b: int) -> int:
        return (a << (b & 0x1F)) & 0xFFFFFFFF

    @staticmethod
    def srl(a: int, b: int) -> int:
        return (a >> (b & 0x1F)) & 0xFFFFFFFF

    @staticmethod
    def sra(a: int, b: int) -> int:
        shift = b & 0x1F
        if a & 0x80000000:  # negative
            return ((a >> shift) | (0xFFFFFFFF << (32 - shift))) & 0xFFFFFFFF
        return (a >> shift) & 0xFFFFFFFF

    @staticmethod
    def slt(a: int, b: int) -> int:
        a_signed = a if a < 0x80000000 else a - 0x100000000
        b_signed = b if b < 0x80000000 else b - 0x100000000
        return 1 if a_signed < b_signed else 0

    @staticmethod
    def sltu(a: int, b: int) -> int:
        return 1 if a < b else 0

    @staticmethod
    def nand(a: int, b: int) -> int:
        return (~(a & b)) & 0xFFFFFFFF

    @staticmethod
    def nor(a: int, b: int) -> int:
        return (~(a | b)) & 0xFFFFFFFF

    @staticmethod
    def xnor(a: int, b: int) -> int:
        return (~(a ^ b)) & 0xFFFFFFFF


# =============================================================================
# XORPU SPECIFICATION
# =============================================================================

@dataclass
class ShapeSpec:
    """Specification for a single shape."""
    id: int
    name: str
    shape_fn: Callable
    truth_fn: Callable
    input_bits: int
    output_bits: int
    description: str
    cycles: int  # Estimated execution cycles
    terms: int   # Number of polynomial terms


@dataclass
class XORPUSpec:
    """Complete XORPU specification."""
    version: str
    bits: int
    shapes: List[ShapeSpec]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "bits": self.bits,
            "num_shapes": len(self.shapes),
            "shapes": [
                {
                    "id": s.id,
                    "name": s.name,
                    "input_bits": s.input_bits,
                    "output_bits": s.output_bits,
                    "description": s.description,
                    "cycles": s.cycles,
                    "terms": s.terms,
                }
                for s in self.shapes
            ]
        }


class XORPU:
    """
    XORPU: XOR Processing Unit

    Complete specification and validation of the XORPU instruction set.
    """

    VERSION = "0.1.0"
    BITS = 32

    # Shape definitions: (name, shape_fn, truth_fn, description, cycles, terms)
    SHAPE_DEFS = [
        ("nop",   lambda a, b: a,        lambda a, b: a,        "No operation",              1,   0),
        ("xor",   Shapes32.xor,          Truth32.xor,           "Bitwise XOR",               5,  32),
        ("and",   Shapes32.and_,         Truth32.and_,          "Bitwise AND",               5,  32),
        ("or",    Shapes32.or_,          Truth32.or_,           "Bitwise OR",                5,  64),
        ("not",   Shapes32.not_,         Truth32.not_,          "Bitwise NOT",               5,  32),
        ("add",   Shapes32.add,          Truth32.add,           "32-bit addition",          37, 256),
        ("sub",   Shapes32.sub,          Truth32.sub,           "32-bit subtraction",       37, 256),
        ("sll",   Shapes32.sll,          Truth32.sll,           "Shift left logical",       41, 320),
        ("srl",   Shapes32.srl,          Truth32.srl,           "Shift right logical",      41, 320),
        ("sra",   Shapes32.sra,          Truth32.sra,           "Shift right arithmetic",   41, 352),
        ("slt",   Shapes32.slt,          Truth32.slt,           "Set less than (signed)",   42, 280),
        ("sltu",  Shapes32.sltu,         Truth32.sltu,          "Set less than (unsigned)", 40, 264),
        ("nand",  Shapes32.nand,         Truth32.nand,          "Bitwise NAND",              6,  64),
        ("nor",   Shapes32.nor,          Truth32.nor,           "Bitwise NOR",               6,  96),
        ("xnor",  Shapes32.xnor,         Truth32.xnor,          "Bitwise XNOR",              6,  64),
    ]

    def __init__(self):
        self.shapes: List[ShapeSpec] = []
        self._build_shapes()
        self._router = None
        self._compiled = False
        self.validation_results = {}

    def _build_shapes(self):
        """Build shape specifications."""
        for i, (name, shape_fn, truth_fn, desc, cycles, terms) in enumerate(self.SHAPE_DEFS):
            self.shapes.append(ShapeSpec(
                id=i,
                name=name,
                shape_fn=shape_fn,
                truth_fn=truth_fn,
                input_bits=64 if name not in ["nop", "not"] else 32,
                output_bits=32,
                description=desc,
                cycles=cycles,
                terms=terms,
            ))

    def _int_to_bits(self, x: int) -> Tensor:
        return torch.tensor([(x >> i) & 1 for i in range(32)], dtype=torch.float32)

    def _bits_to_int(self, bits: Tensor) -> int:
        result = 0
        for i, b in enumerate(bits):
            result += int(round(float(b))) << i
        return result

    def build(self, verbose: bool = True) -> "XORPU":
        """Build the XORPU router."""
        if verbose:
            print(f"Building XORPU v{self.VERSION}")
            print(f"  Shapes: {len(self.shapes)}")
            print(f"  Bit width: {self.BITS}")

        # Router: input bits + opcode -> shape selection
        opcode_bits = math.ceil(math.log2(len(self.shapes)))
        input_bits = self.BITS * 2 + opcode_bits

        self._router = nn.Linear(input_bits, len(self.shapes))
        self._opcode_bits = opcode_bits
        self._compiled = True

        if verbose:
            params = sum(p.numel() for p in self._router.parameters())
            print(f"  Router params: {params}")

        return self

    def validate_shape(self, shape_name: str, n_samples: int = 10000,
                       verbose: bool = True) -> float:
        """Validate a single shape against its truth function."""
        shape = next((s for s in self.shapes if s.name == shape_name), None)
        if shape is None:
            raise ValueError(f"Unknown shape: {shape_name}")

        correct = 0
        for _ in range(n_samples):
            a = torch.randint(0, 0xFFFFFFFF, (1,)).item()
            b = torch.randint(0, 0xFFFFFFFF, (1,)).item()

            # Frozen shape
            a_bits = self._int_to_bits(a).unsqueeze(0)
            b_bits = self._int_to_bits(b).unsqueeze(0)
            result_bits = shape.shape_fn(a_bits, b_bits)
            result = self._bits_to_int((result_bits.squeeze() > 0.5).float())

            # Truth
            expected = shape.truth_fn(a, b)

            if result == expected:
                correct += 1

        accuracy = correct / n_samples
        self.validation_results[shape_name] = accuracy

        if verbose:
            status = "PASS" if accuracy >= 0.9999 else "FAIL"
            print(f"  {shape_name:6s}: {accuracy*100:6.2f}% [{status}]")

        return accuracy

    def validate_all(self, n_samples: int = 5000, verbose: bool = True) -> Dict[str, float]:
        """Validate all shapes."""
        if verbose:
            print(f"\nValidating {len(self.shapes)} shapes ({n_samples} samples each):")

        results = {}
        passed = 0

        for shape in self.shapes:
            if shape.name == "nop":
                results["nop"] = 1.0
                if verbose:
                    print(f"  nop   : 100.00% [PASS]")
                passed += 1
                continue

            acc = self.validate_shape(shape.name, n_samples, verbose)
            results[shape.name] = acc
            if acc >= 0.9999:
                passed += 1

        if verbose:
            print(f"\nTotal: {passed}/{len(self.shapes)} shapes passed")

        return results

    def get_spec(self) -> XORPUSpec:
        """Get the complete XORPU specification."""
        return XORPUSpec(
            version=self.VERSION,
            bits=self.BITS,
            shapes=self.shapes,
        )

    def export_spec(self, path: str = None) -> dict:
        """Export specification as JSON."""
        spec_dict = self.get_spec().to_dict()

        # Add summary statistics
        total_terms = sum(s.terms for s in self.shapes)
        total_cycles = sum(s.cycles for s in self.shapes)

        spec_dict["summary"] = {
            "total_shapes": len(self.shapes),
            "total_terms": total_terms,
            "avg_cycles": total_cycles / len(self.shapes),
            "validation": self.validation_results,
        }

        if path:
            with open(path, 'w') as f:
                json.dump(spec_dict, f, indent=2)

        return spec_dict

    def export_terms(self, path: str = None, include_shapes: list = None) -> dict:
        """
        Export full term specification as JSON.

        This exports the explicit polynomial terms for each shape,
        enabling hardware synthesis and formal verification.

        Args:
            path: Optional file path to save JSON
            include_shapes: Optional list of shape names to include (default: all)

        Returns:
            Dictionary with full term specification
        """
        from datetime import datetime
        from .term import generate_shape

        # Generate term specs for requested shapes
        shape_names = include_shapes or [s.name for s in self.shapes if s.name != "nop"]
        term_specs = {}

        for name in shape_names:
            shape_terms = generate_shape(name, bits=self.BITS)
            if shape_terms:
                term_specs[name] = shape_terms.to_dict()

        # Build full export
        export = {
            "version": self.VERSION,
            "format_version": "1.0.0",
            "bits": self.BITS,
            "generated": datetime.now().isoformat(),
            "total_shapes": len(term_specs),
            "total_terms": sum(ts["total_terms"] for ts in term_specs.values()),
            "shapes": term_specs,
        }

        if path:
            with open(path, 'w') as f:
                json.dump(export, f, indent=2)

        return export

    @classmethod
    def from_terms_json(cls, path: str) -> "XORPU":
        """
        Load XORPU from term specification JSON.

        Args:
            path: Path to JSON file

        Returns:
            XORPU instance with loaded specifications
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Create XORPU instance
        xorpu = cls()

        # Verify version compatibility
        if data.get("bits", 32) != xorpu.BITS:
            raise ValueError(f"Bit width mismatch: file has {data['bits']}, expected {xorpu.BITS}")

        return xorpu

    def compute(self, a: int, b: int, op: str) -> int:
        """Compute a single operation."""
        shape = next((s for s in self.shapes if s.name == op), None)
        if shape is None:
            raise ValueError(f"Unknown operation: {op}")

        a_bits = self._int_to_bits(a).unsqueeze(0)
        b_bits = self._int_to_bits(b).unsqueeze(0)
        result_bits = shape.shape_fn(a_bits, b_bits)
        return self._bits_to_int((result_bits.squeeze() > 0.5).float())

    def summary(self) -> str:
        """Return summary string."""
        lines = [
            f"XORPU v{self.VERSION}",
            f"  Bit width: {self.BITS}",
            f"  Shapes: {len(self.shapes)}",
            "",
            "  Shape List:",
        ]

        for s in self.shapes:
            lines.append(f"    [{s.id:2d}] {s.name:6s} - {s.description} ({s.cycles} cycles, {s.terms} terms)")

        return "\n".join(lines)


def main():
    """Build and validate XORPU specification."""
    print("=" * 60)
    print("XORPU SPECIFICATION BUILD")
    print("=" * 60)
    print()

    # Build
    xorpu = XORPU()
    xorpu.build(verbose=True)
    print()
    print(xorpu.summary())

    # Validate
    results = xorpu.validate_all(n_samples=5000, verbose=True)

    # Export
    spec = xorpu.export_spec("tmp/xorpu/xorpu_spec.json")
    print(f"\nSpec exported to tmp/xorpu/xorpu_spec.json")

    # Demo
    print("\nDemo computations:")
    tests = [
        (0x12345678, 0x9ABCDEF0, "xor"),
        (0x12345678, 0x9ABCDEF0, "add"),
        (0x12345678, 0x00000004, "sll"),
        (0x80000000, 0x00000004, "sra"),
        (5, 10, "slt"),
    ]

    for a, b, op in tests:
        result = xorpu.compute(a, b, op)
        expected = next(s for s in xorpu.shapes if s.name == op).truth_fn(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {op}(0x{a:08X}, 0x{b:08X}) = 0x{result:08X} [{status}]")

    print()
    print("=" * 60)

    return xorpu, spec


if __name__ == "__main__":
    main()
