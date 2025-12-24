"""
MDS Parallel Carry - Read positions, don't compute chains.

Every carry is a direct lookup. Zero dependencies.
"""

from dataclasses import dataclass
from typing import List
import time


@dataclass
class CarryGeometry:
    """The carry exists at a position. We read it."""
    bits: int

    def read_carry(self, position: int, a: int, b: int) -> int:
        """Read carry at position directly. No chain. Just geometry."""
        if position == 0:
            return 0

        # Generate and Propagate for all bits up to position
        G = [(a >> i) & 1 & ((b >> i) & 1) for i in range(position)]
        P = [((a >> i) & 1) ^ ((b >> i) & 1) for i in range(position)]

        # Carry = OR of all generate paths
        carry = 0
        for k in range(position):
            path = G[k]
            for j in range(k + 1, position):
                path &= P[j]
            carry |= path

        return carry

    def read_all_carries(self, a: int, b: int) -> List[int]:
        """Read ALL carries in parallel. Each is independent."""
        return [self.read_carry(i, a, b) for i in range(self.bits + 1)]

    def add(self, a: int, b: int) -> int:
        """Addition via parallel carry reading."""
        carries = self.read_all_carries(a, b)

        result = 0
        for i in range(self.bits):
            a_bit = (a >> i) & 1
            b_bit = (b >> i) & 1
            sum_bit = a_bit ^ b_bit ^ carries[i]
            result |= sum_bit << i

        result |= carries[self.bits] << self.bits
        return result


def verify():
    """Verify parallel carry reading matches arithmetic."""
    print("PARALLEL CARRY VERIFICATION")
    print("=" * 50)

    for bits in [4, 8]:
        geom = CarryGeometry(bits)
        max_val = 2 ** bits
        errors = 0

        for a in range(max_val):
            for b in range(max_val):
                if geom.add(a, b) != a + b:
                    errors += 1

        print(f"{bits:>3}-bit: {max_val * max_val:>6} cases - {'PASS' if errors == 0 else 'FAIL'}")

    # Sample test for larger
    import random
    for bits in [16, 32]:
        geom = CarryGeometry(bits)
        max_val = 2 ** bits
        errors = 0
        samples = 10000

        for _ in range(samples):
            a = random.randint(0, max_val - 1)
            b = random.randint(0, max_val - 1)
            if geom.add(a, b) != a + b:
                errors += 1

        print(f"{bits:>3}-bit: {samples:>6} samples - {'PASS' if errors == 0 else 'FAIL'}")


def benchmark():
    """Benchmark parallel vs sequential."""
    from lattice import build_adder_lattice

    print()
    print("BENCHMARK: Sequential vs Parallel Geometry")
    print("=" * 50)

    for bits in [4, 8]:
        max_val = 2 ** bits
        iterations = 10 if bits == 8 else 100

        # Sequential lattice
        lattice = build_adder_lattice(bits)
        start = time.perf_counter()
        for _ in range(iterations):
            for a in range(max_val):
                for b in range(max_val):
                    inputs = {f'a{i}': (a >> i) & 1 for i in range(bits)}
                    inputs.update({f'b{i}': (b >> i) & 1 for i in range(bits)})
                    lattice.navigate(inputs)
        seq_time = time.perf_counter() - start

        # Parallel geometry
        geom = CarryGeometry(bits)
        start = time.perf_counter()
        for _ in range(iterations):
            for a in range(max_val):
                for b in range(max_val):
                    geom.add(a, b)
        par_time = time.perf_counter() - start

        ops = iterations * max_val * max_val
        print(f"\n{bits}-bit ({ops:,} ops):")
        print(f"  Sequential: {seq_time:.2f}s ({seq_time/ops*1e6:.1f} us/op)")
        print(f"  Parallel:   {par_time:.2f}s ({par_time/ops*1e6:.1f} us/op)")
        print(f"  Speedup:    {seq_time / par_time:.1f}x")


if __name__ == "__main__":
    verify()
    benchmark()
    print()
    print("=" * 50)
    print("Every carry READ from geometry.")
    print("No sequential dependencies.")
    print("Positions, not computations.")
