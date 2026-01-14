"""
Atomic Shapes - Wrappers around TriX native operations.

This module provides the 32 fundamental shapes as thin wrappers around:
- Python's native bitwise operators (for scalar integer ops)
- NumPy's vectorized operations (for array ops)
- TriX native ops (for SIMD-accelerated batch processing)

Categories:
- Logic (7): XOR, AND, OR, NOT, NAND, NOR, XNOR
- Arithmetic (7): ADD, SUB, MUL, DIV, MOD, NEG, ABS
- Shift (7): SHL, SHR, SAR, ROL, ROR, RCL, RCR
- Compare (6): EQ, NE, LT, LE, GT, GE
- Memory (2): LOAD, STORE
- Routing (3): MUX, DEMUX, SELECT

For training (differentiable), use polynomial forms from trix.native.ops.
For inference (frozen), use the binary forms here.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Dict, Optional, Union
from enum import Enum
import numpy as np

# Try to import native ops for SIMD-accelerated batch processing
# Use direct path import to avoid triggering full trix package
_HAS_NATIVE_OPS = False
_HAS_VULKAN_GPU = False
TrixOps = None
get_ops = None
GilliesVulkan = None

try:
    import importlib.util
    import os
    _ops_path = os.path.join(os.path.dirname(__file__), '..', 'native', 'ops', '__init__.py')
    if os.path.exists(_ops_path):
        _spec = importlib.util.spec_from_file_location('trix_native_ops', _ops_path)
        _ops_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ops_module)
        TrixOps = _ops_module.TrixOps
        get_ops = _ops_module.get_ops
        _HAS_NATIVE_OPS = True
except (ImportError, RuntimeError, PermissionError, OSError):
    pass

# Try to import GILLIES Vulkan GPU shapes (17B ops/sec)
try:
    import importlib.util
    import os
    _vulkan_path = os.path.join(os.path.dirname(__file__), '..', 'native', 'vulkan', 'gillies.py')
    if os.path.exists(_vulkan_path):
        _spec = importlib.util.spec_from_file_location('gillies_vulkan', _vulkan_path)
        _vulkan_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vulkan_module)
        GilliesVulkan = _vulkan_module.GilliesVulkan
        _HAS_VULKAN_GPU = True
except (ImportError, RuntimeError, PermissionError, OSError):
    pass


class ShapeCategory(Enum):
    LOGIC = "logic"
    ARITHMETIC = "arithmetic"
    SHIFT = "shift"
    COMPARE = "compare"
    MEMORY = "memory"
    ROUTING = "routing"


@dataclass
class Shape:
    """An atomic shape definition."""
    name: str
    category: ShapeCategory
    inputs: List[str]
    outputs: List[str]
    fn: Callable
    description: str

    def __call__(self, *args, **kwargs):
        """Execute the shape."""
        return self.fn(*args, **kwargs)

    @property
    def arity(self) -> int:
        """Number of inputs."""
        return len(self.inputs)


# =============================================================================
# Logic Shapes (7) - Native bitwise operations
# =============================================================================

def _xor(a: int, b: int) -> int:
    """Bitwise XOR - uses Python's native ^ operator."""
    return a ^ b


def _and(a: int, b: int) -> int:
    """Bitwise AND - uses Python's native & operator."""
    return a & b


def _or(a: int, b: int) -> int:
    """Bitwise OR - uses Python's native | operator."""
    return a | b


def _not(a: int) -> int:
    """Bitwise NOT - uses Python's native ~ operator."""
    return (~a) & 0xFFFFFFFFFFFFFFFF


def _nand(a: int, b: int) -> int:
    """Bitwise NAND."""
    return (~(a & b)) & 0xFFFFFFFFFFFFFFFF


def _nor(a: int, b: int) -> int:
    """Bitwise NOR."""
    return (~(a | b)) & 0xFFFFFFFFFFFFFFFF


def _xnor(a: int, b: int) -> int:
    """Bitwise XNOR (equivalence)."""
    return (~(a ^ b)) & 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Arithmetic Shapes (7) - Native Python operations
# =============================================================================

def _add(a: int, b: int) -> int:
    """Integer addition."""
    return a + b


def _sub(a: int, b: int) -> int:
    """Integer subtraction."""
    return a - b


def _mul(a: int, b: int) -> int:
    """Integer multiplication."""
    return a * b


def _div(a: int, b: int) -> int:
    """Integer division (floor)."""
    return a // b if b != 0 else 0


def _mod(a: int, b: int) -> int:
    """Integer modulo."""
    return a % b if b != 0 else 0


def _neg(a: int) -> int:
    """Integer negation."""
    return -a


def _abs(a: int) -> int:
    """Absolute value."""
    return abs(a)


# =============================================================================
# Shift Shapes (7) - Native Python operations
# =============================================================================

def _shl(a: int, n: int) -> int:
    """Shift left logical."""
    return (a << n) & 0xFFFFFFFFFFFFFFFF


def _shr(a: int, n: int) -> int:
    """Shift right logical."""
    return (a & 0xFFFFFFFFFFFFFFFF) >> n


def _sar(a: int, n: int) -> int:
    """Shift right arithmetic (sign-extending)."""
    if a & 0x8000000000000000:
        mask = ((1 << n) - 1) << (64 - n)
        return ((a >> n) | mask) & 0xFFFFFFFFFFFFFFFF
    return a >> n


def _rol(a: int, n: int, width: int = 64) -> int:
    """Rotate left."""
    n = n % width
    return ((a << n) | (a >> (width - n))) & ((1 << width) - 1)


def _ror(a: int, n: int, width: int = 64) -> int:
    """Rotate right."""
    n = n % width
    return ((a >> n) | (a << (width - n))) & ((1 << width) - 1)


def _rcl(a: int, n: int, carry: int, width: int = 64) -> Tuple[int, int]:
    """Rotate left through carry."""
    n = n % (width + 1)
    extended = (a << 1) | carry
    rotated = ((extended << n) | (extended >> (width + 1 - n))) & ((1 << (width + 1)) - 1)
    new_carry = (rotated >> width) & 1
    result = rotated & ((1 << width) - 1)
    return result, new_carry


def _rcr(a: int, n: int, carry: int, width: int = 64) -> Tuple[int, int]:
    """Rotate right through carry."""
    n = n % (width + 1)
    extended = a | (carry << width)
    rotated = ((extended >> n) | (extended << (width + 1 - n))) & ((1 << (width + 1)) - 1)
    new_carry = rotated & 1
    result = (rotated >> 1) & ((1 << width) - 1)
    return result, new_carry


# =============================================================================
# Compare Shapes (6) - Native Python comparisons
# =============================================================================

def _eq(a: int, b: int) -> int:
    """Equal."""
    return 1 if a == b else 0


def _ne(a: int, b: int) -> int:
    """Not equal."""
    return 1 if a != b else 0


def _lt(a: int, b: int) -> int:
    """Less than."""
    return 1 if a < b else 0


def _le(a: int, b: int) -> int:
    """Less than or equal."""
    return 1 if a <= b else 0


def _gt(a: int, b: int) -> int:
    """Greater than."""
    return 1 if a > b else 0


def _ge(a: int, b: int) -> int:
    """Greater than or equal."""
    return 1 if a >= b else 0


# =============================================================================
# Memory Shapes (2)
# =============================================================================

def _load(addr: int, memory: Dict[int, int] = None) -> int:
    """Load from memory."""
    if memory is None:
        return 0
    return memory.get(addr, 0)


def _store(addr: int, value: int, memory: Dict[int, int] = None) -> None:
    """Store to memory."""
    if memory is not None:
        memory[addr] = value


# =============================================================================
# Routing Shapes (3)
# =============================================================================

def _mux(sel: int, a: Any, b: Any) -> Any:
    """2-to-1 multiplexer. sel=0 selects a, sel=1 selects b."""
    return b if sel else a


def _demux(sel: int, val: Any) -> Tuple[Any, Any]:
    """1-to-2 demultiplexer. Routes val to output selected by sel."""
    return (0, val) if sel else (val, 0)


def _select(idx: int, *values) -> Any:
    """N-to-1 selector. Returns values[idx]."""
    return values[idx] if 0 <= idx < len(values) else 0


# =============================================================================
# Shape Registry
# =============================================================================

SHAPES: Dict[str, Shape] = {
    # Logic
    "XOR": Shape("XOR", ShapeCategory.LOGIC, ["a", "b"], ["y"], _xor, "Bitwise XOR"),
    "AND": Shape("AND", ShapeCategory.LOGIC, ["a", "b"], ["y"], _and, "Bitwise AND"),
    "OR": Shape("OR", ShapeCategory.LOGIC, ["a", "b"], ["y"], _or, "Bitwise OR"),
    "NOT": Shape("NOT", ShapeCategory.LOGIC, ["a"], ["y"], _not, "Bitwise NOT"),
    "NAND": Shape("NAND", ShapeCategory.LOGIC, ["a", "b"], ["y"], _nand, "Bitwise NAND"),
    "NOR": Shape("NOR", ShapeCategory.LOGIC, ["a", "b"], ["y"], _nor, "Bitwise NOR"),
    "XNOR": Shape("XNOR", ShapeCategory.LOGIC, ["a", "b"], ["y"], _xnor, "Bitwise XNOR"),

    # Arithmetic
    "ADD": Shape("ADD", ShapeCategory.ARITHMETIC, ["a", "b"], ["y"], _add, "Integer addition"),
    "SUB": Shape("SUB", ShapeCategory.ARITHMETIC, ["a", "b"], ["y"], _sub, "Integer subtraction"),
    "MUL": Shape("MUL", ShapeCategory.ARITHMETIC, ["a", "b"], ["y"], _mul, "Integer multiplication"),
    "DIV": Shape("DIV", ShapeCategory.ARITHMETIC, ["a", "b"], ["y"], _div, "Integer division"),
    "MOD": Shape("MOD", ShapeCategory.ARITHMETIC, ["a", "b"], ["y"], _mod, "Integer modulo"),
    "NEG": Shape("NEG", ShapeCategory.ARITHMETIC, ["a"], ["y"], _neg, "Integer negation"),
    "ABS": Shape("ABS", ShapeCategory.ARITHMETIC, ["a"], ["y"], _abs, "Absolute value"),

    # Shift
    "SHL": Shape("SHL", ShapeCategory.SHIFT, ["a", "n"], ["y"], _shl, "Shift left logical"),
    "SHR": Shape("SHR", ShapeCategory.SHIFT, ["a", "n"], ["y"], _shr, "Shift right logical"),
    "SAR": Shape("SAR", ShapeCategory.SHIFT, ["a", "n"], ["y"], _sar, "Shift right arithmetic"),
    "ROL": Shape("ROL", ShapeCategory.SHIFT, ["a", "n"], ["y"], _rol, "Rotate left"),
    "ROR": Shape("ROR", ShapeCategory.SHIFT, ["a", "n"], ["y"], _ror, "Rotate right"),
    "RCL": Shape("RCL", ShapeCategory.SHIFT, ["a", "n", "c"], ["y", "c"], _rcl, "Rotate left through carry"),
    "RCR": Shape("RCR", ShapeCategory.SHIFT, ["a", "n", "c"], ["y", "c"], _rcr, "Rotate right through carry"),

    # Compare
    "EQ": Shape("EQ", ShapeCategory.COMPARE, ["a", "b"], ["y"], _eq, "Equal"),
    "NE": Shape("NE", ShapeCategory.COMPARE, ["a", "b"], ["y"], _ne, "Not equal"),
    "LT": Shape("LT", ShapeCategory.COMPARE, ["a", "b"], ["y"], _lt, "Less than"),
    "LE": Shape("LE", ShapeCategory.COMPARE, ["a", "b"], ["y"], _le, "Less than or equal"),
    "GT": Shape("GT", ShapeCategory.COMPARE, ["a", "b"], ["y"], _gt, "Greater than"),
    "GE": Shape("GE", ShapeCategory.COMPARE, ["a", "b"], ["y"], _ge, "Greater than or equal"),

    # Memory
    "LOAD": Shape("LOAD", ShapeCategory.MEMORY, ["addr"], ["y"], _load, "Load from memory"),
    "STORE": Shape("STORE", ShapeCategory.MEMORY, ["addr", "val"], [], _store, "Store to memory"),

    # Routing
    "MUX": Shape("MUX", ShapeCategory.ROUTING, ["sel", "a", "b"], ["y"], _mux, "2-to-1 multiplexer"),
    "DEMUX": Shape("DEMUX", ShapeCategory.ROUTING, ["sel", "val"], ["a", "b"], _demux, "1-to-2 demultiplexer"),
    "SELECT": Shape("SELECT", ShapeCategory.ROUTING, ["idx", "*values"], ["y"], _select, "N-to-1 selector"),
}


def get_shape(name: str) -> Optional[Shape]:
    """Get a shape by name."""
    return SHAPES.get(name.upper())


def list_shapes(category: Optional[ShapeCategory] = None) -> List[str]:
    """List all shape names, optionally filtered by category."""
    if category is None:
        return list(SHAPES.keys())
    return [name for name, shape in SHAPES.items() if shape.category == category]


# =============================================================================
# Vectorized Shapes (NumPy / Native Ops)
# =============================================================================

class VectorShapes:
    """
    Vectorized shape operations for batch processing.

    Backend priority:
    1. GILLIES Vulkan GPU (17B ops/sec)
    2. Native SIMD (NEON/AVX2)
    3. NumPy (fallback)
    """

    def __init__(self, use_gpu: bool = True, use_native: bool = True):
        self._gpu = None
        self._native = None

        # Try GPU first (GILLIES Vulkan)
        if use_gpu and _HAS_VULKAN_GPU:
            try:
                self._gpu = GilliesVulkan(max_elements=16_777_216)
            except (RuntimeError, OSError):
                pass

        # Then native SIMD
        if use_native and _HAS_NATIVE_OPS:
            try:
                self._native = get_ops()
            except RuntimeError:
                pass

    @property
    def backend(self) -> str:
        """Return active backend."""
        if self._gpu:
            return "vulkan (GILLIES)"
        if self._native:
            return f"native ({self._native.simd})"
        return "numpy"

    def close(self):
        """Release GPU resources."""
        if self._gpu:
            self._gpu.close()
            self._gpu = None

    def __del__(self):
        self.close()

    # Logic - vectorized (GPU accelerated when available)
    def xor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized XOR - GPU accelerated."""
        if self._gpu:
            return self._gpu.xor(
                np.ascontiguousarray(a, dtype=np.float32),
                np.ascontiguousarray(b, dtype=np.float32)
            )
        if a.dtype == np.uint8 and b.dtype == np.uint8:
            return np.bitwise_xor(a, b)
        # Polynomial form for floats
        return a + b - 2 * a * b

    def and_(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized AND."""
        if a.dtype == np.uint8 and b.dtype == np.uint8:
            return np.bitwise_and(a, b)
        return a * b

    def or_(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized OR."""
        if a.dtype == np.uint8 and b.dtype == np.uint8:
            return np.bitwise_or(a, b)
        return a + b - a * b

    def not_(self, a: np.ndarray) -> np.ndarray:
        """Vectorized NOT."""
        if a.dtype == np.uint8:
            return np.bitwise_not(a)
        return 1 - a

    # Arithmetic - vectorized
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    # Reductions using native ops if available
    def sum(self, x: np.ndarray) -> float:
        if self._native:
            return self._native.sum(x.astype(np.float32))
        return float(np.sum(x))

    def norm(self, x: np.ndarray) -> float:
        if self._native:
            return self._native.norm(x.astype(np.float32))
        return float(np.linalg.norm(x))


# Global vectorized shapes instance
_vector_shapes: Optional[VectorShapes] = None

def get_vector_shapes() -> VectorShapes:
    """Get or create singleton VectorShapes instance."""
    global _vector_shapes
    if _vector_shapes is None:
        _vector_shapes = VectorShapes()
    return _vector_shapes


# =============================================================================
# Composed Shapes (Built from Atomics)
# =============================================================================

def compose_full_adder(a: int, b: int, cin: int) -> Tuple[int, int]:
    """Full adder composed from atomic shapes."""
    xor1 = _xor(a, b)
    sum_out = _xor(xor1, cin)
    and1 = _and(a, b)
    and2 = _and(xor1, cin)
    cout = _or(and1, and2)
    return sum_out, cout


def compose_adder_n(a: int, b: int, n_bits: int = 8) -> int:
    """N-bit ripple carry adder."""
    result = 0
    carry = 0
    for i in range(n_bits):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        sum_bit, carry = compose_full_adder(a_bit, b_bit, carry)
        result |= (sum_bit << i)
    return result


def compose_alu(a: int, b: int, op: int) -> int:
    """Simple ALU composed from atomic shapes."""
    operations = [_add, _sub, _and, _or, _xor, _shl, _shr, _mul]
    if 0 <= op < len(operations):
        return operations[op](a, b)
    return 0


COMPOSED_SHAPES = {
    "FULL_ADDER": compose_full_adder,
    "ADDER8": lambda a, b: compose_adder_n(a, b, 8),
    "ADDER16": lambda a, b: compose_adder_n(a, b, 16),
    "ADDER32": lambda a, b: compose_adder_n(a, b, 32),
    "ALU": compose_alu,
}


# =============================================================================
# Shape Stats
# =============================================================================

def shape_stats() -> Dict[str, int]:
    """Return statistics about the shape library."""
    stats = {"total": len(SHAPES)}
    for cat in ShapeCategory:
        stats[cat.value] = len(list_shapes(cat))
    return stats


if __name__ == "__main__":
    print("Shape Fabric - Atomic Shapes")
    print("=" * 50)

    # Check backend
    vs = get_vector_shapes()
    print(f"Vector backend: {vs.backend}")

    if _HAS_NATIVE_OPS:
        ops = get_ops()
        print(f"Native ops version: {ops.version}")
        print(f"SIMD: {ops.simd}")

    stats = shape_stats()
    print(f"\nTotal shapes: {stats['total']}")
    for cat in ShapeCategory:
        print(f"  {cat.value}: {stats[cat.value]}")

    print("\nTesting atomic shapes:")
    assert _xor(0b1010, 0b1100) == 0b0110, "XOR failed"
    assert _and(0b1010, 0b1100) == 0b1000, "AND failed"
    assert _or(0b1010, 0b1100) == 0b1110, "OR failed"
    print("  Logic: PASS")

    assert _add(5, 3) == 8, "ADD failed"
    assert _sub(5, 3) == 2, "SUB failed"
    assert _mul(5, 3) == 15, "MUL failed"
    print("  Arithmetic: PASS")

    s, c = compose_full_adder(1, 1, 0)
    assert s == 0 and c == 1, "FullAdder failed"
    assert compose_adder_n(15, 1, 8) == 16, "Adder8 failed"
    print("  Composed: PASS")

    print("\nAll tests passed!")
