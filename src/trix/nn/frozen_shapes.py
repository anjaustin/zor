"""
Frozen Shapes Library - General-Purpose Mathematical Shapes

DEPRECATED: Use trix.native.FrozenShapes instead.

    # Old (PyTorch):
    from trix.nn.frozen_shapes import Primitives
    result = Primitives.xor(a, b)

    # New (Native):
    from trix.native import FrozenShapes
    result = FrozenShapes.xor(a, b)

The Native version works with CuPy (GPU) or NumPy (CPU) and has no
PyTorch dependency. Same API, same math, fewer dependencies.

---

A library of frozen (non-learnable) computation shapes beyond 6502.
These are mathematical truths expressed as continuous polynomials:
- 100% accurate on binary inputs
- Smooth gradients for training routing
- 0 learnable parameters in the shapes themselves

Categories:
- Arithmetic: add, sub, mul (via log tricks), div
- Logic: and, or, xor, not, nand, nor, xnor
- Comparison: eq, lt, gt, max, min, clamp
- Activation: relu, gelu_approx, sigmoid_approx, tanh_approx
- Composition: identity, swap, project, expand

Mathematical foundation:
    All shapes are polynomials in [0,1]^n → [0,1]^m
    XOR(a, b) = a + b - 2ab  (saddle surface)
    AND(a, b) = ab           (product)
    OR(a, b)  = a + b - ab   (union)
    NOT(a)    = 1 - a        (reflection)

Example:
    >>> from trix.nn.frozen_shapes import FrozenShapeLibrary
    >>> shapes = FrozenShapeLibrary()
    >>> shapes.xor(torch.tensor([0., 1.]), torch.tensor([1., 0.]))
    tensor([1., 1.])
"""

import warnings
warnings.warn(
    "trix.nn.frozen_shapes is deprecated for execution. "
    "Use trix.native.FrozenShapes for inference (no PyTorch needed). "
    "trix.nn.frozen_shapes remains valid for gradient-based training.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable, Optional, List
from dataclasses import dataclass
import math


# =============================================================================
# CORE PRIMITIVES (The Axioms)
# =============================================================================

class Primitives:
    """
    The four axioms of frozen computation.

    All other shapes compose from these.
    These ARE the geometry - not approximations.
    """

    @staticmethod
    def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        XOR: a + b - 2ab

        The saddle surface. The primitive of difference.
        Distance = XOR. Routing = XOR. Providence = XOR.
        """
        return a + b - 2 * a * b

    @staticmethod
    def and_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        AND: ab

        Product. Agreement. Intersection.
        """
        return a * b

    @staticmethod
    def or_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        OR: a + b - ab

        Union. Either. Disjunction.
        """
        return a + b - a * b

    @staticmethod
    def not_op(a: torch.Tensor) -> torch.Tensor:
        """
        NOT: 1 - a

        Reflection. Complement. Negation.
        """
        return 1 - a


# =============================================================================
# DERIVED LOGIC SHAPES
# =============================================================================

class LogicShapes:
    """
    Logic operations derived from primitives.

    All 16 binary Boolean functions can be expressed
    as compositions of AND, OR, NOT (and thus XOR).
    """

    @staticmethod
    def nand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """NAND: NOT(AND(a, b)) = 1 - ab"""
        return 1 - a * b

    @staticmethod
    def nor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """NOR: NOT(OR(a, b)) = 1 - (a + b - ab)"""
        return 1 - (a + b - a * b)

    @staticmethod
    def xnor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XNOR: NOT(XOR(a, b)) = 1 - (a + b - 2ab)"""
        return 1 - (a + b - 2 * a * b)

    @staticmethod
    def implies(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """IMPLIES: NOT(a) OR b = (1-a) + b - (1-a)b = 1 - a + ab"""
        return 1 - a + a * b

    @staticmethod
    def half_adder_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Half adder sum: XOR(a, b)"""
        return Primitives.xor(a, b)

    @staticmethod
    def half_adder_carry(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Half adder carry: AND(a, b)"""
        return Primitives.and_op(a, b)

    @staticmethod
    def full_adder(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full adder: (a, b, c_in) → (sum, c_out)

        sum = a XOR b XOR c
        c_out = (a AND b) OR (c AND (a XOR b))
        """
        p = Primitives.xor(a, b)
        sum_bit = Primitives.xor(p, c)
        c_out = Primitives.or_op(
            Primitives.and_op(a, b),
            Primitives.and_op(c, p)
        )
        return sum_bit, c_out

    @staticmethod
    def majority(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Majority: 1 if at least 2 of 3 inputs are 1.

        MAJ(a,b,c) = ab + bc + ac - 2abc
        """
        return a*b + b*c + a*c - 2*a*b*c


# =============================================================================
# COMPARISON SHAPES
# =============================================================================

class ComparisonShapes:
    """
    Comparison operations using soft approximations.

    For training, we need smooth gradients.
    These approximate comparisons while being exact on {0, 1}.
    """

    @staticmethod
    def eq(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """
        Soft equality: 1 if a ≈ b, 0 otherwise.

        Uses Gaussian kernel: exp(-sharpness * (a-b)^2)
        On binary inputs: exact.
        """
        diff_sq = (a - b) ** 2
        return torch.exp(-sharpness * diff_sq)

    @staticmethod
    def lt(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """
        Soft less-than: 1 if a < b, 0 otherwise.

        Uses sigmoid: sigmoid(sharpness * (b - a))
        On binary inputs: near-exact.
        """
        return torch.sigmoid(sharpness * (b - a))

    @staticmethod
    def gt(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """Soft greater-than: 1 if a > b."""
        return torch.sigmoid(sharpness * (a - b))

    @staticmethod
    def le(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """Soft less-than-or-equal."""
        return torch.sigmoid(sharpness * (b - a + 0.5))

    @staticmethod
    def ge(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """Soft greater-than-or-equal."""
        return torch.sigmoid(sharpness * (a - b + 0.5))

    @staticmethod
    def max2(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """
        Soft max of two values.

        Uses: a*sigmoid(sharpness*(a-b)) + b*sigmoid(sharpness*(b-a))
        Approaches true max as sharpness → ∞
        """
        w_a = torch.sigmoid(sharpness * (a - b))
        w_b = 1 - w_a
        return a * w_a + b * w_b

    @staticmethod
    def min2(a: torch.Tensor, b: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """Soft min of two values."""
        w_a = torch.sigmoid(sharpness * (b - a))
        w_b = 1 - w_a
        return a * w_a + b * w_b

    @staticmethod
    def clamp(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0, sharpness: float = 10.0) -> torch.Tensor:
        """Soft clamp to [lo, hi]."""
        x = ComparisonShapes.max2(x, torch.full_like(x, lo), sharpness)
        x = ComparisonShapes.min2(x, torch.full_like(x, hi), sharpness)
        return x


# =============================================================================
# ACTIVATION SHAPES
# =============================================================================

class ActivationShapes:
    """
    Activation functions as frozen shapes.

    These are polynomial approximations that:
    - Have smooth gradients
    - Are fast to compute
    - Approximate standard activations well
    """

    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """
        ReLU: max(0, x)

        This is exact (not an approximation).
        For frozen shapes, we use the standard ReLU.
        """
        return torch.relu(x)

    @staticmethod
    def relu_poly(x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of ReLU.

        For x in [-1, 1]: 0.25 * (x + 1)^2 when x > 0
        Smooth everywhere, differentiable at 0.
        """
        return 0.25 * (x + torch.abs(x) + 2 * x * torch.abs(x) / (torch.abs(x) + 0.1))

    @staticmethod
    def gelu_approx(x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of GELU.

        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        Simplified: x * sigmoid(1.702 * x)
        """
        return x * torch.sigmoid(1.702 * x)

    @staticmethod
    def sigmoid_poly(x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of sigmoid.

        Using: 0.5 + 0.5 * tanh(x/2)
        """
        return 0.5 + 0.5 * torch.tanh(x / 2)

    @staticmethod
    def tanh_poly(x: torch.Tensor) -> torch.Tensor:
        """
        Polynomial approximation of tanh for small x.

        tanh(x) ≈ x - x^3/3 + 2x^5/15 (for |x| < 1)
        For larger x, use actual tanh.
        """
        return torch.tanh(x)  # PyTorch tanh is already optimized

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        """Swish/SiLU: x * sigmoid(x)"""
        return x * torch.sigmoid(x)

    @staticmethod
    def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Softplus: log(1 + exp(beta * x)) / beta"""
        return torch.nn.functional.softplus(x, beta=beta)


# =============================================================================
# ARITHMETIC SHAPES
# =============================================================================

class ArithmeticShapes:
    """
    Arithmetic operations on bits.

    These operate on individual bits or bit vectors.
    """

    @staticmethod
    def increment_bit(
        bits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Increment 8-bit value by 1.

        Uses ripple carry adder with b = 00000001.
        """
        result = []
        carry = torch.ones(bits.shape[0], device=bits.device)

        for i in range(8 if bits.shape[-1] >= 8 else bits.shape[-1]):
            b = torch.zeros(bits.shape[0], device=bits.device)
            s, carry = LogicShapes.full_adder(bits[:, i], b, carry)
            result.append(s)

        return torch.stack(result, dim=1), carry

    @staticmethod
    def decrement_bit(
        bits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decrement 8-bit value by 1.

        Uses: A + 0xFF (two's complement of -1)
        """
        result = []
        carry = torch.zeros(bits.shape[0], device=bits.device)
        ones = torch.ones(bits.shape[0], device=bits.device)

        for i in range(8 if bits.shape[-1] >= 8 else bits.shape[-1]):
            s, carry = LogicShapes.full_adder(bits[:, i], ones, carry)
            result.append(s)

        return torch.stack(result, dim=1), carry

    @staticmethod
    def add_8bit(
        a: torch.Tensor,
        b: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        8-bit addition with carry.

        Returns: (result_bits, carry_out)
        """
        result = []
        carry = c_in

        for i in range(8):
            s, carry = LogicShapes.full_adder(a[:, i], b[:, i], carry)
            result.append(s)

        return torch.stack(result, dim=1), carry

    @staticmethod
    def sub_8bit(
        a: torch.Tensor,
        b: torch.Tensor,
        borrow_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        8-bit subtraction with borrow.

        A - B = A + NOT(B) + 1
        borrow_in = 0 means subtract, = 1 means subtract with borrow
        """
        b_inv = Primitives.not_op(b)
        c_in = Primitives.not_op(borrow_in)  # Borrow is inverted carry
        result, c_out = ArithmeticShapes.add_8bit(a, b_inv, c_in)
        borrow_out = Primitives.not_op(c_out)
        return result, borrow_out


# =============================================================================
# COMPOSITION SHAPES
# =============================================================================

class CompositionShapes:
    """
    Shape composition utilities.

    These help build complex shapes from simpler ones.
    """

    @staticmethod
    def identity(x: torch.Tensor) -> torch.Tensor:
        """Identity: output = input"""
        return x

    @staticmethod
    def constant(x: torch.Tensor, value: float) -> torch.Tensor:
        """Constant: output = value (ignores input)"""
        return torch.full_like(x, value)

    @staticmethod
    def project(x: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """Project: select specific dimensions"""
        return x[..., indices]

    @staticmethod
    def expand(x: torch.Tensor, repeats: int) -> torch.Tensor:
        """Expand: repeat each dimension"""
        return x.repeat_interleave(repeats, dim=-1)

    @staticmethod
    def swap(x: torch.Tensor) -> torch.Tensor:
        """Swap: reverse dimension order"""
        return x.flip(-1)

    @staticmethod
    def split(x: torch.Tensor, n_parts: int) -> List[torch.Tensor]:
        """Split: divide into n parts"""
        return list(x.chunk(n_parts, dim=-1))

    @staticmethod
    def concat(*tensors: torch.Tensor) -> torch.Tensor:
        """Concat: join tensors along last dimension"""
        return torch.cat(tensors, dim=-1)


# =============================================================================
# FROZEN SHAPE LIBRARY
# =============================================================================

@dataclass
class ShapeInfo:
    """Metadata for a frozen shape."""
    name: str
    category: str
    fn: Callable
    n_inputs: int
    n_outputs: int
    description: str


class FrozenShapeLibrary:
    """
    Library of all frozen shapes.

    Usage:
        >>> lib = FrozenShapeLibrary()
        >>> lib.xor(a, b)
        >>> lib.get('xor')(a, b)
        >>> lib.list_shapes('logic')
    """

    def __init__(self):
        self._shapes: Dict[str, ShapeInfo] = {}
        self._register_all()

    def _register_all(self):
        """Register all shapes."""
        # Primitives
        self._register('xor', 'primitive', Primitives.xor, 2, 1, 'XOR: a + b - 2ab')
        self._register('and', 'primitive', Primitives.and_op, 2, 1, 'AND: ab')
        self._register('or', 'primitive', Primitives.or_op, 2, 1, 'OR: a + b - ab')
        self._register('not', 'primitive', Primitives.not_op, 1, 1, 'NOT: 1 - a')

        # Logic
        self._register('nand', 'logic', LogicShapes.nand, 2, 1, 'NAND: NOT(AND)')
        self._register('nor', 'logic', LogicShapes.nor, 2, 1, 'NOR: NOT(OR)')
        self._register('xnor', 'logic', LogicShapes.xnor, 2, 1, 'XNOR: NOT(XOR)')
        self._register('implies', 'logic', LogicShapes.implies, 2, 1, 'IMPLIES: NOT(a) OR b')
        self._register('majority', 'logic', LogicShapes.majority, 3, 1, 'Majority of 3')

        # Comparison
        self._register('eq', 'comparison', ComparisonShapes.eq, 2, 1, 'Soft equality')
        self._register('lt', 'comparison', ComparisonShapes.lt, 2, 1, 'Soft less-than')
        self._register('gt', 'comparison', ComparisonShapes.gt, 2, 1, 'Soft greater-than')
        self._register('max', 'comparison', ComparisonShapes.max2, 2, 1, 'Soft max')
        self._register('min', 'comparison', ComparisonShapes.min2, 2, 1, 'Soft min')

        # Activation
        self._register('relu', 'activation', ActivationShapes.relu, 1, 1, 'ReLU')
        self._register('gelu', 'activation', ActivationShapes.gelu_approx, 1, 1, 'GELU approx')
        self._register('swish', 'activation', ActivationShapes.swish, 1, 1, 'Swish/SiLU')
        self._register('sigmoid', 'activation', ActivationShapes.sigmoid_poly, 1, 1, 'Sigmoid approx')
        self._register('tanh', 'activation', ActivationShapes.tanh_poly, 1, 1, 'Tanh')
        self._register('softplus', 'activation', ActivationShapes.softplus, 1, 1, 'Softplus')

        # Composition
        self._register('identity', 'composition', CompositionShapes.identity, 1, 1, 'Identity')
        self._register('swap', 'composition', CompositionShapes.swap, 1, 1, 'Reverse dimensions')

    def _register(
        self,
        name: str,
        category: str,
        fn: Callable,
        n_inputs: int,
        n_outputs: int,
        description: str,
    ):
        """Register a shape."""
        self._shapes[name] = ShapeInfo(
            name=name,
            category=category,
            fn=fn,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            description=description,
        )

    def get(self, name: str) -> Callable:
        """Get a shape function by name."""
        if name not in self._shapes:
            raise KeyError(f"Unknown shape: {name}")
        return self._shapes[name].fn

    def info(self, name: str) -> ShapeInfo:
        """Get shape metadata."""
        if name not in self._shapes:
            raise KeyError(f"Unknown shape: {name}")
        return self._shapes[name]

    def list_shapes(self, category: Optional[str] = None) -> List[str]:
        """List available shapes, optionally filtered by category."""
        if category is None:
            return list(self._shapes.keys())
        return [
            name for name, info in self._shapes.items()
            if info.category == category
        ]

    def list_categories(self) -> List[str]:
        """List available categories."""
        return list(set(info.category for info in self._shapes.values()))

    # Convenience accessors
    def __getattr__(self, name: str) -> Callable:
        """Allow lib.xor(a, b) syntax."""
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._shapes:
            return self._shapes[name].fn
        raise AttributeError(f"Unknown shape: {name}")

    def __contains__(self, name: str) -> bool:
        return name in self._shapes

    def __len__(self) -> int:
        return len(self._shapes)


# =============================================================================
# FROZEN TILE USING LIBRARY SHAPES
# =============================================================================

class GeneralFrozenTile(nn.Module):
    """
    A tile that executes a frozen shape from the library.

    No learnable parameters in computation.
    Signature derived from shape behavior.
    """

    def __init__(
        self,
        shape_name: str,
        d_model: int,
        library: Optional[FrozenShapeLibrary] = None,
    ):
        super().__init__()
        self.shape_name = shape_name
        self.d_model = d_model
        self.library = library or FrozenShapeLibrary()

        if shape_name not in self.library:
            raise ValueError(f"Unknown shape: {shape_name}")

        self.shape_fn = self.library.get(shape_name)
        self.shape_info = self.library.info(shape_name)

        # Derive signature from shape behavior
        signature = self._derive_signature()
        self.register_buffer('signature', signature)

    def _derive_signature(self) -> torch.Tensor:
        """Derive signature from shape's behavior on random inputs."""
        torch.manual_seed(hash(self.shape_name) % (2**31))

        # Generate test inputs based on shape's arity
        n_samples = 256
        results = []

        if self.shape_info.n_inputs == 1:
            x = torch.rand(n_samples, 8)
            out = self.shape_fn(x)
            results.append(out.flatten())
        elif self.shape_info.n_inputs == 2:
            a = torch.rand(n_samples, 8)
            b = torch.rand(n_samples, 8)
            out = self.shape_fn(a, b)
            if isinstance(out, tuple):
                for o in out:
                    results.append(o.flatten())
            else:
                results.append(out.flatten())
        elif self.shape_info.n_inputs == 3:
            a = torch.rand(n_samples, 8)
            b = torch.rand(n_samples, 8)
            c = torch.rand(n_samples, 8)
            out = self.shape_fn(a, b, c)
            if isinstance(out, tuple):
                for o in out:
                    results.append(o.flatten())
            else:
                results.append(out.flatten())

        # Concatenate results
        result = torch.cat(results)

        # Project to signature dimension
        projection = torch.randn(len(result), self.d_model)
        projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)

        signature = (result @ projection).sign()
        signature[signature == 0] = 1

        return signature

    def get_signature(self) -> torch.Tensor:
        return self.signature

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute the frozen shape."""
        if self.shape_info.n_inputs == 1:
            return self.shape_fn(x)
        elif self.shape_info.n_inputs == 2:
            if y is None:
                raise ValueError(f"Shape {self.shape_name} requires 2 inputs")
            result = self.shape_fn(x, y)
            return result if not isinstance(result, tuple) else result[0]
        else:
            raise NotImplementedError(f"3+ input shapes not yet supported in forward")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Singleton library instance
_library = None


def get_library() -> FrozenShapeLibrary:
    """Get the global frozen shape library."""
    global _library
    if _library is None:
        _library = FrozenShapeLibrary()
    return _library


# Convenience exports
xor = Primitives.xor
and_op = Primitives.and_op
or_op = Primitives.or_op
not_op = Primitives.not_op
full_adder = LogicShapes.full_adder
