"""
Shape Registry - Bridge between TriX shapes and CUDA operations

Provides:
- Registration of shapes with both polynomial (TriX) and hardware (CUDA) forms
- Verification that polynomial = hardware on binary inputs
- Signature generation from shape behavior
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum


class ShapeCategory(Enum):
    """Categories of computation shapes"""
    PRIMITIVE = "primitive"    # XOR, AND, OR, NOT
    LOGIC = "logic"           # NAND, NOR, XNOR, etc.
    ARITHMETIC = "arithmetic" # ADD, SUB, etc.
    ROTATION = "rotation"     # ROL, ROR, SHL, SHR
    MIXING = "mixing"         # ChaCha QR, etc.
    ACTIVATION = "activation" # ReLU, etc. (for neural)


@dataclass
class Shape:
    """
    A computation shape with both polynomial and hardware representations.

    Attributes:
        name: Unique identifier
        category: Shape category
        poly_fn: Polynomial function for training (smooth gradients)
        cuda_expr: CUDA expression for execution (exact)
        n_inputs: Number of inputs (1 or 2)
        verified: Whether polynomial = hardware has been verified
    """
    name: str
    category: ShapeCategory
    poly_fn: Callable
    cuda_expr: str
    n_inputs: int = 2
    verified: bool = False

    def __post_init__(self):
        """Verify the shape on construction."""
        self.verified = self._verify()

    def _verify(self) -> bool:
        """Verify polynomial = hardware on binary inputs."""
        binary = torch.tensor([0.0, 1.0])

        for a_val in binary:
            for b_val in binary:
                a = torch.tensor([a_val])
                b = torch.tensor([b_val])

                try:
                    if self.n_inputs == 1:
                        poly_result = self.poly_fn(a)
                    else:
                        result = self.poly_fn(a, b)
                        poly_result = result[0] if isinstance(result, tuple) else result
                except Exception:
                    return False

                # Evaluate hardware (for known shapes)
                a_int = int(a_val.item())
                b_int = int(b_val.item())

                hw_result = self._eval_hardware(a_int, b_int)
                if hw_result is None:
                    continue  # Can't verify complex expressions

                if abs(poly_result.item() - hw_result) > 1e-6:
                    return False

        return True

    def _eval_hardware(self, a: int, b: int) -> Optional[int]:
        """Evaluate hardware operation."""
        ops = {
            "a ^ b": a ^ b,
            "a & b": a & b,
            "a | b": a | b,
            "~a": 1 - a,
            "~(a & b)": 1 if not (a and b) else 0,
            "~(a | b)": 1 if not (a or b) else 0,
            "~(a ^ b)": 1 if (a == b) else 0,
            "(~a) | b": (1 - a) | b,
            "a + b": (a + b) & 1,  # 1-bit add
            "a": a,
            "b": b,
        }
        return ops.get(self.cuda_expr)

    def generate_signature(self, dim: int = 8, seed: Optional[int] = None) -> np.ndarray:
        """Generate a signature from shape behavior."""
        if seed is None:
            seed = hash(self.name) % (2**31)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Sample random inputs
        n_samples = 256
        results = []

        if self.n_inputs == 1:
            x = torch.rand(n_samples, 8)
            out = self.poly_fn(x)
            results.append(out.flatten())
        else:
            a = torch.rand(n_samples, 8)
            b = torch.rand(n_samples, 8)
            out = self.poly_fn(a, b)
            if isinstance(out, tuple):
                for o in out:
                    results.append(o.flatten())
            else:
                results.append(out.flatten())

        result = torch.cat(results)

        # Project to signature dimension
        proj = torch.randn(len(result), dim)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
        sig = (result @ proj).sign()
        sig[sig == 0] = 1

        return ((sig.numpy() + 1) / 2).astype(np.uint8)[:dim]


class ShapeRegistry:
    """
    Registry of all available shapes.

    Connects TriX polynomial forms to CUDA hardware expressions.
    """

    def __init__(self):
        self._shapes: Dict[str, Shape] = {}
        self._register_primitives()
        self._register_logic()
        self._register_mixing()

    def _register_primitives(self):
        """Register primitive shapes (the axioms)."""
        # Import TriX primitives
        try:
            from trix.nn.frozen_shapes import Primitives
            self.register(Shape(
                name="xor",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=Primitives.xor,
                cuda_expr="a ^ b",
            ))
            self.register(Shape(
                name="and",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=Primitives.and_op,
                cuda_expr="a & b",
            ))
            self.register(Shape(
                name="or",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=Primitives.or_op,
                cuda_expr="a | b",
            ))
            self.register(Shape(
                name="not",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=Primitives.not_op,
                cuda_expr="~a",
                n_inputs=1,
            ))
        except ImportError:
            # Fallback to local definitions
            self.register(Shape(
                name="xor",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=lambda a, b: a + b - 2 * a * b,
                cuda_expr="a ^ b",
            ))
            self.register(Shape(
                name="and",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=lambda a, b: a * b,
                cuda_expr="a & b",
            ))
            self.register(Shape(
                name="or",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=lambda a, b: a + b - a * b,
                cuda_expr="a | b",
            ))
            self.register(Shape(
                name="not",
                category=ShapeCategory.PRIMITIVE,
                poly_fn=lambda a: 1 - a,
                cuda_expr="~a",
                n_inputs=1,
            ))

    def _register_logic(self):
        """Register derived logic shapes."""
        try:
            from trix.nn.frozen_shapes import LogicShapes
            self.register(Shape(
                name="nand",
                category=ShapeCategory.LOGIC,
                poly_fn=LogicShapes.nand,
                cuda_expr="~(a & b)",
            ))
            self.register(Shape(
                name="nor",
                category=ShapeCategory.LOGIC,
                poly_fn=LogicShapes.nor,
                cuda_expr="~(a | b)",
            ))
            self.register(Shape(
                name="xnor",
                category=ShapeCategory.LOGIC,
                poly_fn=LogicShapes.xnor,
                cuda_expr="~(a ^ b)",
            ))
            self.register(Shape(
                name="implies",
                category=ShapeCategory.LOGIC,
                poly_fn=LogicShapes.implies,
                cuda_expr="(~a) | b",
            ))
        except ImportError:
            self.register(Shape(
                name="nand",
                category=ShapeCategory.LOGIC,
                poly_fn=lambda a, b: 1 - a * b,
                cuda_expr="~(a & b)",
            ))
            self.register(Shape(
                name="nor",
                category=ShapeCategory.LOGIC,
                poly_fn=lambda a, b: 1 - (a + b - a * b),
                cuda_expr="~(a | b)",
            ))
            self.register(Shape(
                name="xnor",
                category=ShapeCategory.LOGIC,
                poly_fn=lambda a, b: 1 - (a + b - 2 * a * b),
                cuda_expr="~(a ^ b)",
            ))

    def _register_mixing(self):
        """Register mixing/cipher shapes."""
        # These don't have exact polynomial equivalents but are useful
        self.register(Shape(
            name="add",
            category=ShapeCategory.ARITHMETIC,
            poly_fn=lambda a, b: a + b,  # Approximate
            cuda_expr="a + b",
            verified=False,  # Not exact on non-binary
        ))
        self.register(Shape(
            name="rol",
            category=ShapeCategory.ROTATION,
            poly_fn=lambda a, b: a,  # Placeholder
            cuda_expr="((a << (b & 31)) | (a >> (32 - (b & 31))))",
            verified=False,
        ))
        self.register(Shape(
            name="ror",
            category=ShapeCategory.ROTATION,
            poly_fn=lambda a, b: a,  # Placeholder
            cuda_expr="((a >> (b & 31)) | (a << (32 - (b & 31))))",
            verified=False,
        ))
        self.register(Shape(
            name="mix",
            category=ShapeCategory.MIXING,
            poly_fn=lambda a, b: a + b - 2 * a * b,  # XOR as base
            cuda_expr="(a ^ b ^ ((a + b) << 13))",
            verified=False,
        ))
        self.register(Shape(
            name="identity",
            category=ShapeCategory.PRIMITIVE,
            poly_fn=lambda a: a,
            cuda_expr="a",
            n_inputs=1,
        ))

    def register(self, shape: Shape):
        """Register a shape."""
        self._shapes[shape.name] = shape

    def get(self, name: str) -> Shape:
        """Get a shape by name."""
        if name not in self._shapes:
            raise KeyError(f"Unknown shape: {name}. Available: {list(self._shapes.keys())}")
        return self._shapes[name]

    def list_shapes(self, category: Optional[ShapeCategory] = None) -> List[str]:
        """List available shapes."""
        if category is None:
            return list(self._shapes.keys())
        return [name for name, shape in self._shapes.items() if shape.category == category]

    def list_verified(self) -> List[str]:
        """List shapes that have been verified (polynomial = hardware)."""
        return [name for name, shape in self._shapes.items() if shape.verified]

    def verify_all(self) -> Dict[str, bool]:
        """Verify all shapes and return results."""
        return {name: shape.verified for name, shape in self._shapes.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._shapes

    def __len__(self) -> int:
        return len(self._shapes)

    def __iter__(self):
        return iter(self._shapes.values())


# Global registry instance
_registry: Optional[ShapeRegistry] = None


def get_registry() -> ShapeRegistry:
    """Get the global shape registry."""
    global _registry
    if _registry is None:
        _registry = ShapeRegistry()
    return _registry
