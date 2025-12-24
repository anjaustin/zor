"""
Frozen Mixing Shapes for Token-to-Token Interaction.

Mesa 16.1: The Sacred Foundry

These shapes define HOW tokens mix with their partners.
Each shape is a frozen mathematical function - no learnable parameters.

Shapes:
    copy: a → a (ignore partner)
    zero: → 0 (ignore both)
    gate: g*a + (1-g)*b (weighted blend, g from routing)
    avg: (a + b) / 2 (average)
    diff: a - b (difference)
    add: a + b (sum)
    max_mix: max(a, b) (element-wise max)
    min_mix: min(a, b) (element-wise min)
    hadamard: a * b (element-wise product)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MixingShapeInfo:
    """Metadata about a mixing shape."""
    name: str
    n_inputs: int  # 1 = unary (ignore partner), 2 = binary
    description: str
    formula: str
    commutative: bool  # Is f(a,b) == f(b,a)?


class MixingShapes:
    """
    Library of frozen mixing operations.

    Each shape takes two token tensors and returns the mixed result.
    All shapes are frozen - no learnable parameters.
    """

    @staticmethod
    def copy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Copy: ignore partner, return self. f(a,b) = a"""
        return a

    @staticmethod
    def zero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Zero: return zeros. f(a,b) = 0"""
        return torch.zeros_like(a)

    @staticmethod
    def avg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Average: mean of both. f(a,b) = (a + b) / 2"""
        return (a + b) / 2

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add: sum of both. f(a,b) = a + b"""
        return a + b

    @staticmethod
    def diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Difference: a minus b. f(a,b) = a - b"""
        return a - b

    @staticmethod
    def hadamard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hadamard: element-wise product. f(a,b) = a * b"""
        return a * b

    @staticmethod
    def max_mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Max: element-wise maximum. f(a,b) = max(a, b)"""
        return torch.maximum(a, b)

    @staticmethod
    def min_mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Min: element-wise minimum. f(a,b) = min(a, b)"""
        return torch.minimum(a, b)

    @staticmethod
    def gate(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Gate: weighted blend. f(a,b,g) = g*a + (1-g)*b

        Args:
            a: First token
            b: Second token (partner)
            g: Gate weight in [0, 1]
        """
        return g * a + (1 - g) * b

    @staticmethod
    def gate_sigmoid(a: torch.Tensor, b: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
        """
        Gate with sigmoid: blend controlled by logit.
        f(a,b,logit) = sigmoid(logit)*a + (1-sigmoid(logit))*b
        """
        g = torch.sigmoid(logit)
        return g * a + (1 - g) * b

    @staticmethod
    def interpolate(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """
        Interpolate: fixed blend ratio. f(a,b) = t*a + (1-t)*b
        """
        return t * a + (1 - t) * b

    @staticmethod
    def softmax_blend(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Softmax blend: weight by relative magnitude.
        Larger values get more weight.
        """
        weights = F.softmax(torch.stack([a, b], dim=-1), dim=-1)
        return weights[..., 0] * a + weights[..., 1] * b


class FrozenMixingLibrary:
    """
    Registry of frozen mixing shapes.

    Provides:
        - Shape lookup by name
        - Signature derivation for routing
        - Shape metadata
    """

    def __init__(self):
        self._shapes: Dict[str, Callable] = {}
        self._info: Dict[str, MixingShapeInfo] = {}
        self._register_core_shapes()

    def _register_core_shapes(self):
        """Register the core mixing shapes."""

        # Unary (ignore partner)
        self.register(
            'copy',
            MixingShapes.copy,
            MixingShapeInfo(
                name='copy',
                n_inputs=1,
                description='Identity - return self, ignore partner',
                formula='f(a,b) = a',
                commutative=False,
            )
        )

        self.register(
            'zero',
            MixingShapes.zero,
            MixingShapeInfo(
                name='zero',
                n_inputs=1,
                description='Zero - ignore both inputs',
                formula='f(a,b) = 0',
                commutative=True,
            )
        )

        # Binary symmetric
        self.register(
            'avg',
            MixingShapes.avg,
            MixingShapeInfo(
                name='avg',
                n_inputs=2,
                description='Average of both tokens',
                formula='f(a,b) = (a + b) / 2',
                commutative=True,
            )
        )

        self.register(
            'add',
            MixingShapes.add,
            MixingShapeInfo(
                name='add',
                n_inputs=2,
                description='Sum of both tokens',
                formula='f(a,b) = a + b',
                commutative=True,
            )
        )

        self.register(
            'hadamard',
            MixingShapes.hadamard,
            MixingShapeInfo(
                name='hadamard',
                n_inputs=2,
                description='Element-wise product',
                formula='f(a,b) = a * b',
                commutative=True,
            )
        )

        self.register(
            'max',
            MixingShapes.max_mix,
            MixingShapeInfo(
                name='max',
                n_inputs=2,
                description='Element-wise maximum',
                formula='f(a,b) = max(a, b)',
                commutative=True,
            )
        )

        self.register(
            'min',
            MixingShapes.min_mix,
            MixingShapeInfo(
                name='min',
                n_inputs=2,
                description='Element-wise minimum',
                formula='f(a,b) = min(a, b)',
                commutative=True,
            )
        )

        # Binary asymmetric
        self.register(
            'diff',
            MixingShapes.diff,
            MixingShapeInfo(
                name='diff',
                n_inputs=2,
                description='Difference (a minus b)',
                formula='f(a,b) = a - b',
                commutative=False,
            )
        )

    def register(self, name: str, shape: Callable, info: MixingShapeInfo):
        """Register a new mixing shape."""
        self._shapes[name] = shape
        self._info[name] = info

    def get(self, name: str) -> Callable:
        """Get a shape function by name."""
        if name not in self._shapes:
            raise KeyError(f"Unknown mixing shape: {name}")
        return self._shapes[name]

    def info(self, name: str) -> MixingShapeInfo:
        """Get shape metadata."""
        if name not in self._info:
            raise KeyError(f"Unknown mixing shape: {name}")
        return self._info[name]

    def list_shapes(self) -> List[str]:
        """List all available shape names."""
        return list(self._shapes.keys())

    def derive_signature(self, name: str, d_model: int) -> torch.Tensor:
        """
        Derive a deterministic signature for a shape.

        The signature is derived from the shape's behavior on probe inputs.
        """
        shape = self.get(name)
        info = self.info(name)

        # Use deterministic probe vectors
        torch.manual_seed(hash(name) % (2**32))
        probe_a = torch.randn(1, d_model)
        probe_b = torch.randn(1, d_model)

        # Get shape's response
        if info.n_inputs == 1:
            response = shape(probe_a, probe_b)  # Will ignore b
        else:
            response = shape(probe_a, probe_b)

        # Signature is sign of response
        signature = response.sign().squeeze(0)

        return signature

    def get_all_signatures(self, d_model: int) -> torch.Tensor:
        """Get signatures for all shapes stacked."""
        sigs = []
        for name in self.list_shapes():
            sigs.append(self.derive_signature(name, d_model))
        return torch.stack(sigs)


# Singleton instance
_mixing_library = None


def get_mixing_library() -> FrozenMixingLibrary:
    """Get the singleton mixing library."""
    global _mixing_library
    if _mixing_library is None:
        _mixing_library = FrozenMixingLibrary()
    return _mixing_library
