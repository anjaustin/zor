"""
Geocadesia: Pooling Kingdom

Reduction operations that summarize.
Trade resolution for robustness.

"Pooling shapes reduce dimensionality while preserving important information."
"""

from __future__ import annotations
from typing import List
from .catalog import shape, Shape, Kingdom, Arity, ShapeType, FrozenStatus


@shape(
    name="max_pool",
    kingdom=Kingdom.POOLING,
    formula="max(x)",
    definition="Take the maximum value.",
    arity=Arity.NARY,
    see_also=["avg_pool", "min_pool"],
)
def max_pool(x: List[float]) -> float:
    """Maximum pooling."""
    return max(x)


class MaxPool:
    """
    Maximum Pooling

    Takes the maximum value from a region.
    Preserves the strongest activation.
    """

    def __call__(self, x: List[float]) -> float:
        return max_pool(x)

    def __repr__(self):
        return "<MaxPool: max(x)>"


@shape(
    name="avg_pool",
    kingdom=Kingdom.POOLING,
    formula="mean(x)",
    definition="Take the mean value.",
    arity=Arity.NARY,
    see_also=["max_pool", "sum_pool"],
)
def avg_pool(x: List[float]) -> float:
    """Average pooling."""
    return sum(x) / len(x)


class AvgPool:
    """
    Average Pooling

    Takes the mean of all values.
    Smoother than max pooling.
    """

    def __call__(self, x: List[float]) -> float:
        return avg_pool(x)

    def __repr__(self):
        return "<AvgPool: mean(x)>"


@shape(
    name="sum_pool",
    kingdom=Kingdom.POOLING,
    formula="Σx",
    definition="Sum all values.",
    arity=Arity.NARY,
    see_also=["avg_pool"],
)
def sum_pool(x: List[float]) -> float:
    """Sum pooling."""
    return sum(x)


class SumPool:
    """
    Sum Pooling

    Sums all values. Unlike avg_pool, not normalized.
    Used when magnitude matters.
    """

    def __call__(self, x: List[float]) -> float:
        return sum_pool(x)

    def __repr__(self):
        return "<SumPool: Σx>"


@shape(
    name="min_pool",
    kingdom=Kingdom.POOLING,
    formula="min(x)",
    definition="Take the minimum value.",
    arity=Arity.NARY,
    see_also=["max_pool"],
)
def min_pool(x: List[float]) -> float:
    """Minimum pooling."""
    return min(x)


class MinPool:
    """
    Minimum Pooling

    Takes the minimum value.
    Less common than max pooling.
    """

    def __call__(self, x: List[float]) -> float:
        return min_pool(x)

    def __repr__(self):
        return "<MinPool: min(x)>"


# =============================================================================
# Index-Returning Reductions (FrozenDB Support)
# =============================================================================

@shape(
    name="argmin",
    kingdom=Kingdom.POOLING,
    formula="argmin(x) = index of min(x)",
    definition="Return the index of the minimum value.",
    arity=Arity.NARY,
    see_also=["min_pool", "argmax"],
    examples=[
        ([3, 1, 4, 1, 5], 1),
        ([5, 4, 3, 2, 1], 4),
        ([1], 0),
    ],
)
def argmin(x: List[float]) -> int:
    """
    Index of minimum value.

    Essential for FrozenDB: after computing Hamming distances
    to all signatures, argmin finds the closest match.
    """
    min_idx = 0
    min_val = x[0]
    for i, val in enumerate(x):
        if val < min_val:
            min_val = val
            min_idx = i
    return min_idx


class Argmin:
    """
    Argmin: Index of Minimum

    Returns the INDEX of the minimum value, not the value itself.
    The final step in FrozenDB vector search.

    hamming_distances = [12, 5, 8, 3, 15]
    argmin(hamming_distances) = 3  # Index of closest match
    """

    def __call__(self, x: List[float]) -> int:
        return argmin(x)

    @staticmethod
    def with_value(x: List[float]) -> tuple:
        """Return both index and value."""
        idx = argmin(x)
        return idx, x[idx]

    @staticmethod
    def top_k(x: List[float], k: int) -> List[int]:
        """Return indices of k smallest values."""
        indexed = [(val, i) for i, val in enumerate(x)]
        indexed.sort()
        return [i for _, i in indexed[:k]]

    def __repr__(self):
        return "<Argmin: index of min(x)>"


@shape(
    name="argmax",
    kingdom=Kingdom.POOLING,
    formula="argmax(x) = index of max(x)",
    definition="Return the index of the maximum value.",
    arity=Arity.NARY,
    see_also=["max_pool", "argmin"],
    examples=[
        ([3, 1, 4, 1, 5], 4),
        ([5, 4, 3, 2, 1], 0),
        ([1], 0),
    ],
)
def argmax(x: List[float]) -> int:
    """
    Index of maximum value.

    Used when searching by similarity (higher = better)
    rather than distance (lower = better).
    """
    max_idx = 0
    max_val = x[0]
    for i, val in enumerate(x):
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx


class Argmax:
    """
    Argmax: Index of Maximum

    Returns the INDEX of the maximum value.
    Used for similarity-based search.

    similarity_scores = [0.2, 0.8, 0.5, 0.9, 0.3]
    argmax(similarity_scores) = 3  # Index of best match
    """

    def __call__(self, x: List[float]) -> int:
        return argmax(x)

    @staticmethod
    def with_value(x: List[float]) -> tuple:
        """Return both index and value."""
        idx = argmax(x)
        return idx, x[idx]

    @staticmethod
    def top_k(x: List[float], k: int) -> List[int]:
        """Return indices of k largest values."""
        indexed = [(val, i) for i, val in enumerate(x)]
        indexed.sort(reverse=True)
        return [i for _, i in indexed[:k]]

    def __repr__(self):
        return "<Argmax: index of max(x)>"
