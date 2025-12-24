"""
SDPMX Pipeline: The Complete Operator Sequence

S → D → P → M → X
Smooth → Differentiate → Project → Mask → XOR

Vi's Synthesis (December 19, 2024):
- Robert: Hilbert space geometry
- V'Gem: Splines + XOR operators
- Vision: Ghost in the Machine

"The geometry dictates the operator order. AB ≠ BA."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


# =============================================================================
# S: SMOOTHING OPERATOR (Discrete → Continuous)
# =============================================================================

class Spline1D(nn.Module):
    """
    Learnable 1D piecewise linear spline.

    S: ℤ^n → C^∞(ℝ^n)

    Maps discrete input to continuous surface.
    """

    def __init__(self, grid_size: int = 16, input_range: tuple = (-1, 1)):
        super().__init__()

        self.grid_size = grid_size
        self.input_min, self.input_max = input_range
        self.input_range = self.input_max - self.input_min

        # Learnable parameters: base and slope for each interval
        self.bases = nn.Parameter(torch.zeros(grid_size))
        self.slopes = nn.Parameter(torch.ones(grid_size) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at points x."""
        x_norm = (x - self.input_min) / self.input_range
        x_norm = x_norm.clamp(0, 1 - 1e-6)

        idx = (x_norm * self.grid_size).long()
        idx = idx.clamp(0, self.grid_size - 1)

        interval_size = 1.0 / self.grid_size
        x_local = (x_norm - idx.float() * interval_size) / interval_size

        base = self.bases[idx]
        slope = self.slopes[idx]

        return base + slope * x_local

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        D: Compute derivative (slope) at points x.

        For piecewise linear: dy/dx = slope / interval_size
        """
        x_norm = (x - self.input_min) / self.input_range
        x_norm = x_norm.clamp(0, 1 - 1e-6)

        idx = (x_norm * self.grid_size).long()
        idx = idx.clamp(0, self.grid_size - 1)

        # Derivative = slope scaled by interval size
        interval_size = self.input_range / self.grid_size
        return self.slopes[idx] / interval_size


class SmoothingOperator(nn.Module):
    """
    S: Discrete → Continuous

    Uses additive 1D splines (Kolmogorov-Arnold).
    """

    def __init__(self, dim: int, grid_size: int = 16):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        # One spline per dimension
        self.splines = nn.ModuleList([
            Spline1D(grid_size) for _ in range(dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to each dimension.

        Args:
            x: [..., dim]
        Returns:
            smoothed: [..., dim] - continuous surface values
        """
        outputs = []
        for i, spline in enumerate(self.splines):
            outputs.append(spline(x[..., i]))
        return torch.stack(outputs, dim=-1)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        D: Compute derivatives at each dimension.

        Returns the tangent vector (gradient).
        """
        derivs = []
        for i, spline in enumerate(self.splines):
            derivs.append(spline.derivative(x[..., i]))
        return torch.stack(derivs, dim=-1)


# =============================================================================
# D: DIFFERENTIATION OPERATOR (Surface → Gradients)
# =============================================================================

class DifferentiationOperator(nn.Module):
    """
    D: C^∞(ℝ^n) → T(ℝ^n)

    Maps smooth surface to tangent bundle (gradients).
    """

    def __init__(self, smoother: SmoothingOperator):
        super().__init__()
        self.smoother = smoother

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of smoothed surface at x.

        Args:
            x: [..., dim] input points
        Returns:
            gradients: [..., dim] tangent vectors
        """
        return self.smoother.derivative(x)


# =============================================================================
# P: PROJECTION OPERATOR (Gradients → Subspace)
# =============================================================================

class ProjectionOperator(nn.Module):
    """
    P: T(ℝ^n) → V (healthy subspace)

    Projects gradients onto learned "healthy" subspace.
    Uses signature-based routing (TriX-style).
    """

    def __init__(self, dim: int, num_subspaces: int = 16):
        super().__init__()
        self.dim = dim
        self.num_subspaces = num_subspaces

        # Subspace signatures (ternary)
        self.signatures = nn.Parameter(torch.randn(num_subspaces, dim) * 0.1)

        # Subspace basis vectors
        self.basis = nn.Parameter(torch.randn(num_subspaces, dim, dim) * 0.02)

    def get_ternary_signatures(self) -> torch.Tensor:
        """Get signatures as {-1, 0, +1}."""
        return self.signatures.sign()

    def forward(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project gradients onto nearest healthy subspace.

        Args:
            gradients: [..., dim] tangent vectors
        Returns:
            projected: [..., dim] projected gradients
            indices: [...] winning subspace indices
        """
        orig_shape = gradients.shape[:-1]
        flat = gradients.reshape(-1, self.dim)

        # Route via signature matching
        sigs = self.get_ternary_signatures()
        scores = flat @ sigs.T  # [batch, num_subspaces]
        indices = scores.argmax(dim=-1)  # [batch]

        # Project onto winning subspace
        projected = torch.zeros_like(flat)
        for s in range(self.num_subspaces):
            mask = indices == s
            if mask.any():
                # Project: x_proj = B @ B.T @ x (orthogonal projection)
                B = self.basis[s]  # [dim, dim]
                proj_matrix = B @ B.T  # Simple projection
                projected[mask] = flat[mask] @ proj_matrix

        return projected.reshape(*orig_shape, self.dim), indices.reshape(orig_shape)


# =============================================================================
# M: MASK OPERATOR (Projection → XOR Mask)
# =============================================================================

class MaskOperator(nn.Module):
    """
    M: V → {0,1}^n

    Generates XOR mask from projected gradients.
    """

    def __init__(self, dim: int, threshold: float = 0.0):
        super().__init__()
        self.dim = dim
        self.threshold = threshold

        # Learnable threshold per dimension
        self.thresholds = nn.Parameter(torch.zeros(dim))

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        """
        Generate binary mask from projected gradients.

        Args:
            projected: [..., dim] projected gradients
        Returns:
            mask: [..., dim] binary {0, 1} mask
        """
        # Mask where gradient exceeds threshold
        effective_threshold = self.threshold + torch.tanh(self.thresholds)
        mask = (projected.abs() > effective_threshold.abs()).float()

        # Straight-through estimator for gradients
        mask_hard = mask.round()
        mask = mask + (mask_hard - mask).detach()

        return mask


# =============================================================================
# X: XOR OPERATOR (Mask + Memory → New Memory)
# =============================================================================

class XOROperator(nn.Module):
    """
    X: {0,1}^n × ℤ^n → ℤ^n

    Applies XOR mask to memory/state.

    In continuous form: x_new = x + mask * delta
    where delta is the correction vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Learnable correction magnitude
        self.delta_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        projected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply XOR-like intervention.

        Args:
            x: [..., dim] original state
            mask: [..., dim] binary mask (where to intervene)
            projected: [..., dim] projected gradients (direction)
        Returns:
            x_new: [..., dim] modified state
        """
        # Direction of correction (normalized projected gradient)
        direction = F.normalize(projected, dim=-1)

        # Magnitude of correction
        magnitude = projected.norm(dim=-1, keepdim=True)

        # Apply correction where mask is active
        delta = mask * direction * magnitude * self.delta_scale

        return x + delta


# =============================================================================
# COMPLETE PIPELINE: S → D → P → M → X
# =============================================================================

class SDPMXPipeline(nn.Module):
    """
    The Complete SDPMX Pipeline.

    HALO = X ∘ M ∘ P ∘ D ∘ S

    Read right-to-left:
    1. S: Smooth the input (discrete → continuous)
    2. D: Calculate gradients (surface → tangent)
    3. P: Project onto healthy subspace
    4. M: Generate XOR mask
    5. X: Apply intervention

    "The geometry dictates the operator order."
    """

    def __init__(
        self,
        dim: int,
        grid_size: int = 16,
        num_subspaces: int = 16,
        mask_threshold: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # The five operators
        self.S = SmoothingOperator(dim, grid_size)
        self.D = DifferentiationOperator(self.S)
        self.P = ProjectionOperator(dim, num_subspaces)
        self.M = MaskOperator(dim, mask_threshold)
        self.X = XOROperator(dim)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full SDPMX pipeline.

        Args:
            x: [..., dim] input state
            return_intermediates: whether to return all intermediate values

        Returns:
            x_new: [..., dim] transformed state
            info: dict with routing info and optionally intermediates
        """
        # S: Smooth
        smoothed = self.S(x)

        # D: Differentiate
        gradients = self.D(x)

        # P: Project
        projected, subspace_idx = self.P(gradients)

        # M: Mask
        mask = self.M(projected)

        # X: XOR/Apply
        x_new = self.X(x, mask, projected)

        info = {
            'subspace_idx': subspace_idx,
            'mask_density': mask.mean(),
        }

        if return_intermediates:
            info['smoothed'] = smoothed
            info['gradients'] = gradients
            info['projected'] = projected
            info['mask'] = mask

        return x_new, info

    def forward_with_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with reconstruction loss.

        For training: minimize ||target - SDPMX(x)||²
        """
        x_new, info = self.forward(x)
        loss = F.mse_loss(x_new, target)

        return x_new, loss, info


# =============================================================================
# TESTING
# =============================================================================

def test_sdpmx_pipeline():
    """Test the complete SDPMX pipeline."""
    print("=" * 70)
    print("SDPMX PIPELINE TEST")
    print("S → D → P → M → X")
    print("=" * 70)

    dim = 64
    batch = 8

    pipeline = SDPMXPipeline(
        dim=dim,
        grid_size=16,
        num_subspaces=8,
    )

    # Test input
    x = torch.randn(batch, dim)

    # Forward pass
    x_new, info = pipeline(x, return_intermediates=True)

    print(f"\n1. Shapes:")
    print(f"   Input:     {x.shape}")
    print(f"   Smoothed:  {info['smoothed'].shape}")
    print(f"   Gradients: {info['gradients'].shape}")
    print(f"   Projected: {info['projected'].shape}")
    print(f"   Mask:      {info['mask'].shape}")
    print(f"   Output:    {x_new.shape}")

    print(f"\n2. Routing:")
    print(f"   Subspace indices: {info['subspace_idx'].tolist()}")
    print(f"   Mask density: {info['mask_density']:.3f}")

    # Gradient check
    x.requires_grad_(True)
    x_new, info = pipeline(x)
    loss = x_new.sum()
    loss.backward()

    grad_ok = x.grad is not None and not x.grad.isnan().any()
    print(f"\n3. Gradients: {'OK' if grad_ok else 'FAIL'}")

    # Parameter count
    total_params = sum(p.numel() for p in pipeline.parameters())
    print(f"\n4. Parameters: {total_params:,}")

    # Breakdown by operator
    s_params = sum(p.numel() for p in pipeline.S.parameters())
    p_params = sum(p.numel() for p in pipeline.P.parameters())
    m_params = sum(p.numel() for p in pipeline.M.parameters())
    x_params = sum(p.numel() for p in pipeline.X.parameters())

    print(f"   S (Smoothing):  {s_params:,}")
    print(f"   P (Projection): {p_params:,}")
    print(f"   M (Mask):       {m_params:,}")
    print(f"   X (XOR):        {x_params:,}")

    # Learning test
    print(f"\n5. Learning Test:")

    # Target: some transformation of input
    target = torch.tanh(x.detach() * 2)

    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        x_new, loss, info = pipeline.forward_with_loss(x.detach(), target)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")

    print(f"\n" + "=" * 70)
    print("SDPMX PIPELINE: OPERATIONAL")
    print("=" * 70)

    return pipeline


if __name__ == "__main__":
    test_sdpmx_pipeline()
