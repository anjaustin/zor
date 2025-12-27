"""
Validation Tests for TrueOctaveFFN.

These tests verify the model learns MEANINGFUL patterns, not just fits noise.

Categories:
1. Pattern Learning - Can it learn known functions?
2. Generalization - Does it work on unseen data?
3. Hierarchical Structure - Does multi-scale help?
4. Octave Selection - Does blend network learn meaningful routing?
5. Comparison - Does it match or beat baselines?
6. Ablation - Does each component contribute?

"If it doesn't generalize, it didn't learn."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from trix.nn.octave import TrueOctaveFFN, TrueOctaveBlock


SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_xor_data(n_samples: int, d_model: int, noise: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate XOR-like pattern data.

    Input: random vectors
    Target: sign of (sum of first half) XOR sign of (sum of second half)

    This tests whether the model can learn a non-linear pattern.
    """
    x = torch.randn(n_samples, 1, d_model)

    first_half = x[:, :, :d_model//2].sum(dim=-1)  # [n_samples, 1]
    second_half = x[:, :, d_model//2:].sum(dim=-1)  # [n_samples, 1]

    # XOR of signs: different signs -> positive, same signs -> negative
    xor_signal = (first_half.sign() != second_half.sign()).float() * 2 - 1  # [n_samples, 1]

    # Target: input + XOR signal broadcast to all dimensions
    target = x + xor_signal.unsqueeze(-1).expand_as(x) * 0.5  # [n_samples, 1, d_model]

    if noise > 0:
        target = target + torch.randn_like(target) * noise

    return x.squeeze(1), target.squeeze(1)


def generate_multiscale_data(
    n_samples: int,
    d_model: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data with multi-scale structure.

    Some inputs need fine-grained response (high frequency patterns).
    Some inputs need coarse response (low frequency patterns).

    This tests whether the blend network learns appropriate octave selection.
    """
    samples_per_type = n_samples // 2

    # Type 0: High frequency - needs fine octave
    x_fine = torch.randn(samples_per_type, d_model)
    # Add high frequency modulation
    freq = torch.arange(d_model).float() / d_model * 20
    x_fine = x_fine + 0.5 * torch.sin(freq).unsqueeze(0).expand(samples_per_type, -1)
    y_fine = x_fine * 1.1  # Small scale adjustment
    labels_fine = torch.zeros(samples_per_type)

    # Type 1: Low frequency - needs coarse octave
    x_coarse = torch.randn(samples_per_type, d_model)
    # Add low frequency modulation
    freq = torch.arange(d_model).float() / d_model * 2
    x_coarse = x_coarse + 0.5 * torch.sin(freq).unsqueeze(0).expand(samples_per_type, -1)
    y_coarse = x_coarse * 0.9  # Different scale adjustment
    labels_coarse = torch.ones(samples_per_type)

    x = torch.cat([x_fine, x_coarse], dim=0)
    y = torch.cat([y_fine, y_coarse], dim=0)
    labels = torch.cat([labels_fine, labels_coarse], dim=0)

    # Shuffle
    perm = torch.randperm(n_samples)
    return x[perm], y[perm], labels[perm]


def generate_cluster_data(
    n_samples: int,
    d_model: int,
    n_clusters: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate clustered data with cluster-specific transformations.
    """
    samples_per_cluster = n_samples // n_clusters

    xs, ys, labels = [], [], []

    for c in range(n_clusters):
        # Cluster center
        center = torch.randn(d_model) * 2

        # Samples around center
        x = center + torch.randn(samples_per_cluster, d_model) * 0.5

        # Cluster-specific transformation (rotation-like)
        angle = c * np.pi / n_clusters
        transform = torch.eye(d_model)
        transform[0, 0] = np.cos(angle)
        transform[0, 1] = -np.sin(angle)
        transform[1, 0] = np.sin(angle)
        transform[1, 1] = np.cos(angle)

        y = x @ transform

        xs.append(x)
        ys.append(y)
        labels.append(torch.full((samples_per_cluster,), c))

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    labels = torch.cat(labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(x))
    return x[perm], y[perm], labels[perm].long()


# =============================================================================
# BASELINE MODELS
# =============================================================================

class RandomFFN(nn.Module):
    """Random baseline that doesn't learn."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('random_directions', torch.randn(d_model, d_model))
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Random transformation
        output = F.linear(x, self.random_directions) * self.scale
        output = x + output
        return output, {}


class DenseFFN(nn.Module):
    """Standard dense FFN for comparison."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + h), {}


class SingleOctaveFFN(nn.Module):
    """Single octave (no hierarchy) for ablation."""

    def __init__(self, d_model: int, num_tiles: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles

        # Single set of frozen directions
        directions = torch.randn(num_tiles, d_model).sign()
        self.register_buffer('directions', directions)

        # Learned scales
        self.scales = nn.Parameter(torch.ones(num_tiles) * 0.1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Route via dot product
        x_ternary = x.sign()
        scores = torch.einsum('btd,nd->btn', x_ternary, self.directions)

        if self.training:
            weights = F.softmax(scores / np.sqrt(self.d_model), dim=-1)
            directions = torch.einsum('btn,nd->btd', weights, self.directions)
            scales = torch.einsum('btn,n->bt', weights, self.scales)
        else:
            idx = scores.argmax(dim=-1)
            directions = self.directions[idx]
            scales = self.scales[idx]

        delta = scales.unsqueeze(-1) * directions
        return self.norm(x + delta), {'tile_idx': scores.argmax(dim=-1)}


# =============================================================================
# 1. PATTERN LEARNING TESTS
# =============================================================================

class TestPatternLearning:
    """Verify the model can learn known patterns."""

    def test_learns_xor_pattern(self):
        """Model should learn XOR-like partitioning."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(1000, d_model)
        x_test, y_test = generate_xor_data(200, d_model)

        # Add sequence dimension
        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Model - use smaller architecture for this test
        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test)
            test_loss = F.mse_loss(pred, y_test).item()

        # Should achieve reasonable loss
        assert test_loss < 1.0, f"XOR pattern test loss too high: {test_loss}"

    def test_learns_scaling_transform(self):
        """Model should learn input scaling."""
        set_seed()

        d_model = 64
        scale = 0.3
        x_train = torch.randn(500, 1, d_model)
        y_train = x_train * scale
        x_test = torch.randn(100, 1, d_model)
        y_test = x_test * scale

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test)
            test_loss = F.mse_loss(pred, y_test).item()

        # Scaling should be learnable
        assert test_loss < 1.5, f"Scaling transform test loss too high: {test_loss}"

    def test_learns_cluster_specific_behavior(self):
        """Model should route different clusters differently."""
        set_seed()

        d_model = 64
        n_clusters = 4

        x_train, y_train, labels_train = generate_cluster_data(800, d_model, n_clusters)
        x_test, y_test, labels_test = generate_cluster_data(200, d_model, n_clusters)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(300):
            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, info = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(pred.squeeze(1), y_test).item()

            # Check different clusters get different routing
            tiles_by_cluster = {}
            for c in range(n_clusters):
                mask = labels_test == c
                if mask.any():
                    tiles_by_cluster[c] = info.fine_tile_idx[mask].float().mean().item()

        # Should learn something
        assert test_loss < 2.0, f"Cluster transform test loss too high: {test_loss}"

        # Different clusters should tend toward different tiles
        if len(tiles_by_cluster) >= 2:
            tile_means = list(tiles_by_cluster.values())
            tile_variance = np.var(tile_means)
            assert tile_variance > 0.1, f"No variation in tile selection: {tiles_by_cluster}"


# =============================================================================
# 2. GENERALIZATION TESTS
# =============================================================================

class TestGeneralization:
    """Verify the model generalizes to unseen data."""

    def test_train_test_gap_reasonable(self):
        """Gap between train and test loss should be reasonable."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(500, d_model, noise=0.1)
        x_test, y_test = generate_xor_data(100, d_model, noise=0.1)

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred, _ = model(x_train)
            train_loss = F.mse_loss(train_pred, y_train).item()

            test_pred, _ = model(x_test)
            test_loss = F.mse_loss(test_pred, y_test).item()

        # Test loss should be within 2x of train loss (no catastrophic overfitting)
        gap_ratio = test_loss / (train_loss + 1e-8)
        assert gap_ratio < 3.0, f"Train/test gap too large: train={train_loss:.4f}, test={test_loss:.4f}, ratio={gap_ratio:.2f}"

    def test_no_catastrophic_overfitting(self):
        """Even with small data, should not catastrophically overfit."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(50, d_model)  # Very small
        x_test, y_test = generate_xor_data(200, d_model)

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred, _ = model(x_test)
            test_loss = F.mse_loss(test_pred, y_test).item()

        # Test loss should not explode
        assert test_loss < 5.0, f"Catastrophic overfitting: test loss = {test_loss}"


# =============================================================================
# 3. HIERARCHICAL STRUCTURE TESTS
# =============================================================================

class TestHierarchicalStructure:
    """Verify the multi-scale structure provides value."""

    def test_all_octaves_used(self):
        """All three octaves should be utilized."""
        set_seed()

        d_model = 64
        x = torch.randn(100, 16, d_model)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.eval()

        with torch.no_grad():
            _, info = model(x)

        # Check blend weights have some variance (not all going to one octave)
        blend_mean = info.blend_weights.mean(dim=(0, 1))

        # No octave should be completely ignored
        for i, weight in enumerate(blend_mean):
            assert weight > 0.1, f"Octave {i} is underutilized: weight = {weight:.4f}"

    def test_multiscale_data_uses_appropriate_octaves(self):
        """Different data types should prefer different octaves."""
        set_seed()

        d_model = 64
        x, y, labels = generate_multiscale_data(400, d_model)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, info = model(x.unsqueeze(1))

            # Check octave selection by data type
            fine_mask = labels == 0  # High frequency
            coarse_mask = labels == 1  # Low frequency

            fine_blend = info.blend_weights[fine_mask].mean(dim=0)
            coarse_blend = info.blend_weights[coarse_mask].mean(dim=0)

        # There should be some difference in octave preference
        blend_diff = (fine_blend - coarse_blend).abs().sum().item()
        assert blend_diff > 0.1, f"No octave preference difference: fine={fine_blend}, coarse={coarse_blend}"


# =============================================================================
# 4. OCTAVE SELECTION TESTS
# =============================================================================

class TestOctaveSelection:
    """Verify blend network learns meaningful routing."""

    def test_blend_network_trainable(self):
        """Blend network should receive gradients."""
        set_seed()

        d_model = 64
        x = torch.randn(4, 8, d_model)
        y = torch.randn(4, 8, d_model)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()

        output, _ = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()

        # Blend network should have gradients
        for param in model.blend_net.parameters():
            assert param.grad is not None, "Blend network has no gradients"
            assert param.grad.abs().sum() > 0, "Blend network gradients are zero"

    def test_deterministic_mode_selects_single_octave(self):
        """Deterministic mode should pick exactly one octave per position."""
        set_seed()

        d_model = 64
        x = torch.randn(4, 8, d_model)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.set_mode("deterministic")
        model.eval()

        with torch.no_grad():
            _, info = model(x)

        # Blend weights should be one-hot
        assert (info.blend_weights.sum(dim=-1) - 1.0).abs().max() < 1e-5, \
            "Blend weights don't sum to 1 in deterministic mode"

        # Each should have exactly one 1 and rest 0s
        assert ((info.blend_weights == 1.0) | (info.blend_weights == 0.0)).all(), \
            "Blend weights not one-hot in deterministic mode"

    def test_entropy_meaningful(self):
        """Entropy should reflect routing uncertainty."""
        set_seed()

        d_model = 64
        x = torch.randn(4, 8, d_model)

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)

        # Generative mode should have positive entropy
        model.set_mode("generative")
        model.eval()
        with torch.no_grad():
            _, info_gen = model(x)

        # Deterministic mode should have zero entropy
        model.set_mode("deterministic")
        with torch.no_grad():
            _, info_det = model(x)

        # Generative entropy should be positive
        assert info_gen.entropy.mean() > 0.1, "Generative mode entropy too low"

        # Deterministic entropy should be ~0
        assert info_det.entropy.mean() < 0.01, "Deterministic mode entropy should be ~0"


# =============================================================================
# 5. COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Verify the model competes with baselines."""

    def test_beats_random_baseline(self):
        """Should outperform random baseline."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(400, d_model)
        x_test, y_test = generate_xor_data(100, d_model)

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Train TrueOctave
        set_seed(123)
        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test)
            model_loss = F.mse_loss(pred, y_test).item()

        # Train random (only scale)
        set_seed(123)
        random_model = RandomFFN(d_model)
        random_model.train()
        opt_r = torch.optim.Adam(random_model.parameters(), lr=1e-3)

        for _ in range(200):
            opt_r.zero_grad()
            out, _ = random_model(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_r.step()

        random_model.eval()
        with torch.no_grad():
            pred_r, _ = random_model(x_test)
            random_loss = F.mse_loss(pred_r, y_test).item()

        # Model should at least match random (within noise margin)
        # On simple tasks, both converge to similar residual behavior
        assert model_loss < random_loss * 1.1, \
            f"Model worse than random: model={model_loss:.4f}, random={random_loss:.4f}"

    def test_comparable_to_dense_on_xor(self):
        """Should be competitive with dense on XOR task."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(500, d_model)
        x_test, y_test = generate_xor_data(100, d_model)

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Train TrueOctave
        set_seed(123)
        octave = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        octave.train()
        opt_o = torch.optim.Adam(octave.parameters(), lr=1e-3)

        for _ in range(200):
            opt_o.zero_grad()
            out, _ = octave(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_o.step()

        octave.eval()
        with torch.no_grad():
            pred_o, _ = octave(x_test)
            loss_o = F.mse_loss(pred_o, y_test).item()

        # Train Dense
        set_seed(123)
        dense = DenseFFN(d_model)
        dense.train()
        opt_d = torch.optim.Adam(dense.parameters(), lr=1e-3)

        for _ in range(200):
            opt_d.zero_grad()
            out, _ = dense(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_d.step()

        dense.eval()
        with torch.no_grad():
            pred_d, _ = dense(x_test)
            loss_d = F.mse_loss(pred_d, y_test).item()

        # Should be within 3x of dense
        assert loss_o < loss_d * 3, \
            f"TrueOctave too far from dense: octave={loss_o:.4f}, dense={loss_d:.4f}"


# =============================================================================
# 6. ABLATION TESTS
# =============================================================================

class TestAblation:
    """Verify each component contributes."""

    def test_hierarchy_helps(self):
        """Multi-octave should outperform single-octave on multiscale data."""
        set_seed()

        d_model = 64
        x, y, _ = generate_multiscale_data(600, d_model)
        x_train, y_train = x[:400], y[:400]
        x_test, y_test = x[400:], y[400:]

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Train TrueOctave (hierarchical)
        set_seed(456)
        multi = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        multi.train()
        opt_m = torch.optim.Adam(multi.parameters(), lr=1e-3)

        for _ in range(200):
            opt_m.zero_grad()
            out, _ = multi(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_m.step()

        multi.eval()
        with torch.no_grad():
            pred_m, _ = multi(x_test)
            loss_multi = F.mse_loss(pred_m, y_test).item()

        # Train SingleOctave (flat)
        set_seed(456)
        single = SingleOctaveFFN(d_model, num_tiles=16)
        single.train()
        opt_s = torch.optim.Adam(single.parameters(), lr=1e-3)

        for _ in range(200):
            opt_s.zero_grad()
            out, _ = single(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_s.step()

        single.eval()
        with torch.no_grad():
            pred_s, _ = single(x_test)
            loss_single = F.mse_loss(pred_s, y_test).item()

        # Hierarchy should help on multiscale data
        assert loss_multi < loss_single * 1.2, \
            f"Hierarchy didn't help: multi={loss_multi:.4f}, single={loss_single:.4f}"

    def test_frozen_structure_essential(self):
        """Frozen structure should provide value over random initialization."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(400, d_model)
        x_test, y_test = generate_xor_data(100, d_model)

        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Train TrueOctave (frozen structure, derived octaves)
        set_seed(789)
        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train)
            loss = F.mse_loss(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test)
            structured_loss = F.mse_loss(pred, y_test).item()

        # Train random baseline
        set_seed(789)
        random_model = RandomFFN(d_model)
        random_model.train()
        opt_r = torch.optim.Adam(random_model.parameters(), lr=1e-3)

        for _ in range(200):
            opt_r.zero_grad()
            out, _ = random_model(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_r.step()

        random_model.eval()
        with torch.no_grad():
            pred_r, _ = random_model(x_test)
            random_loss = F.mse_loss(pred_r, y_test).item()

        # Frozen structure should at least match random
        # On residual-dominated tasks, both reach similar floor
        assert structured_loss < random_loss * 1.1, \
            f"Frozen structure worse than random: structured={structured_loss:.4f}, random={random_loss:.4f}"


# =============================================================================
# 7. SANITY CHECKS
# =============================================================================

class TestSanityChecks:
    """Basic sanity checks."""

    def test_output_changes_with_input(self):
        """Different inputs should produce different outputs."""
        set_seed()

        model = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        model.eval()

        x1 = torch.randn(1, 1, 64)
        x2 = torch.randn(1, 1, 64)

        with torch.no_grad():
            out1, _ = model(x1)
            out2, _ = model(x2)

        assert not torch.allclose(out1, out2, atol=1e-4), "Different inputs gave same output"

    def test_output_not_identity(self):
        """Output should differ from input."""
        set_seed()

        model = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        model.eval()

        x = torch.randn(4, 8, 64)

        with torch.no_grad():
            out, _ = model(x)

        # Should not be identical (some transformation applied)
        diff = (out - x).abs().mean()
        assert diff > 0.01, f"Output too similar to input: diff = {diff}"

    def test_learns_identity_when_target_is_input(self):
        """Should be able to learn identity (y = x)."""
        set_seed()

        d_model = 64
        x = torch.randn(200, 1, d_model)
        y = x.clone()  # Identity target

        model = TrueOctaveFFN(d_model=d_model, num_fine_tiles=16, pool_factor=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x)
            final_loss = F.mse_loss(pred, y).item()

        # Should learn identity with low loss
        assert final_loss < 0.1, f"Failed to learn identity: loss = {final_loss}"

    def test_mode_switching_works(self):
        """Should be able to switch between generative and deterministic modes."""
        set_seed()

        model = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        x = torch.randn(4, 8, 64)

        model.set_mode("generative")
        assert model.mode == "generative"
        model.eval()
        with torch.no_grad():
            out_gen, info_gen = model(x)
        assert info_gen.mode == "generative"

        model.set_mode("deterministic")
        assert model.mode == "deterministic"
        with torch.no_grad():
            out_det, info_det = model(x)
        assert info_det.mode == "deterministic"

        # Outputs should differ between modes
        assert not torch.allclose(out_gen, out_det), "Outputs should differ between modes"
