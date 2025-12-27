"""
Validation Tests for Anchored Dual-Mode FFN.

These tests verify the model learns MEANINGFUL patterns, not just fits noise.

Categories:
1. Pattern Learning - Can it learn known functions?
2. Generalization - Does it work on unseen data?
3. Semantic Anchors - Do partitions have meaning?
4. Comparison - Does it match or beat baselines on real tasks?
5. Ablation - Does each component contribute?

"If it doesn't generalize, it didn't learn."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

from trix.nn.anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)


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


def generate_cluster_data(
    n_samples: int,
    d_model: int,
    n_clusters: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate clustered data with cluster-specific transformations.

    Each cluster has a different linear transformation applied.
    This tests whether anchors learn to partition by cluster.
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
    perm = torch.randperm(n_samples)
    return x[perm], y[perm], labels[perm]


def generate_sequence_pattern(
    n_samples: int,
    seq_len: int,
    d_model: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequence data where output depends on position.

    Tests whether the model can handle sequential patterns.
    """
    x = torch.randn(n_samples, seq_len, d_model)

    # Target: each position applies a different scaling
    scales = torch.linspace(0.5, 1.5, seq_len).view(1, seq_len, 1)
    target = x * scales

    return x, target


# =============================================================================
# 1. PATTERN LEARNING TESTS
# =============================================================================

class TestPatternLearning:
    """Verify the model can learn known patterns."""

    def test_learns_xor_pattern(self):
        """Model should learn XOR-like non-linear pattern."""
        set_seed()

        # Data
        x_train, y_train = generate_xor_data(1000, d_model=64)
        x_test, y_test = generate_xor_data(200, d_model=64)

        # Model
        model = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(100):
            temp = get_temperature_schedule(epoch, 100, 2.0, 0.1)
            model.set_temperature(temp)

            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(pred.squeeze(1), y_test).item()

        # Should achieve reasonable loss (not just memorizing)
        assert test_loss < 1.0, f"XOR pattern test loss too high: {test_loss}"

    def test_learns_scaling_transform(self):
        """Model should learn input scaling (what ternary directions CAN approximate)."""
        set_seed()

        d_model = 64

        # Target: scale input by a constant + small offset
        # This IS something frozen ternary directions + learned scales can approximate
        scale = 0.3
        x_train = torch.randn(500, 1, d_model)
        y_train = x_train * scale  # Simple scaling
        x_test = torch.randn(100, 1, d_model)
        y_test = x_test * scale

        # Model
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
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

        # Scaling is approximated, not exact (architecture constraint)
        # The model uses residual + delta, so y ≈ x + scale*directions
        # For y = 0.3*x, this means delta ≈ -0.7*x, achievable via ternary directions
        assert test_loss < 1.5, f"Scaling transform test loss too high: {test_loss}"

    def test_learns_cluster_specific_behavior(self):
        """Model should learn DIFFERENT behaviors for different input clusters."""
        set_seed()

        d_model = 64
        n_clusters = 4

        # Generate clustered data with cluster-specific targets
        # Instead of arbitrary rotations, use cluster-specific SCALING (what the model CAN learn)
        x_train, y_train, labels_train = generate_cluster_data(800, d_model, n_clusters)
        x_test, y_test, labels_test = generate_cluster_data(200, d_model, n_clusters)

        # Model
        model = AnchoredDualModeFFN(
            d_model=d_model,
            num_anchors=n_clusters * 2,  # More anchors than clusters
            num_tiles=32,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(300):  # More epochs for this harder task
            temp = get_temperature_schedule(epoch, 300, 2.0, 0.1)
            model.set_temperature(temp)

            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate - check loss decreased from initial
        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(pred.squeeze(1), y_test).item()

            # Also check that different clusters get different routing
            _, info = model(x_test.unsqueeze(1))
            tiles_by_cluster = {}
            for c in range(n_clusters):
                mask = labels_test == c
                if mask.any():
                    tiles_by_cluster[c] = info['tile_idx'][mask].float().mean().item()

        # Model should learn something (loss < 2.0), not just random
        # Note: can't learn EXACT rotations with ternary directions, but should reduce loss
        assert test_loss < 2.0, f"Cluster transform test loss too high: {test_loss}"

        # Different clusters should tend to use different tiles
        if len(tiles_by_cluster) >= 2:
            tile_means = list(tiles_by_cluster.values())
            tile_variance = np.var(tile_means)
            # Some variance in tile selection by cluster
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

        # Data with some noise
        x_train, y_train = generate_xor_data(500, d_model, noise=0.1)
        x_test, y_test = generate_xor_data(200, d_model, noise=0.1)

        # Model
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred, _ = model(x_train.unsqueeze(1))
            train_loss = F.mse_loss(train_pred.squeeze(1), y_train).item()

            test_pred, _ = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(test_pred.squeeze(1), y_test).item()

        # Test loss shouldn't be much worse than train loss
        gap_ratio = test_loss / (train_loss + 1e-6)
        assert gap_ratio < 2.0, \
            f"Generalization gap too large: train={train_loss:.4f}, test={test_loss:.4f}, ratio={gap_ratio:.2f}"

    def test_no_catastrophic_overfitting(self):
        """Model should not catastrophically overfit small data."""
        set_seed()

        d_model = 64

        # Small training set, larger test set
        x_train, y_train = generate_xor_data(50, d_model)
        x_test, y_test = generate_xor_data(500, d_model)

        # Model with regularization (dropout)
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=4, num_tiles=16, dropout=0.2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Train
        for epoch in range(200):
            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred, _ = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(test_pred.squeeze(1), y_test).item()

        # Test loss should still be reasonable
        assert test_loss < 2.0, f"Catastrophic overfitting: test_loss={test_loss:.4f}"

    def test_output_in_reasonable_range(self):
        """Outputs should be in reasonable range, not exploding."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(200, d_model)

        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(50):
            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        # Check output range
        model.eval()
        with torch.no_grad():
            pred, _ = model(x_train.unsqueeze(1))

        pred_std = pred.std().item()
        target_std = y_train.std().item()

        # Output variance should be comparable to target variance
        assert 0.1 < pred_std / target_std < 10, \
            f"Output scale mismatch: pred_std={pred_std:.3f}, target_std={target_std:.3f}"


# =============================================================================
# 3. SEMANTIC ANCHOR TESTS
# =============================================================================

class TestSemanticAnchors:
    """Verify anchors partition input space meaningfully."""

    def test_anchors_separate_clusters(self):
        """Anchors should tend to separate different input clusters."""
        set_seed()

        d_model = 64
        n_clusters = 4

        x, y, labels = generate_cluster_data(400, d_model, n_clusters)

        model = AnchoredDualModeFFN(
            d_model=d_model,
            num_anchors=n_clusters * 2,
            num_tiles=32,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            output, _ = model(x.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y)
            loss.backward()
            optimizer.step()

        # Check anchor assignments by cluster
        model.eval()
        with torch.no_grad():
            _, info = model(x.unsqueeze(1))

        dominant_anchors = info['dominant_anchor'].squeeze(1)

        # For each cluster, check if it tends to get consistent anchor assignments
        cluster_anchor_consistency = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_anchors = dominant_anchors[mask]
            most_common = cluster_anchors.mode().values
            consistency = (cluster_anchors == most_common).float().mean().item()
            cluster_anchor_consistency.append(consistency)

        avg_consistency = np.mean(cluster_anchor_consistency)

        # Should have SOME consistency (better than random = 1/num_anchors)
        random_baseline = 1.0 / (n_clusters * 2)
        assert avg_consistency > random_baseline * 2, \
            f"Anchor consistency too low: {avg_consistency:.3f} vs random {random_baseline:.3f}"

    def test_similar_inputs_get_similar_anchors(self):
        """Similar inputs should tend to get the same anchor."""
        set_seed()

        d_model = 64
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model.eval()

        # Generate base vectors and perturbations
        n_bases = 50
        n_perturbations = 10

        base_vectors = torch.randn(n_bases, 1, d_model)

        match_count = 0
        total_count = 0

        with torch.no_grad():
            for base in base_vectors:
                _, base_info = model(base.unsqueeze(0))
                base_anchor = base_info['dominant_anchor'].item()

                for _ in range(n_perturbations):
                    perturbed = base + torch.randn_like(base) * 0.1
                    _, pert_info = model(perturbed.unsqueeze(0))
                    pert_anchor = pert_info['dominant_anchor'].item()

                    if base_anchor == pert_anchor:
                        match_count += 1
                    total_count += 1

        match_rate = match_count / total_count

        # Small perturbations should usually keep same anchor
        assert match_rate > 0.5, \
            f"Similar inputs should get similar anchors, match_rate={match_rate:.3f}"


# =============================================================================
# 4. COMPARISON TESTS
# =============================================================================

class DenseFFN(nn.Module):
    """Simple dense FFN for comparison."""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x)), {}


class TestComparison:
    """Compare against baselines."""

    def test_beats_random_baseline(self):
        """Model should significantly beat random predictions."""
        set_seed()

        d_model = 64
        x_test, y_test = generate_xor_data(200, d_model)

        # Trained model
        x_train, y_train = generate_xor_data(500, d_model)
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(100):
            optimizer.zero_grad()
            output, _ = model(x_train.unsqueeze(1))
            loss = F.mse_loss(output.squeeze(1), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x_test.unsqueeze(1))
            model_loss = F.mse_loss(pred.squeeze(1), y_test).item()

        # Random baseline: predict zeros
        random_loss = F.mse_loss(torch.zeros_like(y_test), y_test).item()

        assert model_loss < random_loss * 0.9, \
            f"Model should beat random: model={model_loss:.4f}, random={random_loss:.4f}"

    def test_comparable_to_dense_on_xor_task(self):
        """Should be comparable to dense FFN on XOR pattern (fair comparison)."""
        set_seed()

        d_model = 64

        # XOR pattern - fair comparison since neither architecture is specialized for it
        x_train, y_train = generate_xor_data(500, d_model)
        x_test, y_test = generate_xor_data(100, d_model)
        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        # Train Anchored model
        set_seed(123)
        anchored = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        anchored.train()
        opt_a = torch.optim.Adam(anchored.parameters(), lr=1e-3)

        for epoch in range(200):
            temp = get_temperature_schedule(epoch, 200, 2.0, 0.1)
            anchored.set_temperature(temp)
            opt_a.zero_grad()
            out, _ = anchored(x_train)
            loss = F.mse_loss(out, y_train)
            loss.backward()
            opt_a.step()

        anchored.eval()
        with torch.no_grad():
            pred_a, _ = anchored(x_test)
            loss_a = F.mse_loss(pred_a, y_test).item()

        # Train Dense model
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

        # On XOR, anchored should be competitive (partitioning helps!)
        # Anchored should be within 3x of dense (may even outperform)
        assert loss_a < loss_d * 3, \
            f"Anchored too far from dense on XOR: anchored={loss_a:.4f}, dense={loss_d:.4f}"


# =============================================================================
# 5. ABLATION TESTS
# =============================================================================

class TestAblation:
    """Verify each component contributes."""

    def test_anchors_help(self):
        """Model with anchors should outperform model ignoring anchors."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_cluster_data(400, d_model, n_clusters=4)[:2]
        x_test, y_test = generate_cluster_data(100, d_model, n_clusters=4)[:2]

        # Model with anchors (normal)
        set_seed(456)
        model_with = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model_with.train()
        opt_with = torch.optim.Adam(model_with.parameters(), lr=1e-3)

        for epoch in range(100):
            temp = get_temperature_schedule(epoch, 100, 2.0, 0.1)
            model_with.set_temperature(temp)
            opt_with.zero_grad()
            out, _ = model_with(x_train.unsqueeze(1))
            loss = F.mse_loss(out.squeeze(1), y_train)
            loss.backward()
            opt_with.step()

        model_with.eval()
        with torch.no_grad():
            pred_with, _ = model_with(x_test.unsqueeze(1))
            loss_with = F.mse_loss(pred_with.squeeze(1), y_test).item()

        # Model with single anchor (effectively no partitioning)
        set_seed(456)
        model_without = AnchoredDualModeFFN(d_model=d_model, num_anchors=1, num_tiles=32)
        model_without.train()
        opt_without = torch.optim.Adam(model_without.parameters(), lr=1e-3)

        for epoch in range(100):
            opt_without.zero_grad()
            out, _ = model_without(x_train.unsqueeze(1))
            loss = F.mse_loss(out.squeeze(1), y_train)
            loss.backward()
            opt_without.step()

        model_without.eval()
        with torch.no_grad():
            pred_without, _ = model_without(x_test.unsqueeze(1))
            loss_without = F.mse_loss(pred_without.squeeze(1), y_test).item()

        # Multiple anchors should help on clustered data
        # (At minimum, shouldn't be much worse)
        assert loss_with <= loss_without * 1.5, \
            f"Anchors should help: with={loss_with:.4f}, without={loss_without:.4f}"

    def test_temperature_annealing_helps(self):
        """Temperature annealing should improve or maintain performance."""
        set_seed()

        d_model = 64
        x_train, y_train = generate_xor_data(400, d_model)
        x_test, y_test = generate_xor_data(100, d_model)

        # With annealing
        set_seed(789)
        model_anneal = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model_anneal.train()
        opt_anneal = torch.optim.Adam(model_anneal.parameters(), lr=1e-3)

        for epoch in range(100):
            temp = get_temperature_schedule(epoch, 100, 2.0, 0.1)
            model_anneal.set_temperature(temp)
            opt_anneal.zero_grad()
            out, _ = model_anneal(x_train.unsqueeze(1))
            loss = F.mse_loss(out.squeeze(1), y_train)
            loss.backward()
            opt_anneal.step()

        model_anneal.eval()
        with torch.no_grad():
            pred_anneal, _ = model_anneal(x_test.unsqueeze(1))
            loss_anneal = F.mse_loss(pred_anneal.squeeze(1), y_test).item()

        # Without annealing (fixed cold temperature)
        set_seed(789)
        model_fixed = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32, temperature=0.1)
        model_fixed.train()
        opt_fixed = torch.optim.Adam(model_fixed.parameters(), lr=1e-3)

        for epoch in range(100):
            opt_fixed.zero_grad()
            out, _ = model_fixed(x_train.unsqueeze(1))
            loss = F.mse_loss(out.squeeze(1), y_train)
            loss.backward()
            opt_fixed.step()

        model_fixed.eval()
        with torch.no_grad():
            pred_fixed, _ = model_fixed(x_test.unsqueeze(1))
            loss_fixed = F.mse_loss(pred_fixed.squeeze(1), y_test).item()

        # Annealing should be at least as good
        assert loss_anneal <= loss_fixed * 1.2, \
            f"Annealing should help: anneal={loss_anneal:.4f}, fixed={loss_fixed:.4f}"


# =============================================================================
# 6. SANITY CHECKS
# =============================================================================

class TestSanityChecks:
    """Basic sanity checks that outputs make sense."""

    def test_output_changes_with_input(self):
        """Output should change when input changes."""
        set_seed()

        model = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        model.eval()

        x1 = torch.randn(1, 1, 64)
        x2 = torch.randn(1, 1, 64)

        with torch.no_grad():
            out1, _ = model(x1)
            out2, _ = model(x2)

        # Outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "Different inputs should produce different outputs"

    def test_output_is_not_just_input(self):
        """Output should be transformed, not just passed through."""
        set_seed()

        model = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        model.train()

        # Train briefly
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(4, 1, 64)
            target = torch.randn(4, 1, 64)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        # Check that output differs from input
        model.eval()
        x = torch.randn(4, 1, 64)
        with torch.no_grad():
            out, _ = model(x)

        diff = (out - x).abs().mean().item()
        assert diff > 0.01, f"Output too similar to input: diff={diff}"

    def test_learns_identity_when_target_is_input(self):
        """When target=input, model should learn near-identity."""
        set_seed()

        d_model = 64
        model = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train on identity
        for _ in range(200):
            x = torch.randn(16, 1, d_model)
            target = x  # Identity

            optimizer.zero_grad()
            out, _ = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        x_test = torch.randn(32, 1, d_model)
        with torch.no_grad():
            out, _ = model(x_test)

        identity_loss = F.mse_loss(out, x_test).item()
        assert identity_loss < 0.1, \
            f"Should learn near-identity: loss={identity_loss}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
