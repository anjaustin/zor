"""
Tests for Mesa 16: The Sacred Foundry.

Tests cover:
    - Frozen mixing shapes
    - Token mixer (tokens route to tokens)
    - Unified Providence block
    - Full transformer integration

Philosophy: Tests prove invariants, not find bugs.
"""

import torch
import torch.nn as nn
import pytest
from typing import Dict

from trix.foundry import (
    # Mixing shapes
    MixingShapes,
    FrozenMixingLibrary,
    get_mixing_library,
    # Token mixer
    TokenMixer,
    create_token_mixer,
    hamming_distance,
    binarize_ste,
    # Unified blocks
    UnifiedProvidenceBlock,
    UnifiedProvidenceTransformer,
    create_unified_block,
    create_unified_transformer,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def d_model():
    return 64


# =============================================================================
# Mixing Shapes Tests
# =============================================================================

class TestMixingShapes:
    """Tests for frozen mixing shape operations."""

    def test_copy_ignores_partner(self):
        """Copy shape returns first input, ignores second."""
        a = torch.randn(4, 32)
        b = torch.randn(4, 32)
        result = MixingShapes.copy(a, b)
        assert torch.equal(result, a)

    def test_zero_returns_zeros(self):
        """Zero shape returns zeros."""
        a = torch.randn(4, 32)
        b = torch.randn(4, 32)
        result = MixingShapes.zero(a, b)
        assert torch.equal(result, torch.zeros_like(a))

    def test_avg_is_average(self):
        """Avg shape returns (a + b) / 2."""
        a = torch.tensor([[2.0, 4.0]])
        b = torch.tensor([[4.0, 8.0]])
        result = MixingShapes.avg(a, b)
        expected = torch.tensor([[3.0, 6.0]])
        assert torch.allclose(result, expected)

    def test_add_is_sum(self):
        """Add shape returns a + b."""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        result = MixingShapes.add(a, b)
        expected = torch.tensor([[4.0, 6.0]])
        assert torch.allclose(result, expected)

    def test_diff_is_difference(self):
        """Diff shape returns a - b."""
        a = torch.tensor([[5.0, 8.0]])
        b = torch.tensor([[2.0, 3.0]])
        result = MixingShapes.diff(a, b)
        expected = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(result, expected)

    def test_hadamard_is_product(self):
        """Hadamard shape returns element-wise product."""
        a = torch.tensor([[2.0, 3.0]])
        b = torch.tensor([[4.0, 5.0]])
        result = MixingShapes.hadamard(a, b)
        expected = torch.tensor([[8.0, 15.0]])
        assert torch.allclose(result, expected)

    def test_max_is_maximum(self):
        """Max shape returns element-wise maximum."""
        a = torch.tensor([[1.0, 5.0]])
        b = torch.tensor([[3.0, 2.0]])
        result = MixingShapes.max_mix(a, b)
        expected = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(result, expected)

    def test_min_is_minimum(self):
        """Min shape returns element-wise minimum."""
        a = torch.tensor([[1.0, 5.0]])
        b = torch.tensor([[3.0, 2.0]])
        result = MixingShapes.min_mix(a, b)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_gate_with_zero(self):
        """Gate with g=0 returns b."""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        g = torch.tensor([[0.0, 0.0]])
        result = MixingShapes.gate(a, b, g)
        assert torch.allclose(result, b)

    def test_gate_with_one(self):
        """Gate with g=1 returns a."""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        g = torch.tensor([[1.0, 1.0]])
        result = MixingShapes.gate(a, b, g)
        assert torch.allclose(result, a)

    def test_gate_interpolates(self):
        """Gate interpolates between a and b."""
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[10.0, 10.0]])
        g = torch.tensor([[0.5, 0.5]])
        result = MixingShapes.gate(a, b, g)
        expected = torch.tensor([[5.0, 5.0]])
        assert torch.allclose(result, expected)


class TestFrozenMixingLibrary:
    """Tests for the mixing shape library."""

    def test_library_has_core_shapes(self):
        """Library contains all core mixing shapes."""
        lib = FrozenMixingLibrary()
        shapes = lib.list_shapes()

        core_shapes = ['copy', 'zero', 'avg', 'add', 'diff', 'hadamard', 'max', 'min']
        for shape in core_shapes:
            assert shape in shapes, f"Missing shape: {shape}"

    def test_get_shape(self):
        """Can retrieve shapes by name."""
        lib = FrozenMixingLibrary()
        avg = lib.get('avg')
        assert callable(avg)

        a = torch.tensor([[2.0]])
        b = torch.tensor([[4.0]])
        assert torch.allclose(avg(a, b), torch.tensor([[3.0]]))

    def test_get_unknown_raises(self):
        """Getting unknown shape raises KeyError."""
        lib = FrozenMixingLibrary()
        with pytest.raises(KeyError):
            lib.get('nonexistent_shape')

    def test_shape_info(self):
        """Can get shape metadata."""
        lib = FrozenMixingLibrary()
        info = lib.info('avg')

        assert info.name == 'avg'
        assert info.n_inputs == 2
        assert info.commutative == True
        assert 'average' in info.description.lower()

    def test_derive_signature(self):
        """Signatures are derived deterministically."""
        lib = FrozenMixingLibrary()
        sig1 = lib.derive_signature('avg', 64)
        sig2 = lib.derive_signature('avg', 64)

        assert sig1.shape == (64,)
        assert torch.equal(sig1, sig2)  # Deterministic

    def test_different_shapes_different_signatures(self):
        """Different shapes have different signatures."""
        lib = FrozenMixingLibrary()
        sig_avg = lib.derive_signature('avg', 64)
        sig_diff = lib.derive_signature('diff', 64)

        # They shouldn't be identical
        assert not torch.equal(sig_avg, sig_diff)

    def test_get_all_signatures(self):
        """Can get all signatures stacked."""
        lib = FrozenMixingLibrary()
        sigs = lib.get_all_signatures(64)

        num_shapes = len(lib.list_shapes())
        assert sigs.shape == (num_shapes, 64)

    def test_singleton_library(self):
        """get_mixing_library returns singleton."""
        lib1 = get_mixing_library()
        lib2 = get_mixing_library()
        assert lib1 is lib2


# =============================================================================
# Hamming Distance Tests
# =============================================================================

class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_identical_vectors_zero_distance(self):
        """Identical vectors have zero Hamming distance."""
        a = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        b = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])
        dist = hamming_distance(a, b)
        assert dist.item() == 0

    def test_opposite_vectors_max_distance(self):
        """Opposite vectors have maximum Hamming distance."""
        a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        b = torch.tensor([[[-1.0, -1.0, -1.0, -1.0]]])
        dist = hamming_distance(a, b)
        assert dist.item() == 4  # All 4 positions differ

    def test_half_different(self):
        """Half different has half max distance."""
        a = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        b = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        dist = hamming_distance(a, b)
        assert dist.item() == 2  # 2 positions differ

    def test_batch_distances(self):
        """Computes distances for batched inputs."""
        a = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])  # [2, 2]
        b = torch.tensor([
            [[1.0, 1.0], [-1.0, -1.0]],  # For first query
            [[1.0, 1.0], [-1.0, -1.0]],  # For second query
        ])  # [2, 2, 2]

        dist = hamming_distance(a, b)
        assert dist.shape == (2, 2)


class TestBinarizeSTE:
    """Tests for straight-through estimator binarization."""

    def test_positive_becomes_one(self):
        """Positive values become +1."""
        x = torch.tensor([0.5, 1.0, 0.001])
        result = binarize_ste(x)
        assert torch.allclose(result, torch.tensor([1.0, 1.0, 1.0]))

    def test_negative_becomes_minus_one(self):
        """Negative values become -1."""
        x = torch.tensor([-0.5, -1.0, -0.001])
        result = binarize_ste(x)
        assert torch.allclose(result, torch.tensor([-1.0, -1.0, -1.0]))

    def test_gradient_flows(self):
        """Gradients flow through STE."""
        x = torch.tensor([0.5, -0.5], requires_grad=True)
        result = binarize_ste(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# =============================================================================
# Token Mixer Tests
# =============================================================================

class TestTokenMixer:
    """Tests for the token mixer."""

    def test_creation(self, d_model):
        """TokenMixer can be created."""
        mixer = TokenMixer(d_model=d_model)
        assert mixer.d_model == d_model

    def test_forward_shape(self, batch_size, seq_len, d_model):
        """Forward produces correct shape."""
        mixer = TokenMixer(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = mixer(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_2d(self, batch_size, d_model):
        """Handles 2D input."""
        mixer = TokenMixer(d_model=d_model)
        x = torch.randn(batch_size, d_model)

        output, state, info = mixer(x)

        assert output.shape == (batch_size, d_model)

    def test_causal_masking(self, batch_size, seq_len, d_model):
        """Causal masking prevents future token access."""
        mixer = TokenMixer(d_model=d_model, use_causal=True)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = mixer(x)
        partner_idx = info['partner_idx']

        # Each token's partner should be <= its position
        for t in range(seq_len):
            assert (partner_idx[:, t] <= t).all(), f"Token {t} accessed future token"

    def test_partner_finding(self, d_model):
        """Partner finding works correctly."""
        mixer = TokenMixer(d_model=d_model, use_causal=False)

        # Set sig_proj to identity-like so similar inputs -> similar signatures
        with torch.no_grad():
            mixer.sig_proj.weight.copy_(torch.eye(d_model))
            mixer.sig_proj.bias.zero_()

        # Create tokens with distinct patterns that survive LayerNorm
        # Use alternating patterns so variance is preserved
        x = torch.zeros(1, 3, d_model)

        # t0: positive in first half, negative in second half
        x[0, 0, :d_model//2] = 1.0
        x[0, 0, d_model//2:] = -1.0

        # t1: opposite pattern (negative first, positive second)
        x[0, 1, :d_model//2] = -1.0
        x[0, 1, d_model//2:] = 1.0

        # t2: same pattern as t0
        x[0, 2, :d_model//2] = 1.0
        x[0, 2, d_model//2:] = -1.0

        output, state, info = mixer(x)
        partner_idx = info['partner_idx']

        # t0 should partner with t2 (most similar)
        assert partner_idx[0, 0].item() == 2
        # t2 should partner with t0 (most similar)
        assert partner_idx[0, 2].item() == 0

    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through mixer."""
        mixer = TokenMixer(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, state, info = mixer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_state_persistence(self, batch_size, seq_len, d_model):
        """State persists across calls."""
        mixer = TokenMixer(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        state = mixer.init_state(batch_size)
        _, state1, _ = mixer(x, state)
        _, state2, _ = mixer(x, state1)

        # States should be different
        assert not torch.equal(state1['hidden'], state['hidden'])

    def test_routing_info(self, batch_size, seq_len, d_model):
        """Routing info is returned."""
        mixer = TokenMixer(d_model=d_model, num_mixing_shapes=4)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = mixer(x)

        assert 'partner_idx' in info
        assert 'shape_idx' in info
        assert 'shape_weights' in info
        assert info['partner_idx'].shape == (batch_size, seq_len)
        assert info['shape_idx'].shape == (batch_size, seq_len)


# =============================================================================
# Unified Providence Block Tests
# =============================================================================

class TestUnifiedProvidenceBlock:
    """Tests for the unified block."""

    def test_creation(self, d_model):
        """Block can be created."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        assert block.d_model == d_model

    def test_forward_shape(self, batch_size, seq_len, d_model):
        """Forward produces correct shape."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = block(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_2d(self, batch_size, d_model):
        """Handles 2D input."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, d_model)

        output, state, info = block(x)

        assert output.shape == (batch_size, d_model)

    def test_has_both_phases(self, d_model):
        """Block has both mixer and transform."""
        block = UnifiedProvidenceBlock(d_model=d_model)

        assert hasattr(block, 'token_mixer')
        assert hasattr(block, 'transform_ffn')

    def test_state_initialization(self, batch_size, d_model):
        """State initializes correctly."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        state = block.init_state(batch_size)

        assert 'mixer_state' in state
        assert 'transform_state' in state

    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through block."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, state, info = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan_output(self, batch_size, seq_len, d_model):
        """Output has no NaN values."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = block(x)

        assert not torch.isnan(output).any()

    def test_routing_info_both_phases(self, batch_size, seq_len, d_model):
        """Info contains both mixer and transform routing."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output, state, info = block(x)

        assert 'mixer' in info
        assert 'transform' in info


# =============================================================================
# Unified Providence Transformer Tests
# =============================================================================

class TestUnifiedProvidenceTransformer:
    """Tests for the full transformer."""

    def test_creation(self, d_model):
        """Transformer can be created."""
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=2,
        )
        assert model.d_model == d_model
        assert model.n_layers == 2

    def test_forward_embeddings(self, batch_size, seq_len, d_model):
        """Forward works with embedding input."""
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=2,
        )
        x = torch.randn(batch_size, seq_len, d_model)

        output, states, info = model(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_token_ids(self, batch_size, seq_len, d_model):
        """Forward works with token ids."""
        vocab_size = 1000
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=2,
            vocab_size=vocab_size,
        )
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output, states, info = model(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_state_per_layer(self, batch_size, seq_len, d_model):
        """State is maintained per layer."""
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=3,
        )
        states = model.init_state(batch_size)

        assert len(states) == 3
        for state in states:
            assert 'mixer_state' in state
            assert 'transform_state' in state

    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through full model."""
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=2,
        )
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, states, info = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_no_nan_deep(self, batch_size, seq_len, d_model):
        """No NaN even with multiple layers."""
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=4,
        )
        x = torch.randn(batch_size, seq_len, d_model)

        output, states, info = model(x)

        assert not torch.isnan(output).any()


# =============================================================================
# Integration Tests
# =============================================================================

class TestFoundryIntegration:
    """Integration tests for the Sacred Foundry."""

    def test_full_pipeline(self, batch_size, d_model):
        """Full pipeline works end to end."""
        seq_len = 32
        vocab_size = 256

        # Create model
        model = UnifiedProvidenceTransformer(
            d_model=d_model,
            n_layers=2,
            vocab_size=vocab_size,
            num_mixing_shapes=4,
            num_transform_tiles=8,
        )

        # Generate input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        output, states, info = model(x)

        # Verify output
        assert output.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(output).any()

        # Verify we can compute loss
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = nn.functional.cross_entropy(
            output.view(-1, vocab_size),
            targets.view(-1),
        )

        # Verify gradient flows
        loss.backward()

        # Check some parameters got gradients
        params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1

        assert params_with_grad > 0

    def test_drop_in_replacement(self, batch_size, seq_len, d_model):
        """UnifiedProvidenceBlock can replace transformer block."""
        # Create block
        block = create_unified_block(
            d_model=d_model,
            num_mixing_shapes=4,
            num_transform_tiles=8,
        )

        # Same interface as transformer block
        x = torch.randn(batch_size, seq_len, d_model)
        output, state, info = block(x)

        # Output same shape as input
        assert output.shape == x.shape

    def test_deterministic_eval(self, batch_size, seq_len, d_model):
        """Eval mode is deterministic."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        block.eval()

        x = torch.randn(batch_size, seq_len, d_model)

        output1, _, _ = block(x)
        output2, _, _ = block(x)

        assert torch.equal(output1, output2)

    def test_convenience_functions(self, d_model):
        """Convenience functions work."""
        mixer = create_token_mixer(d_model=d_model)
        assert isinstance(mixer, TokenMixer)

        block = create_unified_block(d_model=d_model)
        assert isinstance(block, UnifiedProvidenceBlock)

        model = create_unified_transformer(d_model=d_model, n_layers=2)
        assert isinstance(model, UnifiedProvidenceTransformer)


# =============================================================================
# Edge Cases
# =============================================================================

class TestFoundryEdgeCases:
    """Edge case tests."""

    def test_single_token_sequence(self, batch_size, d_model):
        """Works with single token sequence."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, 1, d_model)

        output, state, info = block(x)

        assert output.shape == (batch_size, 1, d_model)

    def test_batch_size_one(self, seq_len, d_model):
        """Works with batch size 1."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(1, seq_len, d_model)

        output, state, info = block(x)

        assert output.shape == (1, seq_len, d_model)

    def test_large_sequence(self, batch_size, d_model):
        """Works with large sequence."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, 256, d_model)

        output, state, info = block(x)

        assert output.shape == (batch_size, 256, d_model)

    def test_extreme_values(self, batch_size, seq_len, d_model):
        """Handles extreme input values."""
        block = UnifiedProvidenceBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model) * 100

        output, state, info = block(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
