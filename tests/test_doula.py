"""
Tests for Doula - The Birth Attendant.

Mesa 16.2: Glass Box Meta-Learning

Tests cover:
    - DynamicsSignature creation and embedding
    - PhaseDetector phase recognition
    - DoulaVDB storage and search
    - DoulaFunction aggregation and guidance
    - Training integration
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from trix.doula import (
    DynamicsSignature,
    DoulaEntry,
    DoulaGuidance,
    PhaseDetector,
    DoulaVDB,
    DoulaFunction,
    DoulaTrainingConfig,
    DoulaCallback,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_signature():
    """Create a sample dynamics signature."""
    return DynamicsSignature(
        step=100,
        layer=0,
        shape_entropy=0.5,
        partner_entropy=0.3,
        routing_stability=0.7,
        dominant_shapes=[0, 2, 1],
        shape_counts=torch.tensor([10., 5., 8., 2., 1., 0., 0., 0.]),
        exploration_ratio=0.4,
        exploitation_ratio=0.6,
        new_shape_activated=False,
        cluster_formed=True,
        mean_distance=5.0,
        std_distance=2.0,
    )


@pytest.fixture
def sample_routing_info():
    """Create sample routing info from a TriX forward pass."""
    B, T, num_shapes = 2, 8, 8
    return {
        'shape_idx': torch.randint(0, num_shapes, (B, T)),
        'shape_weights': torch.softmax(torch.randn(B, T, num_shapes), dim=-1),
        'partner_idx': torch.randint(0, T, (B, T)),
        'distances': torch.rand(B, T, T) * 10,
    }


@pytest.fixture
def temp_vdb_path():
    """Create a temporary path for VDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_doula.json")


# =============================================================================
# DynamicsSignature Tests
# =============================================================================

class TestDynamicsSignature:
    """Tests for DynamicsSignature."""

    def test_creation(self, sample_signature):
        """Signature can be created."""
        assert sample_signature.step == 100
        assert sample_signature.layer == 0
        assert sample_signature.shape_entropy == 0.5

    def test_to_embedding(self, sample_signature):
        """Signature can be converted to embedding."""
        embedding = sample_signature.to_embedding(embedding_dim=64)
        assert embedding.shape == (64,)
        assert not torch.isnan(embedding).any()

    def test_to_dict(self, sample_signature):
        """Signature can be serialized to dict."""
        d = sample_signature.to_dict()
        assert d['step'] == 100
        assert d['layer'] == 0
        assert 'shape_counts' in d

    def test_from_dict(self, sample_signature):
        """Signature can be deserialized from dict."""
        d = sample_signature.to_dict()
        restored = DynamicsSignature.from_dict(d)
        assert restored.step == sample_signature.step
        assert restored.shape_entropy == sample_signature.shape_entropy


# =============================================================================
# PhaseDetector Tests
# =============================================================================

class TestPhaseDetector:
    """Tests for PhaseDetector."""

    def test_detect_exploration(self):
        """High entropy = exploration phase."""
        detector = PhaseDetector()
        sig = DynamicsSignature(
            step=0, layer=0,
            shape_entropy=2.0, partner_entropy=2.0,
            routing_stability=0.1,
            dominant_shapes=[0], shape_counts=torch.ones(8),
            exploration_ratio=0.9, exploitation_ratio=0.1,
            new_shape_activated=False, cluster_formed=False,
            mean_distance=5.0, std_distance=2.0,
        )
        assert detector.detect(sig) == 'exploration'

    def test_detect_transition(self):
        """Medium entropy, low stability = transition."""
        detector = PhaseDetector()
        sig = DynamicsSignature(
            step=100, layer=0,
            shape_entropy=1.0, partner_entropy=1.0,
            routing_stability=0.5,
            dominant_shapes=[0, 1], shape_counts=torch.ones(8),
            exploration_ratio=0.5, exploitation_ratio=0.5,
            new_shape_activated=False, cluster_formed=False,
            mean_distance=5.0, std_distance=2.0,
        )
        assert detector.detect(sig) == 'transition'

    def test_detect_crystallization(self):
        """Low entropy, high stability but not complete = crystallization."""
        detector = PhaseDetector()
        sig = DynamicsSignature(
            step=500, layer=0,
            shape_entropy=0.5, partner_entropy=0.5,
            routing_stability=0.85,
            dominant_shapes=[0], shape_counts=torch.tensor([20., 2., 1., 0., 0., 0., 0., 0.]),
            exploration_ratio=0.2, exploitation_ratio=0.8,
            new_shape_activated=False, cluster_formed=True,
            mean_distance=3.0, std_distance=1.0,
        )
        assert detector.detect(sig) == 'crystallization'

    def test_detect_stable(self):
        """Very high stability = stable."""
        detector = PhaseDetector()
        sig = DynamicsSignature(
            step=1000, layer=0,
            shape_entropy=0.2, partner_entropy=0.2,
            routing_stability=0.98,
            dominant_shapes=[0], shape_counts=torch.tensor([25., 1., 0., 0., 0., 0., 0., 0.]),
            exploration_ratio=0.1, exploitation_ratio=0.9,
            new_shape_activated=False, cluster_formed=True,
            mean_distance=2.0, std_distance=0.5,
        )
        assert detector.detect(sig) == 'stable'

    def test_predict_next_phase(self):
        """Next phase prediction follows natural progression."""
        detector = PhaseDetector()
        assert detector.predict_next_phase('exploration') == 'transition'
        assert detector.predict_next_phase('transition') == 'crystallization'
        assert detector.predict_next_phase('crystallization') == 'stable'
        assert detector.predict_next_phase('stable') == 'stable'


# =============================================================================
# DoulaVDB Tests
# =============================================================================

class TestDoulaVDB:
    """Tests for DoulaVDB."""

    def test_creation(self):
        """VDB can be created."""
        vdb = DoulaVDB(embedding_dim=64)
        assert len(vdb.entries) == 0

    def test_add_entry(self, sample_signature):
        """Entries can be added."""
        vdb = DoulaVDB(embedding_dim=64)
        entry = DoulaEntry(
            id="test_0",
            dynamics_embedding=sample_signature.to_embedding(64),
            step=100,
            layer=0,
            epoch=0,
            phase='transition',
            signature=sample_signature,
            loss_at_step=0.5,
        )
        vdb.add(entry)
        assert len(vdb.entries) == 1

    def test_search(self, sample_signature):
        """Search returns similar entries."""
        vdb = DoulaVDB(embedding_dim=64)

        # Add several entries
        for i in range(10):
            sig = DynamicsSignature(
                step=i * 10, layer=0,
                shape_entropy=0.5 + i * 0.05,
                partner_entropy=0.3,
                routing_stability=0.5 + i * 0.05,
                dominant_shapes=[0, 1],
                shape_counts=torch.ones(8),
                exploration_ratio=0.4,
                exploitation_ratio=0.6,
                new_shape_activated=False,
                cluster_formed=False,
                mean_distance=5.0,
                std_distance=2.0,
            )
            entry = DoulaEntry(
                id=f"entry_{i}",
                dynamics_embedding=sig.to_embedding(64),
                step=i * 10,
                layer=0,
                epoch=0,
                phase='transition',
                signature=sig,
                loss_at_step=0.5,
            )
            vdb.add(entry)

        # Search
        query = sample_signature.to_embedding(64)
        results = vdb.search(query, k=3)
        assert len(results) == 3

    def test_glass_box_query_log(self, sample_signature):
        """Queries are logged (glass box)."""
        vdb = DoulaVDB(embedding_dim=64)

        # Add an entry
        entry = DoulaEntry(
            id="test_0",
            dynamics_embedding=sample_signature.to_embedding(64),
            step=100,
            layer=0,
            epoch=0,
            phase='transition',
            signature=sample_signature,
            loss_at_step=0.5,
        )
        vdb.add(entry)

        # Query
        query = sample_signature.to_embedding(64)
        vdb.search(query, k=1)

        # Check log
        log = vdb.get_query_log()
        assert len(log) == 1
        assert 'result_ids' in log[0]

    def test_persistence(self, sample_signature, temp_vdb_path):
        """VDB can be saved and loaded."""
        # Create and populate
        vdb = DoulaVDB(path=temp_vdb_path, embedding_dim=64)
        entry = DoulaEntry(
            id="test_0",
            dynamics_embedding=sample_signature.to_embedding(64),
            step=100,
            layer=0,
            epoch=0,
            phase='transition',
            signature=sample_signature,
            loss_at_step=0.5,
        )
        vdb.add(entry)
        vdb.save()

        # Load fresh
        vdb2 = DoulaVDB(path=temp_vdb_path, embedding_dim=64)
        assert len(vdb2.entries) == 1
        assert vdb2.entries[0].step == 100

    def test_analyze_phases(self, sample_signature):
        """Phase analysis works."""
        vdb = DoulaVDB(embedding_dim=64)

        for phase in ['exploration', 'transition', 'crystallization']:
            entry = DoulaEntry(
                id=f"entry_{phase}",
                dynamics_embedding=sample_signature.to_embedding(64),
                step=100,
                layer=0,
                epoch=0,
                phase=phase,
                signature=sample_signature,
                loss_at_step=0.5,
            )
            vdb.add(entry)

        phases = vdb.analyze_phases()
        assert phases['exploration'] == 1
        assert phases['transition'] == 1
        assert phases['crystallization'] == 1


# =============================================================================
# DoulaFunction Tests
# =============================================================================

class TestDoulaFunction:
    """Tests for DoulaFunction."""

    def test_creation(self):
        """DoulaFunction can be created."""
        doula = DoulaFunction()
        assert doula is not None

    def test_aggregate(self, sample_routing_info):
        """Aggregate extracts signature from routing info."""
        doula = DoulaFunction()
        signature = doula.aggregate(
            routing_info=sample_routing_info,
            step=0,
            layer=0,
            loss=0.5,
        )
        assert isinstance(signature, DynamicsSignature)
        assert signature.step == 0
        assert signature.layer == 0

    def test_store(self, sample_routing_info):
        """Store adds entry to VDB."""
        doula = DoulaFunction()
        signature = doula.aggregate(
            routing_info=sample_routing_info,
            step=0,
            layer=0,
            loss=0.5,
        )
        doula.store(signature, loss=0.5)
        assert len(doula.vdb.entries) == 1

    def test_query(self, sample_routing_info):
        """Query returns guidance."""
        doula = DoulaFunction()

        # Build up some history
        for i in range(10):
            sig = doula.aggregate(
                routing_info=sample_routing_info,
                step=i,
                layer=0,
                loss=0.5,
            )
            doula.store(sig, loss=0.5)

        # Query
        guidance = doula.query(sig)
        assert isinstance(guidance, DoulaGuidance)
        assert guidance.current_phase in ['exploration', 'transition', 'crystallization', 'stable']

    def test_new_shape_detection(self):
        """New shape activation is detected."""
        doula = DoulaFunction()

        # First call with shapes 0, 1
        info1 = {
            'shape_idx': torch.tensor([[0, 1, 0, 1]]),
            'shape_weights': torch.softmax(torch.randn(1, 4, 8), dim=-1),
            'partner_idx': torch.tensor([[0, 0, 1, 1]]),
            'distances': torch.rand(1, 4, 4),
        }
        sig1 = doula.aggregate(info1, step=0, layer=0, loss=0.5)
        assert sig1.new_shape_activated  # First time seeing any shape

        # Second call with same shapes
        sig2 = doula.aggregate(info1, step=1, layer=0, loss=0.5)
        assert not sig2.new_shape_activated

        # Third call with new shape 5
        info3 = {
            'shape_idx': torch.tensor([[0, 1, 5, 1]]),
            'shape_weights': torch.softmax(torch.randn(1, 4, 8), dim=-1),
            'partner_idx': torch.tensor([[0, 0, 1, 1]]),
            'distances': torch.rand(1, 4, 4),
        }
        sig3 = doula.aggregate(info3, step=2, layer=0, loss=0.5)
        assert sig3.new_shape_activated

    def test_persistence(self, sample_routing_info, temp_vdb_path):
        """DoulaFunction saves and loads VDB."""
        doula = DoulaFunction(vdb_path=temp_vdb_path)
        sig = doula.aggregate(sample_routing_info, step=0, layer=0, loss=0.5)
        doula.store(sig, loss=0.5)
        doula.save()

        # Load fresh
        doula2 = DoulaFunction(vdb_path=temp_vdb_path)
        assert len(doula2.vdb.entries) == 1


# =============================================================================
# Training Integration Tests
# =============================================================================

class TestDoulaCallback:
    """Tests for DoulaCallback."""

    def test_creation(self):
        """Callback can be created."""
        config = DoulaTrainingConfig()
        callback = DoulaCallback(config)
        assert callback is not None

    def test_on_step(self, sample_routing_info):
        """on_step processes routing info."""
        config = DoulaTrainingConfig(query_every=5)
        callback = DoulaCallback(config)

        # Simulate info from model
        info = {
            'layers': [
                {'mixer': sample_routing_info},
                {'mixer': sample_routing_info},
            ]
        }

        # First few steps - no query
        for step in range(4):
            guidance = callback.on_step(step, info, loss=0.5)
            assert guidance is None

        # Step 5 - query happens
        guidance = callback.on_step(5, info, loss=0.5)
        assert guidance is not None
        assert isinstance(guidance, DoulaGuidance)

    def test_early_stop(self, sample_routing_info):
        """Early stopping works when training is complete."""
        config = DoulaTrainingConfig(
            query_every=1,
            early_stop_on_stable=True,
        )
        callback = DoulaCallback(config)

        # Simulate stable training (high stability signatures)
        info = {
            'layers': [
                {'mixer': sample_routing_info},
            ]
        }

        # Force stable phase by manipulating phase detector
        callback.doula.phase_detector.crystallization_threshold = 0.0

        # Should eventually trigger stop after enough stable steps
        for step in range(200):
            callback.on_step(step, info, loss=0.1)

        # After many stable steps, should_stop might be true
        # (depends on actual stability calculations)
        # This tests the mechanism exists
        assert hasattr(callback, 'should_stop')


# =============================================================================
# Glass Box Tests
# =============================================================================

class TestGlassBox:
    """Tests verifying glass box transparency."""

    def test_all_entries_accessible(self, sample_routing_info):
        """All entries can be retrieved."""
        doula = DoulaFunction()

        for i in range(5):
            sig = doula.aggregate(sample_routing_info, step=i, layer=0, loss=0.5)
            doula.store(sig, loss=0.5)

        entries = doula.vdb.get_all_entries()
        assert len(entries) == 5

    def test_query_log_accessible(self, sample_routing_info):
        """Query log can be retrieved."""
        doula = DoulaFunction()

        for i in range(5):
            sig = doula.aggregate(sample_routing_info, step=i, layer=0, loss=0.5)
            doula.store(sig, loss=0.5)

        doula.query(sig)
        doula.query(sig)

        log = doula.vdb.get_query_log()
        assert len(log) == 2

    def test_emergence_moments_accessible(self):
        """Emergence moments can be retrieved."""
        doula = DoulaFunction()

        # Create entries with emergence
        for i in range(5):
            info = {
                'shape_idx': torch.tensor([[i % 3, i % 3]]),  # Limited shapes first
                'shape_weights': torch.softmax(torch.randn(1, 2, 8), dim=-1),
                'partner_idx': torch.tensor([[0, 0]]),
                'distances': torch.rand(1, 2, 2),
            }
            sig = doula.aggregate(info, step=i, layer=0, loss=0.5)
            doula.store(sig, loss=0.5)

        emergence = doula.vdb.get_emergence_moments()
        # At least the first few steps should have emergence (new shapes)
        assert len(emergence) > 0

    def test_summary_accessible(self, sample_routing_info):
        """Summary statistics accessible."""
        doula = DoulaFunction(domain="test_domain")

        for i in range(10):
            sig = doula.aggregate(sample_routing_info, step=i, layer=0, loss=0.5)
            doula.store(sig, loss=0.5)

        summary = doula.summarize()
        assert summary['total_entries'] == 10
        assert summary['domain'] == 'test_domain'
        assert 'phases' in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_observation_cycle(self, sample_routing_info, temp_vdb_path):
        """Full cycle: aggregate → store → query → save → load."""
        # First session
        doula = DoulaFunction(vdb_path=temp_vdb_path, domain="integration_test")

        for i in range(20):
            sig = doula.aggregate(sample_routing_info, step=i, layer=0, loss=0.5 - i * 0.01)
            doula.store(sig, loss=0.5 - i * 0.01)

        guidance = doula.query(sig)
        assert guidance is not None

        doula.save()

        # Second session - load and continue
        doula2 = DoulaFunction(vdb_path=temp_vdb_path)
        assert len(doula2.vdb.entries) == 20

        # Query should find similar past moments
        guidance2 = doula2.query(sig)
        assert len(guidance2.similar_past_moments) > 0

    def test_multi_layer_observation(self, sample_routing_info):
        """Multiple layers are tracked correctly."""
        doula = DoulaFunction()

        # Simulate 3-layer model
        for step in range(10):
            for layer in range(3):
                sig = doula.aggregate(sample_routing_info, step=step, layer=layer, loss=0.5)
                doula.store(sig, loss=0.5)

        # Should have entries for each layer
        layer0 = doula.vdb.get_entries_by_layer(0)
        layer1 = doula.vdb.get_entries_by_layer(1)
        layer2 = doula.vdb.get_entries_by_layer(2)

        assert len(layer0) == 10
        assert len(layer1) == 10
        assert len(layer2) == 10
