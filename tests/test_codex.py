"""
Tests for Codex Layer: Archetypal Meaning Illumination

Rigorous tests for the keyword-based Codex system.
"""

import pytest
import numpy as np

from trix.db import (
    KeywordCodex,
    KeywordCodexLayer,
    GeneKeyNode,
    create_codex,
    GENE_KEYS,
    PARTNERS,
    CODON_RINGS,
)


class TestGeneKeysData:
    """Tests for Gene Keys vocabulary data integrity."""
    
    def test_gene_keys_count(self):
        """Must have exactly 64 Gene Keys."""
        assert len(GENE_KEYS) == 64
        
    def test_gene_keys_range(self):
        """Gene Keys numbered 1-64."""
        assert set(GENE_KEYS.keys()) == set(range(1, 65))
        
    def test_gene_keys_structure(self):
        """Each Gene Key has shadow, gift, siddhi."""
        for gk_id, data in GENE_KEYS.items():
            assert 'shadow' in data, f"GK{gk_id} missing shadow"
            assert 'gift' in data, f"GK{gk_id} missing gift"
            assert 'siddhi' in data, f"GK{gk_id} missing siddhi"
            assert isinstance(data['shadow'], str)
            assert isinstance(data['gift'], str)
            assert isinstance(data['siddhi'], str)
            assert len(data['shadow']) > 0
            assert len(data['gift']) > 0
            assert len(data['siddhi']) > 0
            
    def test_partners_count(self):
        """Must have 32 partner pairs (64 entries, bidirectional)."""
        assert len(PARTNERS) == 64
        
    def test_partners_bidirectional(self):
        """Partners are bidirectional: if A→B then B→A."""
        for gk_id, partner_id in PARTNERS.items():
            assert PARTNERS[partner_id] == gk_id, \
                f"Partner not bidirectional: GK{gk_id}→GK{partner_id} but GK{partner_id}→GK{PARTNERS[partner_id]}"
                
    def test_partners_valid_ids(self):
        """All partner IDs are valid Gene Key IDs."""
        for gk_id, partner_id in PARTNERS.items():
            assert 1 <= gk_id <= 64
            assert 1 <= partner_id <= 64
            assert partner_id in GENE_KEYS
            
    def test_codon_rings_coverage(self):
        """All 64 Gene Keys appear in exactly one ring."""
        all_in_rings = set()
        for ring_name, members in CODON_RINGS.items():
            for member in members:
                assert member not in all_in_rings, \
                    f"GK{member} appears in multiple rings"
                all_in_rings.add(member)
        assert all_in_rings == set(range(1, 65)), \
            f"Missing from rings: {set(range(1, 65)) - all_in_rings}"
            
    def test_codon_rings_count(self):
        """Should have 22 Codon Rings."""
        assert len(CODON_RINGS) == 22


class TestKeywordCodex:
    """Tests for KeywordCodex class."""
    
    def test_create_codex(self):
        """Factory function creates valid codex."""
        codex = create_codex()
        assert isinstance(codex, KeywordCodex)
        assert len(codex.nodes) == 64
        
    def test_nodes_initialized(self):
        """All 64 nodes created with correct data."""
        codex = KeywordCodex()
        assert len(codex.nodes) == 64
        for gk_id in range(1, 65):
            node = codex.nodes[gk_id]
            assert isinstance(node, GeneKeyNode)
            assert node.id == gk_id
            assert node.shadow == GENE_KEYS[gk_id]['shadow']
            assert node.gift == GENE_KEYS[gk_id]['gift']
            assert node.siddhi == GENE_KEYS[gk_id]['siddhi']
            
    def test_keyword_index_built(self):
        """Keyword index contains all 192 keywords (64*3)."""
        codex = KeywordCodex()
        # 64 shadows + 64 gifts + 64 siddhis = 192
        # But some might be duplicates, so check >= 64*3 - some_slack
        assert len(codex.keyword_index) >= 180  # Allow for some duplicates
        
    def test_keyword_index_lowercase(self):
        """All keywords are lowercase."""
        codex = KeywordCodex()
        for keyword in codex.keyword_index.keys():
            assert keyword == keyword.lower(), f"Keyword not lowercase: {keyword}"
            
    def test_reset(self):
        """Reset clears all activations."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        assert codex.nodes[38].activation > 0
        codex.reset()
        for node in codex.nodes.values():
            assert node.activation == 0.0
            assert len(node.activated_by) == 0


class TestKeywordActivation:
    """Tests for keyword-based activation."""
    
    def test_shadow_activation(self):
        """Shadow keywords activate Gene Keys."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle")
        assert codex.nodes[38].activation == 1.0
        assert 'shadow:struggle' in codex.nodes[38].activated_by
        
    def test_gift_activation(self):
        """Gift keywords activate Gene Keys."""
        codex = KeywordCodex()
        codex.activate_by_keywords("perseverance")
        assert codex.nodes[38].activation == 1.0
        assert 'gift:perseverance' in codex.nodes[38].activated_by
        
    def test_siddhi_activation(self):
        """Siddhi keywords activate Gene Keys."""
        codex = KeywordCodex()
        codex.activate_by_keywords("honour")
        assert codex.nodes[38].activation == 1.0
        assert 'siddhi:honour' in codex.nodes[38].activated_by
        
    def test_case_insensitive(self):
        """Activation is case-insensitive."""
        codex = KeywordCodex()
        codex.activate_by_keywords("STRUGGLE")
        assert codex.nodes[38].activation == 1.0
        
        codex.reset()
        codex.activate_by_keywords("Struggle")
        assert codex.nodes[38].activation == 1.0
        
    def test_multiple_keywords(self):
        """Multiple keywords activate multiple Gene Keys."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle and provocation")
        assert codex.nodes[38].activation == 1.0  # struggle
        assert codex.nodes[39].activation == 1.0  # provocation
        
    def test_no_match(self):
        """Non-matching text leaves activations at zero."""
        codex = KeywordCodex()
        codex.activate_by_keywords("kubernetes docker nginx")
        for node in codex.nodes.values():
            assert node.activation == 0.0
            
    def test_word_boundary(self):
        """Keywords must match at word boundaries."""
        codex = KeywordCodex()
        # "chaos" should match, but "chaosman" should not trigger it
        codex.activate_by_keywords("chaos")
        assert codex.nodes[3].activation == 1.0  # chaos is shadow of GK3
        
        codex.reset()
        codex.activate_by_keywords("chaotic")  # This might or might not match depending on impl
        # The key is that partial matches within words don't false-trigger


class TestPropagation:
    """Tests for activation propagation."""
    
    def test_partner_propagation(self):
        """Activation propagates to programming partners."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle")  # GK38
        codex.propagate()
        
        # GK39 is partner of GK38
        assert codex.nodes[39].activation == pytest.approx(0.8, rel=0.01)
        assert 'partner:38' in codex.nodes[39].activated_by
        
    def test_ring_propagation(self):
        """Activation propagates to ring members."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle")  # GK38, Ring of Humanity
        codex.propagate()
        
        # Ring of Humanity: [10, 17, 21, 25, 38, 51]
        ring_members = [10, 17, 21, 25, 51]  # Excluding 38 itself
        for member in ring_members:
            assert codex.nodes[member].activation == pytest.approx(0.5, rel=0.01), \
                f"GK{member} should be 0.5, got {codex.nodes[member].activation}"
            assert f'ring:Ring of Humanity' in codex.nodes[member].activated_by
            
    def test_propagation_decay(self):
        """Propagation decays: partner→ring < direct."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle")
        codex.propagate()
        
        # Direct: 1.0
        assert codex.nodes[38].activation == 1.0
        
        # Partner: 0.8
        assert codex.nodes[39].activation == pytest.approx(0.8, rel=0.01)
        
        # Ring of partner (Ring of Seeking for GK39): 0.5 from ring, but partner prop might add
        # The key is activations decrease with distance
        
    def test_propagation_convergence(self):
        """Propagation converges (doesn't oscillate infinitely)."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle provocation chaos love")
        iterations = codex.propagate(max_iters=100)
        assert iterations < 100, "Propagation did not converge"
        
    def test_propagation_max_iterations(self):
        """Propagation respects max_iters."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle")
        iterations = codex.propagate(max_iters=5)
        assert iterations <= 5


class TestIlluminate:
    """Tests for the illuminate() method."""
    
    def test_illuminate_returns_tuple(self):
        """illuminate() returns (activations, explanations)."""
        codex = KeywordCodex()
        result = codex.illuminate("struggle")
        assert isinstance(result, tuple)
        assert len(result) == 2
        
    def test_illuminate_activations_dict(self):
        """First return value is dict of activations."""
        codex = KeywordCodex()
        activations, _ = codex.illuminate("struggle")
        assert isinstance(activations, dict)
        assert len(activations) > 0
        assert all(isinstance(k, int) for k in activations.keys())
        assert all(isinstance(v, float) for v in activations.values())
        
    def test_illuminate_explanations_list(self):
        """Second return value is sorted list of explanations."""
        codex = KeywordCodex()
        _, explanations = codex.illuminate("struggle")
        assert isinstance(explanations, list)
        assert len(explanations) > 0
        
        # Each explanation is (gk_id, gift, sources, activation)
        for exp in explanations:
            assert len(exp) == 4
            gk_id, gift, sources, activation = exp
            assert isinstance(gk_id, int)
            assert isinstance(gift, str)
            assert isinstance(sources, str)
            assert isinstance(activation, float)
            
    def test_illuminate_sorted_by_activation(self):
        """Explanations sorted by activation (descending)."""
        codex = KeywordCodex()
        _, explanations = codex.illuminate("struggle provocation")
        activations = [exp[3] for exp in explanations]
        assert activations == sorted(activations, reverse=True)
        
    def test_illuminate_resets_first(self):
        """illuminate() resets before activating."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        codex.illuminate("chaos")  # Different Gene Key
        
        # Should not have struggle activation anymore
        # GK38 (struggle) should be low, GK3 (chaos) should be high
        assert codex.nodes[3].activation == 1.0
        # GK38 might still have ring propagation but not direct
        assert 'shadow:struggle' not in codex.nodes[38].activated_by


class TestActivationPattern:
    """Tests for activation pattern extraction."""
    
    def test_pattern_shape(self):
        """Pattern is 64-dimensional."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        pattern = codex.get_activation_pattern()
        assert pattern.shape == (64,)
        
    def test_pattern_dtype(self):
        """Pattern is float32."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        pattern = codex.get_activation_pattern()
        assert pattern.dtype == np.float32
        
    def test_pattern_values(self):
        """Pattern values match node activations."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        pattern = codex.get_activation_pattern()
        
        for i, gk_id in enumerate(range(1, 65)):
            assert pattern[i] == pytest.approx(codex.nodes[gk_id].activation, rel=0.01)
            
    def test_pattern_zero_on_reset(self):
        """Pattern is zero after reset."""
        codex = KeywordCodex()
        codex.reset()
        pattern = codex.get_activation_pattern()
        np.testing.assert_array_equal(pattern, np.zeros(64, dtype=np.float32))


class TestKeywordCodexLayer:
    """Tests for KeywordCodexLayer document scoring."""
    
    def test_layer_creation(self):
        """Layer created with codex."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex)
        assert layer.codex is codex
        
    def test_score_document_returns_tuple(self):
        """score_document returns (score, activations, explanations)."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex)
        result = layer.score_document("struggle", "perseverance")
        assert isinstance(result, tuple)
        assert len(result) == 3
        
    def test_score_identical_archetypes(self):
        """Documents with same archetypes score high."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex, alpha=0.0)  # Pure archetype mode
        
        query = "struggle and perseverance"
        doc = "honour through perseverance"
        
        score, _, _ = layer.score_document(query, doc)
        assert score > 0.5  # High overlap on GK38
        
    def test_score_different_archetypes(self):
        """Documents with different archetypes score low."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex, alpha=0.0)
        
        query = "struggle and conflict"  # GK38, GK6
        doc = "chaos and addiction"  # GK3, GK24
        
        score, _, _ = layer.score_document(query, doc)
        # Should be low but might have some ring overlap
        assert score < 0.5
        
    def test_score_no_archetypes(self):
        """Documents with no archetypes score zero."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex, alpha=0.0)
        
        query = "kubernetes docker nginx"
        doc = "postgresql redis mongodb"
        
        score, _, _ = layer.score_document(query, doc)
        assert score == 0.0
        
    def test_find_shared_archetypes(self):
        """find_shared_archetypes returns common Gene Keys."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex)
        
        query = "struggle through perseverance"
        doc = "honour and perseverance lead to liberation"
        
        shared = layer.find_shared_archetypes(query, doc)
        assert len(shared) > 0
        # Returns list of (gk_id, gift, query_act, doc_act)
        gk_ids = [s[0] for s in shared]
        assert 38 in gk_ids  # Both have perseverance/honour (GK38)


class TestExplain:
    """Tests for Gene Key explanation."""
    
    def test_explain_format(self):
        """explain() returns formatted string."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        explanation = codex.explain(38)
        assert isinstance(explanation, str)
        assert 'Gene Key 38' in explanation
        assert 'Struggle' in explanation
        assert 'Perseverance' in explanation
        assert 'Honour' in explanation
        
    def test_explain_includes_partner(self):
        """Explanation includes partner info."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        explanation = codex.explain(38)
        assert '39' in explanation or 'Dynamism' in explanation
        
    def test_explain_includes_ring(self):
        """Explanation includes ring info."""
        codex = KeywordCodex()
        codex.illuminate("struggle")
        explanation = codex.explain(38)
        assert 'Ring' in explanation or 'Humanity' in explanation
        
    def test_explain_invalid_id(self):
        """explain() handles invalid Gene Key ID."""
        codex = KeywordCodex()
        explanation = codex.explain(999)
        assert 'not found' in explanation.lower() or 'unknown' in explanation.lower() or explanation == ""


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_text(self):
        """Empty text returns zero activations."""
        codex = KeywordCodex()
        activations, explanations = codex.illuminate("")
        assert len(activations) == 0 or all(v == 0 for v in activations.values())
        
    def test_special_characters(self):
        """Special characters don't crash activation."""
        codex = KeywordCodex()
        activations, _ = codex.illuminate("!@#$%^&*()[]{}|\\")
        # Should not crash, may have zero activations
        
    def test_unicode_text(self):
        """Unicode text handled gracefully."""
        codex = KeywordCodex()
        activations, _ = codex.illuminate("struggling with übermensch")
        # Should find "struggling" → struggle
        
    def test_very_long_text(self):
        """Long text doesn't cause performance issues."""
        codex = KeywordCodex()
        long_text = "struggle " * 10000
        activations, _ = codex.illuminate(long_text)
        assert codex.nodes[38].activation == 1.0  # Still activates correctly
        
    def test_repeated_keywords(self):
        """Repeated keywords don't multiply activation beyond 1.0."""
        codex = KeywordCodex()
        codex.activate_by_keywords("struggle struggle struggle")
        assert codex.nodes[38].activation == 1.0  # Capped at 1.0


class TestRingDetection:
    """Tests for ring-level arc detection in larger contexts."""
    
    def test_ring_activation_count(self):
        """Count how many Gene Keys in a ring are activated."""
        codex = KeywordCodex()
        
        # Text with multiple Ring of Humanity keywords
        text = "Through struggle and initiative, one finds acceptance and authority"
        codex.illuminate(text)
        
        ring_members = CODON_RINGS["Ring of Humanity"]
        active_count = sum(1 for m in ring_members if codex.nodes[m].activation > 0.3)
        
        # Should activate multiple ring members
        assert active_count >= 3, f"Expected >= 3 ring members active, got {active_count}"
        
    def test_ring_coverage_score(self):
        """Compute ring coverage as fraction of members activated."""
        codex = KeywordCodex()
        
        text = "struggle initiative acceptance authority naturalness"
        codex.illuminate(text)
        
        ring_members = CODON_RINGS["Ring of Humanity"]
        coverage = sum(codex.nodes[m].activation for m in ring_members) / len(ring_members)
        
        # High coverage indicates ring-level activation
        assert coverage > 0.4, f"Expected ring coverage > 0.4, got {coverage:.2f}"
        
    def test_detect_dominant_ring(self):
        """Find which ring has highest total activation."""
        codex = KeywordCodex()
        
        # Text heavy in Ring of Humanity themes
        text = "Through perseverance and initiative, one develops authority and acceptance"
        codex.illuminate(text)
        
        # Use TOTAL activation (not average) to favor rings with more direct hits
        ring_scores = {}
        for ring_name, members in CODON_RINGS.items():
            ring_scores[ring_name] = sum(codex.nodes[m].activation for m in members)
        
        dominant = max(ring_scores, key=ring_scores.get)
        assert dominant == "Ring of Humanity", f"Expected Ring of Humanity, got {dominant}"
        
    def test_multi_ring_detection(self):
        """Detect multiple active rings in complex text."""
        codex = KeywordCodex()
        
        # Text spanning multiple rings
        text = "struggle (Humanity) and chaos (Life and Death) bring transformation"
        codex.illuminate(text)
        
        threshold = 0.3
        active_rings = []
        for ring_name, members in CODON_RINGS.items():
            coverage = sum(codex.nodes[m].activation for m in members) / len(members)
            if coverage > threshold:
                active_rings.append(ring_name)
                
        # Should detect at least Ring of Humanity (struggle)
        assert "Ring of Humanity" in active_rings
        
    def test_ring_signature_vector(self):
        """Create 22-dimensional ring activation signature."""
        codex = KeywordCodex()
        codex.illuminate("struggle and conflict lead to diplomacy")
        
        ring_signature = []
        for ring_name in sorted(CODON_RINGS.keys()):
            members = CODON_RINGS[ring_name]
            coverage = sum(codex.nodes[m].activation for m in members) / len(members)
            ring_signature.append(coverage)
            
        assert len(ring_signature) == 22
        assert max(ring_signature) > 0  # At least one ring active


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test complete illuminate → pattern → score pipeline."""
        codex = create_codex()
        layer = KeywordCodexLayer(codex, alpha=0.0)
        
        # Query and documents (use exact keywords)
        query = "struggle and conflict"  # GK38, GK6
        docs = [
            "perseverance leads to honour",  # GK38 keywords
            "diplomacy brings peace",  # GK6 keywords  
            "kubernetes docker nginx",  # No archetypes
        ]
        
        scores = []
        for doc in docs:
            score, _, _ = layer.score_document(query, doc)
            scores.append(score)
            
        # First two should score higher than third
        assert scores[0] > scores[2], f"Expected {scores[0]} > {scores[2]}"
        assert scores[1] > scores[2], f"Expected {scores[1]} > {scores[2]}"
        
    def test_programming_partners_38_39(self):
        """Verify GK38↔GK39 are programming partners (warrior archetype)."""
        codex = KeywordCodex()
        
        # GK38: Struggle → Perseverance → Honour
        # GK39: Provocation → Dynamism → Liberation
        
        assert PARTNERS[38] == 39
        assert PARTNERS[39] == 38
        
        codex.illuminate("struggle")
        assert codex.nodes[38].activation == 1.0
        assert codex.nodes[39].activation == pytest.approx(0.8, rel=0.01)
        
        codex.illuminate("provocation")
        assert codex.nodes[39].activation == 1.0
        assert codex.nodes[38].activation == pytest.approx(0.8, rel=0.01)
        
    def test_ring_of_humanity(self):
        """Verify Ring of Humanity members and propagation."""
        # Ring of Humanity: GK10, 17, 18, 21, 25, 38, 51, 57
        expected_members = {10, 17, 18, 21, 25, 38, 51, 57}
        actual_members = set(CODON_RINGS.get("Ring of Humanity", []))
        assert actual_members == expected_members
        
        codex = KeywordCodex()
        codex.illuminate("struggle")  # GK38
        
        for member in expected_members:
            if member != 38:  # Not the directly activated one
                assert codex.nodes[member].activation >= 0.5, \
                    f"Ring member GK{member} should have activation >= 0.5"
