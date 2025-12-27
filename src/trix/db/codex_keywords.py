"""
Codex Layer with Keyword-Based Activation

The Gene Keys vocabulary IS the structure.
Each Gene Key has multiple semantic anchors:
- Shadow (unconscious pattern)
- Gift (conscious expression)  
- Siddhi (transcendent state)
- Codon Ring (amino acid family)
- Programming Partner (paired archetype)

Activation through keywords, propagation through graph.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import re


# ============================================================================
# THE 64 GENE KEYS - COMPLETE VOCABULARY
# ============================================================================

GENE_KEYS = {
    1:  {"shadow": "Entropy", "gift": "Freshness", "siddhi": "Beauty"},
    2:  {"shadow": "Dislocation", "gift": "Orientation", "siddhi": "Unity"},
    3:  {"shadow": "Chaos", "gift": "Innovation", "siddhi": "Innocence"},
    4:  {"shadow": "Intolerance", "gift": "Understanding", "siddhi": "Forgiveness"},
    5:  {"shadow": "Impatience", "gift": "Patience", "siddhi": "Timelessness"},
    6:  {"shadow": "Conflict", "gift": "Diplomacy", "siddhi": "Peace"},
    7:  {"shadow": "Division", "gift": "Guidance", "siddhi": "Virtue"},
    8:  {"shadow": "Mediocrity", "gift": "Style", "siddhi": "Exquisiteness"},
    9:  {"shadow": "Inertia", "gift": "Determination", "siddhi": "Invincibility"},
    10: {"shadow": "Self-Obsession", "gift": "Naturalness", "siddhi": "Being"},
    11: {"shadow": "Obscurity", "gift": "Idealism", "siddhi": "Light"},
    12: {"shadow": "Vanity", "gift": "Discrimination", "siddhi": "Purity"},
    13: {"shadow": "Discord", "gift": "Discernment", "siddhi": "Empathy"},
    14: {"shadow": "Compromise", "gift": "Competence", "siddhi": "Bounteousness"},
    15: {"shadow": "Dullness", "gift": "Magnetism", "siddhi": "Florescence"},
    16: {"shadow": "Indifference", "gift": "Versatility", "siddhi": "Mastery"},
    17: {"shadow": "Opinion", "gift": "Far-Sightedness", "siddhi": "Omniscience"},
    18: {"shadow": "Judgement", "gift": "Integrity", "siddhi": "Perfection"},
    19: {"shadow": "Co-Dependence", "gift": "Sensitivity", "siddhi": "Sacrifice"},
    20: {"shadow": "Superficiality", "gift": "Self-Assurance", "siddhi": "Presence"},
    21: {"shadow": "Control", "gift": "Authority", "siddhi": "Valour"},
    22: {"shadow": "Dishonour", "gift": "Graciousness", "siddhi": "Grace"},
    23: {"shadow": "Complexity", "gift": "Simplicity", "siddhi": "Quintessence"},
    24: {"shadow": "Addiction", "gift": "Invention", "siddhi": "Silence"},
    25: {"shadow": "Constriction", "gift": "Acceptance", "siddhi": "Universal Love"},
    26: {"shadow": "Pride", "gift": "Artfulness", "siddhi": "Invisibility"},
    27: {"shadow": "Selfishness", "gift": "Altruism", "siddhi": "Selflessness"},
    28: {"shadow": "Purposelessness", "gift": "Totality", "siddhi": "Immortality"},
    29: {"shadow": "Half-Heartedness", "gift": "Commitment", "siddhi": "Devotion"},
    30: {"shadow": "Desire", "gift": "Lightness", "siddhi": "Rapture"},
    31: {"shadow": "Arrogance", "gift": "Leadership", "siddhi": "Humility"},
    32: {"shadow": "Failure", "gift": "Preservation", "siddhi": "Veneration"},
    33: {"shadow": "Forgetting", "gift": "Mindfulness", "siddhi": "Revelation"},
    34: {"shadow": "Force", "gift": "Strength", "siddhi": "Majesty"},
    35: {"shadow": "Hunger", "gift": "Adventure", "siddhi": "Boundlessness"},
    36: {"shadow": "Turbulence", "gift": "Humanity", "siddhi": "Compassion"},
    37: {"shadow": "Weakness", "gift": "Equality", "siddhi": "Tenderness"},
    38: {"shadow": "Struggle", "gift": "Perseverance", "siddhi": "Honour"},
    39: {"shadow": "Provocation", "gift": "Dynamism", "siddhi": "Liberation"},
    40: {"shadow": "Exhaustion", "gift": "Resolve", "siddhi": "Divine Will"},
    41: {"shadow": "Fantasy", "gift": "Anticipation", "siddhi": "Emanation"},
    42: {"shadow": "Expectation", "gift": "Detachment", "siddhi": "Celebration"},
    43: {"shadow": "Deafness", "gift": "Insight", "siddhi": "Epiphany"},
    44: {"shadow": "Interference", "gift": "Teamwork", "siddhi": "Synarchy"},
    45: {"shadow": "Dominance", "gift": "Synergy", "siddhi": "Communion"},
    46: {"shadow": "Seriousness", "gift": "Delight", "siddhi": "Ecstasy"},
    47: {"shadow": "Oppression", "gift": "Transmutation", "siddhi": "Transfiguration"},
    48: {"shadow": "Inadequacy", "gift": "Resourcefulness", "siddhi": "Wisdom"},
    49: {"shadow": "Reaction", "gift": "Revolution", "siddhi": "Rebirth"},
    50: {"shadow": "Corruption", "gift": "Equilibrium", "siddhi": "Harmony"},
    51: {"shadow": "Agitation", "gift": "Initiative", "siddhi": "Awakening"},
    52: {"shadow": "Stress", "gift": "Restraint", "siddhi": "Stillness"},
    53: {"shadow": "Immaturity", "gift": "Expansion", "siddhi": "Superabundance"},
    54: {"shadow": "Greed", "gift": "Aspiration", "siddhi": "Ascension"},
    55: {"shadow": "Victimisation", "gift": "Freedom", "siddhi": "Freedom"},
    56: {"shadow": "Distraction", "gift": "Enrichment", "siddhi": "Intoxication"},
    57: {"shadow": "Unease", "gift": "Intuition", "siddhi": "Clarity"},
    58: {"shadow": "Dissatisfaction", "gift": "Vitality", "siddhi": "Bliss"},
    59: {"shadow": "Dishonesty", "gift": "Intimacy", "siddhi": "Transparency"},
    60: {"shadow": "Limitation", "gift": "Realism", "siddhi": "Justice"},
    61: {"shadow": "Psychosis", "gift": "Inspiration", "siddhi": "Sanctity"},
    62: {"shadow": "Intellect", "gift": "Precision", "siddhi": "Impeccability"},
    63: {"shadow": "Doubt", "gift": "Inquiry", "siddhi": "Truth"},
    64: {"shadow": "Confusion", "gift": "Imagination", "siddhi": "Illumination"},
}

# Programming Partners (32 pairs)
PARTNERS = {
    1: 2, 2: 1, 3: 50, 50: 3, 4: 49, 49: 4, 5: 35, 35: 5,
    6: 36, 36: 6, 7: 13, 13: 7, 8: 14, 14: 8, 9: 16, 16: 9,
    10: 15, 15: 10, 11: 12, 12: 11, 17: 18, 18: 17, 19: 33, 33: 19,
    20: 34, 34: 20, 21: 48, 48: 21, 22: 47, 47: 22, 23: 43, 43: 23,
    24: 44, 44: 24, 25: 46, 46: 25, 26: 45, 45: 26, 27: 28, 28: 27,
    29: 30, 30: 29, 31: 41, 41: 31, 32: 42, 42: 32, 37: 40, 40: 37,
    38: 39, 39: 38, 51: 57, 57: 51, 52: 58, 58: 52, 53: 54, 54: 53,
    55: 59, 59: 55, 56: 60, 60: 56, 61: 62, 62: 61, 63: 64, 64: 63,
}

# Codon Rings (21 amino acid families)
CODON_RINGS = {
    "Ring of Fire": [1, 14],
    "Ring of Water": [2, 8],
    "Ring of Life and Death": [3, 20, 23, 24, 27, 42],
    "Ring of Union": [4, 7, 29],
    "Ring of Light": [5, 9, 11, 26],
    "Ring of Alchemy": [6, 40, 47, 64],
    "Ring of Purification": [13, 30],
    "Ring of Secrets": [12],
    "Ring of Seeking": [15, 39, 52, 53, 54, 58],
    "Ring of Prosperity": [16, 45],
    "Ring of Humanity": [10, 17, 21, 25, 38, 51],
    "Ring of Illusion": [28, 32],
    "Ring of Trials": [33, 56],
    "Ring of Destiny": [34],
    "Ring of Miracles": [35],
    "Ring of Divinity": [37],
    "Ring of Origin": [41],
    "Ring of No Return": [31, 62],
    "Ring of Matter": [46, 48],
    "Ring of the Whirlwind": [49],
    "Ring of Gaia": [19, 60, 61],
    "Ring of the Illuminati": [44, 59],
}

# Build reverse mapping
GK_TO_RING = {}
for ring_name, gk_ids in CODON_RINGS.items():
    for gk_id in gk_ids:
        GK_TO_RING[gk_id] = ring_name


@dataclass
class GeneKeyNode:
    """A Gene Key with all its semantic anchors."""
    id: int
    shadow: str
    gift: str
    siddhi: str
    partner: Optional[int]
    codon_ring: Optional[str]
    
    # Activation state
    activation: float = 0.0
    _inbox: float = 0.0
    
    # Which layer triggered activation
    activated_by: List[str] = field(default_factory=list)
    
    @property
    def keywords(self) -> Set[str]:
        """All keywords for this Gene Key."""
        kw = {self.shadow.lower(), self.gift.lower(), self.siddhi.lower()}
        # Handle compound words
        for word in [self.shadow, self.gift, self.siddhi]:
            kw.update(w.lower() for w in word.replace('-', ' ').split())
        return kw
    
    def receive(self, amount: float, source: str = ""):
        """Receive activation from propagation."""
        self._inbox = max(self._inbox, amount)
        if source and source not in self.activated_by:
            self.activated_by.append(source)
    
    def process(self) -> float:
        """Process inbox, return change in activation."""
        prev = self.activation
        self.activation = max(self.activation, self._inbox)
        self._inbox = 0.0
        return self.activation - prev
    
    def reset(self):
        self.activation = 0.0
        self._inbox = 0.0
        self.activated_by = []


class KeywordCodex:
    """
    Codex with keyword-based activation.
    
    Keywords trigger Gene Keys directly.
    Activation propagates through partner and ring relationships.
    """
    
    # Activation weights by layer
    SHADOW_WEIGHT = 1.0   # Shadow keywords = full activation
    GIFT_WEIGHT = 1.0     # Gift keywords = full activation
    SIDDHI_WEIGHT = 1.0   # Siddhi keywords = full activation
    
    # Propagation weights
    PARTNER_WEIGHT = 0.8  # Programming partners
    RING_WEIGHT = 0.5     # Codon ring members
    
    # Convergence
    CONVERGENCE_THRESHOLD = 1e-4
    MAX_ITERATIONS = 10
    
    def __init__(self):
        # Build nodes
        self.nodes: Dict[int, GeneKeyNode] = {}
        for gk_id, data in GENE_KEYS.items():
            self.nodes[gk_id] = GeneKeyNode(
                id=gk_id,
                shadow=data['shadow'],
                gift=data['gift'],
                siddhi=data['siddhi'],
                partner=PARTNERS.get(gk_id),
                codon_ring=GK_TO_RING.get(gk_id),
            )
        
        # Build keyword index (keyword -> list of (gk_id, layer, weight))
        self.keyword_index: Dict[str, List[Tuple[int, str, float]]] = {}
        self._build_keyword_index()
        
        # Build ring membership
        self.ring_members: Dict[str, List[int]] = CODON_RINGS.copy()
    
    def _build_keyword_index(self):
        """Index all keywords for fast lookup."""
        for gk_id, node in self.nodes.items():
            # Shadow
            for kw in self._tokenize(node.shadow):
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                self.keyword_index[kw].append((gk_id, 'shadow', self.SHADOW_WEIGHT))
            
            # Gift
            for kw in self._tokenize(node.gift):
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                self.keyword_index[kw].append((gk_id, 'gift', self.GIFT_WEIGHT))
            
            # Siddhi
            for kw in self._tokenize(node.siddhi):
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                self.keyword_index[kw].append((gk_id, 'siddhi', self.SIDDHI_WEIGHT))
    
    def _tokenize(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Lowercase, split on non-alpha, filter short words
        words = re.findall(r'[a-z]+', text.lower())
        return [w for w in words if len(w) >= 3]
    
    def reset(self):
        """Reset all activations."""
        for node in self.nodes.values():
            node.reset()
    
    def activate_by_keywords(self, text: str) -> Dict[int, float]:
        """
        Activate Gene Keys based on keywords in text.
        
        Returns dict of gk_id -> activation strength.
        """
        tokens = set(self._tokenize(text))
        activations = {}
        
        for token in tokens:
            if token in self.keyword_index:
                for gk_id, layer, weight in self.keyword_index[token]:
                    if gk_id not in activations:
                        activations[gk_id] = 0.0
                    activations[gk_id] = max(activations[gk_id], weight)
                    self.nodes[gk_id].activation = max(self.nodes[gk_id].activation, weight)
                    self.nodes[gk_id].activated_by.append(f"{layer}:{token}")
        
        return activations
    
    def propagate_step(self) -> float:
        """
        One step of propagation through partners and rings.
        Returns max change.
        """
        # Collect messages
        messages: Dict[int, Tuple[float, str]] = {}
        
        for gk_id, node in self.nodes.items():
            if node.activation < 0.01:
                continue
            
            # Propagate to partner
            if node.partner and node.partner in self.nodes:
                amount = node.activation * self.PARTNER_WEIGHT
                partner_node = self.nodes[node.partner]
                if node.partner not in messages or messages[node.partner][0] < amount:
                    messages[node.partner] = (amount, f"partner:{gk_id}")
            
            # Propagate to ring members
            if node.codon_ring and node.codon_ring in self.ring_members:
                for ring_gk_id in self.ring_members[node.codon_ring]:
                    if ring_gk_id != gk_id:
                        amount = node.activation * self.RING_WEIGHT
                        if ring_gk_id not in messages or messages[ring_gk_id][0] < amount:
                            messages[ring_gk_id] = (amount, f"ring:{node.codon_ring}")
        
        # Apply messages
        for gk_id, (amount, source) in messages.items():
            self.nodes[gk_id].receive(amount, source)
        
        # Process and get max change
        max_change = 0.0
        for node in self.nodes.values():
            change = node.process()
            max_change = max(max_change, change)
        
        return max_change
    
    def propagate(self, max_iters: Optional[int] = None) -> int:
        """Propagate until convergence. Returns iterations."""
        max_iters = max_iters or self.MAX_ITERATIONS
        
        for i in range(max_iters):
            change = self.propagate_step()
            if change < self.CONVERGENCE_THRESHOLD:
                return i + 1
        
        return max_iters
    
    def illuminate(self, text: str) -> Tuple[Dict[int, float], List[Tuple[int, str, str, float]]]:
        """
        Full illumination pipeline.
        
        Args:
            text: Input text to analyze
            
        Returns:
            (activation_dict, explanations)
            explanations: [(gk_id, gift, activated_by, activation), ...]
        """
        self.reset()
        
        # Keyword activation
        direct = self.activate_by_keywords(text)
        
        # Propagate
        iters = self.propagate()
        
        # Build explanations
        explanations = []
        for gk_id, node in sorted(self.nodes.items(), key=lambda x: -x[1].activation):
            if node.activation > 0.1:
                sources = ", ".join(node.activated_by[:3])
                explanations.append((gk_id, node.gift, sources, node.activation))
        
        # Build activation dict
        activations = {gk_id: node.activation for gk_id, node in self.nodes.items()}
        
        return activations, explanations
    
    def get_activation_pattern(self) -> np.ndarray:
        """Get current activation as [64] vector."""
        pattern = np.zeros(64, dtype=np.float32)
        for gk_id, node in self.nodes.items():
            pattern[gk_id - 1] = node.activation
        return pattern
    
    def explain(self, gk_id: int) -> str:
        """Explain why a Gene Key is activated."""
        if gk_id not in self.nodes:
            return f"Gene Key {gk_id} not found"
        
        node = self.nodes[gk_id]
        lines = [
            f"Gene Key {gk_id}:",
            f"  Shadow: {node.shadow}",
            f"  Gift: {node.gift}",
            f"  Siddhi: {node.siddhi}",
            f"  Partner: {node.partner} ({self.nodes[node.partner].gift if node.partner else 'none'})",
            f"  Ring: {node.codon_ring}",
            f"  Activation: {node.activation:.3f}",
            f"  Activated by: {', '.join(node.activated_by) if node.activated_by else 'none'}",
        ]
        return "\n".join(lines)


class KeywordCodexLayer:
    """
    Integration layer combining keyword activation with embedding similarity.
    """
    
    def __init__(self, codex: KeywordCodex, alpha: float = 0.5):
        """
        Args:
            codex: KeywordCodex instance
            alpha: Weight for embedding similarity (0-1)
                   0 = pure keyword, 1 = pure embedding
        """
        self.codex = codex
        self.alpha = alpha
    
    def score_document(
        self,
        query_text: str,
        doc_text: str,
        query_embedding: Optional[np.ndarray] = None,
        doc_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[int, float], List[str]]:
        """
        Score a document against a query using keywords + optional embeddings.
        
        Returns:
            (combined_score, activation_pattern, explanations)
        """
        # Get query activation pattern
        self.codex.reset()
        query_acts, query_expl = self.codex.illuminate(query_text)
        query_pattern = self.codex.get_activation_pattern()
        
        # Get document activation pattern
        self.codex.reset()
        doc_acts, doc_expl = self.codex.illuminate(doc_text)
        doc_pattern = self.codex.get_activation_pattern()
        
        # Archetypal alignment (cosine similarity of patterns)
        q_norm = query_pattern / (np.linalg.norm(query_pattern) + 1e-8)
        d_norm = doc_pattern / (np.linalg.norm(doc_pattern) + 1e-8)
        archetype_score = float(np.dot(q_norm, d_norm))
        
        # Embedding similarity (if provided)
        if query_embedding is not None and doc_embedding is not None:
            q_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            d_emb = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)
            embedding_score = float(np.dot(q_emb, d_emb))
        else:
            embedding_score = 0.0
            self.alpha = 0.0  # Pure keyword mode
        
        # Combined score
        combined = self.alpha * embedding_score + (1 - self.alpha) * archetype_score
        
        # Explanations
        explanations = []
        for gk_id, gift, sources, act in query_expl[:3]:
            if doc_acts.get(gk_id, 0) > 0.1:
                explanations.append(f"Both activate GK{gk_id} ({gift})")
        
        return combined, doc_acts, explanations
    
    def find_shared_archetypes(
        self, 
        query_text: str, 
        doc_text: str
    ) -> List[Tuple[int, str, float, float]]:
        """
        Find Gene Keys activated by both query and document.
        
        Returns:
            [(gk_id, gift, query_activation, doc_activation), ...]
        """
        self.codex.reset()
        query_acts, _ = self.codex.illuminate(query_text)
        
        self.codex.reset()
        doc_acts, _ = self.codex.illuminate(doc_text)
        
        shared = []
        for gk_id in range(1, 65):
            q_act = query_acts.get(gk_id, 0)
            d_act = doc_acts.get(gk_id, 0)
            if q_act > 0.1 and d_act > 0.1:
                gift = self.codex.nodes[gk_id].gift
                shared.append((gk_id, gift, q_act, d_act))
        
        return sorted(shared, key=lambda x: -(x[2] + x[3]))


def create_codex() -> KeywordCodex:
    """Factory function to create a KeywordCodex."""
    return KeywordCodex()
