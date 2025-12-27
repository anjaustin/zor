"""
DB Cooper: Possibility Space Explorer

Transparent, end-to-end semantic understanding.
No black boxes. Every decision explained.

The insight: Don't just return similar vectors.
Understand what the query is ABOUT and return documents about that.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

from .core import octave_quantize


@dataclass
class Candidate:
    """A candidate document with full transparency."""
    id: str
    idx: int
    concept: Optional[str]
    
    # Scores at each stage
    weighted_score: float
    exact_score: float
    
    # Ranking at each stage
    weighted_rank: int
    exact_rank: int
    
    # Why included/excluded
    included: bool = True
    reason: str = ""


@dataclass
class ConceptCluster:
    """A cluster of candidates sharing a concept."""
    concept: str
    count: int
    avg_score: float
    max_score: float
    min_score: float
    candidates: List[Candidate] = field(default_factory=list)


@dataclass
class ExplorationResult:
    """Full transparent result of possibility space exploration."""
    
    # The query
    query_id: str
    
    # What we explored
    total_docs: int
    candidates_explored: int
    
    # What we found
    concepts_found: Dict[str, ConceptCluster]
    dominant_concept: str
    dominant_confidence: float  # How confident are we?
    
    # Final results
    results: List[Candidate]
    
    # Decision trace
    decisions: List[str] = field(default_factory=list)
    
    def explain(self) -> str:
        """Human-readable explanation of the entire process."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"EXPLORATION REPORT: {self.query_id}")
        lines.append("=" * 70)
        lines.append("")
        
        # Stage 1: Exploration
        lines.append(f"STAGE 1: EXPLORATION")
        lines.append(f"  Total documents: {self.total_docs}")
        lines.append(f"  Candidates explored: {self.candidates_explored}")
        lines.append("")
        
        # Stage 2: Structure Analysis
        lines.append("STAGE 2: STRUCTURE ANALYSIS")
        lines.append("  Concepts found in candidate set:")
        for concept, cluster in sorted(
            self.concepts_found.items(), 
            key=lambda x: -x[1].avg_score
        ):
            lines.append(
                f"    {concept}: {cluster.count} docs, "
                f"avg={cluster.avg_score:.3f}, "
                f"range=[{cluster.min_score:.3f}, {cluster.max_score:.3f}]"
            )
        lines.append("")
        
        # Stage 3: Understanding
        lines.append("STAGE 3: UNDERSTANDING")
        lines.append(f"  Dominant concept: {self.dominant_concept}")
        lines.append(f"  Confidence: {self.dominant_confidence:.0%}")
        lines.append("")
        
        # Decision trace
        lines.append("DECISION TRACE:")
        for i, decision in enumerate(self.decisions, 1):
            lines.append(f"  {i}. {decision}")
        lines.append("")
        
        # Final results
        lines.append("FINAL RESULTS:")
        for i, r in enumerate(self.results, 1):
            status = "✓" if r.included else "✗"
            lines.append(
                f"  {i}. [{status}] {r.id} ({r.concept}) "
                f"score={r.exact_score:.3f} "
                f"[{r.reason}]"
            )
        
        lines.append("=" * 70)
        return "\n".join(lines)


class PossibilityExplorer:
    """
    Explores the possibility space to understand aboutness.
    
    Not a black box. Every decision is traced and explainable.
    """
    
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        doc_signs: np.ndarray,
        doc_mags: np.ndarray,
        doc_concepts: List[str],
        doc_ids: List[str],
    ):
        """
        Initialize with document data.
        
        Args:
            doc_embeddings: Original float embeddings [N, D]
            doc_signs: Quantized signs [N, D]
            doc_mags: Quantized magnitudes [N, D]
            doc_concepts: Concept label for each doc
            doc_ids: ID for each doc
        """
        self.embeddings = doc_embeddings
        self.signs = doc_signs
        self.mags = doc_mags
        self.concepts = doc_concepts
        self.ids = doc_ids
        self.n_docs = len(doc_ids)
    
    def explore(
        self,
        query: np.ndarray,
        query_id: str = "query",
        explore_k: int = 400,
        top_k: int = 10,
        gap_threshold: float = 0.096369,
    ) -> ExplorationResult:
        """
        Explore the possibility space for a query.
        
        Returns a fully transparent result with decision trace.
        """
        decisions = []
        
        # Quantize query
        q_signs, q_mags = octave_quantize(query)
        q_norm = query / np.linalg.norm(query)
        
        decisions.append(
            f"Query quantized: {np.sum(q_signs == 1)} positive, "
            f"{np.sum(q_signs == -1)} negative, "
            f"{np.sum(q_signs == 0)} zero"
        )
        
        # Stage 1: Broad exploration via weighted similarity
        weights = np.sqrt(q_mags * self.mags)
        weighted_scores = np.sum(weights * (q_signs * self.signs), axis=1)
        
        candidate_indices = np.argsort(weighted_scores)[::-1][:explore_k]
        
        decisions.append(
            f"Explored top {explore_k} by weighted similarity "
            f"(score range: {weighted_scores[candidate_indices[0]]:.2f} to "
            f"{weighted_scores[candidate_indices[-1]]:.2f})"
        )
        
        # Stage 2: Exact cosine on candidates
        exact_scores = self.embeddings[candidate_indices] @ q_norm
        exact_order = np.argsort(exact_scores)[::-1]
        
        # Build candidates with full info
        candidates = []
        for rank, local_idx in enumerate(exact_order):
            idx = candidate_indices[local_idx]
            candidates.append(Candidate(
                id=self.ids[idx],
                idx=idx,
                concept=self.concepts[idx],
                weighted_score=float(weighted_scores[idx]),
                exact_score=float(exact_scores[local_idx]),
                weighted_rank=int(np.where(candidate_indices == idx)[0][0]) + 1,
                exact_rank=rank + 1,
            ))
        
        # Stage 3: Analyze structure - what concepts are present?
        concept_clusters = {}
        for c in candidates:
            if c.concept not in concept_clusters:
                concept_clusters[c.concept] = ConceptCluster(
                    concept=c.concept,
                    count=0,
                    avg_score=0,
                    max_score=float('-inf'),
                    min_score=float('inf'),
                    candidates=[],
                )
            cluster = concept_clusters[c.concept]
            cluster.count += 1
            cluster.max_score = max(cluster.max_score, c.exact_score)
            cluster.min_score = min(cluster.min_score, c.exact_score)
            cluster.candidates.append(c)
        
        # Compute averages
        for cluster in concept_clusters.values():
            scores = [c.exact_score for c in cluster.candidates]
            cluster.avg_score = np.mean(scores)
        
        decisions.append(
            f"Found {len(concept_clusters)} distinct concepts in candidates"
        )
        
        # Stage 4: Determine dominant concept
        # INSIGHT: Trust the score when confident, use concept detection when not.
        
        # First, check for a clear winner (one concept's best significantly exceeds others)
        concept_best = {
            concept: cluster.max_score 
            for concept, cluster in concept_clusters.items()
        }
        sorted_by_best = sorted(concept_best.items(), key=lambda x: -x[1])
        
        clear_winner = False
        if len(sorted_by_best) >= 2:
            best_concept, best_score = sorted_by_best[0]
            second_concept, second_score = sorted_by_best[1]
            gap = best_score - second_score
            
            if gap >= gap_threshold:
                # Clear winner - trust the similarity score
                clear_winner = True
                dominant_concept = best_concept
                dominant_cluster = concept_clusters[dominant_concept]
                confidence = min(1.0, 0.5 + gap)  # Higher gap = higher confidence
                
                decisions.append(
                    f"Clear winner detected: '{best_concept}' "
                    f"(best={best_score:.3f}) vs '{second_concept}' "
                    f"(best={second_score:.3f}), gap={gap:.3f} >= {gap_threshold}"
                )
        
        if not clear_winner:
            # Ambiguous - fall back to avg-based concept detection
            concept_ranking = sorted(
                concept_clusters.items(),
                key=lambda x: x[1].avg_score,
                reverse=True
            )
            
            dominant_concept = concept_ranking[0][0]
            dominant_cluster = concept_ranking[0][1]
            
            # Confidence based on avg gap
            if len(concept_ranking) > 1:
                runner_up = concept_ranking[1][1]
                avg_gap = dominant_cluster.avg_score - runner_up.avg_score
                confidence = min(1.0, max(0.0, avg_gap / 0.2 + 0.5))
            else:
                confidence = 1.0
            
            decisions.append(
                f"Ambiguous (gap < {gap_threshold}), using avg-based detection: "
                f"'{dominant_concept}' (avg={dominant_cluster.avg_score:.3f})"
            )
        
        decisions.append(
            f"Dominant concept: '{dominant_concept}' "
            f"(max={dominant_cluster.max_score:.3f}, "
            f"avg={dominant_cluster.avg_score:.3f}, "
            f"{dominant_cluster.count} candidates)"
        )
        decisions.append(f"Confidence: {confidence:.0%}")
        
        # Stage 5: Select results from dominant concept
        results = []
        for c in candidates:
            if c.concept == dominant_concept:
                c.included = True
                c.reason = f"matches dominant concept '{dominant_concept}'"
                results.append(c)
            else:
                c.included = False
                c.reason = f"concept '{c.concept}' != dominant '{dominant_concept}'"
        
        # Take top_k from dominant concept
        results = results[:top_k]
        
        decisions.append(
            f"Selected top {len(results)} from dominant concept"
        )
        
        # Count rejections
        rejected = sum(1 for c in candidates[:top_k] if not c.included)
        if rejected > 0:
            decisions.append(
                f"Rejected {rejected} confusers that would have been in top {top_k}"
            )
        
        return ExplorationResult(
            query_id=query_id,
            total_docs=self.n_docs,
            candidates_explored=explore_k,
            concepts_found=concept_clusters,
            dominant_concept=dominant_concept,
            dominant_confidence=confidence,
            results=results,
            decisions=decisions,
        )


def compare_approaches(
    explorer: PossibilityExplorer,
    query: np.ndarray,
    true_concept: str,
    confuser_concept: str,
    query_id: str = "query",
) -> Dict[str, Any]:
    """
    Compare Cooper exploration vs naive top-k.
    
    Returns transparent comparison.
    """
    q_norm = query / np.linalg.norm(query)
    
    # Naive approach: just return top-k by cosine
    naive_scores = explorer.embeddings @ q_norm
    naive_top10 = np.argsort(naive_scores)[::-1][:10]
    
    naive_correct = sum(
        1 for idx in naive_top10 
        if explorer.concepts[idx] == true_concept
    )
    naive_confuser = sum(
        1 for idx in naive_top10 
        if explorer.concepts[idx] == confuser_concept
    )
    
    # Cooper exploration
    result = explorer.explore(query, query_id=query_id)
    
    cooper_correct = sum(
        1 for r in result.results 
        if r.concept == true_concept
    )
    cooper_confuser = sum(
        1 for r in result.results 
        if r.concept == confuser_concept
    )
    
    return {
        'query_id': query_id,
        'true_concept': true_concept,
        'confuser_concept': confuser_concept,
        'naive': {
            'correct': naive_correct,
            'confuser': naive_confuser,
            'purity': naive_correct / (naive_correct + naive_confuser + 0.001),
        },
        'cooper': {
            'correct': cooper_correct,
            'confuser': cooper_confuser,
            'purity': cooper_correct / (cooper_correct + cooper_confuser + 0.001),
            'understood': result.dominant_concept == true_concept,
            'confidence': result.dominant_confidence,
        },
        'explanation': result.explain(),
    }
