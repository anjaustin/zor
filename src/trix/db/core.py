"""
DB Cooper: Core functions for Octave DB.

Ternary quantization, pooling, and similarity.
"""

import numpy as np
from typing import Tuple, Optional


def ternary_quantize(
    embeddings: np.ndarray,
    threshold: float = 0.0,
    sparsity_target: Optional[float] = None,
) -> np.ndarray:
    """
    Quantize float embeddings to ternary {-1, 0, +1}.
    
    Args:
        embeddings: Float array of shape [..., D]
        threshold: Values with |x| <= threshold become 0
        sparsity_target: If set, auto-compute threshold to achieve target sparsity
                        (fraction of zeros). Overrides threshold.
    
    Returns:
        Ternary array of same shape, dtype int8
    
    The 0 is not absence. It's "not activated" - Prime Meaning.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if sparsity_target is not None:
        # Find threshold that gives target sparsity
        abs_vals = np.abs(embeddings).flatten()
        threshold = np.percentile(abs_vals, sparsity_target * 100)
    
    # Quantize: sign where |x| > threshold, else 0
    ternary = np.zeros_like(embeddings, dtype=np.int8)
    ternary[embeddings > threshold] = 1
    ternary[embeddings < -threshold] = -1
    
    return ternary


def derive_coarse(
    fine: np.ndarray,
    pool_factor: int = 4,
) -> np.ndarray:
    """
    Derive coarse signature from fine by pooling.
    
    coarse = sign(mean(fine_chunks))
    
    Args:
        fine: Ternary array of shape [..., D]
        pool_factor: How many dimensions to pool together
    
    Returns:
        Coarser ternary array of shape [..., D // pool_factor]
    
    This is the core insight: coarse is a VIEW of fine, not independent.
    """
    fine = np.asarray(fine, dtype=np.int8)
    *batch_dims, d = fine.shape
    
    if d % pool_factor != 0:
        # Pad to make divisible
        pad_size = pool_factor - (d % pool_factor)
        fine = np.pad(fine, [*[(0, 0)] * len(batch_dims), (0, pad_size)])
        d = fine.shape[-1]
    
    # Reshape to [..., D//pool_factor, pool_factor]
    reshaped = fine.reshape(*batch_dims, d // pool_factor, pool_factor)
    
    # Mean pool and sign
    pooled = reshaped.mean(axis=-1)
    coarse = np.sign(pooled).astype(np.int8)
    
    return coarse


def derive_hierarchy(
    fine: np.ndarray,
    pool_factor: int = 4,
    num_levels: int = 3,
) -> Tuple[np.ndarray, ...]:
    """
    Derive full octave hierarchy from fine embeddings.
    
    Args:
        fine: Ternary array of shape [..., D]
        pool_factor: Pooling factor between levels
        num_levels: Number of levels (including fine)
    
    Returns:
        Tuple of (fine, medium, coarse, ...) from finest to coarsest
    """
    levels = [fine]
    current = fine
    
    for _ in range(num_levels - 1):
        current = derive_coarse(current, pool_factor)
        levels.append(current)
    
    return tuple(levels)


def pack_ternary(ternary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack ternary values into two bitmasks for fast similarity.
    
    Args:
        ternary: Array of {-1, 0, +1} values
    
    Returns:
        (positive_mask, negative_mask) as uint8 packed bits
    
    For dimension d:
        positive_mask[d] = 1 if ternary[d] == +1
        negative_mask[d] = 1 if ternary[d] == -1
        (zeros are not represented - they vanish)
    """
    ternary = np.asarray(ternary, dtype=np.int8)
    positive = (ternary == 1)
    negative = (ternary == -1)
    
    # Pack bits into bytes
    pos_packed = np.packbits(positive.astype(np.uint8), axis=-1)
    neg_packed = np.packbits(negative.astype(np.uint8), axis=-1)
    
    return pos_packed, neg_packed


def unpack_ternary(
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    num_dims: int,
) -> np.ndarray:
    """
    Unpack bitmasks back to ternary values.
    
    Args:
        positive_mask: Packed bits for +1 values
        negative_mask: Packed bits for -1 values
        num_dims: Original number of dimensions
    
    Returns:
        Ternary array of {-1, 0, +1}
    """
    positive = np.unpackbits(positive_mask, axis=-1)[..., :num_dims]
    negative = np.unpackbits(negative_mask, axis=-1)[..., :num_dims]
    
    ternary = positive.astype(np.int8) - negative.astype(np.int8)
    return ternary


def ternary_similarity(
    query: np.ndarray,
    document: np.ndarray,
) -> np.ndarray:
    """
    Compute similarity between ternary vectors.
    
    score = agreement - conflict
          = (q·d where both non-zero and same sign) 
          - (q·d where both non-zero and opposite sign)
    
    This is equivalent to dot product on ternary values.
    
    Args:
        query: Ternary array of shape [D] or [Q, D]
        document: Ternary array of shape [D] or [N, D]
    
    Returns:
        Similarity scores
    """
    query = np.asarray(query, dtype=np.int8)
    document = np.asarray(document, dtype=np.int8)
    
    # Dot product on ternary is just sum of products
    # (+1)(+1) = 1, (-1)(-1) = 1, (+1)(-1) = -1, (0)(x) = 0
    if query.ndim == 1 and document.ndim == 1:
        return np.sum(query * document)
    elif query.ndim == 1:
        return np.sum(query * document, axis=-1)
    elif document.ndim == 1:
        return np.sum(query * document, axis=-1)
    else:
        # [Q, D] @ [D, N] -> [Q, N]
        return query @ document.T


def ternary_similarity_packed(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    d_pos: np.ndarray,
    d_neg: np.ndarray,
) -> int:
    """
    Compute similarity using packed bitmasks. Fast bit operations.
    
    score = popcount(q_pos & d_pos) + popcount(q_neg & d_neg)
          - popcount(q_pos & d_neg) - popcount(q_neg & d_pos)
    
    Args:
        q_pos, q_neg: Query positive/negative masks (packed uint8)
        d_pos, d_neg: Document positive/negative masks (packed uint8)
    
    Returns:
        Integer similarity score
    """
    # Agreement: both positive or both negative
    agree_pos = np.unpackbits(q_pos & d_pos).sum()
    agree_neg = np.unpackbits(q_neg & d_neg).sum()
    
    # Conflict: one positive, one negative
    conflict_1 = np.unpackbits(q_pos & d_neg).sum()
    conflict_2 = np.unpackbits(q_neg & d_pos).sum()
    
    return int(agree_pos + agree_neg - conflict_1 - conflict_2)


def explain_match(
    query: np.ndarray,
    document: np.ndarray,
) -> dict:
    """
    Explain WHY query and document match. Glassbox.
    
    Args:
        query: Ternary array [D]
        document: Ternary array [D]
    
    Returns:
        {
            'agreement': indices where both agree (same non-zero sign),
            'conflict': indices where both disagree (opposite signs),
            'query_only': indices where only query is non-zero,
            'document_only': indices where only document is non-zero,
            'both_zero': indices where both are zero,
            'score': total similarity score
        }
    """
    query = np.asarray(query, dtype=np.int8)
    document = np.asarray(document, dtype=np.int8)
    
    q_nonzero = query != 0
    d_nonzero = document != 0
    same_sign = query == document
    opposite_sign = (query == -document) & q_nonzero & d_nonzero
    
    return {
        'agreement': np.where(same_sign & q_nonzero & d_nonzero)[0].tolist(),
        'conflict': np.where(opposite_sign)[0].tolist(),
        'query_only': np.where(q_nonzero & ~d_nonzero)[0].tolist(),
        'document_only': np.where(d_nonzero & ~q_nonzero)[0].tolist(),
        'both_zero': np.where(~q_nonzero & ~d_nonzero)[0].tolist(),
        'score': int(np.sum(query * document)),
    }
