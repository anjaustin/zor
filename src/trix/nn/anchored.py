"""
Anchored Dual-Mode FFN: Partition First, Search Within

The anchor doesn't predict the answer. It partitions the question.
Speedup comes from NOT searching, not from searching faster.

Architecture:
    Input ──┬──→ [Anchor Shapes] ──→ partition (fast, frozen)
            │         │
            │         ↓
            └──→ [Router(x, anchor)] ──→ tile selection (learned)
                          │
                          ↓
                [Tile Execution] ──→ output (frozen + scale)

Two deployment modes from one substrate:
    - Chips: shapes only (deterministic, synthesizable)
    - Modal-models: both paths (shapes constrain probabilistic search)

Born from the Lincoln Manifold Method, December 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class AnchoredDualModeFFN(nn.Module):
    """
    Dual-mode FFN with frozen anchors and learned routing.

    Anchors partition the input space (shoreline).
    Router searches within the partition (navigation).
    Tiles execute with learned scales (port operations).

    The anchor is the shoreline. The routing is the voyage.
    You can't sail through land, and that's the point.

    Args:
        d_model: Model dimension
        num_anchors: Number of anchor partitions (frozen signatures)
        num_tiles: Number of execution tiles
        temperature: Softmax temperature for anchor (lower = harder partitions)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        num_anchors: int = 16,
        num_tiles: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.num_tiles = num_tiles
        self.temperature = temperature

        # ═══════════════════════════════════════════
        # FROZEN COMPONENTS (discovered, not learned)
        # ═══════════════════════════════════════════

        # Anchor shapes: partition the input space
        # Each anchor is a ternary signature for Hamming matching
        anchor_init = torch.randn(num_anchors, d_model)
        self.register_buffer('anchor_signatures', anchor_init.sign())

        # Tile directions: ternary transformation directions
        tile_init = torch.randn(num_tiles, d_model)
        self.register_buffer('tile_directions', tile_init.sign())

        # ═══════════════════════════════════════════
        # LEARNED COMPONENTS (trained with gradients)
        # ═══════════════════════════════════════════

        # Anchor projection: learned interpretation of anchor signal
        self.anchor_proj = nn.Linear(num_anchors, num_anchors)

        # Router: sees input AND anchor, selects tile
        self.router = nn.Sequential(
            nn.Linear(d_model + num_anchors, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_tiles),
        )

        # Per-tile learned scales (magnitude modulation)
        self.tile_scales = nn.Parameter(torch.ones(num_tiles) * 0.1)

        # Output normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Track anchor usage for regularization
        self.register_buffer('anchor_usage', torch.zeros(num_anchors))
        self.register_buffer('usage_count', torch.tensor(0.0))

    def compute_anchor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frozen anchor computation via Hamming similarity.

        Fast matching against anchor signatures.
        Returns soft partition membership.

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            anchor_probs: Partition membership [batch, seq, num_anchors]
            anchor_logits: Raw similarity scores [batch, seq, num_anchors]
        """
        # Ternary quantize input for matching (frozen operation)
        x_ternary = x.sign()

        # Hamming similarity = d_model - 2 * hamming_distance
        # For ternary vectors: dot product gives same ranking as Hamming
        similarity = torch.einsum('bsd,ad->bsa', x_ternary, self.anchor_signatures)

        # Normalize by d_model for stability
        similarity = similarity / math.sqrt(self.d_model)

        # Temperature-controlled softmax
        anchor_probs = F.softmax(similarity / self.temperature, dim=-1)

        return anchor_probs, similarity

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with dual-mode computation.

        1. Anchor path (frozen): compute partition
        2. Router path (learned): select tile given (input, anchor)
        3. Execution (frozen + learned): apply tile with scale

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            output: Transformed tensor [batch, seq, d_model]
            info: Dictionary with routing information
        """
        batch, seq, _ = x.shape

        # ═══════════════════════════════════════════
        # PATH 1: ANCHOR (frozen, fast)
        # ═══════════════════════════════════════════
        anchor_probs, anchor_logits = self.compute_anchor(x)

        # Learned interpretation of anchor signal
        anchor_features = self.anchor_proj(anchor_probs)

        # Track anchor usage (for regularization and monitoring)
        if self.training:
            with torch.no_grad():
                self.anchor_usage += anchor_probs.sum(dim=(0, 1))
                self.usage_count += batch * seq

        # ═══════════════════════════════════════════
        # PATH 2: ROUTING (learned, informed by anchor)
        # ═══════════════════════════════════════════
        routing_input = torch.cat([x, anchor_features], dim=-1)
        tile_logits = self.router(routing_input)  # [batch, seq, num_tiles]
        tile_probs = F.softmax(tile_logits, dim=-1)  # [batch, seq, num_tiles]
        tile_idx = tile_logits.argmax(dim=-1)     # [batch, seq]

        # ═══════════════════════════════════════════
        # PATH 3: EXECUTION (frozen directions, learned scales)
        # ═══════════════════════════════════════════

        if self.training:
            # Training: soft weighted combination (gradients flow through tile_probs)
            # directions: [num_tiles, d_model] -> weighted sum
            # tile_probs: [batch, seq, num_tiles]
            directions = torch.einsum(
                'bst,td->bsd', tile_probs, self.tile_directions
            )  # [batch, seq, d_model]

            # scales: weighted combination of tile scales
            scales = torch.einsum(
                'bst,t->bs', tile_probs, self.tile_scales
            )  # [batch, seq]
        else:
            # Inference: hard selection (fast, deterministic)
            tile_idx_expanded = tile_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            directions = torch.gather(
                self.tile_directions.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1, -1),
                dim=2,
                index=tile_idx_expanded.unsqueeze(2)
            ).squeeze(2)  # [batch, seq, d_model]

            scales = self.tile_scales[tile_idx]  # [batch, seq]

        # Apply transformation: residual + scaled direction
        delta = scales.unsqueeze(-1) * directions
        delta = self.dropout(delta)
        output = self.norm(x + delta)

        # ═══════════════════════════════════════════
        # INFO (for analysis and debugging)
        # ═══════════════════════════════════════════

        # Anchor entropy (lower = sharper partitions)
        anchor_entropy = -(anchor_probs * (anchor_probs + 1e-10).log()).sum(-1).mean()

        # Dominant anchor per position
        dominant_anchor = anchor_probs.argmax(dim=-1)

        # Tile utilization
        unique_tiles = tile_idx.unique()
        tile_utilization = len(unique_tiles) / self.num_tiles

        info = {
            'anchor_probs': anchor_probs,
            'anchor_logits': anchor_logits,
            'anchor_entropy': anchor_entropy,
            'dominant_anchor': dominant_anchor,
            'tile_idx': tile_idx,
            'tile_logits': tile_logits,
            'tile_probs': tile_probs,
            'tile_utilization': tile_utilization,
            'temperature': self.temperature,
        }

        return output, info

    def set_temperature(self, temperature: float):
        """
        Adjust anchor temperature.

        Use during training for annealing:
            - High temperature (>1): soft partitions, exploration
            - Low temperature (<1): hard partitions, commitment

        Args:
            temperature: New temperature value
        """
        self.temperature = temperature

    def get_anchor_utilization(self) -> torch.Tensor:
        """
        Get anchor utilization distribution.

        Returns:
            Normalized usage per anchor [num_anchors]
        """
        if self.usage_count > 0:
            return self.anchor_usage / self.usage_count
        return torch.ones(self.num_anchors) / self.num_anchors

    def reset_usage_stats(self):
        """Reset anchor usage tracking."""
        self.anchor_usage.zero_()
        self.usage_count.zero_()

    def compute_anchor_loss(self, target_entropy: float = 2.0) -> torch.Tensor:
        """
        Compute regularization loss for anchor utilization.

        Encourages:
            1. All anchors to be used (no dead partitions)
            2. Reasonable entropy (not too uniform, not too peaked)

        Args:
            target_entropy: Target entropy for anchor distribution

        Returns:
            Regularization loss scalar
        """
        usage = self.get_anchor_utilization()

        # Encourage uniform usage (all anchors contribute)
        uniform = torch.ones_like(usage) / self.num_anchors
        usage_loss = F.kl_div(
            (usage + 1e-10).log(),
            uniform,
            reduction='batchmean'
        )

        return usage_loss

    def freeze_anchors_from_data(self, data_loader, num_batches: int = 100):
        """
        Discover anchor signatures from data distribution.

        Runs k-means-style clustering on input embeddings,
        then ternarizes centroids to get anchor signatures.

        Args:
            data_loader: DataLoader yielding input tensors
            num_batches: Number of batches to sample
        """
        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                # Flatten to [N, d_model]
                x_flat = x.view(-1, self.d_model)
                all_embeddings.append(x_flat)

            embeddings = torch.cat(all_embeddings, dim=0)

            # Simple k-means initialization: random subset
            indices = torch.randperm(embeddings.size(0))[:self.num_anchors]
            centroids = embeddings[indices]

            # Few iterations of k-means
            for _ in range(10):
                # Assign to nearest centroid
                dists = torch.cdist(embeddings, centroids)
                assignments = dists.argmin(dim=1)

                # Update centroids
                for k in range(self.num_anchors):
                    mask = assignments == k
                    if mask.sum() > 0:
                        centroids[k] = embeddings[mask].mean(dim=0)

            # Ternarize and store
            self.anchor_signatures.copy_(centroids.sign())

        self.train()


class AnchoredDualModeBlock(nn.Module):
    """
    Transformer block using AnchoredDualModeFFN.

    Standard attention + anchored FFN.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_anchors: int = 16,
        num_tiles: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = AnchoredDualModeFFN(
            d_model=d_model,
            num_anchors=num_anchors,
            num_tiles=num_tiles,
            temperature=temperature,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [batch, seq, d_model]
            attn_mask: Optional attention mask

        Returns:
            output: Transformed tensor
            info: FFN routing information
        """
        # Attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # Anchored FFN (includes its own residual and norm)
        x, info = self.ffn(x)

        return x, info

    def set_temperature(self, temperature: float):
        """Set anchor temperature."""
        self.ffn.set_temperature(temperature)


def get_temperature_schedule(
    step: int,
    total_steps: int,
    start_temp: float = 2.0,
    end_temp: float = 0.1,
    schedule: str = 'linear',
) -> float:
    """
    Get temperature for current training step.

    Args:
        step: Current step
        total_steps: Total training steps
        start_temp: Initial temperature (warm, exploratory)
        end_temp: Final temperature (cold, committed)
        schedule: 'linear', 'cosine', or 'exponential'

    Returns:
        Temperature value
    """
    progress = min(step / total_steps, 1.0)

    if schedule == 'linear':
        return start_temp + (end_temp - start_temp) * progress
    elif schedule == 'cosine':
        return end_temp + (start_temp - end_temp) * (1 + math.cos(math.pi * progress)) / 2
    elif schedule == 'exponential':
        return start_temp * (end_temp / start_temp) ** progress
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
