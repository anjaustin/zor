"""
XOR Routing FFN - Content-Addressable Routing via Hamming Distance

Instead of dot product routing (argmax(x @ signatures.T)), we use
Hamming distance routing (argmin(hamming(binarize(x), signatures))).

This connects FFNs to ProvidenceMemory:
- Both use XOR-based matching
- Both treat signatures as content addresses
- Both crystallize to a winner

Benefits:
- O(1) XOR + popcount vs O(d) multiply-accumulate per comparison
- Natural content-addressable semantics
- Direct connection to attention (soft Providence)

Mathematical foundation:
    hamming(a, b) = popcount(a XOR b)
    For ternary: distance = sum(a != b)

    Gradient via straight-through estimator (STE)
    Soft option: softmax(-hamming / temperature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


# =============================================================================
# CORE XOR OPERATIONS
# =============================================================================

def binarize_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Binarize to {-1, +1} with straight-through estimator.

    Forward: sign(x), treating 0 as +1
    Backward: gradient flows through as if identity
    """
    with torch.no_grad():
        binary = x.sign()
        binary[binary == 0] = 1.0
    return x + (binary - x).detach()


def ternarize_ste(x: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Ternarize to {-1, 0, +1} with straight-through estimator.

    Forward: -1 if x < -threshold, +1 if x > threshold, 0 otherwise
    Backward: gradient flows through as if identity
    """
    with torch.no_grad():
        ternary = torch.zeros_like(x)
        ternary[x > threshold] = 1.0
        ternary[x < -threshold] = -1.0
    return x + (ternary - x).detach()


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distance between ternary vectors.

    For ternary {-1, 0, +1}: count positions where values differ.
    This is equivalent to: sum(a != b)

    Args:
        a: [batch, d] or [batch, 1, d]
        b: [n, d] or [1, n, d]

    Returns:
        distances: [batch, n]
    """
    if a.dim() == 2:
        a = a.unsqueeze(1)  # [batch, 1, d]
    if b.dim() == 2:
        b = b.unsqueeze(0)  # [1, n, d]

    # Count mismatches
    diff = (a != b).float().sum(dim=-1)  # [batch, n]
    return diff


def soft_hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft Hamming distance for training.

    Uses: 1 - cosine_similarity as a proxy for normalized Hamming.
    This provides smooth gradients while approximating Hamming behavior.

    Args:
        a: [batch, d]
        b: [n, d]

    Returns:
        distances: [batch, n] soft distances
    """
    # Normalize
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    # Cosine similarity → distance
    similarity = a_norm @ b_norm.T  # [batch, n]
    distance = 1 - similarity  # [0, 2] range

    return distance


# =============================================================================
# XOR ROUTING LAYER
# =============================================================================

class XORRouter(nn.Module):
    """
    XOR-based routing module using Hamming distance.

    Routes inputs to tiles based on minimum Hamming distance
    to tile signatures, mirroring ProvidenceMemory semantics.
    """

    def __init__(
        self,
        d_model: int,
        num_tiles: int,
        use_soft_training: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.use_soft_training = use_soft_training
        self.temperature = temperature

        # Learnable signatures (will be ternarized)
        self.signatures = nn.Parameter(torch.randn(num_tiles, d_model) * 0.5)

        # EMA for stable routing
        self.register_buffer('ema_signatures', None)
        self.ema_decay = 0.99

    def get_ternary_signatures(self) -> torch.Tensor:
        """Get ternarized signatures."""
        return ternarize_ste(self.signatures)

    def forward(
        self,
        x: torch.Tensor,
        return_distances: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to tiles via XOR matching.

        Args:
            x: [batch, d_model] input
            return_distances: if True, return distances instead of one-hot

        Returns:
            tile_idx: [batch] winning tile indices
            gate: [batch, num_tiles] one-hot or soft weights
        """
        batch = x.shape[0]
        device = x.device

        # Ternarize input and signatures
        x_tern = ternarize_ste(x)
        sigs = self.get_ternary_signatures()

        # Update EMA during training
        if self.training:
            if self.ema_signatures is None:
                self.ema_signatures = sigs.detach().clone()
            else:
                self.ema_signatures = (
                    self.ema_decay * self.ema_signatures +
                    (1 - self.ema_decay) * sigs.detach()
                )

        # Compute distances
        if self.training and self.use_soft_training:
            # Soft distance for gradient flow
            distances = soft_hamming_distance(x, self.signatures)
        else:
            # Hard Hamming distance
            distances = hamming_distance(x_tern, sigs)

        # Route to minimum distance
        tile_idx = distances.argmin(dim=-1)

        if return_distances:
            return tile_idx, distances

        # Create gate (one-hot or soft)
        if self.training and self.use_soft_training:
            # Soft routing via softmax(-distance/temp)
            gate = F.softmax(-distances / self.temperature, dim=-1)
        else:
            # Hard one-hot routing
            gate = F.one_hot(tile_idx, self.num_tiles).float()

        return tile_idx, gate


# =============================================================================
# XOR ROUTING FFN
# =============================================================================

class XORRoutingFFN(nn.Module):
    """
    Feed-forward network with XOR-based (Hamming distance) routing.

    Architecture:
        1. Ternarize input
        2. Compute Hamming distance to tile signatures
        3. Route to minimum distance tile
        4. Execute tile computation
        5. Residual connection

    This connects the FFN to ProvidenceMemory:
    - Signatures = content addresses
    - Tiles = values (computations)
    - Routing = lookup

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension per tile
        num_tiles: Number of routing tiles
        dropout: Dropout rate
        use_soft_training: Use soft routing during training
        temperature: Softmax temperature for soft routing
        use_residual: Add residual connection

    Example:
        >>> ffn = XORRoutingFFN(d_model=128, num_tiles=16)
        >>> output, routing_info, aux = ffn(x)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        num_tiles: int = 16,
        dropout: float = 0.1,
        use_soft_training: bool = True,
        temperature: float = 1.0,
        use_residual: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 4 // num_tiles
        self.num_tiles = num_tiles
        self.use_residual = use_residual

        # XOR router
        self.router = XORRouter(
            d_model=d_model,
            num_tiles=num_tiles,
            use_soft_training=use_soft_training,
            temperature=temperature,
        )

        # Per-tile transformations
        self.tile_up = nn.ModuleList([
            nn.Linear(d_model, self.d_hidden)
            for _ in range(num_tiles)
        ])
        self.tile_down = nn.ModuleList([
            nn.Linear(self.d_hidden, d_model)
            for _ in range(num_tiles)
        ])

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Usage tracking
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('total_count', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with XOR routing.

        Args:
            x: [batch, d_model] or [batch, seq, d_model]

        Returns:
            output: Same shape as input
            routing_info: Dict with tile_idx, gate, distances
            aux_losses: Dict with balance loss
        """
        orig_shape = x.shape
        residual = x

        # Flatten if 3D
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            residual = residual.view(B * T, C)

        batch = x.shape[0]
        device = x.device

        # Normalize
        x_norm = self.norm(x)

        # XOR routing
        tile_idx, gate = self.router(x_norm)

        # Sparse computation
        output = torch.zeros(batch, self.d_model, device=device)

        if self.training and self.router.use_soft_training:
            # Soft routing: weighted combination
            all_outputs = []
            for t in range(self.num_tiles):
                hidden = F.relu(self.tile_up[t](x_norm))
                tile_out = self.tile_down[t](hidden)
                all_outputs.append(tile_out)

            all_outputs = torch.stack(all_outputs, dim=1)  # [batch, num_tiles, d_model]
            output = (gate.unsqueeze(-1) * all_outputs).sum(dim=1)
        else:
            # Hard routing: only winner computes
            for t in range(self.num_tiles):
                mask = tile_idx == t
                if not mask.any():
                    continue

                hidden = F.relu(self.tile_up[t](x_norm[mask]))
                tile_out = self.tile_down[t](hidden)
                output[mask] = tile_out

                # Track usage
                if self.training:
                    self.tile_counts[t] += mask.sum().float()

        if self.training:
            self.total_count += batch

        # Apply dropout
        output = self.dropout(output)

        # Residual
        if self.use_residual:
            output = residual + output

        # Reshape if needed
        if len(orig_shape) == 3:
            output = output.view(orig_shape)
            tile_idx = tile_idx.view(orig_shape[0], orig_shape[1])
            gate = gate.view(orig_shape[0], orig_shape[1], -1)

        routing_info = {
            'tile_idx': tile_idx,
            'gate': gate,
        }

        aux_losses = self._compute_aux_losses(tile_idx.flatten(), batch)

        return output, routing_info, aux_losses

    def _compute_aux_losses(self, tile_idx: torch.Tensor, total: int) -> Dict:
        """Compute load balancing loss."""
        counts = torch.zeros(self.num_tiles, device=tile_idx.device)
        for t in range(self.num_tiles):
            counts[t] = (tile_idx == t).sum().float()

        # Balance loss: penalize deviation from uniform
        ideal = total / self.num_tiles
        balance_loss = ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)

        return {
            'balance': balance_loss * 0.01,
            'total_aux': balance_loss * 0.01,
        }

    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}

        usage = self.tile_counts / self.total_count

        return {
            'num_tiles': self.num_tiles,
            'active_tiles': (usage > 0.001).sum().item(),
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
            'usage_max': usage.max().item(),
            'usage_min': usage.min().item(),
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.tile_counts.zero_()
        self.total_count.zero_()

    def get_signatures(self) -> torch.Tensor:
        """Get router signatures."""
        return self.router.get_ternary_signatures()


# =============================================================================
# HIERARCHICAL XOR ROUTING FFN
# =============================================================================

class HierarchicalXORRoutingFFN(nn.Module):
    """
    Hierarchical XOR routing for O(sqrt(n)) scaling.

    Two-level routing:
        Level 1: Route to cluster via XOR matching
        Level 2: Route to tile within cluster via XOR matching

    For n tiles with k per cluster:
        Comparisons = n/k + k = O(sqrt(n)) when k = sqrt(n)

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension per tile
        num_tiles: Total number of tiles
        tiles_per_cluster: Tiles in each cluster
        dropout: Dropout rate
        use_soft_training: Use soft routing during training
        temperature: Softmax temperature
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        use_soft_training: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 4 // num_tiles
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_clusters = num_tiles // tiles_per_cluster

        assert num_tiles % tiles_per_cluster == 0

        # Cluster router
        self.cluster_router = XORRouter(
            d_model=d_model,
            num_tiles=self.num_clusters,
            use_soft_training=use_soft_training,
            temperature=temperature,
        )

        # Per-cluster tile routers
        self.tile_routers = nn.ModuleList([
            XORRouter(
                d_model=d_model,
                num_tiles=tiles_per_cluster,
                use_soft_training=use_soft_training,
                temperature=temperature,
            )
            for _ in range(self.num_clusters)
        ])

        # Per-tile transformations
        self.tile_up = nn.ModuleList([
            nn.Linear(d_model, self.d_hidden)
            for _ in range(num_tiles)
        ])
        self.tile_down = nn.ModuleList([
            nn.Linear(self.d_hidden, d_model)
            for _ in range(num_tiles)
        ])

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Usage tracking
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('cluster_counts', torch.zeros(self.num_clusters))
        self.register_buffer('total_count', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical XOR routing."""
        orig_shape = x.shape
        residual = x

        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            residual = residual.view(B * T, C)

        batch = x.shape[0]
        device = x.device

        # Normalize
        x_norm = self.norm(x)

        # Level 1: Cluster routing
        cluster_idx, cluster_gate = self.cluster_router(x_norm)

        # Level 2: Tile routing within clusters
        tile_idx = torch.zeros(batch, dtype=torch.long, device=device)
        tile_gate = torch.zeros(batch, self.num_tiles, device=device)

        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue

            local_idx, local_gate = self.tile_routers[c](x_norm[mask])
            global_idx = c * self.tiles_per_cluster + local_idx
            tile_idx[mask] = global_idx

            # Map local gate to global tile positions
            start = c * self.tiles_per_cluster
            end = start + self.tiles_per_cluster
            tile_gate[mask, start:end] = local_gate

        # Sparse computation
        output = torch.zeros(batch, self.d_model, device=device)

        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue

            hidden = F.relu(self.tile_up[t](x_norm[mask]))
            tile_out = self.tile_down[t](hidden)
            output[mask] = tile_out

            if self.training:
                self.tile_counts[t] += mask.sum().float()
                self.cluster_counts[t // self.tiles_per_cluster] += mask.sum().float()

        if self.training:
            self.total_count += batch

        output = self.dropout(output)
        output = residual + output

        # Reshape
        if len(orig_shape) == 3:
            output = output.view(orig_shape)
            tile_idx = tile_idx.view(orig_shape[0], orig_shape[1])
            cluster_idx = cluster_idx.view(orig_shape[0], orig_shape[1])

        routing_info = {
            'tile_idx': tile_idx,
            'cluster_idx': cluster_idx,
            'tile_gate': tile_gate if len(orig_shape) == 2 else tile_gate.view(orig_shape[0], orig_shape[1], -1),
        }

        aux_losses = self._compute_aux_losses(tile_idx.flatten(), cluster_idx.flatten(), batch)

        return output, routing_info, aux_losses

    def _compute_aux_losses(
        self,
        tile_idx: torch.Tensor,
        cluster_idx: torch.Tensor,
        total: int,
    ) -> Dict:
        """Compute hierarchical balance losses."""
        # Tile balance
        tile_counts = torch.zeros(self.num_tiles, device=tile_idx.device)
        for t in range(self.num_tiles):
            tile_counts[t] = (tile_idx == t).sum().float()
        tile_ideal = total / self.num_tiles
        tile_balance = ((tile_counts - tile_ideal) ** 2).mean() / (tile_ideal ** 2 + 1e-8)

        # Cluster balance
        cluster_counts = torch.zeros(self.num_clusters, device=cluster_idx.device)
        for c in range(self.num_clusters):
            cluster_counts[c] = (cluster_idx == c).sum().float()
        cluster_ideal = total / self.num_clusters
        cluster_balance = ((cluster_counts - cluster_ideal) ** 2).mean() / (cluster_ideal ** 2 + 1e-8)

        total_loss = (tile_balance + cluster_balance) * 0.01

        return {
            'tile_balance': tile_balance * 0.01,
            'cluster_balance': cluster_balance * 0.01,
            'total_aux': total_loss,
        }

    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}

        tile_usage = self.tile_counts / self.total_count
        cluster_usage = self.cluster_counts / self.total_count

        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'active_tiles': (tile_usage > 0.001).sum().item(),
            'active_clusters': (cluster_usage > 0.001).sum().item(),
            'tile_usage_std': tile_usage.std().item(),
            'cluster_usage_std': cluster_usage.std().item(),
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.tile_counts.zero_()
        self.cluster_counts.zero_()
        self.total_count.zero_()
