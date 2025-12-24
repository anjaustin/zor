"""
Hierarchical Temporal FFN - O(sqrt(n)) Routing with Persistent State

Combines:
- Hierarchical routing from HierarchicalTriXFFN (O(√n) scaling)
- Temporal state from TemporalTileLayer (state persists across steps)

Each tile:
- Has a signature for routing (hierarchical: cluster → tile)
- Has state that persists across forward passes
- Routes based on (input, state) not just input
- Learns state transition function

This enables:
- Efficient routing at scale (O(√n))
- Temporal pattern recognition
- Regime-aware computation
- State-dependent specialization

Example:
    >>> ffn = HierarchicalTemporalFFN(d_model=128, num_tiles=64, d_state=16)
    >>> state = ffn.init_state(batch_size=4)
    >>> output, state, routing_info, aux = ffn(x, state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# TEMPORAL TILE
# =============================================================================

class TemporalTile(nn.Module):
    """
    A tile with persistent state.

    Each tile:
    - Has a signature (what inputs it responds to)
    - Has state that persists across calls
    - Learns to transform (input, state) → (output, new_state)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_state: int,
        tile_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_state = d_state
        self.tile_id = tile_id

        # Combined input dimension
        d_combined = d_model + d_state

        # Signature projection (for routing)
        self.signature_proj = nn.Parameter(torch.randn(d_model) * 0.02)

        # State-aware signature (for state-dependent routing)
        self.state_signature_proj = nn.Parameter(torch.randn(d_state) * 0.02)

        # Transformation: (input, state) → hidden
        self.up = nn.Linear(d_combined, d_hidden)

        # Output: hidden → output
        self.down = nn.Linear(d_hidden, d_model)

        # State update: (input, hidden, state) → new_state
        self.state_update = nn.Sequential(
            nn.Linear(d_combined + d_hidden, d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, d_state),
        )

        # Learnable scales
        self.output_scale = nn.Parameter(torch.ones(1))

        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))

    def get_signature(self) -> torch.Tensor:
        """Get tile's routing signature."""
        return self.signature_proj.sign()

    def get_combined_signature(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get signature that includes state contribution.

        This allows routing to depend on current state.
        """
        input_sig = self.signature_proj.sign()
        state_contribution = (state @ self.state_signature_proj).sign()
        return torch.cat([input_sig.expand(state.shape[0], -1), state_contribution.unsqueeze(-1)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input [batch, d_model]
            state: Tile state [batch, d_state]

        Returns:
            output: [batch, d_model]
            new_state: [batch, d_state]
        """
        # Combine input and state
        combined = torch.cat([x, state], dim=-1)

        # Transform
        hidden = F.gelu(self.up(combined))

        # Output
        output = self.down(hidden) * self.output_scale

        # State update
        update_input = torch.cat([combined, hidden], dim=-1)
        new_state = self.state_update(update_input)

        return output, new_state

    def update_usage(self, count: int, total: int):
        """Track activation frequency."""
        self.activation_count = self.activation_count + count
        self.total_count = self.total_count + total

    @property
    def usage_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()


# =============================================================================
# HIERARCHICAL TEMPORAL FFN
# =============================================================================

class HierarchicalTemporalFFN(nn.Module):
    """
    Hierarchical routing with per-tile temporal state.

    Combines O(√n) hierarchical routing with temporal state persistence.

    Architecture:
        Level 1: Route to cluster based on (input, global_state)
        Level 2: Route to tile within cluster based on (input, tile_states)
        Execution: Tile transforms input using its state
        Update: Tile state updates based on input/output

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension per tile
        d_state: State dimension per tile
        num_tiles: Total number of tiles
        tiles_per_cluster: Tiles per cluster
        d_global_state: Dimension of global (shared) state
        dropout: Dropout rate
        use_state_routing: Include state in routing decision
        temperature: Softmax temperature for soft routing
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        d_state: int = 16,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_global_state: int = 32,
        dropout: float = 0.1,
        use_state_routing: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 4 // num_tiles
        self.d_state = d_state
        self.d_global_state = d_global_state
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_clusters = num_tiles // tiles_per_cluster
        self.use_state_routing = use_state_routing
        self.temperature = temperature

        assert num_tiles % tiles_per_cluster == 0

        # Create tiles
        self.tiles = nn.ModuleList([
            TemporalTile(
                d_model=d_model,
                d_hidden=self.d_hidden,
                d_state=d_state,
                tile_id=i,
            )
            for i in range(num_tiles)
        ])

        # Cluster signatures (for level-1 routing)
        self.cluster_signatures = nn.Parameter(
            torch.randn(self.num_clusters, d_model) * 0.02
        )

        # Global state encoder (shared across all tiles)
        self.global_state_encoder = nn.Linear(d_model, d_global_state)

        # Global state transition
        self.global_state_update = nn.GRUCell(d_model, d_global_state)

        # State-aware routing projection (if enabled)
        if use_state_routing:
            self.state_routing_proj = nn.Linear(d_global_state, d_model)

        # Normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Tracking
        self.register_buffer('cluster_counts', torch.zeros(self.num_clusters))
        self.register_buffer('total_count', torch.tensor(0.0))

        # Transition tracking
        self.register_buffer(
            'transition_matrix',
            torch.zeros(num_tiles, num_tiles)
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize all states.

        Returns dict with:
        - global_state: [batch, d_global_state]
        - tile_states: [batch, num_tiles, d_state]
        - prev_tile: [batch] previous tile indices (for transition tracking)
        """
        if device is None:
            device = self.cluster_signatures.device

        return {
            'global_state': torch.zeros(batch_size, self.d_global_state, device=device),
            'tile_states': torch.zeros(batch_size, self.num_tiles, self.d_state, device=device),
            'prev_tile': None,
        }

    def _get_tile_signatures(self) -> torch.Tensor:
        """Get signatures from all tiles."""
        return torch.stack([tile.get_signature() for tile in self.tiles])

    def _route_hierarchical(
        self,
        x: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-level hierarchical routing.

        Args:
            x: Normalized input [batch, d_model]
            global_state: Global state [batch, d_global_state]

        Returns:
            tile_idx: [batch] winning tile indices
            cluster_idx: [batch] winning cluster indices
        """
        batch = x.shape[0]
        device = x.device

        # State-aware input for routing
        if self.use_state_routing and global_state is not None:
            state_contribution = self.state_routing_proj(global_state)
            routing_input = x + 0.1 * state_contribution  # Soft contribution
        else:
            routing_input = x

        # Level 1: Cluster routing
        cluster_scores = routing_input @ self.cluster_signatures.T / self.temperature
        cluster_idx = cluster_scores.argmax(dim=-1)

        # Level 2: Tile routing within cluster
        tile_signatures = self._get_tile_signatures()
        tile_idx = torch.zeros(batch, dtype=torch.long, device=device)

        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue

            # Get tiles in this cluster
            start = c * self.tiles_per_cluster
            end = start + self.tiles_per_cluster
            cluster_tile_sigs = tile_signatures[start:end]

            # Route within cluster
            scores = routing_input[mask] @ cluster_tile_sigs.T / self.temperature
            local_winners = scores.argmax(dim=-1)

            # Convert to global indices
            tile_idx[mask] = start + local_winners

        return tile_idx, cluster_idx

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        track_transitions: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict, Dict]:
        """
        Forward pass with hierarchical temporal routing.

        Args:
            x: Input [batch, d_model] or [batch, seq, d_model]
            state: State dict from init_state() or previous forward
            track_transitions: Whether to update transition matrix

        Returns:
            output: Same shape as input
            new_state: Updated state dict
            routing_info: Dict with routing decisions
            aux_losses: Dict with auxiliary losses
        """
        orig_shape = x.shape
        is_3d = x.dim() == 3

        if is_3d:
            B, T, C = x.shape
            x = x.view(B * T, C)

        batch = x.shape[0]
        device = x.device

        # Initialize state if needed
        if state is None:
            actual_batch = orig_shape[0] if is_3d else batch
            state = self.init_state(actual_batch, device)

        # Expand state for 3D input
        if is_3d:
            global_state = state['global_state'].unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            tile_states = state['tile_states']  # Keep as [B, num_tiles, d_state]
        else:
            global_state = state['global_state']
            tile_states = state['tile_states']

        # Normalize input
        x_norm = self.input_norm(x)

        # Hierarchical routing
        tile_idx, cluster_idx = self._route_hierarchical(x_norm, global_state)

        # Execute tiles with state
        output = torch.zeros(batch, self.d_model, device=device)
        new_tile_states = tile_states.clone()

        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue

            # Get tile inputs
            tile_x = x_norm[mask]

            # Get tile states (need to handle batch dimension correctly)
            if is_3d:
                # Map flat indices back to batch indices
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, T).reshape(-1)
                tile_batch_indices = batch_indices[mask]
                tile_state = tile_states[tile_batch_indices, t]
            else:
                tile_state = tile_states[mask, t]

            # Execute tile
            tile_out, new_state = self.tiles[t](tile_x, tile_state)
            output[mask] = tile_out

            # Update states
            if is_3d:
                # Scatter back to original batch positions
                for idx, (bi, new_s) in enumerate(zip(tile_batch_indices, new_state)):
                    new_tile_states[bi, t] = new_s
            else:
                new_tile_states[mask, t] = new_state

            # Track usage
            if self.training:
                self.tiles[t].update_usage(mask.sum().item(), batch)
                self.cluster_counts[t // self.tiles_per_cluster] += mask.sum().float()

        if self.training:
            self.total_count += batch

        # Update global state
        if is_3d:
            # Use mean of outputs for global state update
            output_mean = output.view(B, T, -1).mean(dim=1)
            new_global_state = self.global_state_update(output_mean, state['global_state'])
        else:
            new_global_state = self.global_state_update(output, state['global_state'])

        # Track transitions
        if track_transitions and state['prev_tile'] is not None:
            with torch.no_grad():
                prev = state['prev_tile']
                curr = tile_idx.view(orig_shape[0], -1)[:, -1] if is_3d else tile_idx
                for p, c in zip(prev, curr):
                    self.transition_matrix[p.item(), c.item()] += 1

        # Apply residual and dropout
        output = self.dropout(output)
        if is_3d:
            output = x.view(B * T, C) + output
            output = output.view(orig_shape)
        else:
            output = x + output

        # Prepare new state
        new_state = {
            'global_state': new_global_state,
            'tile_states': new_tile_states,
            'prev_tile': tile_idx.view(orig_shape[0], -1)[:, -1] if is_3d else tile_idx,
        }

        # Routing info
        routing_info = {
            'tile_idx': tile_idx.view(orig_shape[0], orig_shape[1]) if is_3d else tile_idx,
            'cluster_idx': cluster_idx.view(orig_shape[0], orig_shape[1]) if is_3d else cluster_idx,
        }

        # Aux losses
        aux_losses = self._compute_aux_losses(tile_idx, cluster_idx, batch)

        return output, new_state, routing_info, aux_losses

    def _compute_aux_losses(
        self,
        tile_idx: torch.Tensor,
        cluster_idx: torch.Tensor,
        total: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute balance losses."""
        device = tile_idx.device

        # Tile balance
        tile_counts = torch.zeros(self.num_tiles, device=device)
        for t in range(self.num_tiles):
            tile_counts[t] = (tile_idx == t).sum().float()
        tile_ideal = total / self.num_tiles
        tile_balance = ((tile_counts - tile_ideal) ** 2).mean() / (tile_ideal ** 2 + 1e-8)

        # Cluster balance
        cluster_counts = torch.zeros(self.num_clusters, device=device)
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

        tile_usage = torch.tensor([t.usage_rate for t in self.tiles])
        cluster_usage = self.cluster_counts / self.total_count

        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'd_state': self.d_state,
            'active_tiles': (tile_usage > 0.001).sum().item(),
            'active_clusters': (cluster_usage > 0.001).sum().item(),
            'tile_usage_std': tile_usage.std().item(),
            'cluster_usage_std': cluster_usage.std().item(),
        }

    def get_transition_matrix(self, normalize: bool = True) -> torch.Tensor:
        """Get tile transition matrix."""
        matrix = self.transition_matrix.clone()
        if normalize:
            row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1)
            matrix = matrix / row_sums
        return matrix

    def get_regime_analysis(self) -> Dict:
        """Analyze learned regime structure."""
        trans = self.get_transition_matrix(normalize=True)

        # Self-transition rates (stability)
        self_trans = trans.diag()
        stable_tiles = (self_trans > 0.5).nonzero().squeeze(-1).tolist()

        # Transition entropy (hub-ness)
        trans_entropy = -(trans * (trans + 1e-10).log()).sum(dim=1)
        hub_tiles = (trans_entropy > 1.0).nonzero().squeeze(-1).tolist()

        return {
            'transition_matrix': trans,
            'self_transition_rates': self_trans,
            'stable_tiles': stable_tiles,
            'hub_tiles': hub_tiles,
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.cluster_counts.zero_()
        self.total_count.zero_()
        self.transition_matrix.zero_()
        for tile in self.tiles:
            tile.activation_count.zero_()
            tile.total_count.zero_()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hierarchical_temporal_ffn(
    d_model: int,
    num_tiles: int = 64,
    d_state: int = 16,
    **kwargs,
) -> HierarchicalTemporalFFN:
    """Create a HierarchicalTemporalFFN with sensible defaults."""
    tiles_per_cluster = int(num_tiles ** 0.5)
    if num_tiles % tiles_per_cluster != 0:
        tiles_per_cluster = 8  # Fallback

    return HierarchicalTemporalFFN(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=tiles_per_cluster,
        d_state=d_state,
        **kwargs,
    )
