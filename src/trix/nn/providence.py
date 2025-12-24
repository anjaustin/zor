"""
Providence FFN - The Unified Architecture

FFN as Content-Addressable Memory with Frozen Computation.

Providence synthesizes:
- Gap 1: XOR routing (Hamming distance instead of dot product)
- Gap 2: Frozen shapes (0 learnable params in computation)
- Gap 3: Hierarchical temporal (O(√n) + state persistence)

The insight:
    TILE = signature + shape + state
         = address   + transform + memory
         = key       + computation + value

    Attention(Q, K, V) = softmax(QK^T) V
    Providence(query, sigs, shapes) = shapes[argmin(hamming(query, sigs))](query)

    "Attention is soft Providence. Providence is hard attention."

Example:
    >>> ffn = ProvidenceFFN(d_model=128, num_tiles=64)
    >>> state = ffn.init_state(batch_size=4)
    >>> output, state, routing_info, aux = ffn(x, state)

    # With frozen shapes only (0 learnable compute params):
    >>> ffn = ProvidenceFFN(d_model=128, num_tiles=16, use_frozen_shapes=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Callable, Union
from dataclasses import dataclass

# Import from previous gaps
from .xor_ffn import hamming_distance, soft_hamming_distance, ternarize_ste
from .frozen_shapes import FrozenShapeLibrary, get_library, ShapeInfo


# =============================================================================
# PROVIDENCE TILE
# =============================================================================

class ProvidenceTile(nn.Module):
    """
    A tile with: signature (address) + shape (computation) + state (memory).

    This is the atom of Providence:
    - Signature: ternary vector for XOR matching
    - Shape: either frozen (0 params) or learned transform
    - State: persists across forward passes

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension for learned shapes
        d_state: State dimension
        tile_id: Unique tile identifier
        frozen_shape: Optional frozen shape function (if provided, 0 compute params)
        shape_name: Name of frozen shape (for registry)
        shape_n_inputs: Number of inputs the frozen shape expects (1 or 2)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_state: int,
        tile_id: int = 0,
        frozen_shape: Optional[Callable] = None,
        shape_name: str = None,
        shape_n_inputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_state = d_state
        self.tile_id = tile_id
        self.frozen_shape = frozen_shape
        self.shape_name = shape_name or f"tile_{tile_id}"
        self.shape_n_inputs = shape_n_inputs

        # Signature projection
        self.signature_proj = nn.Parameter(torch.randn(d_model) * 0.02)

        # State-aware signature component
        self.state_signature_proj = nn.Parameter(torch.randn(d_state) * 0.02)

        # Shape: either frozen or learned
        if frozen_shape is None:
            # Learned shape (standard FFN transformation)
            self.up = nn.Linear(d_model + d_state, d_hidden)
            self.down = nn.Linear(d_hidden, d_model)
            self.is_frozen = False
        else:
            # Frozen shape (0 learnable compute params)
            # Only need projection to match dimensions
            self.input_proj = nn.Linear(d_model + d_state, d_model)
            self.output_proj = nn.Linear(d_model, d_model)
            self.is_frozen = True

        # State update (always learned - this IS the learning)
        self.state_update = nn.Sequential(
            nn.Linear(d_model + d_state + d_model, d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, d_state),
        )

        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1))

        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))

    def get_signature(self) -> torch.Tensor:
        """Get ternary routing signature."""
        return self.signature_proj.sign()

    def get_combined_signature(self, state: torch.Tensor) -> torch.Tensor:
        """Get signature that includes state contribution."""
        input_sig = self.signature_proj.sign()
        state_contribution = (state @ self.state_signature_proj).sign()
        return torch.cat([
            input_sig.expand(state.shape[0], -1),
            state_contribution.unsqueeze(-1)
        ], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: (input, state) → (output, new_state)

        Args:
            x: Input [batch, d_model]
            state: Tile state [batch, d_state]

        Returns:
            output: [batch, d_model]
            new_state: [batch, d_state]
        """
        # Combine input and state
        combined = torch.cat([x, state], dim=-1)

        if self.is_frozen:
            # Frozen shape path
            projected = self.input_proj(combined)
            # Apply frozen shape element-wise
            # Most frozen shapes work on [0,1], normalize
            proj_norm = torch.sigmoid(projected)

            # Handle unary vs binary shapes
            if self.shape_n_inputs == 1:
                # Unary shape: apply to whole input
                shaped = self.frozen_shape(proj_norm)
            else:
                # Binary shape: split and apply pairwise
                half = self.d_model // 2
                a, b = proj_norm[:, :half], proj_norm[:, half:2*half]
                shaped = self.frozen_shape(a, b)
                # Pad back to d_model
                if shaped.shape[-1] < self.d_model:
                    shaped = F.pad(shaped, (0, self.d_model - shaped.shape[-1]))

            output = self.output_proj(shaped) * self.output_scale
        else:
            # Learned shape path
            hidden = F.gelu(self.up(combined))
            output = self.down(hidden) * self.output_scale

        # State update (learning happens here)
        update_input = torch.cat([x, state, output], dim=-1)
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
# PROVIDENCE FFN
# =============================================================================

class ProvidenceFFN(nn.Module):
    """
    The Unified Architecture: FFN as Content-Addressable Memory.

    Providence = Temporal + Geometric + Sparse + Lookup + FFN

    Each tile IS:
    - A memory cell (state)
    - A frozen/learned computation (shape)
    - An address (signature)

    Routing via XOR (Hamming distance)
    Computation via frozen or learned shapes
    State persists across forward passes
    Hierarchical for O(√n) scaling

    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension per tile (for learned shapes)
        d_state: State dimension per tile
        num_tiles: Total number of tiles
        tiles_per_cluster: Tiles per cluster for hierarchical routing
        d_global_state: Dimension of global state
        dropout: Dropout rate
        use_state_routing: Include state in routing decision
        use_frozen_shapes: Use frozen shapes (0 compute params)
        frozen_shape_names: List of shape names from FrozenShapeLibrary
        use_soft_routing: Use soft routing during training
        temperature: Softmax temperature for routing
        mode: 'hierarchical' or 'flat' routing
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
        use_frozen_shapes: bool = False,
        frozen_shape_names: Optional[List[str]] = None,
        use_soft_routing: bool = True,
        temperature: float = 1.0,
        mode: str = 'hierarchical',
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
        self.use_frozen_shapes = use_frozen_shapes
        self.use_soft_routing = use_soft_routing
        self.temperature = temperature
        self.mode = mode

        assert num_tiles % tiles_per_cluster == 0

        # Get frozen shapes if requested
        frozen_shapes = []
        shape_n_inputs_list = []
        if use_frozen_shapes:
            library = get_library()
            if frozen_shape_names is None:
                # Default: cycle through available shapes that work with our tile
                # (only shapes with 1 or 2 inputs)
                all_shapes = library.list_shapes()
                shape_names = [
                    name for name in all_shapes
                    if library.info(name).n_inputs <= 2
                ]
                frozen_shape_names = [shape_names[i % len(shape_names)] for i in range(num_tiles)]
            for name in frozen_shape_names:
                frozen_shapes.append(library.get(name))
                shape_n_inputs_list.append(library.info(name).n_inputs)

        # Create tiles
        self.tiles = nn.ModuleList([
            ProvidenceTile(
                d_model=d_model,
                d_hidden=self.d_hidden,
                d_state=d_state,
                tile_id=i,
                frozen_shape=frozen_shapes[i] if frozen_shapes else None,
                shape_name=frozen_shape_names[i] if frozen_shape_names else None,
                shape_n_inputs=shape_n_inputs_list[i] if shape_n_inputs_list else 2,
            )
            for i in range(num_tiles)
        ])

        # Cluster signatures (for hierarchical routing)
        self.cluster_signatures = nn.Parameter(
            torch.randn(self.num_clusters, d_model) * 0.02
        )

        # Global state encoder
        self.global_state_encoder = nn.Linear(d_model, d_global_state)

        # Global state transition
        self.global_state_update = nn.GRUCell(d_model, d_global_state)

        # State-aware routing projection
        if use_state_routing:
            self.state_routing_proj = nn.Linear(d_global_state, d_model)

        # Normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Tracking
        self.register_buffer('cluster_counts', torch.zeros(self.num_clusters))
        self.register_buffer('total_count', torch.tensor(0.0))
        self.register_buffer('transition_matrix', torch.zeros(num_tiles, num_tiles))

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """Initialize all states."""
        if device is None:
            device = self.cluster_signatures.device

        return {
            'global_state': torch.zeros(batch_size, self.d_global_state, device=device),
            'tile_states': torch.zeros(batch_size, self.num_tiles, self.d_state, device=device),
            'prev_tile': None,
        }

    def _get_tile_signatures(self) -> torch.Tensor:
        """Get ternary signatures from all tiles."""
        return torch.stack([tile.get_signature() for tile in self.tiles])

    def _providence_match(
        self,
        x: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        XOR-based matching (Providence semantics).

        Returns:
            tile_idx: [batch] winning tile indices
            cluster_idx: [batch] winning cluster indices
            gate: [batch, num_tiles] routing weights
        """
        batch = x.shape[0]
        device = x.device

        # State-aware input
        if self.use_state_routing and global_state is not None:
            state_contribution = self.state_routing_proj(global_state)
            routing_input = x + 0.1 * state_contribution
        else:
            routing_input = x

        # Ternarize for XOR matching
        x_tern = ternarize_ste(routing_input)

        if self.mode == 'hierarchical':
            return self._hierarchical_match(x_tern, routing_input)
        else:
            return self._flat_match(x_tern, routing_input)

    def _hierarchical_match(
        self,
        x_tern: torch.Tensor,
        x_continuous: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hierarchical O(√n) XOR matching."""
        batch = x_tern.shape[0]
        device = x_tern.device

        # Ternarize cluster signatures
        cluster_sigs = ternarize_ste(self.cluster_signatures)

        # Level 1: Cluster routing via Hamming
        if self.training and self.use_soft_routing:
            cluster_dist = soft_hamming_distance(x_continuous, self.cluster_signatures)
            cluster_scores = -cluster_dist / self.temperature
            cluster_idx = cluster_scores.argmax(dim=-1)
        else:
            cluster_dist = hamming_distance(x_tern, cluster_sigs)
            cluster_idx = cluster_dist.argmin(dim=-1)

        # Level 2: Tile routing within cluster
        tile_signatures = self._get_tile_signatures()
        tile_idx = torch.zeros(batch, dtype=torch.long, device=device)
        gate = torch.zeros(batch, self.num_tiles, device=device)

        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue

            # Get tiles in this cluster
            start = c * self.tiles_per_cluster
            end = start + self.tiles_per_cluster
            cluster_tile_sigs = tile_signatures[start:end]

            # XOR match within cluster
            if self.training and self.use_soft_routing:
                # Use continuous signatures for gradient flow
                cont_sigs = torch.stack([
                    self.tiles[i].signature_proj for i in range(start, end)
                ])
                tile_dist = soft_hamming_distance(x_continuous[mask], cont_sigs)
                tile_scores = -tile_dist / self.temperature
                local_idx = tile_scores.argmax(dim=-1)
                gate[mask, start:end] = F.softmax(tile_scores, dim=-1)
            else:
                tile_dist = hamming_distance(x_tern[mask], cluster_tile_sigs)
                local_idx = tile_dist.argmin(dim=-1)
                # One-hot gate
                for i, li in enumerate(local_idx):
                    gate_idx = torch.where(mask)[0][i]
                    gate[gate_idx, start + li] = 1.0

            # Convert to global indices
            tile_idx[mask] = start + local_idx

        return tile_idx, cluster_idx, gate

    def _flat_match(
        self,
        x_tern: torch.Tensor,
        x_continuous: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat O(n) XOR matching."""
        batch = x_tern.shape[0]
        device = x_tern.device

        tile_signatures = self._get_tile_signatures()

        if self.training and self.use_soft_routing:
            # Soft matching for gradient flow
            cont_sigs = torch.stack([t.signature_proj for t in self.tiles])
            distances = soft_hamming_distance(x_continuous, cont_sigs)
            scores = -distances / self.temperature
            tile_idx = scores.argmax(dim=-1)
            gate = F.softmax(scores, dim=-1)
        else:
            # Hard XOR matching
            distances = hamming_distance(x_tern, tile_signatures)
            tile_idx = distances.argmin(dim=-1)
            gate = F.one_hot(tile_idx, self.num_tiles).float()

        cluster_idx = tile_idx // self.tiles_per_cluster

        return tile_idx, cluster_idx, gate

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        track_transitions: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict, Dict]:
        """
        Forward pass with Providence routing.

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
            tile_states = state['tile_states']
        else:
            global_state = state['global_state']
            tile_states = state['tile_states']

        # Normalize input
        x_norm = self.input_norm(x)

        # Providence matching (XOR-based)
        tile_idx, cluster_idx, gate = self._providence_match(x_norm, global_state)

        # Execute tiles with state
        output = torch.zeros(batch, self.d_model, device=device)
        new_tile_states = tile_states.clone()

        if self.training and self.use_soft_routing:
            # Soft routing: weighted combination of all active tiles
            all_outputs = []
            for t in range(self.num_tiles):
                # Get tile states for batch
                if is_3d:
                    batch_indices = torch.arange(orig_shape[0], device=device).unsqueeze(1).expand(-1, T).reshape(-1)
                    tile_state = tile_states[batch_indices, t]
                else:
                    tile_state = tile_states[:, t]

                tile_out, new_state = self.tiles[t](x_norm, tile_state)
                all_outputs.append(tile_out)

                # Update states (weighted by gate)
                if is_3d:
                    for bi in range(orig_shape[0]):
                        idx_start = bi * T
                        idx_end = idx_start + T
                        weight = gate[idx_start:idx_end, t].mean()
                        if weight > 0.01:
                            new_tile_states[bi, t] = new_state[idx_end-1]
                else:
                    for bi in range(batch):
                        if gate[bi, t] > 0.01:
                            new_tile_states[bi, t] = new_state[bi]

            all_outputs = torch.stack(all_outputs, dim=1)  # [batch, num_tiles, d_model]
            output = (gate.unsqueeze(-1) * all_outputs).sum(dim=1)
        else:
            # Hard routing: only winner computes
            for t in range(self.num_tiles):
                mask = tile_idx == t
                if not mask.any():
                    continue

                tile_x = x_norm[mask]

                # Get tile states
                if is_3d:
                    batch_indices = torch.arange(orig_shape[0], device=device).unsqueeze(1).expand(-1, T).reshape(-1)
                    tile_batch_indices = batch_indices[mask]
                    tile_state = tile_states[tile_batch_indices, t]
                else:
                    tile_state = tile_states[mask, t]

                # Execute tile
                tile_out, new_state = self.tiles[t](tile_x, tile_state)
                output[mask] = tile_out

                # Update states
                if is_3d:
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
            output_mean = output.view(orig_shape[0], T, -1).mean(dim=1)
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
            'gate': gate.view(orig_shape[0], orig_shape[1], -1) if is_3d else gate,
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

        frozen_count = sum(1 for t in self.tiles if t.is_frozen)

        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'd_state': self.d_state,
            'active_tiles': (tile_usage > 0.001).sum().item(),
            'active_clusters': (cluster_usage > 0.001).sum().item(),
            'tile_usage_std': tile_usage.std().item(),
            'cluster_usage_std': cluster_usage.std().item(),
            'frozen_tiles': frozen_count,
            'learned_tiles': self.num_tiles - frozen_count,
            'mode': self.mode,
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

        # Self-transition rates
        self_trans = trans.diag()
        stable_tiles = (self_trans > 0.5).nonzero().squeeze(-1).tolist()

        # Transition entropy
        trans_entropy = -(trans * (trans + 1e-10).log()).sum(dim=1)
        hub_tiles = (trans_entropy > 1.0).nonzero().squeeze(-1).tolist()

        # Shape distribution
        frozen_tiles = [i for i, t in enumerate(self.tiles) if t.is_frozen]
        learned_tiles = [i for i, t in enumerate(self.tiles) if not t.is_frozen]

        return {
            'transition_matrix': trans,
            'self_transition_rates': self_trans,
            'stable_tiles': stable_tiles,
            'hub_tiles': hub_tiles,
            'frozen_tiles': frozen_tiles,
            'learned_tiles': learned_tiles,
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.cluster_counts.zero_()
        self.total_count.zero_()
        self.transition_matrix.zero_()
        for tile in self.tiles:
            tile.activation_count.zero_()
            tile.total_count.zero_()

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category."""
        routing_params = 0
        compute_params = 0
        state_params = 0

        for name, param in self.named_parameters():
            count = param.numel()
            if 'signature' in name or 'cluster' in name or 'routing' in name:
                routing_params += count
            elif 'state' in name or 'global' in name:
                state_params += count
            else:
                compute_params += count

        return {
            'total': sum(p.numel() for p in self.parameters()),
            'routing': routing_params,
            'compute': compute_params,
            'state': state_params,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_providence_ffn(
    d_model: int,
    num_tiles: int = 64,
    d_state: int = 16,
    use_frozen_shapes: bool = False,
    **kwargs,
) -> ProvidenceFFN:
    """Create a ProvidenceFFN with sensible defaults."""
    tiles_per_cluster = int(num_tiles ** 0.5)
    if num_tiles % tiles_per_cluster != 0:
        tiles_per_cluster = 8  # Fallback

    return ProvidenceFFN(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=tiles_per_cluster,
        d_state=d_state,
        use_frozen_shapes=use_frozen_shapes,
        **kwargs,
    )


def create_frozen_providence_ffn(
    d_model: int,
    num_tiles: int = 16,
    shape_names: Optional[List[str]] = None,
    **kwargs,
) -> ProvidenceFFN:
    """Create a ProvidenceFFN with all frozen shapes (0 compute params)."""
    return create_providence_ffn(
        d_model=d_model,
        num_tiles=num_tiles,
        use_frozen_shapes=True,
        frozen_shape_names=shape_names,
        **kwargs,
    )


# =============================================================================
# PROVIDENCE BLOCK (For Transformer Integration)
# =============================================================================

class ProvidenceBlock(nn.Module):
    """
    Transformer block using Providence FFN.

    Standard: LayerNorm → Attention → LayerNorm → ProvidenceFFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_tiles: int = 64,
        d_state: int = 16,
        dropout: float = 0.1,
        use_frozen_shapes: bool = False,
        **ffn_kwargs,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = create_providence_ffn(
            d_model=d_model,
            num_tiles=num_tiles,
            d_state=d_state,
            use_frozen_shapes=use_frozen_shapes,
            **ffn_kwargs,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict, Dict]:
        """Forward pass."""
        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # Providence FFN
        x_norm = self.norm2(x)
        ffn_out, new_state, routing_info, aux = self.ffn(x_norm, state)
        x = x + self.dropout(ffn_out - x_norm)  # Already has residual in FFN

        return x, new_state, routing_info, aux
