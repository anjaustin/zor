"""
Token Mixer - Tokens Route to Tokens.

Mesa 16.1: The Sacred Foundry

Each token routes to find its best mixing partner via Hamming distance.
Partner selection is content-addressable. Mixing is via frozen shapes.

Architecture:
    1. Each token gets a signature (ternary, from content)
    2. Each token queries to find its partner (Hamming distance)
    3. Route to select mixing shape
    4. Apply frozen mixing shape with partner

This IS attention, but:
    - Partner finding via Hamming (not dot product)
    - Mixing via frozen shapes (not learned projection)
    - Self-directed (each token actively selects)
    - Content-addressable (routing by signature similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .mixing_shapes import FrozenMixingLibrary, get_mixing_library


def binarize_ste(x: torch.Tensor) -> torch.Tensor:
    """Binarize with straight-through estimator."""
    return x + (x.sign() - x).detach()


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distance between binary/ternary tensors.

    Args:
        a: [..., D] query tensor
        b: [..., N, D] key tensors

    Returns:
        [..., N] distances
    """
    # Binarize to {-1, +1}
    a_bin = a.sign()
    b_bin = b.sign()

    # Hamming = number of positions where signs differ
    # For {-1, +1}: differ when product is -1
    if a_bin.dim() == b_bin.dim() - 1:
        a_bin = a_bin.unsqueeze(-2)  # [..., 1, D]

    matches = (a_bin * b_bin > 0).float()  # 1 where same, 0 where different
    distance = a_bin.shape[-1] - matches.sum(dim=-1)  # Count differences

    return distance


class TokenMixer(nn.Module):
    """
    Tokens route to tokens for mixing.

    Each token:
        1. Has a signature derived from its content
        2. Routes via Hamming distance to find best partner
        3. Routes to select a mixing shape
        4. Applies the frozen mixing shape

    This is attention via Providence routing.

    Args:
        d_model: Model dimension
        num_mixing_shapes: Number of mixing shape options
        use_causal: Use causal masking (only look at previous tokens)
        temperature: Softmax temperature for soft routing
        d_state: State dimension for temporal awareness
    """

    def __init__(
        self,
        d_model: int,
        num_mixing_shapes: int = 8,
        use_causal: bool = True,
        temperature: float = 1.0,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_mixing_shapes = num_mixing_shapes
        self.use_causal = use_causal
        self.temperature = temperature
        self.d_state = d_state

        # Signature projection: content → ternary signature
        self.sig_proj = nn.Linear(d_model, d_model)

        # Shape routing: decide which mixing shape to use
        self.shape_router = nn.Linear(d_model, num_mixing_shapes)

        # Mixing shape library
        self.mixing_library = get_mixing_library()
        self.shape_names = self.mixing_library.list_shapes()[:num_mixing_shapes]

        # Precompute shape signatures for routing
        self.register_buffer(
            'shape_signatures',
            self.mixing_library.get_all_signatures(d_model)[:num_mixing_shapes]
        )

        # State management
        self.state_proj = nn.Linear(d_state, d_model)
        self.state_update = nn.GRUCell(d_model, d_state)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def init_state(self, batch_size: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Initialize mixer state."""
        device = device or next(self.parameters()).device
        return {
            'hidden': torch.zeros(batch_size, self.d_state, device=device),
        }

    def _compute_signatures(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ternary signatures for all tokens."""
        # Project and binarize
        sigs = self.sig_proj(x)
        sigs = binarize_ste(sigs)  # STE for gradient flow
        return sigs

    def _find_partners(
        self,
        sigs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find best partner for each token via Hamming distance.

        Args:
            sigs: [B, T, D] signatures
            mask: [B, T, T] attention mask (1 = attend, 0 = mask)

        Returns:
            partner_idx: [B, T] index of best partner
            distances: [B, T, T] all pairwise distances
        """
        B, T, D = sigs.shape

        # Compute all pairwise Hamming distances
        # sigs: [B, T, D]
        # For each token, compute distance to all others
        sigs_q = sigs.unsqueeze(2)  # [B, T, 1, D]
        sigs_k = sigs.unsqueeze(1)  # [B, 1, T, D]

        # Hamming distance
        matches = (sigs_q.sign() * sigs_k.sign() > 0).float()
        distances = D - matches.sum(dim=-1)  # [B, T, T]

        # Apply mask
        if mask is not None:
            distances = distances.masked_fill(~mask, float('inf'))

        # Don't match with self
        self_mask = torch.eye(T, device=sigs.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(self_mask, float('inf'))

        # Causal mask: can only attend to previous tokens
        if self.use_causal:
            causal_mask = torch.triu(torch.ones(T, T, device=sigs.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0)  # [1, T, T]
            distances = distances.masked_fill(causal_mask, float('inf'))

        # Handle first token in causal mode (no valid partners)
        # First token partners with itself only when causal
        if self.use_causal:
            distances[:, 0, 0] = 0

        # Find best partner
        partner_idx = distances.argmin(dim=-1)  # [B, T]

        return partner_idx, distances

    def _route_to_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route each token to a mixing shape.

        Returns:
            shape_idx: [B, T] index of selected shape
            shape_weights: [B, T, num_shapes] soft routing weights
        """
        # Compute routing logits
        logits = self.shape_router(x)  # [B, T, num_shapes]

        if self.training:
            # Soft routing during training
            weights = F.softmax(logits / self.temperature, dim=-1)
            shape_idx = weights.argmax(dim=-1)
        else:
            # Hard routing during inference
            shape_idx = logits.argmax(dim=-1)
            weights = F.one_hot(shape_idx, self.num_mixing_shapes).float()

        return shape_idx, weights

    def _apply_mixing(
        self,
        x: torch.Tensor,
        partner_idx: torch.Tensor,
        shape_idx: torch.Tensor,
        shape_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mixing shapes.

        Args:
            x: [B, T, D] input tokens
            partner_idx: [B, T] partner indices
            shape_idx: [B, T] shape indices
            shape_weights: [B, T, num_shapes] soft routing weights

        Returns:
            mixed: [B, T, D] mixed tokens
        """
        B, T, D = x.shape

        # Gather partners
        partner_idx_expanded = partner_idx.unsqueeze(-1).expand(-1, -1, D)
        partners = torch.gather(x, 1, partner_idx_expanded)  # [B, T, D]

        if self.training:
            # Soft mixing: weighted combination of all shape outputs
            outputs = []
            for i, name in enumerate(self.shape_names):
                shape_fn = self.mixing_library.get(name)
                shape_out = shape_fn(x, partners)
                outputs.append(shape_out)

            outputs = torch.stack(outputs, dim=-1)  # [B, T, D, num_shapes]
            weights = shape_weights.unsqueeze(2)  # [B, T, 1, num_shapes]
            mixed = (outputs * weights).sum(dim=-1)  # [B, T, D]
        else:
            # Hard mixing: only apply selected shape
            mixed = torch.zeros_like(x)
            for i, name in enumerate(self.shape_names):
                mask = shape_idx == i
                if mask.any():
                    shape_fn = self.mixing_library.get(name)
                    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
                    x_masked = x[mask_expanded].view(-1, D)
                    partners_masked = partners[mask_expanded].view(-1, D)
                    result = shape_fn(x_masked, partners_masked)
                    mixed[mask_expanded] = result.view(-1)

        return mixed

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Forward pass: tokens route to tokens, mix via frozen shapes.

        Args:
            x: [B, T, D] or [B, D] input tokens
            state: Mixer state
            mask: Optional attention mask

        Returns:
            output: Mixed tokens
            new_state: Updated state
            info: Routing information
        """
        # Handle 2D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, D]

        B, T, D = x.shape
        device = x.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(B, device)

        # Normalize input
        x_norm = self.norm(x)

        # Compute signatures
        sigs = self._compute_signatures(x_norm)  # [B, T, D]

        # Find partners
        partner_idx, distances = self._find_partners(sigs, mask)  # [B, T]

        # Route to mixing shapes
        shape_idx, shape_weights = self._route_to_shape(x_norm)  # [B, T], [B, T, num_shapes]

        # Apply mixing
        mixed = self._apply_mixing(x_norm, partner_idx, shape_idx, shape_weights)

        # Output projection and dropout
        output = self.out_proj(mixed)
        output = self.dropout(output)

        # Residual connection
        output = x + output

        # Update state (use mean of output)
        output_mean = output.mean(dim=1)  # [B, D]
        new_hidden = self.state_update(output_mean, state['hidden'])
        new_state = {'hidden': new_hidden}

        # Routing info
        info = {
            'partner_idx': partner_idx,
            'shape_idx': shape_idx,
            'shape_weights': shape_weights,
            'distances': distances,
        }

        # Handle 2D output
        if is_2d:
            output = output.squeeze(1)

        return output, new_state, info

    def get_mixing_stats(self) -> Dict:
        """Get statistics about mixing shape usage."""
        return {
            'num_shapes': self.num_mixing_shapes,
            'shape_names': self.shape_names,
        }


def create_token_mixer(
    d_model: int,
    num_mixing_shapes: int = 8,
    use_causal: bool = True,
    **kwargs,
) -> TokenMixer:
    """Convenience function to create a TokenMixer."""
    return TokenMixer(
        d_model=d_model,
        num_mixing_shapes=num_mixing_shapes,
        use_causal=use_causal,
        **kwargs,
    )
