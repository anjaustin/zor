"""
TrueOctaveFFN: Derived Multi-Resolution Frozen Geometry

Born from the Lincoln Manifold Method, December 2025.
Revised December 2025: Commitment Principle - select, don't blend.

The insight: Octaves are VIEWS of the same structure at different resolutions,
not independent banks. Coarse = pooled(Fine). This is what bit-shift means
in the original Sparse Octave design.

    Fine:   64 tiles - the base truth, discovered at init
    Medium: 16 tiles - derived as sign(pool(fine)), frozen
    Coarse:  4 tiles - derived as sign(pool(medium)), frozen

    Octave selector: learned selection of which octave to use (besides scales)

REVISION (Lincoln Manifold Analysis):
    Original: Compute all 3 octaves, blend results (SLOW)
    Revised:  Select 1 octave, compute only that (FAST)

    The key principle: COMMITMENT IS COMPUTATION
    - Early commitment eliminates parallel paths
    - Selection is cheaper than blend
    - Temperature annealing: soft exploration → hard commitment

The derivation is frozen. The structure is frozen. Only the navigation is learned.
This is Gradient Truth applied to multi-scale architecture.

Two modes:
    - Generative: soft octave selection with temperature (gradients flow)
    - Deterministic: hard octave selection (exact computation)

The same architecture models both exact systems (6502) and fuzzy systems (LLM).
The difference is only in the routing mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal, List
from dataclasses import dataclass


@dataclass 
class OctaveRoutingInfo:
    """Routing information from TrueOctaveFFN."""
    blend_weights: torch.Tensor      # [B, T, 3] - weight per octave
    fine_tile_idx: torch.Tensor      # [B, T] - selected fine tile
    medium_tile_idx: torch.Tensor    # [B, T] - selected medium tile
    coarse_tile_idx: torch.Tensor    # [B, T] - selected coarse tile
    selected_octave: torch.Tensor    # [B, T] - which octave dominated (0/1/2)
    entropy: torch.Tensor            # [B, T] - routing uncertainty
    mode: str                        # "generative" or "deterministic"


class FrozenTile(nn.Module):
    """
    A single frozen ternary tile.
    
    Weights are frozen (discovered at init or derived).
    Only the scale is learned.
    """
    
    def __init__(self, d_model: int, d_hidden: int, init_up: torch.Tensor = None, init_down: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # Initialize or use provided weights
        if init_up is None:
            init_up = torch.sign(torch.randn(d_hidden, d_model))
        if init_down is None:
            init_down = torch.sign(torch.randn(d_model, d_hidden))
        
        # Frozen ternary weights
        self.register_buffer('up_weight', init_up)
        self.register_buffer('down_weight', init_down)
        
        # Learned scale (where gradients flow)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.up_weight)
        h = F.relu(h)
        out = F.linear(h, self.down_weight)
        return out * self.scale
    
    @property
    def signature(self) -> torch.Tensor:
        """Routing signature derived from up weights."""
        return self.up_weight.mean(dim=0).sign()


class Octave(nn.Module):
    """
    One octave (resolution level) containing multiple tiles.
    
    Can be base (random init) or derived (from pooling another octave).
    """
    
    def __init__(self, d_model: int, d_hidden: int, num_tiles: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_tiles = num_tiles
        
        self.tiles = nn.ModuleList()
        self._signatures = None
    
    def add_tile(self, tile: FrozenTile):
        self.tiles.append(tile)
        self._signatures = None  # Invalidate cache
    
    @property
    def signatures(self) -> torch.Tensor:
        """Cached signatures for routing."""
        if self._signatures is None:
            self._signatures = torch.stack([t.signature for t in self.tiles])
        return self._signatures
    
    def route(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route input to tiles."""
        scores = torch.einsum('btd,nd->btn', x, self.signatures)
        
        if hard:
            idx = scores.argmax(dim=-1)
            weights = F.one_hot(idx, self.num_tiles).float()
        else:
            weights = F.softmax(scores / (self.d_model ** 0.5), dim=-1)
            idx = scores.argmax(dim=-1)
        
        return weights, idx
    
    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through octave.
        
        Returns:
            output: [B, T, D]
            tile_idx: [B, T] selected tile indices
        """
        weights, idx = self.route(x, hard=hard)
        
        if hard:
            # Hard: only compute selected tiles
            B, T, D = x.shape
            output = torch.zeros_like(x)
            for t_idx in range(self.num_tiles):
                mask = idx == t_idx
                if mask.any():
                    output[mask] = self.tiles[t_idx](x[mask])
        else:
            # Soft: weighted sum
            tile_outputs = torch.stack([tile(x) for tile in self.tiles], dim=2)  # [B, T, N, D]
            output = torch.einsum('btnd,btn->btd', tile_outputs, weights)
        
        return output, idx


def derive_octave(source: Octave, pool_factor: int, d_hidden_scale: float = 1.0) -> Octave:
    """
    Derive a coarser octave from a finer one.
    
    The key insight: coarse signatures = sign(mean(fine signatures))
    This is analogous to bit-shifting in Sparse Octave.
    
    For signatures to match, we derive up_weight such that:
        derived.signature = sign(mean(source_signatures))
    
    Since signature = up_weight.mean(dim=0).sign(), we need:
        sign(derived_up.mean(dim=0)) = sign(mean(source_sigs))
    
    We achieve this by constructing up_weight to have the correct mean.
    
    Args:
        source: The finer octave to derive from
        pool_factor: How many source tiles per derived tile
        d_hidden_scale: Scale factor for hidden dimension
    """
    assert source.num_tiles % pool_factor == 0
    
    num_derived = source.num_tiles // pool_factor
    d_hidden = int(source.d_hidden * d_hidden_scale)
    
    derived = Octave(source.d_model, d_hidden, num_derived)
    
    with torch.no_grad():
        for i in range(num_derived):
            start = i * pool_factor
            end = start + pool_factor
            
            # Derive the target signature from source signatures
            source_sigs = torch.stack([source.tiles[j].signature for j in range(start, end)])
            target_sig = source_sigs.mean(dim=0).sign()  # This is what signature should be
            
            # Create up_weight that produces this signature
            # signature = up_weight.mean(dim=0).sign()
            # So we make all rows of up_weight equal to target_sig
            # Then mean(dim=0) = target_sig, and sign(target_sig) = target_sig (already ternary)
            derived_up = target_sig.unsqueeze(0).expand(d_hidden, -1).clone()
            
            # For down_weight, pool from source
            source_down = torch.stack([source.tiles[j].down_weight for j in range(start, end)])
            derived_down = source_down.mean(dim=0).sign()
            
            # Handle d_hidden mismatch
            if d_hidden != source.d_hidden:
                if d_hidden > source.d_hidden:
                    repeat_factor = (d_hidden + source.d_hidden - 1) // source.d_hidden
                    derived_down = derived_down.repeat(1, repeat_factor)[:, :d_hidden]
                else:
                    derived_down = derived_down[:, :d_hidden]
            
            tile = FrozenTile(
                source.d_model, 
                d_hidden,
                init_up=derived_up,
                init_down=derived_down,
            )
            derived.add_tile(tile)
    
    return derived


class TrueOctaveFFN(nn.Module):
    """
    True Octave Feed-Forward Network.

    Three octaves with DERIVED structure:
        Fine:   64 tiles (base, random init)
        Medium: 16 tiles (derived from fine, pool_factor=4)
        Coarse:  4 tiles (derived from medium, pool_factor=4)

    The derivation is frozen. Coarse IS a compressed view of Fine.
    This is the bit-shift principle from Sparse Octave, applied to ternary weights.

    REVISED: Selection, not blend (Commitment Principle)
        - Select ONE octave per token, not all three
        - Temperature annealing: soft → hard selection
        - 3x faster than original blend approach

    Modes:
        - Generative: soft octave selection with temperature
        - Deterministic: hard octave selection (argmax)

    The same frozen structure serves both exact and probabilistic computation.
    Only the routing mode changes.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_fine_tiles: int = 64,
        pool_factor: int = 4,
        d_hidden: int = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_fine_tiles = num_fine_tiles
        self.pool_factor = pool_factor
        d_hidden = d_hidden or d_model

        # Build fine octave (base truth)
        self.fine = Octave(d_model, d_hidden, num_fine_tiles)
        for _ in range(num_fine_tiles):
            self.fine.add_tile(FrozenTile(d_model, d_hidden))

        # Derive medium and coarse octaves
        num_medium = num_fine_tiles // pool_factor
        num_coarse = num_medium // pool_factor

        self.medium = derive_octave(self.fine, pool_factor)
        self.coarse = derive_octave(self.medium, pool_factor)

        self.num_medium_tiles = num_medium
        self.num_coarse_tiles = num_coarse

        # Octave selector (REVISED: simple selector, not blend network)
        # Select which octave to use, not blend all three
        self.octave_selector = nn.Linear(d_model, 3)

        # Temperature for soft selection (annealed during training)
        self.temperature = temperature

        # Layer norm and output
        self.norm = nn.LayerNorm(d_model)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)

        # Mode
        self._mode = "generative"

        # Store octaves in a list for indexed access
        self._octaves = [self.fine, self.medium, self.coarse]
    
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: Literal["generative", "deterministic"]):
        assert mode in ("generative", "deterministic")
        self._mode = mode

    def set_temperature(self, temperature: float):
        """Set temperature for soft octave selection."""
        self.temperature = max(temperature, 1e-6)  # Avoid division by zero

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, OctaveRoutingInfo]:
        """
        REVISED forward pass: Select ONE octave, then execute.

        Commitment Principle:
            - Don't compute all octaves and blend
            - Select which octave to use, then compute only that one
            - 3x faster than original approach

        Mode behavior:
            - Deterministic: hard selection, execute only selected octave (fast)
            - Generative: soft selection with temperature (gradients flow)
        """
        B, T, D = x.shape

        # Deterministic mode: always hard selection
        # Generative mode: soft selection (for gradients)
        hard = (self._mode == "deterministic")

        # Normalize
        x_norm = self.norm(x)

        # Octave selection (REVISED: select, don't blend)
        selection_logits = self.octave_selector(x_norm)  # [B, T, 3]

        # Octave selection with straight-through estimator
        # Key insight: ALWAYS select ONE octave (commitment principle)
        # Gradients flow through softmax probabilities (STE trick)
        probs = F.softmax(selection_logits / self.temperature, dim=-1)
        selected = selection_logits.argmax(dim=-1)  # [B, T]

        if hard:
            # Hard selection: no gradients through selection
            selection_weights = F.one_hot(selected, 3).float()
        else:
            # Straight-through estimator: one-hot forward, soft gradients backward
            # This is the key to fast training with gradient flow
            one_hot = F.one_hot(selected, 3).float()
            selection_weights = one_hot - probs.detach() + probs  # STE trick

        # Execute ONLY selected octave (3x faster than computing all)
        output = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        idx_fine = torch.zeros(B, T, dtype=torch.long, device=x.device)
        idx_medium = torch.zeros(B, T, dtype=torch.long, device=x.device)
        idx_coarse = torch.zeros(B, T, dtype=torch.long, device=x.device)

        # Single-octave execution for all modes (COMMITMENT PRINCIPLE)
        for octave_idx in range(3):
            mask = selected == octave_idx
            if mask.any():
                octave = self._octaves[octave_idx]
                # Get positions where this octave is selected
                x_masked = x_norm[mask]  # [N, D] where N = number of selected positions

                # For gradient flow in soft mode, we need tile's soft output
                out_masked, tile_idx = octave(x_masked.unsqueeze(1), hard=hard)

                # Apply STE: multiply by selection weight
                # Forward: weight = 1.0 (identity)
                # Backward: gradients flow through probs for selection learning
                weight = selection_weights[mask, octave_idx:octave_idx+1]  # [N, 1]
                output[mask] = out_masked.squeeze(1) * weight

                # Store tile indices
                if octave_idx == 0:
                    idx_fine[mask] = tile_idx.squeeze(1)
                elif octave_idx == 1:
                    idx_medium[mask] = tile_idx.squeeze(1)
                else:
                    idx_coarse[mask] = tile_idx.squeeze(1)

        # Output processing
        output = output * self.output_scale
        output = self.dropout(output)
        output = x + output  # Residual

        # Report based on mode:
        # - Deterministic: one-hot weights, zero entropy (committed)
        # - Generative: soft probs, soft entropy (exploring)
        if hard:
            reported_weights = F.one_hot(selected, 3).float()
            entropy = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        else:
            reported_weights = probs
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        info = OctaveRoutingInfo(
            blend_weights=reported_weights,
            fine_tile_idx=idx_fine,
            medium_tile_idx=idx_medium,
            coarse_tile_idx=idx_coarse,
            selected_octave=selected,
            entropy=entropy,
            mode=self._mode,
        )

        return output, info
    
    def get_derivation_check(self) -> Dict[str, bool]:
        """Verify octave derivation relationships."""
        checks = {}
        
        # Check medium derived from fine
        for i in range(self.num_medium_tiles):
            start = i * self.pool_factor
            end = start + self.pool_factor
            fine_sigs = torch.stack([self.fine.tiles[j].signature for j in range(start, end)])
            expected = fine_sigs.mean(dim=0).sign()
            actual = self.medium.tiles[i].signature
            checks[f"medium_{i}_from_fine"] = torch.allclose(expected, actual)
        
        # Check coarse derived from medium
        for i in range(self.num_coarse_tiles):
            start = i * self.pool_factor
            end = start + self.pool_factor
            medium_sigs = torch.stack([self.medium.tiles[j].signature for j in range(start, end)])
            expected = medium_sigs.mean(dim=0).sign()
            actual = self.coarse.tiles[i].signature
            checks[f"coarse_{i}_from_medium"] = torch.allclose(expected, actual)
        
        return checks
    
    def rederive(self):
        """Re-derive medium and coarse from fine. Call if fine tiles have changed."""
        self.medium = derive_octave(self.fine, self.pool_factor)
        self.coarse = derive_octave(self.medium, self.pool_factor)
        self._octaves = [self.fine, self.medium, self.coarse]

    # Backward compatibility: alias blend_net to octave_selector
    @property
    def blend_net(self):
        """Backward compatibility alias for octave_selector."""
        return self.octave_selector


class TrueOctaveBlock(nn.Module):
    """Transformer block with TrueOctaveFFN."""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        num_fine_tiles: int = 64,
        pool_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = TrueOctaveFFN(
            d_model=d_model,
            num_fine_tiles=num_fine_tiles,
            pool_factor=pool_factor,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, OctaveRoutingInfo]:
        # Self-attention
        x_norm = self.ln1(x)
        if is_causal:
            T = x.size(1)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # FFN
        x, info = self.ffn(x)
        
        return x, info
    
    def set_mode(self, mode: Literal["generative", "deterministic"]):
        self.ffn.set_mode(mode)

    def set_temperature(self, temperature: float):
        """Set temperature for octave selection."""
        self.ffn.set_temperature(temperature)
