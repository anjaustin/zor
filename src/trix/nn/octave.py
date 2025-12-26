"""
TrueOctaveFFN: Derived Multi-Resolution Frozen Geometry

Born from the Lincoln Manifold Method, December 2025.

The insight: Octaves are VIEWS of the same structure at different resolutions,
not independent banks. Coarse = pooled(Fine). This is what bit-shift means
in the original Sparse Octave design.

    Fine:   64 tiles - the base truth, discovered at init
    Medium: 16 tiles - derived as sign(pool(fine)), frozen
    Coarse:  4 tiles - derived as sign(pool(medium)), frozen
    
    Blend network: the only learned component (besides scales)

The derivation is frozen. The structure is frozen. Only the navigation is learned.
This is Gradient Truth applied to multi-scale architecture.

Two modes:
    - Generative: soft routing, soft blend (probability flows everywhere)
    - Deterministic: hard routing, hard blend (exact computation at all scales)

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
    
    Modes:
        - Generative: soft routing at all octaves, soft blend
        - Deterministic: hard routing at all octaves, hard blend
    
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
        
        # Blend network (the only learned routing)
        self.blend_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 3),  # 3 octaves
        )
        
        # Layer norm and output
        self.norm = nn.LayerNorm(d_model)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        
        # Mode
        self._mode = "generative"
    
    @property
    def mode(self) -> str:
        return self._mode
    
    def set_mode(self, mode: Literal["generative", "deterministic"]):
        assert mode in ("generative", "deterministic")
        self._mode = mode
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, OctaveRoutingInfo]:
        B, T, D = x.shape
        hard = (self._mode == "deterministic")
        
        # Normalize
        x_norm = self.norm(x)
        
        # Forward through each octave
        out_fine, idx_fine = self.fine(x_norm, hard=hard)
        out_medium, idx_medium = self.medium(x_norm, hard=hard)
        out_coarse, idx_coarse = self.coarse(x_norm, hard=hard)
        
        # Blend
        blend_logits = self.blend_net(x_norm)
        
        if hard:
            # Hard blend: pick best octave
            selected = blend_logits.argmax(dim=-1)  # [B, T]
            blend_weights = F.one_hot(selected, 3).float()
            
            # Select output based on best octave
            outputs = torch.stack([out_fine, out_medium, out_coarse], dim=2)  # [B, T, 3, D]
            output = outputs.gather(2, selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)).squeeze(2)
        else:
            # Soft blend
            blend_weights = F.softmax(blend_logits, dim=-1)
            output = (blend_weights[..., 0:1] * out_fine +
                      blend_weights[..., 1:2] * out_medium +
                      blend_weights[..., 2:3] * out_coarse)
            selected = blend_logits.argmax(dim=-1)
        
        # Output processing
        output = output * self.output_scale
        output = self.dropout(output)
        output = x + output  # Residual
        
        # Entropy (uncertainty measure) - use actual blend_weights
        # For one-hot (deterministic), entropy is 0
        # For soft blend (generative), entropy measures uncertainty
        entropy = -(blend_weights * (blend_weights + 1e-10).log()).sum(dim=-1)
        
        info = OctaveRoutingInfo(
            blend_weights=blend_weights,
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
