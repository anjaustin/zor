"""
MultiScaleTriXFFN: Exact where exact, fuzzy where fuzzy.

Multi-scale frozen ternary architecture that can:
- Model deterministic systems exactly (fine scale, hard routing)
- Generate probabilistically (all scales, soft routing)
- Bridge the gap with learned scale blending

The insight: Fuzziness lives in ROUTING and BLENDING, not in the patterns.
Patterns are frozen truth. Selection is learned uncertainty.

Architecture:
    Fine scale:   High-resolution ternary patterns (exact computation)
    Medium scale: Clustered patterns (common operations)  
    Coarse scale: Semantic patterns (categories of computation)
    
    Blend network learns which scale matters for each input.

Usage:
    >>> ffn = MultiScaleTriXFFN(d_model=512, num_tiles=64)
    >>> 
    >>> # Generative mode (default): soft blending across scales
    >>> output, info = ffn(x)
    >>> 
    >>> # Deterministic mode: hard routing to fine scale only
    >>> ffn.set_mode("deterministic")
    >>> output, info = ffn(x)  # Exact computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class MultiScaleRoutingInfo:
    """Information about multi-scale routing decisions."""
    scale_weights: torch.Tensor      # [B, T, num_octaves] blend weights per position
    tile_indices: torch.Tensor       # [B, T, num_octaves] selected tile per scale
    entropy: torch.Tensor            # [B, T] routing entropy (uncertainty measure)
    mode: str                        # "deterministic" or "generative"


class OctaveTile(nn.Module):
    """
    Single tile at one octave (scale level).
    
    Frozen ternary weights, learned scale only.
    """
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # Frozen ternary weights (discovered at init)
        up_init = torch.randn(d_hidden, d_model)
        down_init = torch.randn(d_model, d_hidden)
        
        self.register_buffer('up_weight', torch.sign(up_init))
        self.register_buffer('down_weight', torch.sign(down_init))
        
        # Learned scales (where gradients flow)
        self.up_scale = nn.Parameter(torch.ones(d_hidden))
        self.down_scale = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up projection: ternary matmul + scale
        hidden = F.linear(x, self.up_weight) * self.up_scale
        hidden = F.relu(hidden)
        
        # Down projection: ternary matmul + scale
        out = F.linear(hidden, self.down_weight) * self.down_scale
        return out
    
    def get_signature(self) -> torch.Tensor:
        """Routing signature derived from frozen weights."""
        return self.up_weight.sum(dim=0).sign()


class Octave(nn.Module):
    """
    One scale level containing multiple tiles.
    
    Each octave has different "resolution":
    - Fine: many small tiles (precise patterns)
    - Medium: fewer larger tiles (common patterns)
    - Coarse: few large tiles (semantic categories)
    """
    
    def __init__(
        self,
        d_model: int,
        num_tiles: int,
        d_hidden: int,
        octave_level: int,  # 0=fine, 1=medium, 2=coarse
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.d_hidden = d_hidden
        self.octave_level = octave_level
        
        # Tiles at this scale
        self.tiles = nn.ModuleList([
            OctaveTile(d_model, d_hidden) for _ in range(num_tiles)
        ])
        
        # Precompute signatures for routing
        self.register_buffer('_signatures', None)
    
    @property
    def signatures(self) -> torch.Tensor:
        """Lazily compute and cache signatures."""
        if self._signatures is None:
            sigs = torch.stack([t.get_signature() for t in self.tiles])
            self._signatures = sigs
        return self._signatures
    
    def route(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to tiles.
        
        Args:
            x: Input [B, T, D]
            hard: If True, hard routing (argmax). If False, soft routing (softmax).
        
        Returns:
            weights: Routing weights [B, T, num_tiles]
            indices: Selected tile indices [B, T]
        """
        # Compute similarity to all tile signatures
        scores = torch.einsum('btd,nd->btn', x, self.signatures)
        
        if hard:
            indices = scores.argmax(dim=-1)
            weights = F.one_hot(indices, self.num_tiles).float()
        else:
            weights = F.softmax(scores, dim=-1)
            indices = scores.argmax(dim=-1)
        
        return weights, indices
    
    def forward(
        self, 
        x: torch.Tensor, 
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward through this octave.
        
        Returns:
            output: [B, T, D]
            weights: routing weights [B, T, num_tiles]
            indices: selected tiles [B, T]
        """
        B, T, D = x.shape
        
        # Route
        weights, indices = self.route(x, hard=hard)
        
        # Apply all tiles and blend (soft) or select (hard)
        if hard:
            # Hard: only compute selected tile
            output = torch.zeros_like(x)
            for t_idx in range(self.num_tiles):
                mask = indices == t_idx
                if mask.any():
                    tile_out = self.tiles[t_idx](x[mask])
                    output[mask] = tile_out
        else:
            # Soft: weighted sum of all tiles
            tile_outputs = torch.stack([tile(x) for tile in self.tiles], dim=-2)  # [B,T,num_tiles,D]
            output = torch.einsum('btnd,btn->btd', tile_outputs, weights)
        
        return output, weights, indices


class MultiScaleTriXFFN(nn.Module):
    """
    Multi-Scale Frozen Ternary FFN.
    
    Three octaves with different resolutions:
    - Fine (octave 0): Many small tiles for precise patterns
    - Medium (octave 1): Moderate tiles for common patterns
    - Coarse (octave 2): Few tiles for semantic categories
    
    The blend network learns which scale matters for each input.
    Mode switch controls hard vs soft routing.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_tiles_fine: int = 64,
        num_tiles_medium: int = 16,
        num_tiles_coarse: int = 4,
        d_hidden_fine: Optional[int] = None,
        d_hidden_medium: Optional[int] = None,
        d_hidden_coarse: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_octaves = 3
        
        # Default hidden dims: coarser = larger
        d_hidden_fine = d_hidden_fine or d_model
        d_hidden_medium = d_hidden_medium or d_model * 2
        d_hidden_coarse = d_hidden_coarse or d_model * 4
        
        # Create octaves
        self.octaves = nn.ModuleList([
            Octave(d_model, num_tiles_fine, d_hidden_fine, octave_level=0),
            Octave(d_model, num_tiles_medium, d_hidden_medium, octave_level=1),
            Octave(d_model, num_tiles_coarse, d_hidden_coarse, octave_level=2),
        ])
        
        # Blend network: learns which octave matters for each input
        self.blend_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, self.num_octaves),
        )
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output scale
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Mode: "generative" (soft) or "deterministic" (hard)
        self._mode = "generative"
    
    @property
    def mode(self) -> str:
        return self._mode
    
    def set_mode(self, mode: Literal["generative", "deterministic"]):
        """
        Set routing mode.
        
        - "generative": Soft routing, soft blending (LLM-like)
        - "deterministic": Hard routing, fine scale only (6502-like)
        """
        assert mode in ("generative", "deterministic")
        self._mode = mode
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, MultiScaleRoutingInfo]:
        """
        Forward pass.
        
        Args:
            x: Input [B, T, D]
        
        Returns:
            output: [B, T, D]
            info: MultiScaleRoutingInfo with routing details
        """
        B, T, D = x.shape
        hard = (self._mode == "deterministic")
        
        # Normalize
        x_norm = self.norm(x)
        
        # Get octave blend weights
        if hard:
            # Deterministic: 100% fine scale
            scale_weights = torch.zeros(B, T, self.num_octaves, device=x.device)
            scale_weights[..., 0] = 1.0
        else:
            # Generative: learned blending
            scale_weights = F.softmax(self.blend_net(x_norm), dim=-1)
        
        # Forward through each octave
        octave_outputs = []
        tile_indices = []
        
        for i, octave in enumerate(self.octaves):
            out, weights, indices = octave(x_norm, hard=hard)
            octave_outputs.append(out)
            tile_indices.append(indices)
        
        # Blend octave outputs
        octave_outputs = torch.stack(octave_outputs, dim=-2)  # [B, T, num_octaves, D]
        output = torch.einsum('btod,bto->btd', octave_outputs, scale_weights)
        
        # Scale, dropout, residual
        output = output * self.output_scale
        output = self.dropout(output)
        output = x + output
        
        # Compute routing entropy (uncertainty measure)
        entropy = -(scale_weights * (scale_weights + 1e-10).log()).sum(dim=-1)
        
        # Build routing info
        info = MultiScaleRoutingInfo(
            scale_weights=scale_weights,
            tile_indices=torch.stack(tile_indices, dim=-1),
            entropy=entropy,
            mode=self._mode,
        )
        
        return output, info
    
    def get_scale_usage(self) -> Dict[str, float]:
        """Get average usage of each scale (for analysis)."""
        return {
            "fine": 0.0,    # Would track during training
            "medium": 0.0,
            "coarse": 0.0,
        }
    
    def freeze_for_deterministic(self):
        """
        Freeze blend network and set deterministic mode.
        
        Call after training to lock in exact computation.
        """
        self.set_mode("deterministic")
        for param in self.blend_net.parameters():
            param.requires_grad = False


class MultiScaleTriXBlock(nn.Module):
    """
    Full transformer block with multi-scale FFN.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        num_tiles_fine: int = 64,
        num_tiles_medium: int = 16,
        num_tiles_coarse: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = MultiScaleTriXFFN(
            d_model=d_model,
            num_tiles_fine=num_tiles_fine,
            num_tiles_medium=num_tiles_medium,
            num_tiles_coarse=num_tiles_coarse,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, MultiScaleRoutingInfo]:
        # Self-attention
        x_norm = self.ln1(x)
        if is_causal:
            T = x.size(1)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Multi-scale FFN
        x, info = self.ffn(x)
        
        return x, info
    
    def set_mode(self, mode: Literal["generative", "deterministic"]):
        self.ffn.set_mode(mode)
