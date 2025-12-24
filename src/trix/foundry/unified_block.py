"""
Unified Providence Block - Attention and FFN as One Mechanism.

Mesa 16.1: The Sacred Foundry

The Unified Providence Block dissolves the attention/FFN distinction.
Both phases use the same mechanism: Providence routing to frozen shapes.

Architecture:
    Phase 1 (Mixing): Tokens route to tokens, mix via frozen shapes
    Phase 2 (Transform): Tokens route to tiles, transform via frozen shapes

Same mechanism. Different shape libraries.
No separate attention. No separate FFN. Just Providence.

"Attention is soft Providence. Providence is hard attention."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .token_mixer import TokenMixer, create_token_mixer
from ..nn.providence import ProvidenceFFN, create_providence_ffn


class UnifiedProvidenceBlock(nn.Module):
    """
    Unified Providence Block: Mixing + Transform via Providence.

    Replaces a standard transformer block (attention + FFN) with:
        - Phase 1: TokenMixer (tokens route to tokens)
        - Phase 2: ProvidenceFFN (tokens route to transform tiles)

    Both phases use:
        - Hamming-distance routing
        - Frozen shapes
        - Temporal state
        - Hierarchical O(√n) structure

    Args:
        d_model: Model dimension
        num_mixing_shapes: Number of mixing shape options
        num_transform_tiles: Number of transform tiles
        d_state: State dimension
        use_causal: Use causal masking for mixing
        use_frozen_transforms: Use frozen shapes for transform phase
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_mixing_shapes: int = 8,
        num_transform_tiles: int = 16,
        d_state: int = 16,
        use_causal: bool = True,
        use_frozen_transforms: bool = True,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_mixing_shapes = num_mixing_shapes
        self.num_transform_tiles = num_transform_tiles
        self.d_state = d_state

        # Phase 1: Token Mixing
        # Tokens route to other tokens, mix via frozen shapes
        self.token_mixer = TokenMixer(
            d_model=d_model,
            num_mixing_shapes=num_mixing_shapes,
            use_causal=use_causal,
            temperature=temperature,
            d_state=d_state,
            dropout=dropout,
        )

        # Phase 2: Token Transform
        # Tokens route to transform tiles, apply frozen shapes
        self.transform_ffn = ProvidenceFFN(
            d_model=d_model,
            d_hidden=d_model * 4 // num_transform_tiles,
            d_state=d_state,
            num_tiles=num_transform_tiles,
            use_frozen_shapes=use_frozen_transforms,
            use_soft_routing=True,
            temperature=temperature,
            dropout=dropout,
            mode='hierarchical' if num_transform_tiles >= 16 else 'flat',
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """Initialize block state."""
        device = device or next(self.parameters()).device
        return {
            'mixer_state': self.token_mixer.init_state(batch_size, device),
            'transform_state': self.transform_ffn.init_state(batch_size, device),
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass: Mixing then Transform.

        Args:
            x: [B, T, D] input tokens
            state: Block state
            mask: Optional attention mask

        Returns:
            output: [B, T, D] transformed tokens
            new_state: Updated state
            info: Routing information from both phases
        """
        # Handle 2D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, T, D = x.shape
        device = x.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(B, device)

        # Phase 1: Token Mixing
        x_mixed, mixer_state, mixer_info = self.token_mixer(
            x,
            state=state.get('mixer_state'),
            mask=mask,
        )

        # Phase 2: Token Transform
        x_transformed, transform_state, transform_info, aux_losses = self.transform_ffn(
            x_mixed,
            state=state.get('transform_state'),
        )

        # Combine states
        new_state = {
            'mixer_state': mixer_state,
            'transform_state': transform_state,
        }

        # Combine info
        info = {
            'mixer': mixer_info,
            'transform': transform_info,
            'aux_losses': aux_losses,
        }

        # Handle 2D output
        if is_2d:
            x_transformed = x_transformed.squeeze(1)

        return x_transformed, new_state, info

    def reset_stats(self):
        """Reset tracking statistics."""
        self.transform_ffn.reset_stats()

    def get_stats(self) -> Dict:
        """Get block statistics."""
        return {
            'mixer': self.token_mixer.get_mixing_stats(),
            'transform': self.transform_ffn.analyze_regimes() if hasattr(self.transform_ffn, 'analyze_regimes') else {},
        }


class UnifiedProvidenceTransformer(nn.Module):
    """
    Stack of Unified Providence Blocks.

    A full transformer model using only Providence routing.
    No separate attention mechanism. No separate FFN.
    All computation via content-addressable routing to frozen shapes.

    Args:
        d_model: Model dimension
        n_layers: Number of blocks
        num_mixing_shapes: Mixing shapes per block
        num_transform_tiles: Transform tiles per block
        vocab_size: Vocabulary size (for embeddings)
        max_seq_len: Maximum sequence length
        d_state: State dimension
        use_causal: Use causal masking
        use_frozen_transforms: Use frozen shapes for transform
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_mixing_shapes: int = 8,
        num_transform_tiles: int = 16,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        d_state: int = 16,
        use_causal: bool = True,
        use_frozen_transforms: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings (if vocab_size provided)
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            self.token_embedding = None
            self.position_embedding = None

        # Unified Providence Blocks
        self.blocks = nn.ModuleList([
            UnifiedProvidenceBlock(
                d_model=d_model,
                num_mixing_shapes=num_mixing_shapes,
                num_transform_tiles=num_transform_tiles,
                d_state=d_state,
                use_causal=use_causal,
                use_frozen_transforms=use_frozen_transforms,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

        # Language modeling head (if vocab_size provided)
        if vocab_size is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Tie weights
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = None

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> List[Dict]:
        """Initialize state for all layers."""
        return [
            block.init_state(batch_size, device)
            for block in self.blocks
        ]

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[Dict]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        """
        Forward pass through all blocks.

        Args:
            x: [B, T] token ids or [B, T, D] embeddings
            state: List of block states
            mask: Optional attention mask

        Returns:
            output: [B, T, D] or [B, T, vocab_size] if has lm_head
            new_states: Updated states for all layers
            info: Routing information
        """
        # Embed if needed
        if self.token_embedding is not None and x.dtype == torch.long:
            B, T = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            x = self.token_embedding(x) + self.position_embedding(positions)

        B, T, D = x.shape
        device = x.device

        # Initialize states if needed
        if state is None:
            state = self.init_state(B, device)

        # Forward through blocks
        new_states = []
        all_info = []
        total_aux_loss = 0

        for i, block in enumerate(self.blocks):
            x, new_state, info = block(x, state[i], mask)
            new_states.append(new_state)
            all_info.append(info)
            if 'aux_losses' in info and info['aux_losses']:
                total_aux_loss = total_aux_loss + info['aux_losses'].get('total_aux', 0)

        # Final normalization
        x = self.norm(x)

        # Language modeling head
        if self.lm_head is not None:
            x = self.lm_head(x)

        info = {
            'layers': all_info,
            'total_aux_loss': total_aux_loss,
        }

        return x, new_states, info


def create_unified_block(
    d_model: int,
    num_mixing_shapes: int = 8,
    num_transform_tiles: int = 16,
    **kwargs,
) -> UnifiedProvidenceBlock:
    """Convenience function to create a UnifiedProvidenceBlock."""
    return UnifiedProvidenceBlock(
        d_model=d_model,
        num_mixing_shapes=num_mixing_shapes,
        num_transform_tiles=num_transform_tiles,
        **kwargs,
    )


def create_unified_transformer(
    d_model: int,
    n_layers: int,
    vocab_size: Optional[int] = None,
    **kwargs,
) -> UnifiedProvidenceTransformer:
    """Convenience function to create a UnifiedProvidenceTransformer."""
    return UnifiedProvidenceTransformer(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        **kwargs,
    )
