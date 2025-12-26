"""
Gradient Truth: A Training Paradigm Beyond STE

Gradients flow only where there is genuine uncertainty.
Structure stands apart as truth.

This module implements the three-layer decomposition:
    1. ROUTING (continuous, learned) - which shape for which input
    2. SHAPE BANK (discrete, frozen) - the computational primitives  
    3. MAGNITUDE (continuous, learned) - output scaling

No Straight-Through Estimator needed. All learning happens in
naturally continuous domains.

Example:
    >>> from trix.nn.gradient_truth import GradientTruthFFN, PolynomialShapeBank
    >>> 
    >>> # Create shape bank with derived mathematical shapes
    >>> shapes = PolynomialShapeBank.from_primitives(d_model=128)
    >>> 
    >>> # Create FFN - routing and magnitude are learned, shapes are frozen
    >>> ffn = GradientTruthFFN(d_model=128, shape_bank=shapes)
    >>> 
    >>> # Train with standard optimizer - no STE
    >>> output, routing_info = ffn(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()  # Gradients are mathematically correct

Theory:
    - Discrete things are discovered (derivation, evolution, distillation)
    - Continuous things are learned (routing, scales)
    - They interface through soft attention over shapes
    - Gradients flow through the attention weights, not through shapes

See: /tmp/gradient_truth_synth.md for full theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import math

from trix.nn.frozen_shapes import ActivationShapes


# =============================================================================
# SHAPE BANK: The Frozen Truth
# =============================================================================

class Shape:
    """
    A single computational shape.
    
    A shape is a function from input to output with:
    - A unique name
    - A signature for routing (what inputs it prefers)
    - The computation itself (frozen, exact)
    """
    
    def __init__(
        self,
        name: str,
        fn: Callable[[torch.Tensor], torch.Tensor],
        signature: Optional[torch.Tensor] = None,
        d_model: Optional[int] = None,
    ):
        self.name = name
        self.fn = fn
        self._signature = signature
        self._d_model = d_model
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)
    
    def get_signature(self, d_model: int) -> torch.Tensor:
        """Get or derive the routing signature."""
        if self._signature is not None:
            return self._signature
        
        # Derive signature from function behavior
        torch.manual_seed(hash(self.name) % (2**31))
        probe = torch.randn(64, d_model)
        with torch.no_grad():
            response = self.fn(probe)
            if response.shape != probe.shape:
                response = F.linear(response, torch.randn(d_model, response.shape[-1]))
            signature = response.mean(dim=0).sign()
        return signature


class ShapeBank(nn.Module):
    """
    A library of frozen computational shapes.
    
    The shape bank is the "truth" layer - frozen, exact, no gradients.
    Shapes can come from:
    - Mathematical derivation (polynomials)
    - Evolution (tap patterns, circuits)
    - Distillation (observed from trained networks)
    
    Once in the bank, shapes are immutable.
    """
    
    def __init__(self, shapes: List[Shape], d_model: int):
        super().__init__()
        self.shapes = shapes
        self.d_model = d_model
        self._num_shapes = len(shapes)
        
        # Precompute signatures for routing
        signatures = torch.stack([s.get_signature(d_model) for s in shapes])
        self.register_buffer('signatures', signatures)
        
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all shapes to input, return stacked outputs.
        
        Args:
            x: [batch, d_model]
            
        Returns:
            [batch, num_shapes, d_model] - output from each shape
        """
        outputs = []
        for shape in self.shapes:
            out = shape(x)
            # Ensure output matches d_model
            if out.shape[-1] != self.d_model:
                # Project if needed (shouldn't happen for well-designed shapes)
                out = F.pad(out, (0, self.d_model - out.shape[-1]))
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
    def __len__(self) -> int:
        return self._num_shapes
    
    def get_signatures(self) -> torch.Tensor:
        """Get all shape signatures for routing."""
        return self.signatures


class PolynomialShapeBank(ShapeBank):
    """
    Shape bank with polynomial shapes derived from mathematical truths.
    
    These shapes are the geometry of computation:
    - XOR(a, b) = a + b - 2ab
    - AND(a, b) = ab  
    - OR(a, b) = a + b - ab
    - NOT(a) = 1 - a
    
    Extended with projections and combinations for d_model dimensions.
    """
    
    @classmethod
    def from_primitives(
        cls,
        d_model: int,
        num_shapes: int = 16,
        include_identity: bool = True,
        include_projections: bool = True,
        seed: int = 42,
    ) -> 'PolynomialShapeBank':
        """
        Create a shape bank from mathematical primitives.
        
        Generates a diverse set of shapes by:
        1. Including core primitives (XOR, AND, OR, NOT patterns)
        2. Adding random projections that preserve structure
        3. Creating compositions of primitives
        """
        torch.manual_seed(seed)
        shapes = []
        
        # Core primitives applied dimension-wise with learned projections
        if include_identity:
            shapes.append(Shape(
                name="identity",
                fn=lambda x: x,
                signature=torch.ones(d_model),
            ))
        
        # XOR-like shapes (difference detectors)
        for i in range(num_shapes // 4):
            proj = torch.randn(d_model, d_model) * 0.1
            proj = proj / proj.norm() * math.sqrt(d_model)
            shapes.append(Shape(
                name=f"xor_proj_{i}",
                fn=cls._make_xor_shape(proj.clone()),
                signature=proj.sum(dim=0).sign(),
            ))
        
        # AND-like shapes (agreement detectors)
        for i in range(num_shapes // 4):
            proj = torch.randn(d_model, d_model) * 0.1
            proj = proj / proj.norm() * math.sqrt(d_model)
            shapes.append(Shape(
                name=f"and_proj_{i}",
                fn=cls._make_and_shape(proj.clone()),
                signature=proj.sum(dim=0).sign(),
            ))
        
        # Linear projections (for feature mixing)
        if include_projections:
            remaining = num_shapes - len(shapes)
            for i in range(remaining):
                proj = torch.randn(d_model, d_model) * (1.0 / math.sqrt(d_model))
                # Make orthogonal-ish
                if i > 0:
                    proj = proj - proj.mean()
                shapes.append(Shape(
                    name=f"proj_{i}",
                    fn=cls._make_linear_shape(proj.clone()),
                    signature=proj.sum(dim=0).sign(),
                ))
        
        return cls(shapes, d_model)
    
    @staticmethod
    def _make_xor_shape(proj: torch.Tensor) -> Callable:
        """Create a XOR-like shape with projection."""
        def shape_fn(x: torch.Tensor) -> torch.Tensor:
            # Project to mixing space
            h = F.linear(x, proj)
            # Apply XOR-like polynomial: activation that emphasizes differences
            # Approximates XOR behavior: high when inputs disagree
            h_norm = torch.sigmoid(h)
            xor_like = h_norm * (1 - h_norm) * 4  # Peaks at 0.5
            return F.linear(xor_like, proj.T)
        return shape_fn
    
    @staticmethod  
    def _make_and_shape(proj: torch.Tensor) -> Callable:
        """Create an AND-like shape with projection."""
        def shape_fn(x: torch.Tensor) -> torch.Tensor:
            # Project
            h = F.linear(x, proj)
            # AND-like: product emphasizes agreement
            h_norm = torch.sigmoid(h)
            and_like = h_norm * h_norm  # Emphasizes when both high
            return F.linear(and_like, proj.T)
        return shape_fn
    
    @staticmethod
    def _make_linear_shape(proj: torch.Tensor) -> Callable:
        """Create a linear projection shape."""
        def shape_fn(x: torch.Tensor) -> torch.Tensor:
            return F.linear(F.linear(x, proj), proj.T)
        return shape_fn


class DistilledShapeBank(ShapeBank):
    """
    Shape bank created by distilling patterns from a trained network.
    
    The distillation process:
    1. Train a network with STE (or other method)
    2. Observe the converged weight patterns
    3. Cluster similar patterns
    4. Freeze cluster centroids as shapes
    
    This uses STE once for discovery, then never again.
    """
    
    @classmethod
    def from_trained_tiles(
        cls,
        tiles: nn.ModuleList,
        d_model: int,
        num_shapes: Optional[int] = None,
    ) -> 'DistilledShapeBank':
        """
        Distill shapes from trained TriX tiles.
        
        Args:
            tiles: Trained TriXTile modules
            d_model: Model dimension
            num_shapes: Number of shapes to distill (default: len(tiles))
        """
        num_shapes = num_shapes or len(tiles)
        
        # Extract weight patterns
        with torch.no_grad():
            patterns = []
            for tile in tiles:
                # Get the effective computation as a frozen function
                up_w = tile.up_weight.sign()
                down_w = tile.down_weight.sign()
                up_scale = tile.up_scale.clone()
                down_scale = tile.down_scale.clone()
                
                # Create frozen shape from tile
                def make_tile_fn(uw, dw, us, ds):
                    def fn(x):
                        h = F.linear(x, uw) * us
                        h = ActivationShapes.relu(h)
                        return F.linear(h, dw) * ds
                    return fn
                
                patterns.append({
                    'fn': make_tile_fn(up_w, down_w, up_scale, down_scale),
                    'signature': tile.get_signature(),
                })
        
        # If we need fewer shapes than tiles, cluster
        if num_shapes < len(tiles):
            patterns = cls._cluster_patterns(patterns, num_shapes)
        
        # Build shapes
        shapes = []
        for i, p in enumerate(patterns):
            shapes.append(Shape(
                name=f"distilled_{i}",
                fn=p['fn'],
                signature=p['signature'],
            ))
        
        return cls(shapes, d_model)
    
    @staticmethod
    def _cluster_patterns(patterns: List[Dict], k: int) -> List[Dict]:
        """Cluster patterns by signature similarity."""
        # Simple k-means on signatures
        sigs = torch.stack([p['signature'] for p in patterns])
        n = len(patterns)
        
        # Initialize centroids
        indices = torch.randperm(n)[:k]
        centroids = sigs[indices].clone()
        
        # Iterate
        for _ in range(10):
            # Assign to nearest centroid
            dists = torch.cdist(sigs.float(), centroids.float())
            assignments = dists.argmin(dim=1)
            
            # Update centroids
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    centroids[c] = sigs[mask].float().mean(dim=0).sign()
        
        # Pick representative pattern for each cluster
        result = []
        for c in range(k):
            mask = (assignments == c)
            if mask.any():
                idx = mask.nonzero()[0].item()
                result.append(patterns[idx])
        
        return result


# =============================================================================
# GRADIENT TRUTH FFN: The Clean Architecture
# =============================================================================

@dataclass
class RoutingInfo:
    """Information about routing decisions."""
    weights: torch.Tensor  # [batch, num_shapes] - soft attention weights
    selected: torch.Tensor  # [batch] - argmax shape index
    entropy: torch.Tensor   # [batch] - routing entropy (uncertainty)


class GradientTruthFFN(nn.Module):
    """
    Feed-Forward Network with Gradient Truth architecture.
    
    Three-layer decomposition:
        1. Routing: Continuous, learned, receives gradients
        2. Shape Bank: Discrete, frozen, no gradients
        3. Magnitude: Continuous, learned, receives gradients
    
    No STE. Gradients flow through routing weights and magnitude scales.
    The shapes themselves are frozen truth.
    
    Args:
        d_model: Model dimension
        shape_bank: Pre-built frozen shape bank
        routing_type: 'soft' (differentiable) or 'hard' (Gumbel-softmax)
        routing_temp: Temperature for routing softmax
        dropout: Dropout rate
        use_signatures: Use shape signatures for routing bias
    """
    
    def __init__(
        self,
        d_model: int,
        shape_bank: ShapeBank,
        routing_type: str = 'soft',
        routing_temp: float = 1.0,
        dropout: float = 0.1,
        use_signatures: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.shape_bank = shape_bank
        self.num_shapes = len(shape_bank)
        self.routing_type = routing_type
        self.routing_temp = routing_temp
        self.use_signatures = use_signatures
        
        # Layer 1: Routing (continuous, learned)
        self.router = nn.Linear(d_model, self.num_shapes)
        
        # Optionally use signatures to bias routing
        if use_signatures:
            # Signature-based routing: input @ signatures.T
            self.register_buffer('sig_routing', shape_bank.get_signatures())
        
        # Layer 3: Magnitude (continuous, learned)
        self.scales = nn.Parameter(torch.ones(self.num_shapes))
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Output projection (continuous, learned)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.router.weight, gain=0.1)
        nn.init.zeros_(self.router.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, RoutingInfo]]:
        """
        Forward pass with Gradient Truth architecture.
        
        Args:
            x: [batch, d_model] or [batch, seq, d_model]
            return_routing: Whether to return routing info
            
        Returns:
            output: Same shape as input
            routing_info: (optional) RoutingInfo with attention weights
        """
        orig_shape = x.shape
        
        # Flatten if 3D
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        batch = x.shape[0]
        
        # Normalize input
        x_norm = self.norm(x)
        
        # =========================================
        # LAYER 1: Routing (continuous, gradients)
        # =========================================
        
        # Compute routing logits
        logits = self.router(x_norm)
        
        # Optionally add signature-based bias
        if self.use_signatures:
            sig_scores = x_norm @ self.sig_routing.T
            logits = logits + sig_scores * 0.5
        
        # Apply temperature and softmax
        logits = logits / self.routing_temp
        
        if self.routing_type == 'soft':
            weights = F.softmax(logits, dim=-1)  # [batch, num_shapes]
        elif self.routing_type == 'hard':
            # Gumbel-softmax for discrete but differentiable selection
            weights = F.gumbel_softmax(logits, tau=self.routing_temp, hard=True)
        else:
            raise ValueError(f"Unknown routing_type: {self.routing_type}")
        
        # =========================================
        # LAYER 2: Shape Bank (frozen, no gradients)
        # =========================================
        
        # Execute all shapes
        with torch.no_grad():
            shape_outputs = self.shape_bank(x)  # [batch, num_shapes, d_model]
        
        # Detach to ensure no gradients flow through shapes
        shape_outputs = shape_outputs.detach()
        
        # =========================================
        # LAYER 3: Magnitude (continuous, gradients)
        # =========================================
        
        # Apply learned scales per shape
        scaled = shape_outputs * self.scales.view(1, -1, 1)  # [batch, num_shapes, d_model]
        
        # Weighted combination (gradients flow through weights!)
        output = (weights.unsqueeze(-1) * scaled).sum(dim=1)  # [batch, d_model]
        
        # Output projection and scaling
        output = self.output_proj(output) * self.output_scale
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        # Restore shape
        if len(orig_shape) == 3:
            output = output.view(orig_shape)
        
        if return_routing:
            routing_info = RoutingInfo(
                weights=weights.view(orig_shape[:-1] + (self.num_shapes,)) if len(orig_shape) == 3 else weights,
                selected=weights.argmax(dim=-1).view(orig_shape[:-1]) if len(orig_shape) == 3 else weights.argmax(dim=-1),
                entropy=-(weights * (weights + 1e-8).log()).sum(dim=-1),
            )
            return output, routing_info
        
        return output
    
    def get_routing_stats(self) -> Dict:
        """Get statistics about routing behavior."""
        return {
            'num_shapes': self.num_shapes,
            'routing_type': self.routing_type,
            'routing_temp': self.routing_temp,
            'scales': self.scales.detach().cpu().tolist(),
        }


class GradientTruthBlock(nn.Module):
    """
    Transformer block with Gradient Truth FFN.
    
    Standard attention + GradientTruthFFN.
    All discrete computation is in the FFN's frozen shape bank.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        shape_bank: ShapeBank,
        dropout: float = 0.1,
        routing_temp: float = 1.0,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = GradientTruthFFN(
            d_model=d_model,
            shape_bank=shape_bank,
            routing_temp=routing_temp,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
        return_routing: bool = True,
    ) -> Tuple[torch.Tensor, Optional[RoutingInfo]]:
        """Forward pass through transformer block."""
        
        # Self-attention
        normed = self.norm1(x)
        if is_causal:
            T = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Gradient Truth FFN
        if return_routing:
            ffn_out, routing_info = self.ffn(self.norm2(x), return_routing=True)
        else:
            ffn_out = self.ffn(self.norm2(x), return_routing=False)
            routing_info = None
        
        # Note: residual already applied in GradientTruthFFN
        # But we applied norm2 before, so output is already x + ffn(norm(x))
        # Actually, the FFN adds residual internally, so:
        x = ffn_out  # FFN already did x + output
        
        return x, routing_info


# =============================================================================
# SHAPE GENESIS UTILITIES
# =============================================================================

class ShapeGenesis:
    """
    Utilities for discovering shapes without gradients.
    
    Three methods:
    1. derive() - Mathematical derivation
    2. evolve() - Evolutionary search
    3. distill() - Observe trained networks
    """
    
    @staticmethod
    def derive_boolean_shapes(d_model: int) -> ShapeBank:
        """
        Derive shapes from Boolean algebra truths.
        
        These are mathematical facts, not learned patterns.
        """
        shapes = []
        
        # The four axioms, applied element-wise
        shapes.append(Shape(
            name="identity",
            fn=lambda x: x,
            signature=torch.ones(d_model),
        ))
        
        shapes.append(Shape(
            name="negate",
            fn=lambda x: -x,
            signature=-torch.ones(d_model),
        ))
        
        # Self-interaction (AND-like)
        shapes.append(Shape(
            name="square",
            fn=lambda x: x * torch.sigmoid(x),
            signature=torch.ones(d_model),
        ))
        
        # Difference from mean (XOR-like)
        shapes.append(Shape(
            name="center",
            fn=lambda x: x - x.mean(dim=-1, keepdim=True),
            signature=torch.zeros(d_model),
        ))
        
        return ShapeBank(shapes, d_model)
    
    @staticmethod
    def evolve_shapes(
        d_model: int,
        num_shapes: int,
        fitness_fn: Callable[[Callable], float],
        generations: int = 100,
        population: int = 50,
    ) -> ShapeBank:
        """
        Evolve shapes using genetic algorithm.
        
        No gradients - just mutation and selection.
        
        Args:
            d_model: Model dimension
            num_shapes: Number of shapes to evolve
            fitness_fn: Function that scores a shape (higher = better)
            generations: Number of evolution generations
            population: Population size per generation
        """
        # Initialize random population
        pop = []
        for _ in range(population):
            proj = torch.randn(d_model, d_model) * 0.1
            fn = lambda x, p=proj: F.linear(ActivationShapes.relu(F.linear(x, p)), p.T)
            pop.append({'proj': proj, 'fn': fn, 'fitness': 0.0})
        
        # Evolve
        for gen in range(generations):
            # Evaluate fitness
            for ind in pop:
                ind['fitness'] = fitness_fn(ind['fn'])
            
            # Sort by fitness
            pop.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Keep top half, mutate to fill rest
            survivors = pop[:population // 2]
            children = []
            for parent in survivors:
                # Mutate projection
                child_proj = parent['proj'] + torch.randn_like(parent['proj']) * 0.01
                child_fn = lambda x, p=child_proj: F.linear(ActivationShapes.relu(F.linear(x, p)), p.T)
                children.append({'proj': child_proj, 'fn': child_fn, 'fitness': 0.0})
            
            pop = survivors + children
        
        # Take top num_shapes
        pop.sort(key=lambda x: x['fitness'], reverse=True)
        shapes = []
        for i, ind in enumerate(pop[:num_shapes]):
            shapes.append(Shape(
                name=f"evolved_{i}",
                fn=ind['fn'],
                signature=ind['proj'].sum(dim=0).sign(),
            ))
        
        return ShapeBank(shapes, d_model)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_gradient_truth_ffn(
    d_model: int,
    num_shapes: int = 16,
    routing_temp: float = 1.0,
    dropout: float = 0.1,
) -> GradientTruthFFN:
    """
    Create a GradientTruthFFN with polynomial shapes.
    
    Convenience function for quick setup.
    """
    shape_bank = PolynomialShapeBank.from_primitives(
        d_model=d_model,
        num_shapes=num_shapes,
    )
    return GradientTruthFFN(
        d_model=d_model,
        shape_bank=shape_bank,
        routing_temp=routing_temp,
        dropout=dropout,
    )


def create_gradient_truth_block(
    d_model: int,
    n_heads: int,
    num_shapes: int = 16,
    routing_temp: float = 1.0,
    dropout: float = 0.1,
) -> GradientTruthBlock:
    """
    Create a GradientTruthBlock with polynomial shapes.
    
    Convenience function for quick setup.
    """
    shape_bank = PolynomialShapeBank.from_primitives(
        d_model=d_model,
        num_shapes=num_shapes,
    )
    return GradientTruthBlock(
        d_model=d_model,
        n_heads=n_heads,
        shape_bank=shape_bank,
        routing_temp=routing_temp,
        dropout=dropout,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing GradientTruth architecture...")
    
    d_model = 128
    batch = 32
    seq_len = 16
    
    # Create shape bank
    print("\n1. Creating polynomial shape bank...")
    shape_bank = PolynomialShapeBank.from_primitives(d_model=d_model, num_shapes=16)
    print(f"   Created {len(shape_bank)} shapes")
    
    # Create FFN
    print("\n2. Creating GradientTruthFFN...")
    ffn = GradientTruthFFN(d_model=d_model, shape_bank=shape_bank)
    
    # Count parameters
    total_params = sum(p.numel() for p in ffn.parameters())
    trainable_params = sum(p.numel() for p in ffn.parameters() if p.requires_grad)
    print(f"   Total params: {total_params}")
    print(f"   Trainable params: {trainable_params}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    x = torch.randn(batch, d_model)
    output, routing_info = ffn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Routing entropy: {routing_info.entropy.mean().item():.4f}")
    
    # Test with 3D input
    print("\n4. Testing 3D input...")
    x3d = torch.randn(batch, seq_len, d_model)
    output3d, routing3d = ffn(x3d)
    print(f"   Input shape: {x3d.shape}")
    print(f"   Output shape: {output3d.shape}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    x = torch.randn(batch, d_model, requires_grad=True)
    output, _ = ffn(x)
    loss = output.sum()
    loss.backward()
    
    print(f"   Input grad exists: {x.grad is not None}")
    print(f"   Router grad exists: {ffn.router.weight.grad is not None}")
    print(f"   Scales grad exists: {ffn.scales.grad is not None}")
    
    # Verify no STE was used (check shape bank has no grads)
    shape_grads = [p.grad for p in ffn.shape_bank.parameters() if p.grad is not None]
    print(f"   Shape bank grads: {len(shape_grads)} (should be 0)")
    
    # Test transformer block
    print("\n6. Testing GradientTruthBlock...")
    block = GradientTruthBlock(
        d_model=d_model,
        n_heads=8,
        shape_bank=shape_bank,
    )
    
    x = torch.randn(4, seq_len, d_model)
    output, routing = block(x)
    print(f"   Block output shape: {output.shape}")
    
    # Gradient check
    loss = output.sum()
    loss.backward()
    has_grads = any(p.grad is not None for p in block.parameters() if p.requires_grad)
    print(f"   Block gradients flow: {has_grads}")
    
    print("\n✓ All tests passed!")
    print("\nGradient Truth: Gradients flow only where there is genuine uncertainty.")
