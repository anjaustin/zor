"""
Native Programmable Tiles - CuPy-based Modifiable Computation Units

Port of trix.guardian.programmable_tile to the native (PyTorch-free) stack.
Same interface, CuPy arrays instead of torch tensors.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


@dataclass
class TileModification:
    """Record of a tile modification for history tracking."""
    version: int
    modification_type: str  # 'signature' or 'weights'
    blend: float
    timestamp: float
    reason: str = ""


class NativeProgrammableTile:
    """
    A single programmable tile with read/write interface.
    
    CuPy-native version - no PyTorch dependency.
    
    Exposes its signature (routing address) and weights (computation)
    for inspection and bounded modification by observers.
    """
    
    def __init__(self, d_model: int, d_hidden: int, tile_id: int = 0):
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.tile_id = tile_id
        
        # Core parameters (CuPy arrays)
        self.signature = cp.random.randn(d_model).astype(cp.float32) * 0.1
        self.weights_up = cp.random.randn(d_model, d_hidden).astype(cp.float32) * 0.02
        self.weights_down = cp.random.randn(d_hidden, d_model).astype(cp.float32) * 0.02
        self.bias = cp.zeros(d_hidden, dtype=cp.float32)
        
        # Gradient buffers
        self.d_signature = cp.zeros_like(self.signature)
        self.d_weights_up = cp.zeros_like(self.weights_up)
        self.d_weights_down = cp.zeros_like(self.weights_down)
        self.d_bias = cp.zeros_like(self.bias)
        
        # State tracking
        self._frozen = False
        self._version = 0
        self._history: List[TileModification] = []
        self._initial_signature: Optional[cp.ndarray] = None
        
    def save_initial_state(self):
        """Snapshot initial signature for movement tracking."""
        self._initial_signature = self.signature.copy()
    
    @property
    def signature_movement(self) -> float:
        """How far has the signature moved from initialization?"""
        if self._initial_signature is None:
            return 0.0
        return float(cp.linalg.norm(self.signature - self._initial_signature))
    
    # === Read Interface ===
    
    def read_signature(self) -> cp.ndarray:
        """Read current signature (copy for safety)."""
        return self.signature.copy()
    
    def read_weights(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """Read current weights (copies for safety)."""
        return self.weights_up.copy(), self.weights_down.copy()
    
    def read_gradients(self) -> Dict[str, cp.ndarray]:
        """Read current gradients."""
        return {
            'signature': self.d_signature.copy(),
            'weights_up': self.d_weights_up.copy(),
            'weights_down': self.d_weights_down.copy(),
            'bias': self.d_bias.copy(),
        }
    
    # === Write Interface (Gentle) ===
    
    def write_signature(
        self, 
        new_sig: cp.ndarray, 
        blend: float = 0.1,
        reason: str = ""
    ) -> bool:
        """
        Gently blend new signature with current.
        
        blend=0.0: no change
        blend=1.0: full replacement
        blend=0.1: gentle nudge (recommended)
        """
        if self._frozen:
            return False
        
        self.signature = (1 - blend) * self.signature + blend * cp.asarray(new_sig)
        
        self._version += 1
        self._history.append(TileModification(
            version=self._version,
            modification_type='signature',
            blend=blend,
            timestamp=time.time(),
            reason=reason
        ))
        return True
    
    def write_weights(
        self,
        new_weights_up: cp.ndarray,
        new_weights_down: cp.ndarray,
        blend: float = 0.1,
        reason: str = ""
    ) -> bool:
        """Blend new weights with current weights."""
        if self._frozen:
            return False
        
        self.weights_up = (1 - blend) * self.weights_up + blend * cp.asarray(new_weights_up)
        self.weights_down = (1 - blend) * self.weights_down + blend * cp.asarray(new_weights_down)
        
        self._version += 1
        self._history.append(TileModification(
            version=self._version,
            modification_type='weights',
            blend=blend,
            timestamp=time.time(),
            reason=reason
        ))
        return True
    
    # === Control Interface ===
    
    def freeze(self):
        """Prevent modifications."""
        self._frozen = True
    
    def unfreeze(self):
        """Allow modifications."""
        self._frozen = False
    
    @property
    def is_frozen(self) -> bool:
        return self._frozen
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def history(self) -> List[TileModification]:
        return self._history.copy()
    
    # === Forward/Backward ===
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass through tile.
        
        Args:
            x: [batch, d_model]
            
        Returns:
            [batch, d_model]
        """
        self._cached_x = x
        hidden = cp.dot(x, self.weights_up) + self.bias
        hidden = cp.maximum(hidden, 0)  # ReLU
        self._cached_hidden = hidden
        return cp.dot(hidden, self.weights_down)
    
    def backward(self, d_output: cp.ndarray) -> cp.ndarray:
        """
        Backward pass - accumulate gradients.
        
        Args:
            d_output: [batch, d_model]
            
        Returns:
            d_input: [batch, d_model]
        """
        batch = d_output.shape[0]
        
        # Gradient through down projection
        self.d_weights_down += cp.dot(self._cached_hidden.T, d_output)
        d_hidden = cp.dot(d_output, self.weights_down.T)
        
        # Gradient through ReLU
        d_hidden = d_hidden * (self._cached_hidden > 0)
        
        # Gradient through up projection
        self.d_weights_up += cp.dot(self._cached_x.T, d_hidden)
        self.d_bias += d_hidden.sum(axis=0)
        
        return cp.dot(d_hidden, self.weights_up.T)
    
    def zero_grad(self):
        """Reset gradient buffers."""
        self.d_signature.fill(0)
        self.d_weights_up.fill(0)
        self.d_weights_down.fill(0)
        self.d_bias.fill(0)
    
    def get_params(self) -> Dict[str, cp.ndarray]:
        """Get all parameters for optimizer."""
        return {
            f'tile_{self.tile_id}_signature': self.signature,
            f'tile_{self.tile_id}_weights_up': self.weights_up,
            f'tile_{self.tile_id}_weights_down': self.weights_down,
            f'tile_{self.tile_id}_bias': self.bias,
        }
    
    def get_grads(self) -> Dict[str, cp.ndarray]:
        """Get all gradients for optimizer."""
        return {
            f'tile_{self.tile_id}_signature': self.d_signature,
            f'tile_{self.tile_id}_weights_up': self.d_weights_up,
            f'tile_{self.tile_id}_weights_down': self.d_weights_down,
            f'tile_{self.tile_id}_bias': self.d_bias,
        }


class NativeProgrammableTileBank:
    """
    Collection of programmable tiles with unified interface.
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_tiles: int,
    ):
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_tiles = num_tiles
        
        self.tiles = [
            NativeProgrammableTile(d_model, d_hidden, tile_id=i)
            for i in range(num_tiles)
        ]
        
        # Routing via signature matching
        self._cached_signatures: Optional[cp.ndarray] = None
    
    def get_signatures(self) -> cp.ndarray:
        """Get all tile signatures as matrix [num_tiles, d_model]."""
        return cp.stack([t.signature for t in self.tiles])
    
    def route(self, x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Route inputs to tiles via signature matching.
        
        Args:
            x: [batch, d_model]
            
        Returns:
            tile_idx: [batch] winning tile indices
            scores: [batch] alignment scores
        """
        sigs = self.get_signatures()
        scores = cp.dot(x, sigs.T)  # [batch, num_tiles]
        tile_idx = scores.argmax(axis=-1)
        max_scores = scores.max(axis=-1)
        return tile_idx, max_scores
    
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass with routing.
        
        Args:
            x: [batch, d_model]
            
        Returns:
            output: [batch, d_model]
        """
        batch = x.shape[0]
        tile_idx, _ = self.route(x)
        self._cached_tile_idx = tile_idx
        
        output = cp.zeros((batch, self.d_model), dtype=cp.float32)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            output[mask] = self.tiles[t].forward(x[mask])
        
        return output
    
    def backward(self, d_output: cp.ndarray) -> cp.ndarray:
        """
        Backward pass through routed tiles.
        """
        batch = d_output.shape[0]
        tile_idx = self._cached_tile_idx
        
        d_input = cp.zeros((batch, self.d_model), dtype=cp.float32)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            d_input[mask] = self.tiles[t].backward(d_output[mask])
        
        return d_input
    
    def zero_grad(self):
        """Reset all gradient buffers."""
        for tile in self.tiles:
            tile.zero_grad()
    
    def get_params(self) -> Dict[str, cp.ndarray]:
        """Get all parameters for optimizer."""
        params = {}
        for tile in self.tiles:
            params.update(tile.get_params())
        return params
    
    def get_grads(self) -> Dict[str, cp.ndarray]:
        """Get all gradients for optimizer."""
        grads = {}
        for tile in self.tiles:
            grads.update(tile.get_grads())
        return grads
    
    # === Bank-level observation ===
    
    def get_load_balance(self, tile_idx: cp.ndarray) -> cp.ndarray:
        """Compute load per tile from routing decisions."""
        counts = cp.zeros(self.num_tiles, dtype=cp.float32)
        for t in range(self.num_tiles):
            counts[t] = (tile_idx == t).sum()
        return counts / tile_idx.shape[0]
    
    def get_signature_diversity(self) -> float:
        """Measure how different tile signatures are."""
        sigs = self.get_signatures()
        # Normalize
        norms = cp.linalg.norm(sigs, axis=-1, keepdims=True)
        sigs_norm = sigs / (norms + 1e-8)
        # Pairwise cosine similarity
        sim = cp.dot(sigs_norm, sigs_norm.T)
        # Mean off-diagonal similarity
        mask = 1 - cp.eye(self.num_tiles)
        mean_sim = (sim * mask).sum() / mask.sum()
        # Diversity = 1 - similarity, clamp to [0, 1]
        diversity = float(1 - mean_sim)
        return max(0.0, min(1.0, diversity))
    
    def freeze_all(self):
        """Freeze all tiles."""
        for tile in self.tiles:
            tile.freeze()
    
    def unfreeze_all(self):
        """Unfreeze all tiles."""
        for tile in self.tiles:
            tile.unfreeze()


class NativeTrainingObserver:
    """
    Observer for native programmable tiles.
    
    Monitors training dynamics and can intervene with bounded modifications.
    """
    
    def __init__(
        self,
        tile_bank: NativeProgrammableTileBank,
        diversity_threshold: float = 0.3,
        balance_threshold: float = 0.5,
        intervention_blend: float = 0.1,
    ):
        self.tile_bank = tile_bank
        self.diversity_threshold = diversity_threshold
        self.balance_threshold = balance_threshold
        self.intervention_blend = intervention_blend
        
        self._observations: List[Dict] = []
        self._interventions: List[Dict] = []
    
    def observe(self, tile_idx: cp.ndarray) -> Dict:
        """
        Observe current training state.
        
        Args:
            tile_idx: [batch] routing decisions from last forward pass
            
        Returns:
            Observation dict with metrics
        """
        load = self.tile_bank.get_load_balance(tile_idx)
        diversity = self.tile_bank.get_signature_diversity()
        
        obs = {
            'timestamp': time.time(),
            'load_balance': float(load.std()),
            'load_max': float(load.max()),
            'load_min': float(load.min()),
            'diversity': diversity,
            'num_unused': int((load == 0).sum()),
        }
        
        self._observations.append(obs)
        return obs
    
    def should_intervene(self, obs: Dict) -> Tuple[bool, str]:
        """Check if intervention is warranted."""
        if obs['diversity'] < self.diversity_threshold:
            return True, 'signature_collapse'
        if obs['load_max'] > self.balance_threshold:
            return True, 'load_imbalance'
        if obs['num_unused'] > self.tile_bank.num_tiles // 4:
            return True, 'unused_tiles'
        return False, ''
    
    def intervene(self, reason: str) -> int:
        """
        Apply intervention based on reason.
        
        Returns number of tiles modified.
        """
        modified = 0
        
        if reason == 'signature_collapse':
            # Push similar signatures apart
            sigs = self.tile_bank.get_signatures()
            norms = cp.linalg.norm(sigs, axis=-1, keepdims=True)
            sigs_norm = sigs / (norms + 1e-8)
            sim = cp.dot(sigs_norm, sigs_norm.T)
            
            for i in range(self.tile_bank.num_tiles):
                # Find most similar other tile
                sim[i, i] = -1  # Exclude self
                j = int(sim[i].argmax())
                if sim[i, j] > 0.9:  # Too similar
                    # Add noise to push apart
                    noise = cp.random.randn(self.tile_bank.d_model).astype(cp.float32) * 0.5
                    self.tile_bank.tiles[i].write_signature(
                        self.tile_bank.tiles[i].signature + noise,
                        blend=self.intervention_blend,
                        reason='diversity_push'
                    )
                    modified += 1
        
        elif reason == 'load_imbalance' or reason == 'unused_tiles':
            # Nudge unused tiles toward popular tile's signature
            load = self.tile_bank.get_load_balance(self.tile_bank._cached_tile_idx)
            popular = int(load.argmax())
            popular_sig = self.tile_bank.tiles[popular].read_signature()
            
            for i in range(self.tile_bank.num_tiles):
                if load[i] == 0:
                    # Move toward popular with noise
                    noise = cp.random.randn(self.tile_bank.d_model).astype(cp.float32) * 0.3
                    new_sig = popular_sig + noise
                    self.tile_bank.tiles[i].write_signature(
                        new_sig,
                        blend=self.intervention_blend,
                        reason='load_rebalance'
                    )
                    modified += 1
        
        self._interventions.append({
            'timestamp': time.time(),
            'reason': reason,
            'tiles_modified': modified,
        })
        
        return modified
    
    def step(self, tile_idx: cp.ndarray) -> Dict:
        """
        Full observation + intervention cycle.
        
        Returns observation dict (with 'intervened' field if intervention occurred).
        """
        obs = self.observe(tile_idx)
        should, reason = self.should_intervene(obs)
        
        if should:
            modified = self.intervene(reason)
            obs['intervened'] = True
            obs['intervention_reason'] = reason
            obs['tiles_modified'] = modified
        else:
            obs['intervened'] = False
        
        return obs
    
    @property
    def observations(self) -> List[Dict]:
        return self._observations.copy()
    
    @property
    def interventions(self) -> List[Dict]:
        return self._interventions.copy()
