"""
SparseOctaveLookupFFN - Multi-scale sparse memory for FFN replacement

Replaces standard FFN (y = W2 @ ReLU(W1 @ x + b1) + b2) with multi-scale
content-addressed memory lookup using Providence at multiple octave resolutions.

"Information lives at different scales. Capture it where it lives."

Key concepts:
- Octaves: Natural scale levels (bit, byte, word, block)
- Providence: Content-addressed memory with Hamming distance
- Sparse lookup: Only query top-k nearest neighbors
- Frozen extraction: Bit shifts for octave decomposition (no learned params)
- Learned blending: Combine octave outputs adaptively

Created by: Tripp + Claude
Date: December 2025
"""

import numpy as np
from typing import List, Optional, Tuple

# Try CuPy for GPU support
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = np
    HAS_CUDA = False


# ═══════════════════════════════════════════════════════════════════════════
# Providence: Content-Addressed Memory
# ═══════════════════════════════════════════════════════════════════════════

class Providence:
    """
    Content-addressed memory with Hamming-like distance lookup.

    Named after Providence, Rhode Island - where addresses find you.
    Uses soft lookup with attention-weighted blending.
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 1024,
        precision: str = 'fp32'
    ):
        self.d_model = d_model
        self.memory_size = memory_size
        self.precision = precision

        # Get array module
        self.xp = cp if HAS_CUDA else np

        # Dtype based on precision
        self.dtype = {
            'fp16': np.float16,
            'fp32': np.float32,
            'fp64': np.float64
        }.get(precision, np.float32)

        # Initialize memory: keys and values
        # Keys are what we match against, values are what we retrieve
        scale = 0.02
        self.keys = self.xp.random.randn(memory_size, d_model).astype(self.dtype) * scale
        self.values = self.xp.random.randn(memory_size, d_model).astype(self.dtype) * scale

    def _get_xp(self, arr):
        """Get the array module for the given array."""
        if HAS_CUDA and hasattr(arr, 'device'):
            return cp
        return np

    def hamming_distance(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        Compute Hamming-like distance between query and keys.

        For continuous values, we use L1 distance as a differentiable
        proxy for Hamming distance. On binary inputs, this equals Hamming.

        The Hamming distance is a frozen shape: d(a,b) = Σ |a_i ⊕ b_i|
        """
        # Use the array module matching the input
        xp = self._get_xp(query)

        # Ensure keys are on the same device/type as query
        if xp == cp and not hasattr(self.keys, 'device'):
            self.keys = cp.asarray(self.keys)
        elif xp == np and hasattr(self.keys, 'device'):
            self.keys = cp.asnumpy(self.keys)

        # query: [batch, d_model]
        # keys: [memory_size, d_model]
        # output: [batch, memory_size]
        return xp.sum(xp.abs(query[:, None, :] - self.keys[None, :, :]), axis=-1)

    def soft_lookup(
        self,
        query: np.ndarray,
        k: int = 16,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Sparse soft lookup with attention-weighted blending.

        1. Compute distances to all keys
        2. Select top-k nearest (smallest distance)
        3. Compute attention weights (softmax of negative distance)
        4. Return weighted sum of corresponding values

        This is implementable in pure frozen shapes:
        - Distance: Hamming (XOR + popcount)
        - Top-k: Comparison shapes
        - Weights: Softmax (exp + div)
        - Blend: MUL + ADD

        Args:
            query: Input tensor [batch, d_model]
            k: Number of nearest neighbors to retrieve
            temperature: Softmax temperature (lower = sharper)

        Returns:
            Output tensor [batch, d_model]
        """
        # Get array module matching input
        xp = self._get_xp(query)
        batch_size = query.shape[0]

        # Ensure values match input type
        if xp == cp and not hasattr(self.values, 'device'):
            self.values = cp.asarray(self.values)
        elif xp == np and hasattr(self.values, 'device'):
            self.values = cp.asnumpy(self.values)

        # Compute distances to all keys
        distances = self.hamming_distance(query, self.keys)  # [batch, memory_size]

        # Top-k nearest (smallest distance)
        # Use partition for efficiency
        k = min(k, self.memory_size)
        top_k_idx = xp.argpartition(distances, k, axis=-1)[:, :k]  # [batch, k]

        # Gather top-k distances and values
        batch_idx = xp.arange(batch_size)[:, None]
        top_k_dist = distances[batch_idx, top_k_idx]  # [batch, k]
        top_k_vals = self.values[top_k_idx]           # [batch, k, d_model]

        # Attention weights (inverse distance with temperature)
        # Using exp(-dist/temp) as soft attention
        weights = xp.exp(-top_k_dist / temperature)
        weights = weights / (xp.sum(weights, axis=-1, keepdims=True) + 1e-8)

        # Weighted sum of values
        output = xp.sum(weights[:, :, None] * top_k_vals, axis=1)  # [batch, d_model]

        return output

    def update_memory(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        indices: Optional[np.ndarray] = None
    ):
        """Update memory entries (for training)."""
        if indices is None:
            # Replace all
            self.keys[:] = keys
            self.values[:] = values
        else:
            # Replace specific indices
            self.keys[indices] = keys
            self.values[indices] = values


# ═══════════════════════════════════════════════════════════════════════════
# SparseOctaveLookupFFN
# ═══════════════════════════════════════════════════════════════════════════

class SparseOctaveLookupFFN:
    """
    Multi-scale sparse memory FFN replacement.

    Instead of dense matrix multiplies, uses content-addressed lookup
    at multiple resolution scales (octaves).

    Standard FFN:
        h = ReLU(x @ W1 + b1)
        y = h @ W2 + b2
        Parameters: 8 * d^2

    SparseOctaveLookupFFN:
        for each octave:
            key = extract_octave(x, scale)  # Frozen
            value = providence_lookup(key)   # Sparse lookup
        y = blend(values)                    # Learned
        Parameters: n_octaves * memory_size * d + blend_params
    """

    def __init__(
        self,
        d_model: int,
        n_octaves: int = 3,
        memory_size: int = 1024,
        top_k: int = 16,
        precision: str = 'fp32',
        dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.n_octaves = n_octaves
        self.memory_size = memory_size
        self.top_k = top_k
        self.precision = precision
        self.dropout_rate = dropout_rate

        # Get array module
        self.xp = cp if HAS_CUDA else np
        self.dtype = np.float32 if precision == 'fp32' else np.float16

        # Providence memory at each octave
        self.octaves = [
            Providence(d_model, memory_size, precision)
            for _ in range(n_octaves)
        ]

        # Octave shift amounts (frozen)
        # Octave 0: full precision (shift 0)
        # Octave 1: medium (shift 4 bits)
        # Octave 2: coarse (shift 8 bits)
        # Octave 3: very coarse (shift 12 bits)
        self.shifts = [0, 4, 8, 12][:n_octaves]

        # Blending weights (learned)
        # These determine how much each octave contributes
        self.blend_weights = self.xp.ones(n_octaves, dtype=self.dtype) / n_octaves
        self.blend_bias = self.xp.zeros(d_model, dtype=self.dtype)

        # Output projection (optional, for expressiveness)
        self.output_scale = self.xp.ones(d_model, dtype=self.dtype)

    def _get_xp_for_array(self, arr):
        """Get the array module for the given array."""
        if HAS_CUDA and hasattr(arr, 'device'):
            return cp
        return np

    def extract_octave_key(self, x: np.ndarray, shift: int) -> np.ndarray:
        """
        Extract octave-specific key via bit shift (frozen shape).

        For float values, we simulate bit shifting via quantization.
        This preserves the coarse/fine hierarchy without learning.

        Shift 0: Full precision (all bits)
        Shift 4: 16x coarser
        Shift 8: 256x coarser
        """
        if shift == 0:
            return x

        # Use array module matching input
        xp = self._get_xp_for_array(x)
        # Simulate bit shift by quantizing
        scale = 2.0 ** shift
        return xp.floor(x * 256 / scale) * scale / 256

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through multi-scale sparse lookup.

        Args:
            x: Input tensor [batch, d_model]
            training: Whether to apply octave dropout

        Returns:
            Output tensor [batch, d_model]
        """
        # Use array module matching input
        xp = self._get_xp_for_array(x)
        batch_size = x.shape[0]

        # Ensure blend weights and output params match input type
        if xp == np:
            if hasattr(self.blend_weights, 'device'):
                self.blend_weights = cp.asnumpy(self.blend_weights)
                self.blend_bias = cp.asnumpy(self.blend_bias)
                self.output_scale = cp.asnumpy(self.output_scale)
        else:
            if not hasattr(self.blend_weights, 'device'):
                self.blend_weights = cp.asarray(self.blend_weights)
                self.blend_bias = cp.asarray(self.blend_bias)
                self.output_scale = cp.asarray(self.output_scale)

        octave_outputs = []

        for i, (octave, shift) in enumerate(zip(self.octaves, self.shifts)):
            # Octave dropout during training
            # Randomly drop entire octaves to ensure each learns something useful
            if training and xp.random.rand() < self.dropout_rate:
                octave_outputs.append(xp.zeros((batch_size, self.d_model), dtype=self.dtype))
                continue

            # Extract octave-specific key (frozen shape)
            key = self.extract_octave_key(x, shift)

            # Providence lookup
            value = octave.soft_lookup(key, k=self.top_k)

            octave_outputs.append(value)

        # Stack outputs from all octaves
        stacked = xp.stack(octave_outputs, axis=1)  # [batch, n_octaves, d_model]

        # Compute blending weights (softmax for proper weighting)
        weights = xp.exp(self.blend_weights)
        weights = weights / xp.sum(weights)

        # Weighted combination across octaves
        output = xp.sum(stacked * weights[None, :, None], axis=1)  # [batch, d_model]

        # Apply output scale and bias
        output = output * self.output_scale + self.blend_bias

        return output

    def parameter_count(self) -> int:
        """Count total parameters."""
        # Providence memories (keys + values)
        memory_params = self.n_octaves * 2 * self.memory_size * self.d_model

        # Blend weights + bias + output scale
        blend_params = self.n_octaves + self.d_model + self.d_model

        return memory_params + blend_params

    def frozen_parameter_count(self) -> int:
        """Count frozen (non-learned) parameters."""
        # The octave shifts are frozen
        return self.n_octaves  # Just the shift values

    def learned_parameter_count(self) -> int:
        """Count learned parameters."""
        return self.parameter_count() - self.frozen_parameter_count()

    def get_octave_contributions(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Get individual octave contributions (for visualization/debugging).

        Returns list of outputs from each octave before blending.
        """
        contributions = []
        for octave, shift in zip(self.octaves, self.shifts):
            key = self.extract_octave_key(x, shift)
            value = octave.soft_lookup(key, k=self.top_k)
            contributions.append(value)
        return contributions


# ═══════════════════════════════════════════════════════════════════════════
# Transformer FFN Drop-in Replacement
# ═══════════════════════════════════════════════════════════════════════════

class SparseOctaveTransformerFFN:
    """
    Drop-in replacement for Transformer FFN using SparseOctaveLookup.

    Standard Transformer FFN:
        h = GELU(x @ W1 + b1)   # Up-project
        y = h @ W2 + b2         # Down-project

    This replacement:
        y = SparseOctaveLookup(x)

    Use this when you want to replace standard FFN layers with sparse
    multi-scale lookup for efficiency and interpretability.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,  # Ignored - we use memory_size instead
        n_octaves: int = 3,
        memory_size: int = 1024,
        top_k: int = 16,
        **kwargs
    ):
        # d_ff is ignored - parameter count determined by memory_size
        self.lookup = SparseOctaveLookupFFN(
            d_model=d_model,
            n_octaves=n_octaves,
            memory_size=memory_size,
            top_k=top_k,
            **kwargs
        )

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        return self.lookup.forward(x, training=training)

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(x, training=training)

    def parameter_count(self) -> int:
        return self.lookup.parameter_count()


# ═══════════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

class SparseOctaveTrainer:
    """
    Trainer for SparseOctaveLookupFFN.

    Uses gradient estimation for memory updates and standard
    gradient descent for blending weights.
    """

    def __init__(
        self,
        model: SparseOctaveLookupFFN,
        lr: float = 0.001,
        memory_lr: float = 0.0001
    ):
        self.model = model
        self.lr = lr
        self.memory_lr = memory_lr
        self.xp = model.xp

    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """MSE loss."""
        return float(self.xp.mean((pred - target) ** 2))

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Single training step.

        Returns loss and dictionary of metrics.
        """
        xp = self.xp

        # Forward pass
        pred = self.model.forward(x, training=True)
        loss = self.compute_loss(pred, target)

        # Gradient estimation for blend weights (finite differences)
        eps = 1e-4
        blend_grads = xp.zeros_like(self.model.blend_weights)

        for i in range(self.model.n_octaves):
            # Perturb weight
            self.model.blend_weights[i] += eps
            pred_plus = self.model.forward(x, training=False)
            loss_plus = self.compute_loss(pred_plus, target)
            self.model.blend_weights[i] -= eps

            # Estimate gradient
            blend_grads[i] = (loss_plus - loss) / eps

        # Update blend weights
        self.model.blend_weights -= self.lr * blend_grads

        # Update memory (simplified - move closest keys toward query)
        error = target - pred  # [batch, d_model]
        for octave in self.model.octaves:
            # Find which memory entries were used (approximate)
            # In full implementation, track this during forward pass
            octave.values += self.memory_lr * xp.mean(error, axis=0, keepdims=True)

        # Compute metrics
        metrics = {
            'loss': loss,
            'blend_weights': self.model.blend_weights.tolist() if hasattr(self.model.blend_weights, 'tolist') else list(self.model.blend_weights),
        }

        return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_providence():
    """Test Providence content-addressed memory."""
    print("Testing Providence...")

    d_model = 64
    memory_size = 256
    batch_size = 8

    prov = Providence(d_model, memory_size)

    xp = cp if HAS_CUDA else np
    query = xp.random.randn(batch_size, d_model).astype(np.float32)

    # Soft lookup
    output = prov.soft_lookup(query, k=16)

    assert output.shape == (batch_size, d_model), f"Shape mismatch: {output.shape}"
    print("  Providence PASS")
    return True


def test_sparse_octave():
    """Test SparseOctaveLookupFFN."""
    print("Testing SparseOctaveLookupFFN...")

    d_model = 64
    batch_size = 8

    model = SparseOctaveLookupFFN(
        d_model=d_model,
        n_octaves=3,
        memory_size=256,
        top_k=8
    )

    xp = cp if HAS_CUDA else np
    x = xp.random.randn(batch_size, d_model).astype(np.float32)

    # Forward pass
    y = model.forward(x)
    assert y.shape == (batch_size, d_model), f"Shape mismatch: {y.shape}"

    # Training forward
    y_train = model.forward(x, training=True)
    assert y_train.shape == (batch_size, d_model)

    # Parameter count
    params = model.parameter_count()
    print(f"  Parameters: {params:,}")

    # Compare to equivalent FFN
    ffn_params = d_model * 4 * d_model + 4 * d_model + 4 * d_model * d_model + d_model
    print(f"  Equivalent FFN (4x hidden): {ffn_params:,}")
    print(f"  Ratio: {params / ffn_params:.2f}x")

    print("  SparseOctaveLookupFFN PASS")
    return True


def test_transformer_ffn():
    """Test drop-in transformer FFN replacement."""
    print("Testing SparseOctaveTransformerFFN...")

    d_model = 768  # BERT-base hidden size
    batch_size = 4

    model = SparseOctaveTransformerFFN(
        d_model=d_model,
        n_octaves=3,
        memory_size=512,
        top_k=16
    )

    xp = cp if HAS_CUDA else np
    x = xp.random.randn(batch_size, d_model).astype(np.float32)

    y = model(x)
    assert y.shape == (batch_size, d_model)

    params = model.parameter_count()
    ffn_params = d_model * 4 * d_model * 2  # Standard FFN with 4x expansion

    print(f"  SparseOctave params: {params:,}")
    print(f"  Standard FFN params: {ffn_params:,}")
    print(f"  Compression: {ffn_params / params:.1f}x")

    print("  SparseOctaveTransformerFFN PASS")
    return True


def test_training():
    """Test training loop."""
    print("Testing training...")

    d_model = 32
    batch_size = 4

    model = SparseOctaveLookupFFN(
        d_model=d_model,
        n_octaves=3,
        memory_size=128,
        top_k=8
    )

    trainer = SparseOctaveTrainer(model, lr=0.01)

    xp = cp if HAS_CUDA else np

    # Simple regression task
    x = xp.random.randn(batch_size, d_model).astype(np.float32)
    target = xp.random.randn(batch_size, d_model).astype(np.float32)

    # Training steps
    losses = []
    for step in range(5):
        loss, metrics = trainer.train_step(x, target)
        losses.append(loss)

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print("  Training PASS")
    return True


def test_octave_contributions():
    """Test octave contribution analysis."""
    print("Testing octave contributions...")

    d_model = 64
    batch_size = 4

    model = SparseOctaveLookupFFN(
        d_model=d_model,
        n_octaves=3,
        memory_size=128,
        top_k=8
    )

    xp = cp if HAS_CUDA else np
    x = xp.random.randn(batch_size, d_model).astype(np.float32)

    contributions = model.get_octave_contributions(x)

    assert len(contributions) == 3
    for i, contrib in enumerate(contributions):
        assert contrib.shape == (batch_size, d_model), f"Octave {i} shape mismatch"

    print(f"  Got {len(contributions)} octave contributions")
    print("  Octave contributions PASS")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SparseOctaveLookupFFN Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_providence,
        test_sparse_octave,
        test_transformer_ffn,
        test_training,
        test_octave_contributions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    run_all_tests()
