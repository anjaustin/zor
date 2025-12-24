"""
TriX Native Model - Hollywood Squares

No PyTorch. No TensorFlow. Pure CuPy + CUDA.

The NativeHollywoodSquares model implements:
- Ternary signatures for content-based routing
- Spline-based magnitude modulation
- Full forward/backward in custom CUDA kernels
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from .kernels import NATIVE_KERNELS


class NativeHollywoodSquares:
    """
    Fully native Hollywood Squares model.
    No PyTorch. No frameworks. Pure CUDA.

    Args:
        d_model: Model dimension (must be multiple of 16)
        num_tiles: Number of specialist tiles
        grid_size: Spline grid resolution
        position_spread: Spatial routing spread
        lr: Learning rate (for built-in optimizer)

    Example:
        >>> model = NativeHollywoodSquares(d_model=128, num_tiles=16)
        >>> output = model.forward(x)
        >>> model.backward(grad)
        >>> model.step()
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        grid_size: int = 16,
        position_spread: float = 2.0,
        lr: float = 0.001,
    ):
        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy is required for NativeHollywoodSquares. "
                "Install with: pip install cupy-cuda12x"
            )

        if d_model % 16 != 0:
            raise ValueError(f"d_model must be multiple of 16, got {d_model}")

        self.d_model = d_model
        self.num_tiles = num_tiles
        self.grid_size = grid_size
        self.position_spread = position_spread
        self.output_scale = 0.1
        self.lr = lr

        # Initialize weights
        self._init_weights()

        # Compile kernels
        self._compile_kernels()

        # Gradient buffers
        self.d_directions = cp.zeros_like(self.directions)
        self.d_spline_coeffs = cp.zeros_like(self.spline_coeffs)

        # Adam optimizer state
        self._init_optimizer()

        self.threads_per_block = 256

    def _init_weights(self):
        """Initialize model weights."""
        # Ternary signatures (stored as packed uint32)
        signatures_float = np.random.randn(self.num_tiles, self.d_model).astype(np.float32)
        signatures_ternary = np.sign(signatures_float).astype(np.int8)
        packed_shape = (self.num_tiles, self.d_model // 16)
        self.signatures_packed = np.zeros(packed_shape, dtype=np.uint32)

        for t in range(self.num_tiles):
            for w in range(self.d_model // 16):
                packed = 0
                for i in range(16):
                    val = signatures_ternary[t, w * 16 + i] + 1
                    packed |= (val & 0x3) << (i * 2)
                self.signatures_packed[t, w] = packed

        self.signatures_gpu = cp.asarray(self.signatures_packed)

        # Directions (trainable, float32 for gradients)
        self.directions = cp.random.randn(
            self.num_tiles, self.d_model
        ).astype(cp.float32) * 0.02
        self.dir_scale = 1.0

        # Spline coefficients (trainable)
        self.spline_coeffs = cp.random.randn(
            self.num_tiles, self.grid_size, self.grid_size, 3
        ).astype(cp.float32) * 0.1

    def _compile_kernels(self):
        """Compile CUDA kernels."""
        self.module = cp.RawModule(code=NATIVE_KERNELS)
        self.forward_kernel = self.module.get_function('forward_with_cache')
        self.backward_kernel = self.module.get_function('backward_kernel')
        self.zero_kernel = self.module.get_function('zero_gradients')

    def _init_optimizer(self):
        """Initialize Adam optimizer state."""
        self.optimizer_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # Moment buffers for directions
        self.m_directions = cp.zeros_like(self.directions)
        self.v_directions = cp.zeros_like(self.directions)

        # Moment buffers for spline coefficients
        self.m_spline = cp.zeros_like(self.spline_coeffs)
        self.v_spline = cp.zeros_like(self.spline_coeffs)

    def param_count(self) -> int:
        """Total trainable parameters."""
        return int(self.directions.size + self.spline_coeffs.size)

    def forward(
        self,
        x: "cp.ndarray",
        positions: Optional["cp.ndarray"] = None,
        save_for_backward: bool = True,
    ) -> "cp.ndarray":
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, d_model]
            positions: Position indices [batch_size] (optional)
            save_for_backward: Whether to cache for backward pass

        Returns:
            Output tensor [batch_size, d_model]
        """
        batch_size = x.shape[0]

        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32) % 64

        output = cp.zeros_like(x)

        # Allocate cache if training
        if save_for_backward:
            self.cached_x = x
            self.cached_positions = positions
            self.cached_tile_indices = cp.zeros(batch_size, dtype=cp.int32)
            self.cached_spline_scales = cp.zeros(batch_size, dtype=cp.float32)
            self.cached_spatial_scores = cp.zeros(
                (batch_size, self.num_tiles), dtype=cp.float32
            )
        else:
            tile_indices = cp.zeros(batch_size, dtype=cp.int32)
            spline_scales = cp.zeros(batch_size, dtype=cp.float32)
            spatial_scores = cp.zeros((batch_size, self.num_tiles), dtype=cp.float32)

        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block
        directions_int8 = self.directions.astype(cp.int8)

        self.forward_kernel(
            (blocks,), (self.threads_per_block,),
            (
                x,
                self.signatures_gpu.reshape(-1),
                directions_int8.reshape(-1),
                self.spline_coeffs.reshape(-1),
                positions,
                output,
                self.cached_tile_indices if save_for_backward else tile_indices,
                self.cached_spline_scales if save_for_backward else spline_scales,
                self.cached_spatial_scores if save_for_backward else spatial_scores,
                self.dir_scale,
                batch_size,
                self.d_model,
                self.num_tiles,
                self.grid_size,
                self.position_spread,
                self.output_scale,
            )
        )

        return output

    def backward(self, d_output: "cp.ndarray") -> "cp.ndarray":
        """
        Backward pass - compute gradients.

        Args:
            d_output: Gradient of loss w.r.t. output [batch_size, d_model]

        Returns:
            Gradient of loss w.r.t. input [batch_size, d_model]
        """
        batch_size = d_output.shape[0]
        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block

        # Zero gradients
        self.d_directions.fill(0)
        self.d_spline_coeffs.fill(0)

        d_input = cp.zeros_like(self.cached_x)
        directions_int8 = self.directions.astype(cp.int8)

        self.backward_kernel(
            (blocks,), (self.threads_per_block,),
            (
                d_output,
                self.cached_x,
                self.cached_tile_indices,
                self.cached_spline_scales,
                directions_int8.reshape(-1),
                self.d_directions.reshape(-1),
                self.d_spline_coeffs.reshape(-1),
                d_input,
                self.dir_scale,
                self.output_scale,
                batch_size,
                self.d_model,
                self.num_tiles,
                self.grid_size,
            )
        )

        return d_input

    def step(self):
        """Update weights with Adam optimizer."""
        self.optimizer_t += 1
        beta1_t = self.beta1 ** self.optimizer_t
        beta2_t = self.beta2 ** self.optimizer_t

        # Update directions
        self._adam_update(
            self.directions, self.d_directions,
            self.m_directions, self.v_directions,
            beta1_t, beta2_t
        )

        # Update spline coefficients
        self._adam_update(
            self.spline_coeffs, self.d_spline_coeffs,
            self.m_spline, self.v_spline,
            beta1_t, beta2_t
        )

    def _adam_update(self, param, grad, m, v, beta1_t, beta2_t):
        """Apply Adam update to a parameter."""
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)

        param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero all gradients."""
        self.d_directions.fill(0)
        self.d_spline_coeffs.fill(0)

    def save(self, path: str):
        """Save model weights to .npz file."""
        np.savez(
            path,
            signatures=self.signatures_packed,
            directions=cp.asnumpy(self.directions),
            spline_coeffs=cp.asnumpy(self.spline_coeffs),
            config=np.array([
                self.d_model,
                self.num_tiles,
                self.grid_size,
            ]),
        )

    def load(self, path: str):
        """Load model weights from .npz file."""
        data = np.load(path, allow_pickle=True)
        self.signatures_packed = data['signatures']
        self.signatures_gpu = cp.asarray(self.signatures_packed)
        self.directions = cp.asarray(data['directions'])
        self.spline_coeffs = cp.asarray(data['spline_coeffs'])

        # Reinitialize gradient buffers
        self.d_directions = cp.zeros_like(self.directions)
        self.d_spline_coeffs = cp.zeros_like(self.spline_coeffs)
        self._init_optimizer()

    def __call__(self, x: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Callable interface for forward pass."""
        return self.forward(x, **kwargs)
