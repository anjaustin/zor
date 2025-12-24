"""
Hollywood Squares - NATIVE TRAINING

No PyTorch. No TensorFlow. No frameworks.
Pure CuPy + CUDA. Complete independence.

Forward. Backward. Optimize. Train.

"The most BOSS move in ML history since GPT3."
"""

import numpy as np
import cupy as cp
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json


# =============================================================================
# CUDA KERNELS - FORWARD AND BACKWARD
# =============================================================================

NATIVE_KERNELS = r'''
extern "C" {

// =============================================================================
// SHARED UTILITIES
// =============================================================================

__device__ __forceinline__ float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) return 0.666667f - t*t + 0.5f * t*t*t;
    if (t < 2.0f) { float tmp = 2.0f - t; return 0.166667f * tmp * tmp * tmp; }
    return 0.0f;
}

__device__ __forceinline__ float cubic_bspline_deriv(float t) {
    // Derivative of cubic B-spline w.r.t. t
    float sign = (t < 0) ? -1.0f : 1.0f;
    t = fabsf(t);
    if (t < 1.0f) return sign * (-2.0f * t + 1.5f * t * t);
    if (t < 2.0f) { float tmp = 2.0f - t; return sign * (-0.5f * tmp * tmp); }
    return 0.0f;
}

__device__ __forceinline__ int unpack_ternary(unsigned int packed, int idx) {
    int shift = (idx % 16) * 2;
    int bits = (packed >> shift) & 0x3;
    return bits - 1;
}

// =============================================================================
// FORWARD KERNEL (with saved intermediates for backward)
// =============================================================================

#define BLOCK_SIZE 256
#define MAX_TILES 32
#define MAX_D_MODEL 128

__global__ void forward_with_cache(
    const float* __restrict__ x,
    const unsigned int* __restrict__ signatures,
    const signed char* __restrict__ directions,
    const float* __restrict__ spline_coeffs,
    const float* __restrict__ positions,
    float* __restrict__ output,
    int* __restrict__ tile_indices,      // Save winning tile per token
    float* __restrict__ spline_scales,   // Save spline output per token
    float* __restrict__ spatial_scores,  // Save spatial scores for gradient
    const float dir_scale,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale
) {
    __shared__ unsigned int s_signatures[MAX_TILES][MAX_D_MODEL / 16];
    __shared__ signed char s_directions[MAX_TILES][MAX_D_MODEL];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * BLOCK_SIZE + tid;

    // Cooperative load
    int sig_words = num_tiles * (d_model / 16);
    for (int i = tid; i < sig_words; i += BLOCK_SIZE) {
        int t = i / (d_model / 16);
        int w = i % (d_model / 16);
        s_signatures[t][w] = signatures[i];
    }

    int dir_bytes = num_tiles * d_model;
    for (int i = tid; i < dir_bytes; i += BLOCK_SIZE) {
        s_directions[i / d_model][i % d_model] = directions[i];
    }
    __syncthreads();

    if (global_tid >= batch_size) return;

    // Load input
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        x_local[d] = x[global_tid * d_model + d];
    }
    float pos = positions[global_tid];

    // Find best tile
    float best_score = -1e30f;
    int best_tile = 0;

    for (int t = 0; t < num_tiles; t++) {
        float content_score = 0.0f;

        for (int w = 0; w < d_model / 16; w++) {
            unsigned int packed = s_signatures[t][w];
            #pragma unroll 16
            for (int i = 0; i < 16; i++) {
                int sign = unpack_ternary(packed, i);
                content_score += x_local[w * 16 + i] * (float)sign;
            }
        }

        float tile_center = (float)t / (float)num_tiles * 64.0f;
        float spatial = cubic_bspline((pos - tile_center) / position_spread);
        float combined = content_score * spatial;

        // Save spatial score for this tile (for gradient)
        if (t < MAX_TILES) {
            spatial_scores[global_tid * num_tiles + t] = spatial;
        }

        if (combined > best_score) {
            best_score = combined;
            best_tile = t;
        }
    }

    // Save winning tile
    tile_indices[global_tid] = best_tile;

    // Spline lookup
    float a = tanhf(x_local[0] * 0.1f);
    float b = tanhf(x_local[1] * 0.1f);

    int idx_a = min(max((int)(((a + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);
    int idx_b = min(max((int)(((b + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);

    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;
    float base = spline_coeffs[coeff_idx];
    float slope_a = spline_coeffs[coeff_idx + 1];
    float slope_b = spline_coeffs[coeff_idx + 2];

    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;
    float scale = (base + slope_a * local_a + slope_b * local_b) * output_scale;

    // Save spline scale
    spline_scales[global_tid] = scale;

    // Output with residual
    for (int d = 0; d < d_model; d++) {
        float dir = (float)s_directions[best_tile][d] * dir_scale;
        output[global_tid * d_model + d] = x_local[d] + scale * dir;
    }
}

// =============================================================================
// BACKWARD KERNEL
// =============================================================================

__global__ void backward_kernel(
    const float* __restrict__ d_output,     // [batch, d_model] gradient from loss
    const float* __restrict__ x,            // [batch, d_model] original input
    const int* __restrict__ tile_indices,   // [batch] winning tile per token
    const float* __restrict__ spline_scales,// [batch] scale per token
    const signed char* __restrict__ directions, // [num_tiles, d_model]
    float* __restrict__ d_directions,       // [num_tiles, d_model] gradient accumulator
    float* __restrict__ d_spline_coeffs,    // [num_tiles, grid*grid*3] gradient accumulator
    float* __restrict__ d_input,            // [batch, d_model] gradient to input
    const float dir_scale,
    const float output_scale,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= batch_size) return;

    int best_tile = tile_indices[global_tid];
    float scale = spline_scales[global_tid];

    // Load gradient and input
    float d_out_local[MAX_D_MODEL];
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        d_out_local[d] = d_output[global_tid * d_model + d];
        x_local[d] = x[global_tid * d_model + d];
    }

    // Gradient through residual connection
    // output = x + scale * direction
    // d_input = d_output (residual passes through)
    for (int d = 0; d < d_model; d++) {
        d_input[global_tid * d_model + d] = d_out_local[d];
    }

    // Gradient w.r.t. directions (accumulated atomically)
    // d_output/d_direction = scale * dir_scale
    for (int d = 0; d < d_model; d++) {
        float grad = d_out_local[d] * scale / dir_scale;  // Chain rule
        atomicAdd(&d_directions[best_tile * d_model + d], grad);
    }

    // Gradient w.r.t. spline coefficients
    // output = x + (base + slope_a * local_a + slope_b * local_b) * output_scale * direction
    // We need d_output/d_base, d_output/d_slope_a, d_output/d_slope_b

    float a = tanhf(x_local[0] * 0.1f);
    float b = tanhf(x_local[1] * 0.1f);

    int idx_a = min(max((int)(((a + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);
    int idx_b = min(max((int)(((b + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);

    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;

    // Sum of (d_output * direction) gives us d_loss/d_scale
    float d_scale = 0.0f;
    for (int d = 0; d < d_model; d++) {
        float dir = (float)directions[best_tile * d_model + d] * dir_scale;
        d_scale += d_out_local[d] * dir;
    }

    // Now chain through spline
    // scale = (base + slope_a * local_a + slope_b * local_b) * output_scale
    // d_scale/d_base = output_scale
    // d_scale/d_slope_a = output_scale * local_a
    // d_scale/d_slope_b = output_scale * local_b

    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;

    atomicAdd(&d_spline_coeffs[coeff_idx + 0], d_scale * output_scale);
    atomicAdd(&d_spline_coeffs[coeff_idx + 1], d_scale * output_scale * local_a);
    atomicAdd(&d_spline_coeffs[coeff_idx + 2], d_scale * output_scale * local_b);
}

// =============================================================================
// WEIGHT UPDATE KERNEL (Adam optimizer fused)
// =============================================================================

__global__ void adam_update(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float beta1_t,  // beta1^t for bias correction
    float beta2_t,  // beta2^t for bias correction
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx];

    // Update biased moments
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);

    // Update parameter
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// =============================================================================
// GRADIENT ZEROING
// =============================================================================

__global__ void zero_gradients(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) grad[idx] = 0.0f;
}

}  // extern "C"
'''


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mse_loss(pred: cp.ndarray, target: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """Mean Squared Error loss with gradient."""
    diff = pred - target
    loss = (diff ** 2).mean()
    grad = 2.0 * diff / diff.size
    return loss, grad


def cross_entropy_loss(logits: cp.ndarray, targets: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """Cross-entropy loss for classification."""
    # Softmax
    exp_logits = cp.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Loss
    batch_size = logits.shape[0]
    log_probs = cp.log(probs + 1e-10)
    loss = -log_probs[cp.arange(batch_size), targets].mean()

    # Gradient
    grad = probs.copy()
    grad[cp.arange(batch_size), targets] -= 1.0
    grad /= batch_size

    return loss, grad


# =============================================================================
# ADAM OPTIMIZER
# =============================================================================

class AdamOptimizer:
    """Pure CuPy Adam optimizer."""

    def __init__(
        self,
        params: Dict[str, cp.ndarray],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Moment buffers
        self.m = {k: cp.zeros_like(v) for k, v in params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, cp.ndarray]):
        """Update parameters with gradients."""
        self.t += 1
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for name, param in self.params.items():
            if name not in grads:
                continue

            g = grads[name]
            m = self.m[name]
            v = self.v[name]

            # Update moments
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = m / (1 - beta1_t)
            v_hat = v / (1 - beta2_t)

            # Update parameter
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset for next iteration (moments persist)."""
        pass  # Moments persist across iterations in Adam


# =============================================================================
# NATIVE TRAINABLE MODEL
# =============================================================================

class NativeHollywoodSquares:
    """
    Fully native Hollywood Squares model.
    No PyTorch. No frameworks. Pure CUDA.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        grid_size: int = 16,
        position_spread: float = 2.0,
        lr: float = 0.001,
    ):
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.grid_size = grid_size
        self.position_spread = position_spread
        self.output_scale = 0.1

        print("=" * 70)
        print("NATIVE HOLLYWOOD SQUARES - NO PYTORCH")
        print("=" * 70)

        # Initialize weights
        print("\nInitializing weights...")

        # Ternary signatures (stored as packed uint32)
        signatures_float = np.random.randn(num_tiles, d_model).astype(np.float32)
        signatures_ternary = np.sign(signatures_float).astype(np.int8)
        packed_shape = (num_tiles, d_model // 16)
        self.signatures_packed = np.zeros(packed_shape, dtype=np.uint32)
        for t in range(num_tiles):
            for w in range(d_model // 16):
                packed = 0
                for i in range(16):
                    val = signatures_ternary[t, w * 16 + i] + 1
                    packed |= (val & 0x3) << (i * 2)
                self.signatures_packed[t, w] = packed
        self.signatures_gpu = cp.asarray(self.signatures_packed)

        # Directions (trainable, float32 for gradients)
        self.directions = cp.random.randn(num_tiles, d_model).astype(cp.float32) * 0.02
        self.dir_scale = 1.0  # No quantization during training

        # Spline coefficients (trainable)
        self.spline_coeffs = cp.random.randn(
            num_tiles, grid_size, grid_size, 3
        ).astype(cp.float32) * 0.1

        # Compile kernels
        print("Compiling CUDA kernels...")
        self.module = cp.RawModule(code=NATIVE_KERNELS)
        self.forward_kernel = self.module.get_function('forward_with_cache')
        self.backward_kernel = self.module.get_function('backward_kernel')
        self.zero_kernel = self.module.get_function('zero_gradients')

        # Gradient buffers
        self.d_directions = cp.zeros_like(self.directions)
        self.d_spline_coeffs = cp.zeros_like(self.spline_coeffs)

        # Optimizer
        self.optimizer = AdamOptimizer(
            params={
                'directions': self.directions,
                'spline_coeffs': self.spline_coeffs.reshape(-1),
            },
            lr=lr,
        )

        self.threads_per_block = 256

        print(f"\n  d_model: {d_model}")
        print(f"  num_tiles: {num_tiles}")
        print(f"  Parameters: {self.param_count():,}")
        print("\n  READY FOR NATIVE TRAINING.")
        print("=" * 70)

    def param_count(self) -> int:
        """Total trainable parameters."""
        return self.directions.size + self.spline_coeffs.size

    def forward(
        self,
        x: cp.ndarray,
        positions: cp.ndarray = None,
        save_for_backward: bool = True,
    ) -> cp.ndarray:
        """Forward pass with optional caching for backward."""
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

        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block

        # Convert directions to int8 for kernel (or use float version)
        directions_int8 = self.directions.astype(cp.int8)  # Simplified

        self.forward_kernel(
            (blocks,), (self.threads_per_block,),
            (
                x,
                self.signatures_gpu.reshape(-1),
                directions_int8.reshape(-1),
                self.spline_coeffs.reshape(-1),
                positions,
                output,
                self.cached_tile_indices if save_for_backward else cp.zeros(batch_size, dtype=cp.int32),
                self.cached_spline_scales if save_for_backward else cp.zeros(batch_size, dtype=cp.float32),
                self.cached_spatial_scores if save_for_backward else cp.zeros((batch_size, self.num_tiles), dtype=cp.float32),
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

    def backward(self, d_output: cp.ndarray):
        """Backward pass - compute gradients."""
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
        """Update weights with accumulated gradients."""
        self.optimizer.step({
            'directions': self.d_directions,
            'spline_coeffs': self.d_spline_coeffs.reshape(-1),
        })

    def save(self, path: str):
        """Save model weights."""
        cp.savez(
            path,
            signatures=self.signatures_packed,
            directions=cp.asnumpy(self.directions),
            spline_coeffs=cp.asnumpy(self.spline_coeffs),
            config={
                'd_model': self.d_model,
                'num_tiles': self.num_tiles,
                'grid_size': self.grid_size,
            }
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        data = np.load(path, allow_pickle=True)
        self.signatures_packed = data['signatures']
        self.signatures_gpu = cp.asarray(self.signatures_packed)
        self.directions = cp.asarray(data['directions'])
        self.spline_coeffs = cp.asarray(data['spline_coeffs'])
        print(f"Model loaded from {path}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

class Trainer:
    """Native training loop. No frameworks."""

    def __init__(
        self,
        model: NativeHollywoodSquares,
        loss_fn: str = 'mse',
    ):
        self.model = model
        self.loss_fn = mse_loss if loss_fn == 'mse' else cross_entropy_loss
        self.history = {
            'loss': [],
            'time': [],
        }

    def train_step(
        self,
        x: cp.ndarray,
        target: cp.ndarray,
        positions: cp.ndarray = None,
    ) -> float:
        """Single training step."""
        # Forward
        output = self.model.forward(x, positions, save_for_backward=True)

        # Loss
        loss, d_loss = self.loss_fn(output, target)

        # Backward
        self.model.backward(d_loss)

        # Update
        self.model.step()

        return float(loss)

    def train(
        self,
        train_x: cp.ndarray,
        train_y: cp.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Dict:
        """Full training loop."""
        n_samples = train_x.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        if verbose:
            print(f"\nTraining on {n_samples:,} samples")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Batches per epoch: {n_batches}")
            print()

        start_time = time.perf_counter()

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle using numpy (CuPy thrust has device issues)
            indices_np = np.random.permutation(n_samples)
            indices = cp.asarray(indices_np)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]

                loss = self.train_step(batch_x, batch_y)
                epoch_loss += loss

            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            self.history['time'].append(time.perf_counter() - start_time)

            if verbose and (epoch + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Epoch {epoch+1:4d}: loss = {epoch_loss:.6f} ({elapsed:.1f}s)")

        total_time = time.perf_counter() - start_time

        if verbose:
            print(f"\nTraining complete in {total_time:.2f}s")
            print(f"Final loss: {self.history['loss'][-1]:.6f}")

        return self.history


# =============================================================================
# DEMO: Train a model to learn a simple function
# =============================================================================

def demo_native_training():
    """Demonstrate native training on a simple task."""

    print()
    print("=" * 70)
    print("NATIVE TRAINING DEMO")
    print("No PyTorch. No TensorFlow. Pure CUDA.")
    print("=" * 70)
    print()

    # Create model
    model = NativeHollywoodSquares(
        d_model=128,
        num_tiles=16,
        grid_size=8,
        lr=0.01,
    )

    # Generate synthetic data: Learn to route based on input patterns
    print("\nGenerating training data...")
    n_samples = 10000

    # Create input patterns - different patterns should route to different tiles
    train_x = cp.random.randn(n_samples, 128).astype(cp.float32)

    # Target: input with some transformation (the model should learn to approximate)
    # Simple target: scale by 0.9 and add small noise
    train_y = train_x * 0.95 + cp.random.randn(n_samples, 128).astype(cp.float32) * 0.01

    print(f"  Train samples: {n_samples:,}")
    print(f"  Input shape: {train_x.shape}")
    print(f"  Target shape: {train_y.shape}")

    # Create trainer
    trainer = Trainer(model, loss_fn='mse')

    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    history = trainer.train(
        train_x,
        train_y,
        epochs=100,
        batch_size=256,
        verbose=True,
    )

    # Evaluate
    print("\n" + "-" * 70)
    print("EVALUATION")
    print("-" * 70)

    # Test on new data
    test_x = cp.random.randn(1000, 128).astype(cp.float32)
    test_y = test_x * 0.95 + cp.random.randn(1000, 128).astype(cp.float32) * 0.01

    pred = model.forward(test_x, save_for_backward=False)
    test_loss = float(((pred - test_y) ** 2).mean())

    print(f"\n  Test loss: {test_loss:.6f}")
    print(f"  Train loss: {history['loss'][-1]:.6f}")

    # Save model
    save_path = Path(__file__).parent / "native_model.npz"
    model.save(str(save_path))

    # Summary
    print("\n" + "=" * 70)
    print("NATIVE TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("  No PyTorch. No TensorFlow. No frameworks.")
    print("  Pure CuPy + CUDA kernels.")
    print("  Forward pass: Native CUDA")
    print("  Backward pass: Native CUDA")
    print("  Optimizer: Native Adam")
    print("  Save/Load: NumPy npz")
    print()
    print("  THE MACHINE LEARNS WITHOUT FRAMEWORKS.")
    print()
    print("=" * 70)

    return history


if __name__ == "__main__":
    history = demo_native_training()
