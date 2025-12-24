"""
Qupid 6502 Direct: Native GPU Training with Direct Transformation

Unlike standard Hollywood Squares (residual: output = input + delta),
this uses direct transformation (output = tile_output).

For 6502, we need to transform [opcode, operands] → [result, flags]
which requires a direct mapping, not refinement.

"Position IS Computation. But sometimes you need a clean slate."
"""

import numpy as np
import cupy as cp
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')


# =============================================================================
# CUDA KERNEL - DIRECT TRANSFORMATION (NO RESIDUAL)
# =============================================================================

DIRECT_KERNELS = r'''
extern "C" {

#define BLOCK_SIZE 256
#define MAX_TILES 32
#define MAX_D_MODEL 128

__device__ __forceinline__ float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) return 0.666667f - t*t + 0.5f * t*t*t;
    if (t < 2.0f) { float tmp = 2.0f - t; return 0.166667f * tmp * tmp * tmp; }
    return 0.0f;
}

__device__ __forceinline__ int unpack_ternary(unsigned int packed, int idx) {
    int shift = (idx % 16) * 2;
    int bits = (packed >> shift) & 0x3;
    return bits - 1;
}

// =============================================================================
// FORWARD KERNEL - DIRECT OUTPUT (NO RESIDUAL)
// =============================================================================

__global__ void forward_direct(
    const float* __restrict__ x,
    const unsigned int* __restrict__ signatures,
    const float* __restrict__ tile_outputs,     // [num_tiles, d_model] - learned output per tile
    const float* __restrict__ positions,
    float* __restrict__ output,
    int* __restrict__ tile_indices,
    float* __restrict__ tile_weights,           // soft weights for blending
    int batch_size,
    int d_model,
    int num_tiles,
    float position_spread,
    float temperature
) {
    __shared__ unsigned int s_signatures[MAX_TILES][MAX_D_MODEL / 16];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * BLOCK_SIZE + tid;

    // Cooperative load signatures
    int sig_words = num_tiles * (d_model / 16);
    for (int i = tid; i < sig_words; i += BLOCK_SIZE) {
        int t = i / (d_model / 16);
        int w = i % (d_model / 16);
        s_signatures[t][w] = signatures[i];
    }
    __syncthreads();

    if (global_tid >= batch_size) return;

    // Load input
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        x_local[d] = x[global_tid * d_model + d];
    }
    float pos = positions[global_tid];

    // Compute scores for all tiles
    float scores[MAX_TILES];
    float max_score = -1e30f;

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
        scores[t] = content_score * spatial / temperature;

        if (scores[t] > max_score) max_score = scores[t];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int t = 0; t < num_tiles; t++) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }

    int best_tile = 0;
    float best_weight = 0.0f;
    for (int t = 0; t < num_tiles; t++) {
        scores[t] /= sum_exp;
        tile_weights[global_tid * num_tiles + t] = scores[t];
        if (scores[t] > best_weight) {
            best_weight = scores[t];
            best_tile = t;
        }
    }
    tile_indices[global_tid] = best_tile;

    // Blend tile outputs (soft routing during training)
    for (int d = 0; d < d_model; d++) {
        float out = 0.0f;
        for (int t = 0; t < num_tiles; t++) {
            out += scores[t] * tile_outputs[t * d_model + d];
        }
        output[global_tid * d_model + d] = out;
    }
}

// =============================================================================
// BACKWARD KERNEL - GRADIENT FOR TILE OUTPUTS
// =============================================================================

__global__ void backward_direct(
    const float* __restrict__ d_output,
    const float* __restrict__ tile_weights,
    float* __restrict__ d_tile_outputs,
    int batch_size,
    int d_model,
    int num_tiles
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= batch_size) return;

    // Each sample contributes to tile gradients based on its weights
    for (int t = 0; t < num_tiles; t++) {
        float weight = tile_weights[global_tid * num_tiles + t];
        for (int d = 0; d < d_model; d++) {
            float grad = d_output[global_tid * d_model + d] * weight;
            atomicAdd(&d_tile_outputs[t * d_model + d], grad);
        }
    }
}

}
'''


class DirectHollywoodSquares:
    """
    Hollywood Squares with direct transformation (no residual).

    Each tile has a learned output vector. The model soft-routes to tiles
    and blends their outputs.

    For 6502: tiles specialize to different operations.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        position_spread: float = 2.0,
        temperature: float = 0.5,
        lr: float = 0.01,
    ):
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.position_spread = position_spread
        self.temperature = temperature
        self.lr = lr

        # Compile kernels
        self.module = cp.RawModule(code=DIRECT_KERNELS)
        self.forward_kernel = self.module.get_function('forward_direct')
        self.backward_kernel = self.module.get_function('backward_direct')

        # Initialize weights
        self._init_weights()

        # Gradient accumulators
        self.d_tile_outputs = None

        # Cache for backward
        self.cache = {}

        # Adam state
        self.m_tile_outputs = cp.zeros_like(self.tile_outputs)
        self.v_tile_outputs = cp.zeros_like(self.tile_outputs)
        self.m_signatures = cp.zeros_like(self.signatures_float)
        self.v_signatures = cp.zeros_like(self.signatures_float)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _init_weights(self):
        """Initialize weights."""
        # Ternary signatures (packed)
        self.signatures_float = cp.random.randn(
            self.num_tiles, self.d_model
        ).astype(cp.float32)
        self._update_packed_signatures()

        # Tile outputs - what each tile produces
        self.tile_outputs = cp.random.randn(
            self.num_tiles, self.d_model
        ).astype(cp.float32) * 0.1

    def _update_packed_signatures(self):
        """Pack float signatures to ternary."""
        signs = cp.sign(self.signatures_float).astype(cp.int32)  # {-1, 0, 1}
        encoded = (signs + 1).astype(cp.uint32)  # {0, 1, 2}

        packed = cp.zeros(
            (self.num_tiles, self.d_model // 16), dtype=cp.uint32
        )
        for i in range(16):
            packed |= encoded[:, i::16] << (i * 2)

        self.packed_signatures = packed.flatten()

    def forward(
        self,
        x: cp.ndarray,
        positions: cp.ndarray = None,
        save_for_backward: bool = True,
    ) -> cp.ndarray:
        """Forward pass with soft routing."""
        batch_size = x.shape[0]

        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32)

        # Allocate outputs
        output = cp.zeros((batch_size, self.d_model), dtype=cp.float32)
        tile_indices = cp.zeros(batch_size, dtype=cp.int32)
        tile_weights = cp.zeros((batch_size, self.num_tiles), dtype=cp.float32)

        # Launch kernel
        block_size = 256
        num_blocks = (batch_size + block_size - 1) // block_size

        self.forward_kernel(
            (num_blocks,), (block_size,),
            (
                x,
                self.packed_signatures,
                self.tile_outputs.flatten(),
                positions,
                output,
                tile_indices,
                tile_weights,
                cp.int32(batch_size),
                cp.int32(self.d_model),
                cp.int32(self.num_tiles),
                cp.float32(self.position_spread),
                cp.float32(self.temperature),
            )
        )

        if save_for_backward:
            self.cache = {
                'x': x,
                'positions': positions,
                'tile_weights': tile_weights,
                'tile_indices': tile_indices,
            }

        return output

    def backward(self, d_output: cp.ndarray) -> cp.ndarray:
        """Backward pass."""
        batch_size = d_output.shape[0]
        tile_weights = self.cache['tile_weights']

        # Reset gradient accumulators
        self.d_tile_outputs = cp.zeros_like(self.tile_outputs)

        # Launch backward kernel
        block_size = 256
        num_blocks = (batch_size + block_size - 1) // block_size

        self.backward_kernel(
            (num_blocks,), (block_size,),
            (
                d_output,
                tile_weights,
                self.d_tile_outputs,
                cp.int32(batch_size),
                cp.int32(self.d_model),
                cp.int32(self.num_tiles),
            )
        )

        # No input gradient for this architecture (routing is non-differentiable)
        return cp.zeros_like(self.cache['x'])

    def step(self):
        """Adam optimizer step."""
        self.t += 1

        # Update tile outputs
        grad = self.d_tile_outputs / max(self.cache['x'].shape[0], 1)
        self.m_tile_outputs = self.beta1 * self.m_tile_outputs + (1 - self.beta1) * grad
        self.v_tile_outputs = self.beta2 * self.v_tile_outputs + (1 - self.beta2) * grad**2
        m_hat = self.m_tile_outputs / (1 - self.beta1**self.t)
        v_hat = self.v_tile_outputs / (1 - self.beta2**self.t)
        self.tile_outputs -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

        # Also update signatures based on how well tiles match their assigned samples
        # This is a simplified version - we nudge signatures toward the inputs they should match
        x = self.cache['x']
        tile_indices = self.cache['tile_indices']

        # For each tile, average the inputs that were routed to it
        for t in range(self.num_tiles):
            mask = tile_indices == t
            if mask.sum() > 0:
                avg_input = x[mask].mean(axis=0)
                # Nudge signature toward average input
                grad_sig = (cp.sign(avg_input) - cp.sign(self.signatures_float[t]))
                self.m_signatures[t] = self.beta1 * self.m_signatures[t] + (1 - self.beta1) * grad_sig
                self.v_signatures[t] = self.beta2 * self.v_signatures[t] + (1 - self.beta2) * grad_sig**2
                m_hat = self.m_signatures[t] / (1 - self.beta1**self.t)
                v_hat = self.v_signatures[t] / (1 - self.beta2**self.t)
                self.signatures_float[t] -= self.lr * 0.1 * m_hat / (cp.sqrt(v_hat) + self.eps)

        # Update packed signatures
        self._update_packed_signatures()

    def param_count(self) -> int:
        """Total trainable parameters."""
        return self.tile_outputs.size + self.signatures_float.size

    def save(self, path: str):
        """Save model."""
        np.savez(
            path,
            tile_outputs=cp.asnumpy(self.tile_outputs),
            signatures_float=cp.asnumpy(self.signatures_float),
        )

    def load(self, path: str):
        """Load model."""
        data = np.load(path)
        self.tile_outputs = cp.asarray(data['tile_outputs'])
        self.signatures_float = cp.asarray(data['signatures_float'])
        self._update_packed_signatures()


# =============================================================================
# 6502 GROUND TRUTH
# =============================================================================

def adc_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ADC: A + B + C -> (result, carry_out)"""
    a_int = a.astype(np.int32)
    b_int = b.astype(np.int32)
    c_int = c.astype(np.int32)
    result = a_int + b_int + c_int
    carry_out = (result > 255).astype(np.float32)
    result = (result & 0xFF).astype(np.float32)
    return result, carry_out


def sbc_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SBC: A - B - (1-C) = A + NOT(B) + C"""
    b_inv = 255 - b.astype(np.int32)
    return adc_truth(a, b_inv.astype(np.float32), c)


def and_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """AND: A & B"""
    result = (a.astype(np.int32) & b.astype(np.int32)).astype(np.float32)
    carry_out = np.zeros_like(c)
    return result, carry_out


def ora_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ORA: A | B"""
    result = (a.astype(np.int32) | b.astype(np.int32)).astype(np.float32)
    carry_out = np.zeros_like(c)
    return result, carry_out


def eor_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EOR: A ^ B"""
    result = (a.astype(np.int32) ^ b.astype(np.int32)).astype(np.float32)
    carry_out = np.zeros_like(c)
    return result, carry_out


def asl_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ASL: Shift A left, carry gets bit 7"""
    a_int = a.astype(np.int32)
    carry_out = ((a_int >> 7) & 1).astype(np.float32)
    result = ((a_int << 1) & 0xFF).astype(np.float32)
    return result, carry_out


def lsr_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LSR: Shift A right, carry gets bit 0"""
    a_int = a.astype(np.int32)
    carry_out = (a_int & 1).astype(np.float32)
    result = (a_int >> 1).astype(np.float32)
    return result, carry_out


def rol_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ROL: Rotate left through carry"""
    a_int = a.astype(np.int32)
    c_int = c.astype(np.int32)
    carry_out = ((a_int >> 7) & 1).astype(np.float32)
    result = (((a_int << 1) | c_int) & 0xFF).astype(np.float32)
    return result, carry_out


def ror_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ROR: Rotate right through carry"""
    a_int = a.astype(np.int32)
    c_int = c.astype(np.int32)
    carry_out = (a_int & 1).astype(np.float32)
    result = ((a_int >> 1) | (c_int << 7)).astype(np.float32)
    return result, carry_out


def inc_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """INC: A + 1"""
    a_int = a.astype(np.int32)
    result = ((a_int + 1) & 0xFF).astype(np.float32)
    carry_out = np.zeros_like(c)
    return result, carry_out


def dec_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """DEC: A - 1"""
    a_int = a.astype(np.int32)
    result = ((a_int - 1) & 0xFF).astype(np.float32)
    carry_out = np.zeros_like(c)
    return result, carry_out


OPERATIONS = {
    0: ('ADC', adc_truth),
    1: ('SBC', sbc_truth),
    2: ('AND', and_truth),
    3: ('ORA', ora_truth),
    4: ('EOR', eor_truth),
    5: ('ASL', asl_truth),
    6: ('LSR', lsr_truth),
    7: ('ROL', rol_truth),
    8: ('ROR', ror_truth),
    9: ('INC', inc_truth),
    10: ('DEC', dec_truth),
}

NUM_OPS = len(OPERATIONS)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_6502_dataset(
    n_samples_per_op: int = 5000,
    seed: int = 42,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Generate training data for all 6502 operations."""
    np.random.seed(seed)

    all_x = []
    all_y = []

    for op_id, (op_name, truth_fn) in OPERATIONS.items():
        a = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        b = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        c = np.random.randint(0, 2, n_samples_per_op).astype(np.float32)

        result, carry_out = truth_fn(a, b, c)

        # Encode inputs
        x = np.zeros((n_samples_per_op, 128), dtype=np.float32)
        for i in range(8):
            x[:, i] = ((a.astype(np.int32) >> i) & 1).astype(np.float32)
        for i in range(8):
            x[:, 8 + i] = ((b.astype(np.int32) >> i) & 1).astype(np.float32)
        x[:, 16] = c
        x[:, 17 + op_id] = 1.0

        # Encode targets
        y = np.zeros((n_samples_per_op, 128), dtype=np.float32)
        for i in range(8):
            y[:, i] = ((result.astype(np.int32) >> i) & 1).astype(np.float32)
        y[:, 8] = carry_out

        all_x.append(x)
        all_y.append(y)

    train_x = np.concatenate(all_x, axis=0)
    train_y = np.concatenate(all_y, axis=0)

    perm = np.random.permutation(len(train_x))
    train_x = train_x[perm]
    train_y = train_y[perm]

    return cp.asarray(train_x), cp.asarray(train_y)


def evaluate_accuracy(
    model: DirectHollywoodSquares,
    test_x: cp.ndarray,
    test_y: cp.ndarray,
) -> Dict:
    """Evaluate per-operation accuracy."""
    pred = model.forward(test_x, save_for_backward=False)

    pred_np = cp.asnumpy(pred)
    test_y_np = cp.asnumpy(test_y)
    test_x_np = cp.asnumpy(test_x)

    pred_bits = (pred_np > 0.5).astype(np.float32)
    target_bits = test_y_np

    correct = np.all(pred_bits[:, :9] == target_bits[:, :9], axis=1)
    overall_acc = correct.mean() * 100

    per_op = {}
    for op_id, (op_name, _) in OPERATIONS.items():
        mask = test_x_np[:, 17 + op_id] > 0.5
        if mask.sum() > 0:
            per_op[op_name] = correct[mask].mean() * 100

    return {'overall': overall_acc, 'per_op': per_op}


# =============================================================================
# TRAINING
# =============================================================================

def mse_loss(pred: cp.ndarray, target: cp.ndarray) -> Tuple[float, cp.ndarray]:
    """MSE loss and gradient."""
    diff = pred - target
    loss = float(cp.mean(diff ** 2))
    grad = 2 * diff / diff.size
    return loss, grad


def train_qupid_6502_direct(
    epochs: int = 200,
    samples_per_op: int = 5000,
    batch_size: int = 256,
    lr: float = 0.01,
    num_tiles: int = 16,
    temperature: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
):
    """Train Qupid 6502 with direct transformation."""

    if verbose:
        print()
        print("=" * 70)
        print("QUPID 6502 DIRECT: THE EMERGENCE")
        print("=" * 70)
        print()
        print("Direct transformation. No residual. Pure routing.")
        print()

    # Generate data
    if verbose:
        print("Generating data...")

    train_x, train_y = generate_6502_dataset(samples_per_op, seed)
    test_x, test_y = generate_6502_dataset(samples_per_op // 5, seed + 1000)

    n_train = len(train_x)
    n_test = len(test_x)

    if verbose:
        print(f"  Training: {n_train:,}")
        print(f"  Test: {n_test:,}")
        print()

    # Create model
    if verbose:
        print("Creating model...")

    model = DirectHollywoodSquares(
        d_model=128,
        num_tiles=num_tiles,
        temperature=temperature,
        lr=lr,
    )

    if verbose:
        print(f"  Tiles: {num_tiles}")
        print(f"  Parameters: {model.param_count():,}")
        print()

    # Training loop
    if verbose:
        print("Training...")
        print("-" * 70)
        print(f"{'Epoch':>6} | {'Loss':>10} | {'Test Acc':>10} | {'Best':>10} | {'Time':>8}")
        print("-" * 70)

    loss_history = []
    acc_history = []
    best_accuracy = 0.0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # Shuffle
        perm = np.random.permutation(n_train)
        perm_cp = cp.asarray(perm)
        train_x_shuffled = train_x[perm_cp]
        train_y_shuffled = train_y[perm_cp]

        # Train batches
        epoch_loss = 0.0
        n_batches = (n_train + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)

            batch_x = train_x_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]

            # Forward
            pred = model.forward(batch_x)

            # Loss
            loss, grad = mse_loss(pred, batch_y)
            epoch_loss += loss

            # Backward
            model.backward(grad)

            # Update
            model.step()

        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = evaluate_accuracy(model, test_x, test_y)
            acc_history.append(test_acc['overall'])
            best_accuracy = max(best_accuracy, test_acc['overall'])

            epoch_time = time.perf_counter() - epoch_start

            if verbose:
                print(f"{epoch+1:>6} | {epoch_loss:>10.6f} | {test_acc['overall']:>9.2f}% | {best_accuracy:>9.2f}% | {epoch_time:>7.2f}s")

    train_time = time.perf_counter() - start_time

    # Final evaluation
    final_acc = evaluate_accuracy(model, test_x, test_y)

    if verbose:
        print("-" * 70)
        print()
        print("FINAL RESULTS")
        print("-" * 50)
        print(f"  Overall Accuracy: {final_acc['overall']:.2f}%")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        print(f"  Training Time: {train_time:.2f}s")
        print()
        print("Per-Operation Accuracy:")
        for op_name, acc in sorted(final_acc['per_op'].items()):
            status = "OK" if acc >= 99.0 else "NEEDS WORK"
            print(f"  {op_name:>4}: {acc:>6.2f}% [{status}]")
        print()

    # Compare with baseline
    print()
    print("=" * 70)
    print("COMPARISON WITH MESA 11 BASELINE")
    print("=" * 70)

    baseline_acc = 99.76
    baseline_time = 255.2

    speedup = baseline_time / train_time

    print(f"{'Metric':<20} {'Qupid':<15} {'Mesa 11':<15} {'Ratio':<10}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {final_acc['overall']:<14.2f}% {baseline_acc:<14.2f}% ")
    print(f"{'Time':<20} {train_time:<14.2f}s {baseline_time:<14.2f}s {speedup:.1f}x faster")
    print()

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': epochs,
            'samples_per_op': samples_per_op,
            'num_tiles': num_tiles,
            'temperature': temperature,
            'lr': lr,
        },
        'accuracy': {
            'overall': final_acc['overall'],
            'best': best_accuracy,
            'per_op': final_acc['per_op'],
        },
        'timing': {
            'train_time_sec': train_time,
            'speedup_vs_baseline': speedup,
        },
        'history': {
            'loss': loss_history,
            'accuracy': acc_history,
        },
    }

    with open(output_dir / 'qupid_6502_direct_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'qupid_6502_direct_results.json'}")

    return model, results


if __name__ == "__main__":
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "QUPID 6502 DIRECT".center(68) + "*")
    print("*" + "Direct Transformation - No Residual".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" + "\"Observe the Emergence.\"".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    model, results = train_qupid_6502_direct(
        epochs=200,
        samples_per_op=5000,
        batch_size=256,
        lr=0.02,
        num_tiles=16,
        temperature=0.3,
    )
