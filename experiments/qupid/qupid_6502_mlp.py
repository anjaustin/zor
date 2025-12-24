"""
Qupid 6502 MLP: Native GPU MLP for 6502 Emulation

A simple native GPU MLP to establish what's achievable with pure CUDA.
Then we can see how Hollywood Squares compares.

"First, prove the CUDA path works. Then add the magic."
"""

import numpy as np
import cupy as cp
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')


# =============================================================================
# NATIVE MLP - PURE CUDA
# =============================================================================

class NativeMLP:
    """
    Simple 2-layer MLP in pure CuPy.
    No frameworks. Just matrix multiplication.
    """

    def __init__(
        self,
        d_input: int = 128,
        d_hidden: int = 256,
        d_output: int = 128,
        lr: float = 0.01,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.lr = lr

        # Initialize weights (Xavier)
        self.W1 = cp.random.randn(d_input, d_hidden).astype(cp.float32) * np.sqrt(2.0 / d_input)
        self.b1 = cp.zeros(d_hidden, dtype=cp.float32)
        self.W2 = cp.random.randn(d_hidden, d_output).astype(cp.float32) * np.sqrt(2.0 / d_hidden)
        self.b2 = cp.zeros(d_output, dtype=cp.float32)

        # Adam state
        self.m_W1 = cp.zeros_like(self.W1)
        self.v_W1 = cp.zeros_like(self.W1)
        self.m_b1 = cp.zeros_like(self.b1)
        self.v_b1 = cp.zeros_like(self.b1)
        self.m_W2 = cp.zeros_like(self.W2)
        self.v_W2 = cp.zeros_like(self.W2)
        self.m_b2 = cp.zeros_like(self.b2)
        self.v_b2 = cp.zeros_like(self.b2)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # Cache
        self.cache = {}

    def forward(self, x: cp.ndarray, save_for_backward: bool = True) -> cp.ndarray:
        """Forward pass: x -> h -> y"""
        # Layer 1: x @ W1 + b1 -> ReLU
        z1 = x @ self.W1 + self.b1
        h = cp.maximum(z1, 0)  # ReLU

        # Layer 2: h @ W2 + b2 -> Sigmoid
        z2 = h @ self.W2 + self.b2
        y = 1 / (1 + cp.exp(-z2))  # Sigmoid (for binary outputs)

        if save_for_backward:
            self.cache = {'x': x, 'z1': z1, 'h': h, 'z2': z2, 'y': y}

        return y

    def backward(self, d_y: cp.ndarray):
        """Backward pass."""
        x = self.cache['x']
        z1 = self.cache['z1']
        h = self.cache['h']
        y = self.cache['y']
        batch_size = x.shape[0]

        # Gradient through sigmoid: d_z2 = d_y * y * (1 - y)
        d_z2 = d_y * y * (1 - y)

        # Gradients for W2, b2
        self.d_W2 = h.T @ d_z2 / batch_size
        self.d_b2 = d_z2.mean(axis=0)

        # Gradient through ReLU
        d_h = d_z2 @ self.W2.T
        d_z1 = d_h * (z1 > 0)

        # Gradients for W1, b1
        self.d_W1 = x.T @ d_z1 / batch_size
        self.d_b1 = d_z1.mean(axis=0)

    def step(self):
        """Adam optimizer step."""
        self.t += 1

        for name in ['W1', 'b1', 'W2', 'b2']:
            param = getattr(self, name)
            grad = getattr(self, f'd_{name}')
            m = getattr(self, f'm_{name}')
            v = getattr(self, f'v_{name}')

            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad**2

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# =============================================================================
# 6502 DATA GENERATION
# =============================================================================

def adc_truth(a, b, c):
    result = a.astype(np.int32) + b.astype(np.int32) + c.astype(np.int32)
    carry = (result > 255).astype(np.float32)
    return (result & 0xFF).astype(np.float32), carry

def sbc_truth(a, b, c):
    b_inv = 255 - b.astype(np.int32)
    return adc_truth(a, b_inv.astype(np.float32), c)

def and_truth(a, b, c):
    return (a.astype(np.int32) & b.astype(np.int32)).astype(np.float32), np.zeros_like(c)

def ora_truth(a, b, c):
    return (a.astype(np.int32) | b.astype(np.int32)).astype(np.float32), np.zeros_like(c)

def eor_truth(a, b, c):
    return (a.astype(np.int32) ^ b.astype(np.int32)).astype(np.float32), np.zeros_like(c)

def asl_truth(a, b, c):
    a_int = a.astype(np.int32)
    return ((a_int << 1) & 0xFF).astype(np.float32), ((a_int >> 7) & 1).astype(np.float32)

def lsr_truth(a, b, c):
    a_int = a.astype(np.int32)
    return (a_int >> 1).astype(np.float32), (a_int & 1).astype(np.float32)

def rol_truth(a, b, c):
    a_int, c_int = a.astype(np.int32), c.astype(np.int32)
    return (((a_int << 1) | c_int) & 0xFF).astype(np.float32), ((a_int >> 7) & 1).astype(np.float32)

def ror_truth(a, b, c):
    a_int, c_int = a.astype(np.int32), c.astype(np.int32)
    return ((a_int >> 1) | (c_int << 7)).astype(np.float32), (a_int & 1).astype(np.float32)

def inc_truth(a, b, c):
    return ((a.astype(np.int32) + 1) & 0xFF).astype(np.float32), np.zeros_like(c)

def dec_truth(a, b, c):
    return ((a.astype(np.int32) - 1) & 0xFF).astype(np.float32), np.zeros_like(c)

OPERATIONS = {
    0: ('ADC', adc_truth), 1: ('SBC', sbc_truth), 2: ('AND', and_truth),
    3: ('ORA', ora_truth), 4: ('EOR', eor_truth), 5: ('ASL', asl_truth),
    6: ('LSR', lsr_truth), 7: ('ROL', rol_truth), 8: ('ROR', ror_truth),
    9: ('INC', inc_truth), 10: ('DEC', dec_truth),
}


def generate_6502_dataset(n_samples_per_op: int = 5000, seed: int = 42):
    """Generate training data."""
    np.random.seed(seed)
    all_x, all_y = [], []

    for op_id, (op_name, truth_fn) in OPERATIONS.items():
        a = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        b = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        c = np.random.randint(0, 2, n_samples_per_op).astype(np.float32)

        result, carry_out = truth_fn(a, b, c)

        # Encode inputs
        x = np.zeros((n_samples_per_op, 128), dtype=np.float32)
        for i in range(8):
            x[:, i] = ((a.astype(np.int32) >> i) & 1).astype(np.float32)
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
    return cp.asarray(train_x[perm]), cp.asarray(train_y[perm])


def evaluate_accuracy(model: NativeMLP, test_x: cp.ndarray, test_y: cp.ndarray) -> Dict:
    """Evaluate accuracy."""
    pred = model.forward(test_x, save_for_backward=False)
    pred_np = cp.asnumpy(pred)
    test_y_np = cp.asnumpy(test_y)
    test_x_np = cp.asnumpy(test_x)

    pred_bits = (pred_np > 0.5).astype(np.float32)
    correct = np.all(pred_bits[:, :9] == test_y_np[:, :9], axis=1)
    overall_acc = correct.mean() * 100

    per_op = {}
    for op_id, (op_name, _) in OPERATIONS.items():
        mask = test_x_np[:, 17 + op_id] > 0.5
        if mask.sum() > 0:
            per_op[op_name] = correct[mask].mean() * 100

    return {'overall': overall_acc, 'per_op': per_op}


def bce_loss(pred: cp.ndarray, target: cp.ndarray) -> Tuple[float, cp.ndarray]:
    """Binary cross-entropy loss (better for bit predictions)."""
    eps = 1e-7
    pred = cp.clip(pred, eps, 1 - eps)
    loss = -cp.mean(target * cp.log(pred) + (1 - target) * cp.log(1 - pred))
    grad = (pred - target) / (pred * (1 - pred) + eps)
    return float(loss), grad


# =============================================================================
# TRAINING
# =============================================================================

def train_mlp_6502(
    epochs: int = 200,
    samples_per_op: int = 5000,
    batch_size: int = 256,
    lr: float = 0.001,
    d_hidden: int = 512,
    verbose: bool = True,
):
    """Train native MLP on 6502."""

    if verbose:
        print()
        print("=" * 70)
        print("QUPID 6502 MLP: NATIVE GPU BASELINE")
        print("=" * 70)
        print()
        print("Pure CuPy. No PyTorch. Establishing the CUDA baseline.")
        print()

    # Generate data
    train_x, train_y = generate_6502_dataset(samples_per_op, seed=42)
    test_x, test_y = generate_6502_dataset(samples_per_op // 5, seed=1042)

    n_train = len(train_x)

    if verbose:
        print(f"Training: {n_train:,} samples")
        print(f"Test: {len(test_x):,} samples")
        print()

    # Create model
    model = NativeMLP(d_input=128, d_hidden=d_hidden, d_output=128, lr=lr)

    if verbose:
        print(f"Hidden: {d_hidden}")
        print(f"Parameters: {model.param_count():,}")
        print()

    # Training
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
        perm = cp.asarray(np.random.permutation(n_train))
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]

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

            # Loss (only on first 9 dimensions - result + carry)
            loss, grad = bce_loss(pred[:, :9], batch_y[:, :9])

            # Zero out gradient for unused dimensions
            full_grad = cp.zeros_like(pred)
            full_grad[:, :9] = grad

            # Backward
            model.backward(full_grad)

            # Update
            model.step()

            epoch_loss += loss

        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = evaluate_accuracy(model, test_x, test_y)
            acc_history.append(test_acc['overall'])
            best_accuracy = max(best_accuracy, test_acc['overall'])

            epoch_time = time.perf_counter() - epoch_start

            if verbose:
                print(f"{epoch+1:>6} | {epoch_loss:>10.4f} | {test_acc['overall']:>9.2f}% | {best_accuracy:>9.2f}% | {epoch_time:>7.2f}s")

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
            status = "OK" if acc >= 99.0 else "NEEDS WORK" if acc >= 90 else "LOW"
            print(f"  {op_name:>4}: {acc:>6.2f}% [{status}]")
        print()

    # Compare with baseline
    print()
    print("=" * 70)
    print("COMPARISON WITH MESA 11 BASELINE (PyTorch)")
    print("=" * 70)

    baseline_acc = 99.76
    baseline_time = 255.2
    speedup = baseline_time / train_time

    print(f"{'Metric':<20} {'Native MLP':<15} {'Mesa 11':<15} {'Ratio':<10}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {final_acc['overall']:<14.2f}% {baseline_acc:<14.2f}%")
    print(f"{'Time':<20} {train_time:<14.2f}s {baseline_time:<14.2f}s {speedup:.1f}x faster")
    print()

    # Save results
    output_dir = Path(__file__).parent
    results = {
        'timestamp': datetime.now().isoformat(),
        'architecture': 'NativeMLP',
        'config': {
            'd_hidden': d_hidden,
            'epochs': epochs,
            'samples_per_op': samples_per_op,
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
    }

    with open(output_dir / 'qupid_6502_mlp_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'qupid_6502_mlp_results.json'}")

    return model, results


if __name__ == "__main__":
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "QUPID 6502 MLP".center(68) + "*")
    print("*" + "Native GPU MLP - Establishing Baseline".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    model, results = train_mlp_6502(
        epochs=200,
        samples_per_op=5000,
        batch_size=256,
        lr=0.001,
        d_hidden=512,
    )
