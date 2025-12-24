"""
Qupid 6502: Native GPU Training for MOS 6502 Emulation

Using Hollywood Squares Foundry to learn the 6502 ALU.
No PyTorch. Pure CUDA. Observe the Emergence.

"Position IS Computation. The 6502 knows this too."

Mesa 11 Baseline: 99.76% accuracy in 255 seconds (PyTorch)
Target: Match or exceed accuracy, 10x+ speedup

Operations:
- ADC: Add with Carry
- SBC: Subtract with Carry
- AND: Logical AND
- ORA: Logical OR
- EOR: Exclusive OR (XOR)
- ASL: Arithmetic Shift Left
- LSR: Logical Shift Right
- ROL: Rotate Left through Carry
- ROR: Rotate Right through Carry
- INC: Increment
- DEC: Decrement
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

from trix.foundry.native_training import NativeHollywoodSquares, Trainer, mse_loss


# =============================================================================
# 6502 GROUND TRUTH (Pure Math - No Learning)
# =============================================================================

def adc_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ADC: A + B + C -> (result, carry_out)"""
    result = a.astype(np.int32) + b.astype(np.int32) + c.astype(np.int32)
    carry_out = (result > 255).astype(np.float32)
    result = (result & 0xFF).astype(np.float32)
    return result, carry_out


def sbc_truth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SBC: A - B - (1-C) = A + NOT(B) + C"""
    b_inv = 255 - b
    return adc_truth(a, b_inv, c)


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


# Operation registry
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
    """
    Generate training data for all 6502 operations.

    Input encoding (d_model=128):
        [0:8]   - operand A (8 bits, normalized to [0,1])
        [8:16]  - operand B (8 bits, normalized to [0,1])
        [16]    - carry in
        [17:28] - operation one-hot (11 operations)
        [28:128]- padding zeros

    Target encoding (d_model=128):
        [0:8]   - result (8 bits, normalized to [0,1])
        [8]     - carry out
        [9:128] - padding zeros

    Returns:
        train_x: [n_samples, 128]
        train_y: [n_samples, 128]
    """
    np.random.seed(seed)

    all_x = []
    all_y = []

    for op_id, (op_name, truth_fn) in OPERATIONS.items():
        # Generate random inputs
        a = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        b = np.random.randint(0, 256, n_samples_per_op).astype(np.float32)
        c = np.random.randint(0, 2, n_samples_per_op).astype(np.float32)

        # Compute ground truth
        result, carry_out = truth_fn(a, b, c)

        # Encode inputs
        x = np.zeros((n_samples_per_op, 128), dtype=np.float32)

        # Operand A as 8 bits
        for i in range(8):
            x[:, i] = ((a.astype(np.int32) >> i) & 1).astype(np.float32)

        # Operand B as 8 bits
        for i in range(8):
            x[:, 8 + i] = ((b.astype(np.int32) >> i) & 1).astype(np.float32)

        # Carry in
        x[:, 16] = c

        # Operation one-hot
        x[:, 17 + op_id] = 1.0

        # Encode targets
        y = np.zeros((n_samples_per_op, 128), dtype=np.float32)

        # Result as 8 bits
        for i in range(8):
            y[:, i] = ((result.astype(np.int32) >> i) & 1).astype(np.float32)

        # Carry out
        y[:, 8] = carry_out

        all_x.append(x)
        all_y.append(y)

    # Concatenate and shuffle
    train_x = np.concatenate(all_x, axis=0)
    train_y = np.concatenate(all_y, axis=0)

    # Shuffle
    perm = np.random.permutation(len(train_x))
    train_x = train_x[perm]
    train_y = train_y[perm]

    return cp.asarray(train_x), cp.asarray(train_y)


def decode_output(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode output back to result and carry."""
    result = np.zeros(len(y), dtype=np.int32)
    for i in range(8):
        result += ((y[:, i] > 0.5).astype(np.int32) << i)
    carry = (y[:, 8] > 0.5).astype(np.int32)
    return result, carry


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_accuracy(
    model: NativeHollywoodSquares,
    test_x: cp.ndarray,
    test_y: cp.ndarray,
    n_samples_per_op: int,
) -> Dict:
    """Evaluate per-operation accuracy."""

    # Forward pass
    pred = model.forward(test_x, save_for_backward=False)

    # Move to CPU for evaluation
    pred_np = cp.asnumpy(pred)
    test_y_np = cp.asnumpy(test_y)
    test_x_np = cp.asnumpy(test_x)

    # Threshold predictions
    pred_bits = (pred_np > 0.5).astype(np.float32)
    target_bits = test_y_np

    # Overall accuracy (per sample, all 9 bits must match)
    correct = np.all(pred_bits[:, :9] == target_bits[:, :9], axis=1)
    overall_acc = correct.mean() * 100

    # Per-operation accuracy
    per_op = {}
    for op_id, (op_name, _) in OPERATIONS.items():
        # Find samples for this operation (check one-hot encoding)
        mask = test_x_np[:, 17 + op_id] > 0.5
        if mask.sum() > 0:
            op_correct = correct[mask]
            per_op[op_name] = op_correct.mean() * 100

    return {
        'overall': overall_acc,
        'per_op': per_op,
    }


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    d_model: int = 128
    num_tiles: int = 16
    grid_size: int = 16
    lr: float = 0.01
    epochs: int = 100
    batch_size: int = 256
    samples_per_op: int = 5000
    test_samples_per_op: int = 1000
    seed: int = 42


@dataclass
class TrainingResult:
    """Training result."""
    config: TrainingConfig
    final_accuracy: float
    per_op_accuracy: Dict[str, float]
    best_accuracy: float
    train_time_sec: float
    loss_history: List[float] = field(default_factory=list)
    acc_history: List[float] = field(default_factory=list)


def train_qupid_6502(config: TrainingConfig, verbose: bool = True) -> TrainingResult:
    """
    Train Qupid 6502 using Hollywood Squares Foundry.

    "Observe the Emergence."
    """
    if verbose:
        print()
        print("=" * 70)
        print("QUPID 6502: THE EMERGENCE")
        print("=" * 70)
        print()
        print("Position IS Computation.")
        print("The 6502 shall learn through frozen geometry.")
        print()

    # Generate data
    if verbose:
        print("Generating training data...")

    train_x, train_y = generate_6502_dataset(
        n_samples_per_op=config.samples_per_op,
        seed=config.seed,
    )

    test_x, test_y = generate_6502_dataset(
        n_samples_per_op=config.test_samples_per_op,
        seed=config.seed + 1000,
    )

    n_train = len(train_x)
    n_test = len(test_x)

    if verbose:
        print(f"  Training samples: {n_train:,}")
        print(f"  Test samples: {n_test:,}")
        print()

    # Create model
    if verbose:
        print("Creating model...")

    model = NativeHollywoodSquares(
        d_model=config.d_model,
        num_tiles=config.num_tiles,
        grid_size=config.grid_size,
        lr=config.lr,
    )

    if verbose:
        print(f"  Parameters: {model.param_count():,}")
        print()

    # Training
    if verbose:
        print("Training...")
        print("-" * 70)
        print(f"{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time':>8}")
        print("-" * 70)

    trainer = Trainer(model, loss_fn='mse')

    loss_history = []
    acc_history = []
    best_accuracy = 0.0

    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        # Shuffle
        perm = np.random.permutation(n_train)
        perm_cp = cp.asarray(perm)
        train_x_shuffled = train_x[perm_cp]
        train_y_shuffled = train_y[perm_cp]

        # Train one epoch
        epoch_loss = 0.0
        n_batches = (n_train + config.batch_size - 1) // config.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, n_train)

            batch_x = train_x_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]

            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss

        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = evaluate_accuracy(model, train_x[:5000], train_y[:5000], 500)
            test_acc = evaluate_accuracy(model, test_x, test_y, config.test_samples_per_op)

            acc_history.append(test_acc['overall'])
            best_accuracy = max(best_accuracy, test_acc['overall'])

            epoch_time = time.perf_counter() - epoch_start

            if verbose:
                print(f"{epoch+1:>6} | {epoch_loss:>10.6f} | {train_acc['overall']:>9.2f}% | {test_acc['overall']:>9.2f}% | {epoch_time:>7.2f}s")

    train_time = time.perf_counter() - start_time

    # Final evaluation
    final_acc = evaluate_accuracy(model, test_x, test_y, config.test_samples_per_op)

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

    return TrainingResult(
        config=config,
        final_accuracy=final_acc['overall'],
        per_op_accuracy=final_acc['per_op'],
        best_accuracy=best_accuracy,
        train_time_sec=train_time,
        loss_history=loss_history,
        acc_history=acc_history,
    )


def compare_with_baseline(result: TrainingResult):
    """Compare with Mesa 11 PyTorch baseline."""
    print()
    print("=" * 70)
    print("COMPARISON WITH MESA 11 BASELINE")
    print("=" * 70)
    print()

    # Mesa 11 baseline
    baseline_acc = 99.76
    baseline_time = 255.2  # seconds
    baseline_params = 0  # We'll estimate

    print(f"{'Metric':<25} {'Qupid (Native)':<20} {'Mesa 11 (PyTorch)':<20} {'Winner':<10}")
    print("-" * 75)

    # Accuracy
    acc_winner = "Qupid" if result.final_accuracy >= baseline_acc else "Mesa 11"
    print(f"{'Accuracy':<25} {result.final_accuracy:<19.2f}% {baseline_acc:<19.2f}% {acc_winner:<10}")

    # Time
    speedup = baseline_time / result.train_time_sec
    time_winner = "Qupid" if result.train_time_sec < baseline_time else "Mesa 11"
    print(f"{'Training Time':<25} {result.train_time_sec:<19.2f}s {baseline_time:<19.2f}s {time_winner:<10} ({speedup:.1f}x)")

    # Per-op comparison
    print()
    print("Per-Operation Comparison:")
    mesa11_per_op = {
        'ADC': 99.6, 'AND': 100.0, 'ORA': 100.0, 'EOR': 100.0,
        'ASL': 100.0, 'LSR': 100.0, 'INC': 94.23, 'DEC': 94.23,
    }

    for op_name in ['ADC', 'SBC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC']:
        qupid_acc = result.per_op_accuracy.get(op_name, 0)
        mesa11_acc = mesa11_per_op.get(op_name, 'N/A')
        if isinstance(mesa11_acc, float):
            winner = "=" if abs(qupid_acc - mesa11_acc) < 0.5 else ("Qupid" if qupid_acc > mesa11_acc else "Mesa11")
            print(f"  {op_name:>4}: Qupid {qupid_acc:>6.2f}% | Mesa11 {mesa11_acc:>6.2f}% | {winner}")
        else:
            print(f"  {op_name:>4}: Qupid {qupid_acc:>6.2f}% | Mesa11 N/A")

    print()
    print("=" * 70)
    if result.final_accuracy >= baseline_acc and speedup > 1:
        print("QUPID WINS: Better accuracy AND faster training!")
    elif result.final_accuracy >= baseline_acc:
        print("QUPID MATCHES ACCURACY: Same quality, different speed.")
    elif speedup > 10:
        print(f"QUPID IS {speedup:.0f}x FASTER: Speed vs accuracy tradeoff.")
    else:
        print("MORE TRAINING NEEDED: Continue the Emergence.")
    print("=" * 70)


def save_results(result: TrainingResult, path: Path):
    """Save results to JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'd_model': result.config.d_model,
            'num_tiles': result.config.num_tiles,
            'grid_size': result.config.grid_size,
            'lr': result.config.lr,
            'epochs': result.config.epochs,
            'batch_size': result.config.batch_size,
            'samples_per_op': result.config.samples_per_op,
            'seed': result.config.seed,
        },
        'accuracy': {
            'overall': result.final_accuracy,
            'best': result.best_accuracy,
            'per_op': result.per_op_accuracy,
        },
        'timing': {
            'train_time_sec': result.train_time_sec,
            'samples_per_sec': (result.config.samples_per_op * NUM_OPS * result.config.epochs) / result.train_time_sec,
        },
        'history': {
            'loss': result.loss_history,
            'accuracy': result.acc_history,
        },
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Qupid 6502 training."""
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "QUPID 6502".center(68) + "*")
    print("*" + "Native GPU Training for MOS 6502 Emulation".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" + "\"Observe the Emergence.\"".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Configuration
    config = TrainingConfig(
        d_model=128,
        num_tiles=16,
        grid_size=16,
        lr=0.01,
        epochs=100,
        batch_size=256,
        samples_per_op=5000,
        test_samples_per_op=1000,
        seed=42,
    )

    # Train
    result = train_qupid_6502(config, verbose=True)

    # Compare with baseline
    compare_with_baseline(result)

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(result, output_dir / 'qupid_6502_results.json')

    return result


if __name__ == "__main__":
    result = main()
