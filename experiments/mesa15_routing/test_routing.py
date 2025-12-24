#!/usr/bin/env python3
"""
The Calculator Test: Proving "Learning IS Routing"

Compares:
  A) MLP that learns both routing AND execution
  B) Frozen shapes + learned router (routing only)

Hypothesis: If learning IS routing, then B should:
  - Reach 100% accuracy (frozen shapes are exact)
  - Use far fewer trainable parameters
  - Converge much faster
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("THE CALCULATOR TEST")
print("Proving: Learning IS Routing. Everything Else Can Be Frozen.")
print("=" * 60)
print()

# ============================================================================
# FROZEN SHAPES (Pure Math - No Training)
# ============================================================================

def xor_poly(a, b):
    """XOR as polynomial: a + b - 2ab"""
    return a + b - 2 * a * b

def and_poly(a, b):
    """AND as polynomial: ab"""
    return a * b

def or_poly(a, b):
    """OR as polynomial: a + b - ab"""
    return a + b - a * b

def not_poly(a):
    """NOT as polynomial: 1 - a"""
    return 1 - a

def full_adder(a, b, cin):
    """Full adder from polynomials."""
    p = xor_poly(a, b)
    s = xor_poly(p, cin)
    cout = or_poly(and_poly(a, b), and_poly(cin, p))
    return s, cout

def frozen_add_8bit(a_bits, b_bits):
    """8-bit ripple-carry adder using frozen shapes."""
    result = []
    carry = torch.zeros_like(a_bits[:, 0])
    for i in range(8):
        s, carry = full_adder(a_bits[:, i], b_bits[:, i], carry)
        result.append(s)
    return torch.stack(result, dim=1)

def frozen_sub_8bit(a_bits, b_bits):
    """8-bit subtraction: a - b = a + NOT(b) + 1"""
    # Two's complement: -b = NOT(b) + 1
    not_b = 1 - b_bits
    result = []
    carry = torch.ones_like(a_bits[:, 0])  # Add 1 (carry-in = 1)
    for i in range(8):
        s, carry = full_adder(a_bits[:, i], not_b[:, i], carry)
        result.append(s)
    return torch.stack(result, dim=1)

def frozen_xor_8bit(a_bits, b_bits):
    """8-bit XOR using frozen shape."""
    return xor_poly(a_bits, b_bits)

def frozen_and_8bit(a_bits, b_bits):
    """8-bit AND using frozen shape."""
    return and_poly(a_bits, b_bits)

# ============================================================================
# DATA GENERATION
# ============================================================================

def int_to_bits(x, width=8):
    """Convert integer to bit tensor (LSB first)."""
    return torch.tensor([(x >> i) & 1 for i in range(width)], dtype=torch.float32)

def bits_to_int(bits):
    """Convert bit tensor back to integer."""
    result = 0
    for i, b in enumerate(bits):
        result += int(round(float(b))) << i
    return result

def generate_dataset():
    """Generate all 262,144 test cases."""
    print("Generating exhaustive dataset...")

    inputs = []
    targets = []

    for a in range(256):
        for b in range(256):
            for op in range(4):
                # Input: a bits (8) + b bits (8) + op bits (2) = 18
                a_bits = int_to_bits(a, 8)
                b_bits = int_to_bits(b, 8)
                op_bits = int_to_bits(op, 2)
                inp = torch.cat([a_bits, b_bits, op_bits])
                inputs.append(inp)

                # Target: result of operation
                if op == 0:  # ADD
                    result = (a + b) & 0xFF
                elif op == 1:  # SUB
                    result = (a - b) & 0xFF
                elif op == 2:  # XOR
                    result = a ^ b
                else:  # AND
                    result = a & b

                targets.append(int_to_bits(result, 8))

    X = torch.stack(inputs)
    Y = torch.stack(targets)

    print(f"  Dataset size: {len(X)} cases")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {Y.shape}")

    return X, Y

# ============================================================================
# ARCHITECTURE A: MLP (Learns Everything)
# ============================================================================

class MLP(nn.Module):
    """MLP that must learn both routing AND execution."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# ARCHITECTURE B: Frozen + Router (Learns Only Routing)
# ============================================================================

class FrozenRouter(nn.Module):
    """Frozen shapes + learned router."""

    def __init__(self):
        super().__init__()
        # Router: learns to select shape based on input
        # Only looks at the opcode bits (last 2 of 18)
        self.router = nn.Linear(18, 4)

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract operands
        a_bits = x[:, :8]
        b_bits = x[:, 8:16]

        # Compute all frozen shapes (no gradients needed for shapes themselves)
        add_out = frozen_add_8bit(a_bits, b_bits)
        sub_out = frozen_sub_8bit(a_bits, b_bits)
        xor_out = frozen_xor_8bit(a_bits, b_bits)
        and_out = frozen_and_8bit(a_bits, b_bits)

        # Stack shapes: [batch, 4, 8]
        shapes = torch.stack([add_out, sub_out, xor_out, and_out], dim=1)

        # Router produces weights over shapes
        weights = torch.softmax(self.router(x), dim=1)  # [batch, 4]

        # Weighted combination of shape outputs
        # weights: [batch, 4] -> [batch, 4, 1]
        # shapes: [batch, 4, 8]
        # result: [batch, 8]
        result = torch.einsum('bi,bij->bj', weights, shapes)

        return result

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# TRAINING
# ============================================================================

def calculate_accuracy(model, X, Y):
    """Calculate exact match accuracy."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        pred_bits = (pred > 0.5).float()
        correct = (pred_bits == Y).all(dim=1).sum().item()
    model.train()
    return correct / len(X)

def train_model(model, X, Y, name, max_epochs=100):
    """Train a model and track metrics."""
    print(f"\nTraining {name}...")
    print(f"  Trainable parameters: {model.count_params()}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    batch_size = 256
    n_batches = len(X) // batch_size

    history = {
        'epoch': [],
        'accuracy': [],
        'loss': []
    }

    epoch_95 = None
    epoch_99 = None
    epoch_100 = None

    start_time = time.time()

    for epoch in range(max_epochs):
        # Shuffle
        perm = torch.randperm(len(X))
        X_shuf = X[perm]
        Y_shuf = Y[perm]

        total_loss = 0
        for i in range(n_batches):
            batch_X = X_shuf[i*batch_size:(i+1)*batch_size]
            batch_Y = Y_shuf[i*batch_size:(i+1)*batch_size]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        acc = calculate_accuracy(model, X, Y)

        history['epoch'].append(epoch)
        history['accuracy'].append(acc)
        history['loss'].append(avg_loss)

        if epoch_95 is None and acc >= 0.95:
            epoch_95 = epoch
        if epoch_99 is None and acc >= 0.99:
            epoch_99 = epoch
        if epoch_100 is None and acc >= 0.9999:
            epoch_100 = epoch
            print(f"  Epoch {epoch}: 100% accuracy reached!")
            break

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch}: loss={avg_loss:.6f}, accuracy={acc*100:.2f}%")

    elapsed = time.time() - start_time
    final_acc = calculate_accuracy(model, X, Y)

    print(f"  Final accuracy: {final_acc*100:.4f}%")
    print(f"  Training time: {elapsed:.1f}s")

    return {
        'name': name,
        'params': model.count_params(),
        'final_accuracy': final_acc,
        'epochs_to_95': epoch_95,
        'epochs_to_99': epoch_99,
        'epochs_to_100': epoch_100,
        'total_epochs': len(history['epoch']),
        'time': elapsed
    }

# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == '__main__':
    # Generate data
    X, Y = generate_dataset()

    # Create models
    mlp = MLP()
    frozen = FrozenRouter()

    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    print(f"MLP parameters:    {mlp.count_params()}")
    print(f"Frozen parameters: {frozen.count_params()}")
    print(f"Ratio: {mlp.count_params() / frozen.count_params():.1f}x")

    # Train both
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    results_mlp = train_model(mlp, X, Y, "MLP (learns everything)")
    results_frozen = train_model(frozen, X, Y, "Frozen+Router (learns routing)")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'MLP':<20} {'Frozen+Router':<20}")
    print("-" * 65)
    print(f"{'Trainable Params':<25} {results_mlp['params']:<20} {results_frozen['params']:<20}")
    print(f"{'Final Accuracy':<25} {results_mlp['final_accuracy']*100:.2f}%{'':<15} {results_frozen['final_accuracy']*100:.2f}%")
    print(f"{'Epochs to 95%':<25} {str(results_mlp['epochs_to_95']):<20} {str(results_frozen['epochs_to_95']):<20}")
    print(f"{'Epochs to 99%':<25} {str(results_mlp['epochs_to_99']):<20} {str(results_frozen['epochs_to_99']):<20}")
    print(f"{'Epochs to 100%':<25} {str(results_mlp['epochs_to_100']):<20} {str(results_frozen['epochs_to_100']):<20}")
    print(f"{'Training Time':<25} {results_mlp['time']:.1f}s{'':<15} {results_frozen['time']:.1f}s")

    # Hypothesis evaluation
    print("\n" + "=" * 60)
    print("HYPOTHESIS EVALUATION")
    print("=" * 60)
    print()
    print("Hypothesis: Learning IS Routing. Everything Else Can Be Frozen.")
    print()

    evidence = []

    # Check 1: Frozen reaches 100%
    if results_frozen['final_accuracy'] >= 0.9999:
        evidence.append("+ Frozen+Router reached 100% accuracy")
    else:
        evidence.append(f"- Frozen+Router only reached {results_frozen['final_accuracy']*100:.2f}%")

    # Check 2: Frozen uses fewer params
    if results_frozen['params'] < results_mlp['params']:
        ratio = results_mlp['params'] / results_frozen['params']
        evidence.append(f"+ Frozen uses {ratio:.0f}x fewer trainable parameters")
    else:
        evidence.append("- Frozen uses more parameters than MLP")

    # Check 3: Frozen converges faster
    if results_frozen['epochs_to_100'] is not None:
        if results_mlp['epochs_to_100'] is None:
            evidence.append(f"+ Frozen reached 100% in {results_frozen['epochs_to_100']} epochs; MLP never did")
        elif results_frozen['epochs_to_100'] < results_mlp['epochs_to_100']:
            evidence.append(f"+ Frozen converged faster ({results_frozen['epochs_to_100']} vs {results_mlp['epochs_to_100']} epochs)")

    # Check 4: MLP struggles with 100%
    if results_mlp['epochs_to_100'] is None:
        evidence.append(f"+ MLP could not reach 100% (stuck at {results_mlp['final_accuracy']*100:.2f}%)")

    for e in evidence:
        print(e)

    # Final verdict
    supports = sum(1 for e in evidence if e.startswith('+'))
    challenges = sum(1 for e in evidence if e.startswith('-'))

    print()
    if supports > challenges:
        print("VERDICT: HYPOTHESIS SUPPORTED")
        print()
        print("Learning IS Routing. Frozen shapes handle execution exactly.")
        print("The router learns only WHERE to send data, not HOW to compute.")
    else:
        print("VERDICT: HYPOTHESIS NEEDS MORE INVESTIGATION")

    print()
    print("=" * 60)
