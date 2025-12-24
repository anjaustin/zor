#!/usr/bin/env python3
"""
ADC Micro-NN Experiment: Chop First, Then Sharpen

Train a minimal ternary network to learn the 6502 ADC instruction.

ADC: Add with Carry
  Inputs: A (8-bit), M (8-bit), C_in (1-bit)
  Outputs: Result (8-bit), C_out, V, Z, N (flags)

The function is tiny. Can we learn it with a tiny network?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# =============================================================================
# GROUND TRUTH: 6502 ADC
# =============================================================================

def adc_truth(a: int, m: int, c_in: int) -> tuple:
    """
    6502 ADC in binary mode.

    Returns: (result, c_out, v, z, n)
    """
    sum_val = a + m + c_in
    result = sum_val & 0xFF
    c_out = 1 if sum_val > 255 else 0
    z = 1 if result == 0 else 0
    n = 1 if result & 0x80 else 0
    # Overflow: set if sign of result differs from sign of both operands
    v = 1 if ((a ^ result) & (m ^ result) & 0x80) else 0
    return result, c_out, v, z, n


def generate_adc_dataset(n_samples: int = None, exhaustive: bool = False):
    """Generate ADC training data."""

    if exhaustive:
        # All 2^17 = 131,072 combinations
        inputs = []
        outputs = []
        for a in range(256):
            for m in range(256):
                for c in range(2):
                    result, c_out, v, z, n = adc_truth(a, m, c)
                    inputs.append([a, m, c])
                    outputs.append([result, c_out, v, z, n])
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)
    else:
        # Random samples
        a = torch.randint(0, 256, (n_samples,))
        m = torch.randint(0, 256, (n_samples,))
        c = torch.randint(0, 2, (n_samples,))

        inputs = torch.stack([a, m, c], dim=1).float()
        outputs = []
        for i in range(n_samples):
            result, c_out, v, z, n = adc_truth(a[i].item(), m[i].item(), c[i].item())
            outputs.append([result, c_out, v, z, n])

        return inputs, torch.tensor(outputs, dtype=torch.float32)


# =============================================================================
# MICRO-NN ARCHITECTURES
# =============================================================================

class TinyMLP(nn.Module):
    """Baseline: tiny MLP."""
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),   # A, M, C_in
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),   # Result, C, V, Z, N
        )

    def forward(self, x):
        # Normalize inputs to [-1, 1]
        x_norm = x.clone()
        x_norm[:, :2] = x_norm[:, :2] / 127.5 - 1  # A, M: [0,255] -> [-1,1]
        x_norm[:, 2] = x_norm[:, 2] * 2 - 1        # C: [0,1] -> [-1,1]
        return self.net(x_norm)


class TernaryMicroNN(nn.Module):
    """Ternary micro-NN with XOR-like structure."""
    def __init__(self, hidden: int = 32, num_tiles: int = 4):
        super().__init__()
        self.num_tiles = num_tiles

        # Ternary signatures for routing
        self.signatures = nn.Parameter(torch.randn(num_tiles, 3) * 0.1)

        # Per-tile tiny networks
        self.tiles = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden // num_tiles),
                nn.Tanh(),
                nn.Linear(hidden // num_tiles, 5),
            )
            for _ in range(num_tiles)
        ])

        # Combination layer
        self.combine = nn.Linear(5 * num_tiles, 5)

    def forward(self, x):
        # Normalize
        x_norm = x.clone()
        x_norm[:, :2] = x_norm[:, :2] / 127.5 - 1
        x_norm[:, 2] = x_norm[:, 2] * 2 - 1

        # Route through tiles
        sigs = torch.sign(self.signatures)
        scores = x_norm @ sigs.T  # [batch, num_tiles]
        weights = F.softmax(scores, dim=-1)

        # Get tile outputs
        tile_outputs = torch.stack([tile(x_norm) for tile in self.tiles], dim=1)  # [batch, tiles, 5]

        # Weighted combination
        weighted = (tile_outputs * weights.unsqueeze(-1)).sum(dim=1)

        return weighted


class BitwiseMicroNN(nn.Module):
    """Process each bit position somewhat independently."""
    def __init__(self, hidden_per_bit: int = 8):
        super().__init__()

        # One small network per output bit (8 result bits + 4 flags = 12)
        self.bit_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_per_bit),
                nn.Tanh(),
                nn.Linear(hidden_per_bit, 1),
            )
            for _ in range(12)  # 8 result bits + C, V, Z, N
        ])

        # Combine result bits
        self.result_combine = nn.Linear(8, 1)  # Scale factor

    def forward(self, x):
        x_norm = x.clone()
        x_norm[:, :2] = x_norm[:, :2] / 127.5 - 1
        x_norm[:, 2] = x_norm[:, 2] * 2 - 1

        bit_outputs = [net(x_norm) for net in self.bit_nets]

        # Result: combine 8 bit predictions
        result_bits = torch.cat(bit_outputs[:8], dim=1)  # [batch, 8]
        # Convert to value: each bit weighted by power of 2
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32, device=x.device)
        result = (torch.sigmoid(result_bits) * powers).sum(dim=1, keepdim=True)

        # Flags
        flags = torch.cat(bit_outputs[8:], dim=1)  # [batch, 4]

        return torch.cat([result, flags], dim=1)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_x, train_y, epochs=100, lr=0.01, batch_size=256):
    """Train a model on ADC data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = train_x.shape[0]
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0

        for i in range(0, n, batch_size):
            batch_x = train_x[perm[i:i+batch_size]]
            batch_y = train_y[perm[i:i+batch_size]]

            optimizer.zero_grad()
            pred = model(batch_x)

            # MSE for result, BCE for flags
            result_loss = F.mse_loss(pred[:, 0], batch_y[:, 0])
            flag_loss = F.binary_cross_entropy_with_logits(pred[:, 1:], batch_y[:, 1:])
            loss = result_loss + flag_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(train_x)
                # Result accuracy (within 0.5)
                result_acc = (torch.abs(pred[:, 0] - train_y[:, 0]) < 0.5).float().mean()
                # Flag accuracy
                flag_pred = (pred[:, 1:] > 0).float()
                flag_acc = (flag_pred == train_y[:, 1:]).float().mean()
                # Perfect match (result + all flags)
                result_match = torch.abs(pred[:, 0] - train_y[:, 0]) < 0.5
                flags_match = (flag_pred == train_y[:, 1:]).all(dim=1)
                perfect = (result_match & flags_match).float().mean()

                best_acc = max(best_acc, perfect.item())

            print(f"Epoch {epoch+1:3d}: Loss={total_loss/n*batch_size:.4f} "
                  f"Result={result_acc:.2%} Flags={flag_acc:.2%} Perfect={perfect:.2%}")

    return best_acc


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("ADC Micro-NN Experiment: Chop First, Then Sharpen")
    print("=" * 70)

    # Generate exhaustive dataset (all 131,072 combinations)
    print("\nGenerating exhaustive ADC dataset...")
    train_x, train_y = generate_adc_dataset(exhaustive=True)
    print(f"Dataset: {train_x.shape[0]:,} samples")
    print(f"Input: A (8-bit), M (8-bit), C_in (1-bit)")
    print(f"Output: Result (8-bit), C_out, V, Z, N")

    print("\n" + "-" * 70)

    # Test different architectures
    configs = [
        ("TinyMLP-16", TinyMLP(hidden=16)),
        ("TinyMLP-32", TinyMLP(hidden=32)),
        ("TinyMLP-64", TinyMLP(hidden=64)),
        ("TinyMLP-128", TinyMLP(hidden=128)),
        ("Ternary-4tiles", TernaryMicroNN(hidden=32, num_tiles=4)),
        ("Ternary-8tiles", TernaryMicroNN(hidden=64, num_tiles=8)),
        ("Bitwise-8", BitwiseMicroNN(hidden_per_bit=8)),
        ("Bitwise-16", BitwiseMicroNN(hidden_per_bit=16)),
    ]

    results = []

    for name, model in configs:
        params = count_params(model)
        print(f"\n{name} ({params:,} params)")
        print("-" * 40)

        start = time.time()
        best_acc = train_model(model, train_x, train_y, epochs=50, lr=0.01, batch_size=512)
        elapsed = time.time() - start

        results.append({
            'name': name,
            'params': params,
            'accuracy': best_acc,
            'time': elapsed,
        })

        print(f"Best: {best_acc:.2%} in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>10} {'Accuracy':>10} {'Time':>8}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: -x['accuracy']):
        print(f"{r['name']:<20} {r['params']:>10,} {r['accuracy']:>10.2%} {r['time']:>7.1f}s")

    print("\n" + "=" * 70)
    print("Observations for temp file process:")
    print("- What worked?")
    print("- What surprised?")
    print("- Where does ternary/XOR want to go?")
    print("=" * 70)


if __name__ == "__main__":
    main()
