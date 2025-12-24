#!/usr/bin/env python3
"""
SDPMX 6502 - Testing the Complete Pipeline on Real Data

S → D → P → M → X applied to 6502 emulation.

Date: 2024-12-20
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import time

from trix.nn.sdpmx_pipeline import SDPMXPipeline


# Ground truth functions
def adc_truth(a, b, c):
    result = (a + b + c) & 0xFF
    return result

def and_truth(a, b):
    return a & b

def ora_truth(a, b):
    return a | b

def eor_truth(a, b):
    return a ^ b

def asl_truth(val):
    return (val << 1) & 0xFF

def lsr_truth(val):
    return val >> 1

def inc_truth(val):
    return (val + 1) & 0xFF

def dec_truth(val):
    return (val - 1) & 0xFF


OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


def generate_data(n_samples: int = 5000) -> List[Dict]:
    """Generate 6502 dataset."""
    data = []

    # ADC
    for _ in range(n_samples):
        a, b, c = np.random.randint(256), np.random.randint(256), np.random.randint(2)
        data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': adc_truth(a, b, c)})

    # LOGIC
    for op_name, fn in [('AND', and_truth), ('ORA', ora_truth), ('EOR', eor_truth)]:
        for _ in range(n_samples):
            a, b = np.random.randint(256), np.random.randint(256)
            data.append({'op': op_name, 'a': a, 'b': b, 'c': 0, 'result': fn(a, b)})

    # SHIFT
    for val in range(256):
        data.append({'op': 'ASL', 'a': val, 'b': 0, 'c': 0, 'result': asl_truth(val)})
        data.append({'op': 'LSR', 'a': val, 'b': 0, 'c': 0, 'result': lsr_truth(val)})

    # INCDEC
    for val in range(256):
        data.append({'op': 'INC', 'a': val, 'b': 0, 'c': 0, 'result': inc_truth(val)})
        data.append({'op': 'DEC', 'a': val, 'b': 0, 'c': 0, 'result': dec_truth(val)})

    np.random.shuffle(data)
    return data


class SDPMX6502(nn.Module):
    """SDPMX model for 6502 emulation."""

    def __init__(self, d_model: int = 128, num_subspaces: int = 16, grid_size: int = 16):
        super().__init__()
        self.d_model = d_model

        self.op_embed = nn.Embedding(len(OPCODES), 32)

        # Input: op_embed (32) + a_bits (8) + b_bits (8) + c (1) = 49
        self.input_proj = nn.Linear(49, d_model)

        # SDPMX Pipeline
        self.pipeline = SDPMXPipeline(
            dim=d_model,
            grid_size=grid_size,
            num_subspaces=num_subspaces,
        )

        # Output head
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

    def forward(self, op_idx, a, b, c):
        # Encode inputs
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()

        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x)

        # SDPMX
        x, info = self.pipeline(x)

        # Predict
        result = self.result_head(x)

        return result, info


def prepare_tensors(data: List[Dict], device: str) -> Dict:
    """Convert data to tensors."""
    op_idx = torch.tensor([OP_TO_IDX[d['op']] for d in data], device=device)
    a = torch.tensor([d['a'] for d in data], device=device)
    b = torch.tensor([d['b'] for d in data], device=device)
    c = torch.tensor([d['c'] for d in data], device=device)
    result = torch.tensor([d['result'] for d in data], device=device)
    result_bits = torch.stack([(result >> i) & 1 for i in range(8)], dim=1).float()
    ops = [d['op'] for d in data]

    return {
        'op_idx': op_idx, 'a': a, 'b': b, 'c': c,
        'result': result, 'result_bits': result_bits, 'ops': ops
    }


def train_epoch(model, data, optimizer, batch_size=256):
    """Train one epoch."""
    model.train()
    device = next(model.parameters()).device

    n = len(data['op_idx'])
    perm = torch.randperm(n, device=device)

    total_loss = 0.0
    correct = 0
    n_batches = 0

    for i in range(0, n - batch_size, batch_size):
        idx = perm[i:i+batch_size]

        pred, info = model(
            data['op_idx'][idx],
            data['a'][idx],
            data['b'][idx],
            data['c'][idx],
        )

        loss = F.binary_cross_entropy(pred, data['result_bits'][idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pred_vals = sum((pred[:, i] > 0.5).long() << i for i in range(8))
        correct += (pred_vals == data['result'][idx]).sum().item()

    return total_loss / n_batches, correct / n * 100


def evaluate(model, data, batch_size=256):
    """Evaluate model."""
    model.eval()

    n = len(data['op_idx'])
    correct_by_op = defaultdict(int)
    total_by_op = defaultdict(int)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)

            pred, _ = model(
                data['op_idx'][i:end],
                data['a'][i:end],
                data['b'][i:end],
                data['c'][i:end],
            )

            pred_vals = sum((pred[:, j] > 0.5).long() << j for j in range(8))

            for j in range(end - i):
                op = data['ops'][i + j]
                total_by_op[op] += 1
                if pred_vals[j] == data['result'][i + j]:
                    correct_by_op[op] += 1

    results = {op: correct_by_op[op] / total_by_op[op] * 100 for op in OPCODES if total_by_op[op] > 0}
    results['overall'] = sum(correct_by_op.values()) / sum(total_by_op.values()) * 100

    return results


def main():
    print("=" * 70)
    print("SDPMX 6502 - Complete Pipeline Test")
    print("S → D → P → M → X")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Generate data
    print("Generating data...")
    all_data = generate_data(n_samples=5000)
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]

    train_tensors = prepare_tensors(train_data, device)
    test_tensors = prepare_tensors(test_data, device)
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}\n")

    # Create model
    model = SDPMX6502(d_model=128, num_subspaces=16, grid_size=16).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"{'Epoch':>5} {'Loss':>10} {'Train%':>10} {'Test%':>10}")
    print("-" * 45)

    for epoch in range(100):
        loss, train_acc = train_epoch(model, train_tensors, optimizer)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            test_results = evaluate(model, test_tensors)
            print(f"{epoch+1:5d} {loss:10.4f} {train_acc:10.2f} {test_results['overall']:10.2f}")

    # Final evaluation
    print("\nFinal Results:")
    results = evaluate(model, test_tensors)
    for op in OPCODES:
        bar = '█' * int(results.get(op, 0) / 5)
        print(f"  {op:4s}: {bar:20s} {results.get(op, 0):6.1f}%")
    print(f"\n  Overall: {results['overall']:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
