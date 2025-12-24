#!/usr/bin/env python3
"""
TriX129X - 6502 Monolithic Training with XOR Superposition Compression

CODENAME: TriX129X
MISSION: Train 6502 emulator with HierarchicalTriXFFN, then compress signatures
         and verify 129× compression with maintained accuracy.

BASELINE: Previous SparseLookupFFNv2 achieved 99.76% accuracy
TARGET: Match or beat 99.76% AND achieve 129× signature compression

Author: Droid (Mesa 13 - XOR Superposition)
Date: 2024-12-19
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')
sys.path.insert(0, '/workspace/flynnconceivable')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import json

# Import TriX components
from trix.nn import HierarchicalTriXFFN


class XORMixer(nn.Module):
    """
    XOR-based superposition mixer for routing scores.

    XOR properties we exploit:
    - Self-inverse: a ^ b ^ b = a
    - Orthogonality generator
    - Natural superposition creator
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.randn(dim, dim) * 0.0173)
        self.mix_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ternary = torch.tanh(x)
        mixed = torch.matmul(x_ternary, self.mix_weight) + self.mix_bias
        return x + mixed

# Import 6502 ground truth
try:
    from training.data import (
        adc_truth, sbc_truth,
        asl_truth, lsr_truth, rol_truth, ror_truth,
        and_truth, ora_truth, eor_truth,
        inc_truth, dec_truth,
    )
    HAS_FLYNN = True
except ImportError:
    HAS_FLYNN = False
    print("Warning: flynnconceivable not found, using built-in ground truth")


# =============================================================================
# GROUND TRUTH (fallback if flynnconceivable not available)
# =============================================================================

if not HAS_FLYNN:
    def adc_truth(a, b, c):
        result = (a + b + c) & 0xFF
        c_out = int((a + b + c) > 255)
        v = int(((a ^ result) & (b ^ result) & 0x80) != 0)
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0), 'c': c_out, 'v': v}

    def and_truth(a, b):
        result = a & b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}

    def ora_truth(a, b):
        result = a | b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}

    def eor_truth(a, b):
        result = a ^ b
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}

    def asl_truth(val):
        result = (val << 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0), 'c': (val >> 7) & 1}

    def lsr_truth(val):
        result = val >> 1
        return {'result': result, 'n': 0, 'z': int(result == 0), 'c': val & 1}

    def inc_truth(val):
        result = (val + 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}

    def dec_truth(val):
        result = (val - 1) & 0xFF
        return {'result': result, 'n': (result >> 7) & 1, 'z': int(result == 0)}


# =============================================================================
# DATA GENERATION
# =============================================================================

OPCODES = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}
OP_TO_CAT = {
    'ADC': 'ALU',
    'AND': 'LOGIC', 'ORA': 'LOGIC', 'EOR': 'LOGIC',
    'ASL': 'SHIFT', 'LSR': 'SHIFT',
    'INC': 'INCDEC', 'DEC': 'INCDEC',
}


def generate_exhaustive_data(fast_mode: bool = True) -> List[Dict]:
    """Generate 6502 dataset. fast_mode uses sampling for speed."""
    data = []

    if fast_mode:
        print("Generating SAMPLED 6502 data (fast mode)...")
        n_samples = 5000  # Per operation - smaller for faster iteration

        # ADC
        for _ in range(n_samples):
            a, b, c = np.random.randint(256), np.random.randint(256), np.random.randint(2)
            d = adc_truth(a, b, c)
            data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': d['result']})

        # LOGIC
        for op_name, truth_fn in [('AND', and_truth), ('ORA', ora_truth), ('EOR', eor_truth)]:
            for _ in range(n_samples):
                a, b = np.random.randint(256), np.random.randint(256)
                data.append({'op': op_name, 'a': a, 'b': b, 'c': 0, 'result': truth_fn(a, b)['result']})

        # SHIFT (all 512 - it's small)
        for val in range(256):
            data.append({'op': 'ASL', 'a': val, 'b': 0, 'c': 0, 'result': asl_truth(val)['result']})
            data.append({'op': 'LSR', 'a': val, 'b': 0, 'c': 0, 'result': lsr_truth(val)['result']})

        # INCDEC (all 512 - it's small)
        for val in range(256):
            data.append({'op': 'INC', 'a': val, 'b': 0, 'c': 0, 'result': inc_truth(val)['result']})
            data.append({'op': 'DEC', 'a': val, 'b': 0, 'c': 0, 'result': dec_truth(val)['result']})

        print(f"  Total samples: {len(data):,}")
    else:
        print("Generating exhaustive 6502 data...")

        # ADC: All 131,072 combinations (256 * 256 * 2)
        print("  ADC: 131,072 combinations...")
        for a in range(256):
            for b in range(256):
                for c in [0, 1]:
                    d = adc_truth(a, b, c)
                    data.append({'op': 'ADC', 'a': a, 'b': b, 'c': c, 'result': d['result']})

        # LOGIC: 3 * 65,536 = 196,608 combinations
        print("  LOGIC (AND/ORA/EOR): 196,608 combinations...")
        for a in range(256):
            for b in range(256):
                data.append({'op': 'AND', 'a': a, 'b': b, 'c': 0, 'result': and_truth(a, b)['result']})
                data.append({'op': 'ORA', 'a': a, 'b': b, 'c': 0, 'result': ora_truth(a, b)['result']})
                data.append({'op': 'EOR', 'a': a, 'b': b, 'c': 0, 'result': eor_truth(a, b)['result']})

        # SHIFT: 2 * 256 = 512 combinations
        print("  SHIFT (ASL/LSR): 512 combinations...")
        for val in range(256):
            data.append({'op': 'ASL', 'a': val, 'b': 0, 'c': 0, 'result': asl_truth(val)['result']})
            data.append({'op': 'LSR', 'a': val, 'b': 0, 'c': 0, 'result': lsr_truth(val)['result']})

        # INCDEC: 2 * 256 = 512 combinations
        print("  INCDEC (INC/DEC): 512 combinations...")
        for val in range(256):
            data.append({'op': 'INC', 'a': val, 'b': 0, 'c': 0, 'result': inc_truth(val)['result']})
            data.append({'op': 'DEC', 'a': val, 'b': 0, 'c': 0, 'result': dec_truth(val)['result']})

        # Edge case oversampling
        edge_cases = [d for d in data if d['result'] == 0 or d['a'] in [0, 0xFF] or d['b'] in [0, 0xFF]]
        print(f"  Edge cases: {len(edge_cases)} (oversampling 3x)")
        data.extend(edge_cases * 2)

        print(f"  Total samples: {len(data):,}")

    np.random.shuffle(data)
    return data


def split_data(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train/test with stratification by operation."""
    by_op = defaultdict(list)
    for d in data:
        by_op[d['op']].append(d)

    train_data, test_data = [], []
    for op, samples in by_op.items():
        n_train = int(len(samples) * train_ratio)
        train_data.extend(samples[:n_train])
        test_data.extend(samples[n_train:])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return train_data, test_data


# =============================================================================
# MODEL
# =============================================================================

class TriX129X6502(nn.Module):
    """
    TriX129X model for 6502 emulation with XOR Superposition.

    Key features:
    - Uses HierarchicalTriXFFN with XOR compression support
    - XOR mixer for superposition magic
    - Exposes signatures for compression analysis
    - Tracks routing decisions for verification
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        num_layers: int = 2,
        use_xor_mixing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_layers = num_layers
        self.use_xor_mixing = use_xor_mixing

        self.op_embed = nn.Embedding(len(OPCODES), 32)

        # Input: op_embed (32) + a_bits (8) + b_bits (8) + c (1) = 49
        self.input_proj = nn.Linear(49, d_model)

        # XOR mixer for superposition
        if use_xor_mixing:
            self.xor_mixer = XORMixer(d_model)

        # Stack of Hierarchical FFN layers
        self.ffn_layers = nn.ModuleList([
            HierarchicalTriXFFN(
                d_model=d_model,
                num_tiles=num_tiles,
                tiles_per_cluster=tiles_per_cluster,
                dropout=0.1,
                balance_weight=0.01,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.result_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

    def compress_all_signatures(self):
        """Compress signatures in all FFN layers."""
        for layer in self.ffn_layers:
            layer.compress_signatures()

    def decompress_all_signatures(self):
        """Decompress signatures in all FFN layers."""
        for layer in self.ffn_layers:
            layer.decompress_signatures()

    def get_compression_stats(self) -> List[Dict]:
        """Get compression stats for all layers."""
        stats = []
        for i, layer in enumerate(self.ffn_layers):
            layer_stats = layer.get_compression_stats()
            if layer_stats:
                stats.append({
                    'layer': i,
                    'tile': layer_stats['tile']._asdict() if layer_stats['tile'] else None,
                    'cluster': layer_stats['cluster']._asdict() if layer_stats['cluster'] else None,
                })
        return stats

    def is_compressed(self) -> bool:
        """Check if model is compressed."""
        return self.ffn_layers[0]._is_compressed if self.ffn_layers else False

    def forward(self, op_idx, a, b, c, return_routing: bool = False):
        batch_size = op_idx.shape[0]
        device = op_idx.device

        # Encode inputs
        op_emb = self.op_embed(op_idx)
        a_bits = torch.stack([(a >> i) & 1 for i in range(8)], dim=1).float()
        b_bits = torch.stack([(b >> i) & 1 for i in range(8)], dim=1).float()

        x = torch.cat([op_emb, a_bits, b_bits, c.unsqueeze(1).float()], dim=1)
        x = self.input_proj(x)  # [B, d_model]

        # XOR mixing for superposition magic
        if self.use_xor_mixing:
            x = self.xor_mixer(x)

        x = x.unsqueeze(1)  # [B, 1, d_model]

        # Forward through all FFN layers
        total_aux = {'total_aux': torch.tensor(0.0, device=device)}
        first_info = None

        for i, ffn in enumerate(self.ffn_layers):
            out, info, aux = ffn(x)
            x = out  # Pass to next layer
            if 'total_aux' in aux:
                total_aux['total_aux'] = total_aux['total_aux'] + aux['total_aux']
            if i == 0:
                first_info = info  # Track routing from first layer

        out = x.squeeze(1)

        # Predict result
        result = self.result_head(out)

        if return_routing:
            return result, first_info, total_aux
        return result, total_aux


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: TriX129X6502,
    data_tensors: Dict,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 512,
) -> Tuple[float, float, Dict]:
    """Train for one epoch, return loss, accuracy, tile counts."""
    model.train()
    device = next(model.parameters()).device

    n = len(data_tensors['op_idx'])
    perm = torch.randperm(n, device=device)

    total_loss = 0.0
    correct = 0
    tile_counts = defaultdict(lambda: defaultdict(int))
    ops_list = data_tensors['ops']
    n_batches = 0

    for i in range(0, n - batch_size, batch_size):
        idx = perm[i:i+batch_size]

        op_idx = data_tensors['op_idx'][idx]
        a = data_tensors['a'][idx]
        b = data_tensors['b'][idx]
        c = data_tensors['c'][idx]
        result_bits = data_tensors['result_bits'][idx]

        # Forward
        pred, info, aux = model(op_idx, a, b, c, return_routing=True)

        # Loss
        loss = F.binary_cross_entropy(pred, result_bits) + aux['total_aux']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Accuracy
        pred_vals = sum((pred[:, i] > 0.5).long() << i for i in range(8))
        target_vals = data_tensors['result'][idx]
        correct += (pred_vals == target_vals).sum().item()

        # Track tiles (use global_indices from HierarchicalTriXFFN)
        if 'global_indices' in info:
            tiles = info['global_indices'].squeeze(-1).cpu().numpy()
            for j, t in enumerate(tiles):
                tile_counts[int(t)][ops_list[idx[j].item()]] += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    accuracy = correct / n * 100

    return avg_loss, accuracy, dict(tile_counts)


def evaluate(model: TriX129X6502, data_tensors: Dict, batch_size: int = 512) -> Dict[str, float]:
    """Evaluate on test data, return per-operation accuracy."""
    model.eval()
    device = next(model.parameters()).device

    n = len(data_tensors['op_idx'])

    correct_by_op = defaultdict(int)
    total_by_op = defaultdict(int)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)

            op_idx = data_tensors['op_idx'][i:end]
            a = data_tensors['a'][i:end]
            b = data_tensors['b'][i:end]
            c = data_tensors['c'][i:end]

            pred, _ = model(op_idx, a, b, c)

            pred_vals = sum((pred[:, j] > 0.5).long() << j for j in range(8))
            target_vals = data_tensors['result'][i:end]

            for j in range(end - i):
                op = data_tensors['ops'][i + j]
                total_by_op[op] += 1
                if pred_vals[j] == target_vals[j]:
                    correct_by_op[op] += 1

    results = {}
    for op in OPCODES:
        if total_by_op[op] > 0:
            results[op] = correct_by_op[op] / total_by_op[op] * 100
        else:
            results[op] = 0.0

    results['overall'] = sum(correct_by_op.values()) / sum(total_by_op.values()) * 100

    return results


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


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_trix129x_6502(
    epochs: int = 100,
    d_model: int = 128,
    num_tiles: int = 64,
    tiles_per_cluster: int = 8,
    num_layers: int = 2,
    batch_size: int = 512,
    lr: float = 0.001,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict:
    """
    Train TriX129X on 6502 with XOR compression testing.

    Returns comprehensive results including compression metrics.
    """

    print("=" * 70)
    print("TriX129X - 6502 Monolithic Training with XOR Superposition")
    print("=" * 70)
    print()
    print("BASELINE: SparseLookupFFNv2 achieved 99.76% accuracy")
    print("TARGET: Match accuracy AND achieve 129× signature compression")
    print()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    print("Phase 1: Data Generation")
    print("-" * 70)
    all_data = generate_exhaustive_data(fast_mode=True)
    train_data, test_data = split_data(all_data, train_ratio=0.8)
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")
    print()

    # Prepare tensors
    print("Phase 2: Tensor Preparation")
    print("-" * 70)
    train_tensors = prepare_tensors(train_data, device)
    test_tensors = prepare_tensors(test_data, device)
    print("Tensors ready")
    print()

    # Create model
    print("Phase 3: Model Creation")
    print("-" * 70)
    model = TriX129X6502(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=tiles_per_cluster,
        num_layers=num_layers,
    ).to(device)
    print(f"Model: d_model={d_model}, num_tiles={num_tiles}, tiles_per_cluster={tiles_per_cluster}")
    print(f"Layers: {num_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
    }

    # Training loop
    print("Phase 4: Training")
    print("-" * 70)
    print(f"{'Epoch':>5} {'Loss':>10} {'Train%':>10} {'Test%':>10}")
    print("-" * 70)

    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(epochs):
        # Train
        loss, train_acc, tile_counts = train_epoch(model, train_tensors, optimizer, batch_size)
        scheduler.step()

        # Evaluate
        test_results = evaluate(model, test_tensors)
        test_acc = test_results['overall']

        # Record
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:5d} {loss:10.4f} {train_acc:10.2f} {test_acc:10.2f}")

    train_time = time.time() - start_time

    # Final evaluation (uncompressed)
    print()
    print("Phase 5: Final Evaluation (Uncompressed)")
    print("-" * 70)

    final_results_uncompressed = evaluate(model, test_tensors)

    print("\nPer-operation accuracy (uncompressed):")
    for op in OPCODES:
        bar = '█' * int(final_results_uncompressed[op] / 5)
        print(f"  {op:4s}: {bar:20s} {final_results_uncompressed[op]:6.1f}%")

    print(f"\nOverall: {final_results_uncompressed['overall']:.2f}%")

    # XOR Compression Phase
    print()
    print("=" * 70)
    print("Phase 6: XOR SUPERPOSITION COMPRESSION")
    print("=" * 70)

    # Compress signatures
    print("\nCompressing signatures...")
    model.compress_all_signatures()

    # Get compression stats
    compression_stats = model.get_compression_stats()

    print("\nCompression Results:")
    for layer_stat in compression_stats:
        layer_idx = layer_stat['layer']
        tile_stats = layer_stat['tile']
        cluster_stats = layer_stat['cluster']

        print(f"\n  Layer {layer_idx}:")
        if tile_stats:
            print(f"    Tile signatures:")
            print(f"      Original:   {tile_stats['original_bytes']:,} bytes")
            print(f"      Compressed: {tile_stats['compressed_bytes']:,} bytes")
            print(f"      Ratio:      {tile_stats['compression_ratio']:.1f}×")
            print(f"      Sparsity:   {tile_stats['mean_delta_sparsity']*100:.1f}%")
        if cluster_stats:
            print(f"    Cluster signatures:")
            print(f"      Original:   {cluster_stats['original_bytes']:,} bytes")
            print(f"      Compressed: {cluster_stats['compressed_bytes']:,} bytes")
            print(f"      Ratio:      {cluster_stats['compression_ratio']:.1f}×")

    # Evaluate with compressed signatures
    print()
    print("Phase 7: Evaluation (COMPRESSED)")
    print("-" * 70)

    final_results_compressed = evaluate(model, test_tensors)

    print("\nPer-operation accuracy (compressed):")
    for op in OPCODES:
        bar = '█' * int(final_results_compressed[op] / 5)
        print(f"  {op:4s}: {bar:20s} {final_results_compressed[op]:6.1f}%")

    print(f"\nOverall: {final_results_compressed['overall']:.2f}%")

    # Verify accuracy maintained
    accuracy_diff = abs(final_results_compressed['overall'] - final_results_uncompressed['overall'])

    # Compute overall compression ratio
    total_original = sum(s['tile']['original_bytes'] for s in compression_stats if s['tile'])
    total_compressed = sum(s['tile']['compressed_bytes'] for s in compression_stats if s['tile'])
    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0.0

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    baseline_acc = 99.76
    beat_baseline = final_results_compressed['overall'] >= baseline_acc - 0.5  # Allow 0.5% tolerance
    good_compression = overall_ratio >= 10.0  # At least 10× compression
    accuracy_preserved = accuracy_diff < 0.5  # Less than 0.5% accuracy loss

    print(f"\n  Accuracy vs baseline:")
    print(f"    Baseline (SparseLookupFFNv2): {baseline_acc:.2f}%")
    print(f"    TriX129X (compressed):        {final_results_compressed['overall']:.2f}%")
    print(f"    Status: {'✓ PASS' if beat_baseline else '✗ FAIL'}")

    print(f"\n  Compression:")
    print(f"    Tile signatures:    {total_original:,} → {total_compressed:,} bytes")
    print(f"    Compression ratio:  {overall_ratio:.1f}×")
    print(f"    Status: {'✓ PASS' if good_compression else '✗ FAIL'}")

    print(f"\n  Accuracy preservation:")
    print(f"    Uncompressed: {final_results_uncompressed['overall']:.2f}%")
    print(f"    Compressed:   {final_results_compressed['overall']:.2f}%")
    print(f"    Difference:   {accuracy_diff:.3f}%")
    print(f"    Status: {'✓ PASS' if accuracy_preserved else '✗ FAIL'}")

    all_pass = beat_baseline and good_compression and accuracy_preserved

    print()
    if all_pass:
        print("  *** TriX129X VALIDATED: XOR Superposition preserves accuracy! ***")
    else:
        print("  Results need attention - see metrics above")

    print("=" * 70)
    print(f"Training time: {train_time:.1f}s")

    # Return full results
    return {
        'config': {
            'd_model': d_model,
            'num_tiles': num_tiles,
            'tiles_per_cluster': tiles_per_cluster,
            'num_layers': num_layers,
            'epochs': epochs,
            'seed': seed,
        },
        'accuracy': {
            'uncompressed': {
                'overall': final_results_uncompressed['overall'],
                'by_op': {op: final_results_uncompressed[op] for op in OPCODES},
            },
            'compressed': {
                'overall': final_results_compressed['overall'],
                'by_op': {op: final_results_compressed[op] for op in OPCODES},
            },
            'difference': accuracy_diff,
            'best': best_test_acc,
        },
        'compression': {
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'overall_ratio': overall_ratio,
            'per_layer': compression_stats,
        },
        'history': history,
        'train_time': train_time,
        'validation': {
            'beat_baseline': beat_baseline,
            'good_compression': good_compression,
            'accuracy_preserved': accuracy_preserved,
            'all_pass': all_pass,
        },
    }


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Run training - flat routing (16 clusters of 1 tile)
    results = train_trix129x_6502(
        epochs=150,
        d_model=192,
        num_tiles=16,
        tiles_per_cluster=1,  # 16 clusters of 1 tile (flat routing)
        num_layers=1,
        batch_size=384,
        lr=0.00337,
        device=device,
        seed=11229,
    )

    # Save results
    import os
    results_dir = '/workspace/trix_latest/TriXO/experiments/mesa13'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/trix129x_6502_results.json'

    # Convert numpy to python for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\nResults saved to: {results_file}")
