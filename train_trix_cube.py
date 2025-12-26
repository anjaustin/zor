#!/usr/bin/env python3
"""
TriX LLM with CUBIC TILE LAYERS

Instead of flat tiles, each layer is a 4x4x4 cube:
- Axis 0: Sequence chunks (4 chunks of seq_len/4)
- Axis 1: Feature groups (4 groups of d_model/4)
- Axis 2: Representation depth (4 levels)

Tiles interact with 3D neighbors - like a cellular automaton.
Context flows through the cube structure.
"""

import numpy as np
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trix.native.vulkan.shapes import GilliesFrozenShapes
from trix.data.slimpajama import SlimPajamaLoader, ByteTokenizer


class CubicTile:
    """
    A tile in a 3D cube structure.

    Each tile has:
    - Position (x, y, z) in the cube
    - Learnable polynomial coefficients
    - Connections to 6 neighbors (±x, ±y, ±z)
    """

    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 6):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        # coeffs maps from basis to output
        self.coeffs = np.random.randn(out_dim, n_basis).astype(np.float32) * 0.1

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply polynomial transformation.
        a, b: [batch, in_dim/2] each
        returns: [batch, out_dim]
        """
        # Build polynomial basis - shape [batch, n_basis]
        basis = np.stack([
            np.ones(a.shape[0]),          # 1
            a.mean(axis=-1),              # mean(a)
            b.mean(axis=-1),              # mean(b)
            (a * b).mean(axis=-1),        # XOR-like interaction
            (a * a).mean(axis=-1),        # a²
            (b * b).mean(axis=-1),        # b²
        ], axis=-1)  # [batch, 6]

        # Linear combination of basis functions
        output = basis @ self.coeffs.T  # [batch, out_dim]
        return np.tanh(output)


class CubicLayer:
    """
    A 4x4x4 cube of tiles.

    Structure:
    - cube[seq_chunk, feat_group, depth] = tile
    - Each tile processes a portion of the input
    - Neighbors exchange information via XOR mixing
    """

    def __init__(self, d_model: int, seq_len: int, cube_size: int = 4):
        self.d_model = d_model
        self.seq_len = seq_len
        self.cube_size = cube_size

        self.chunk_size = seq_len // cube_size
        self.feat_size = d_model // cube_size

        # Output dim per tile (what each tile produces)
        self.tile_out_dim = max(self.feat_size, 8)

        # 4x4x4 cube of tiles
        self.tiles = {}
        for x in range(cube_size):
            for y in range(cube_size):
                for z in range(cube_size):
                    # in_dim is feat_size, out_dim is tile_out_dim
                    self.tiles[(x, y, z)] = CubicTile(self.feat_size, self.tile_out_dim)

        # Neighbor mixing weights (learned)
        self.neighbor_weights = np.random.randn(6).astype(np.float32) * 0.1

        # Output projection
        self.proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.05

    def _get_neighbors(self, x: int, y: int, z: int) -> list:
        """Get valid neighbor coordinates."""
        neighbors = []
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.cube_size and 0 <= ny < self.cube_size and 0 <= nz < self.cube_size:
                neighbors.append((nx, ny, nz))
        return neighbors

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process input through the cubic structure.

        x: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model]

        The cube structure:
        - sx (0..3): which sequence chunk
        - fy (0..3): which feature group
        - dz (0..3): depth level (multiple passes within same region)
        """
        batch, seq, d = x.shape
        cs = self.cube_size

        # Reshape into cube structure
        # [batch, 4 seq_chunks, chunk_size, 4 feat_groups, feat_size]
        x_cube = x.reshape(batch, cs, self.chunk_size, cs, self.feat_size)

        # Process each tile and collect outputs
        # Key: (sx, fy), Value: accumulated output [batch, chunk_size, tile_out_dim]
        accumulated = {}

        for coords, tile in self.tiles.items():
            sx, fy, dz = coords

            # Get input for this tile (sx=sequence chunk, fy=feature group)
            tile_input = x_cube[:, sx, :, fy, :]  # [batch, chunk_size, feat_size]
            tile_input_flat = tile_input.reshape(batch * self.chunk_size, self.feat_size)

            # Split features for polynomial: first half = a, second half = b
            half = self.feat_size // 2
            a = 1.0 / (1.0 + np.exp(-tile_input_flat[:, :half]))
            b = 1.0 / (1.0 + np.exp(-tile_input_flat[:, half:]))

            # Apply tile polynomial
            out = tile.forward(a, b)  # [batch*chunk_size, tile_out_dim]
            out = out.reshape(batch, self.chunk_size, self.tile_out_dim)

            # Accumulate outputs from all depth levels for each (sx, fy)
            key = (sx, fy)
            if key not in accumulated:
                accumulated[key] = out / cs  # Average over depth
            else:
                accumulated[key] = accumulated[key] + out / cs

        # Mix with neighbors using XOR-style combination
        mixed_outputs = {}
        for (sx, fy), out in accumulated.items():
            neighbor_sum = np.zeros_like(out)
            n_neighbors = 0

            # Check 2D neighbors (in sequence and feature dimensions)
            for dsx, dfy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nsx, nfy = sx + dsx, fy + dfy
                if (nsx, nfy) in accumulated:
                    w = np.exp(self.neighbor_weights[n_neighbors % 6])
                    neighbor_sum = neighbor_sum + w * accumulated[(nsx, nfy)]
                    n_neighbors += 1

            # Mix: self + weighted neighbor contribution
            if n_neighbors > 0:
                mixed = out + 0.1 * neighbor_sum / n_neighbors
            else:
                mixed = out
            mixed_outputs[(sx, fy)] = mixed

        # Reassemble into output tensor
        # Output shape matches input: [batch, seq_len, d_model]
        output = np.zeros((batch, seq, d), dtype=np.float32)

        for (sx, fy), out in mixed_outputs.items():
            # out shape: [batch, chunk_size, tile_out_dim]
            seq_start = sx * self.chunk_size
            seq_end = seq_start + self.chunk_size
            feat_start = fy * self.feat_size
            feat_end = min(feat_start + self.tile_out_dim, d)
            actual_width = feat_end - feat_start

            output[:, seq_start:seq_end, feat_start:feat_end] = out[:, :, :actual_width]

        # Final projection
        output = output @ self.proj

        return output


class TriXCubeLM:
    """
    TriX LLM with cubic tile layers.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 seq_len: int = 64, n_layers: int = 4, cube_size: int = 4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.cube_size = cube_size

        # Embeddings
        self.token_emb = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1
        self.pos_emb = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

        # Cubic layers
        self.layers = [
            CubicLayer(d_model, seq_len, cube_size)
            for _ in range(n_layers)
        ]

        # Output projection
        self.out_proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

        # GILLIES
        self.shapes = GilliesFrozenShapes(max_elements=1_000_000)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch, seq = input_ids.shape

        # Embed
        x = self.token_emb[input_ids] + self.pos_emb[:seq]

        # Apply cubic layers
        for layer in self.layers:
            x_norm = self._layer_norm(x)
            out = layer.forward(x_norm)
            x = x + out * (0.5 / np.sqrt(self.n_layers))

        # Final norm and projection
        x = self._layer_norm(x)
        x = x @ self.out_proj

        # LM head
        logits = x @ self.token_emb.T
        return logits

    def compute_loss(self, input_ids: np.ndarray, target_ids: np.ndarray) -> Tuple[float, Dict]:
        batch, seq = input_ids.shape

        logits = self.forward(input_ids)

        # Cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_ids.reshape(-1)

        logits_max = logits_flat.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Label smoothing
        smooth = 0.1
        probs_smooth = probs * (1 - smooth) + smooth / self.vocab_size

        log_probs = np.log(probs_smooth + 1e-10)
        loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()

        # Gradients (simplified)
        d_logits = probs.copy()
        d_logits[np.arange(len(targets_flat)), targets_flat] -= 1.0
        d_logits /= (batch * seq)

        # Embedding gradient
        x = self.token_emb[input_ids] + self.pos_emb[:seq]
        for layer in self.layers:
            x_norm = self._layer_norm(x)
            out = layer.forward(x_norm)
            x = x + out * (0.5 / np.sqrt(self.n_layers))
        x = self._layer_norm(x)
        final_x = x @ self.out_proj

        x_flat = final_x.reshape(-1, self.d_model)
        d_emb_head = d_logits.T @ x_flat

        d_x = d_logits @ self.token_emb
        d_emb_input = np.zeros_like(self.token_emb)
        for i, tok in enumerate(input_ids.reshape(-1)):
            d_emb_input[tok] += d_x[i]

        # Update cube layers (simplified gradient)
        d_final_x = (d_logits @ self.token_emb).reshape(batch, seq, self.d_model)
        scale = 0.5 / np.sqrt(self.n_layers)

        for layer in self.layers:
            # Update projection
            d_proj = np.einsum('bsd,bse->de', x, d_final_x * scale) / (batch * seq)
            layer.proj -= 0.03 * d_proj

            # Update tile coefficients
            for coords, tile in layer.tiles.items():
                d_tile = d_final_x.mean() * scale * 0.01
                tile.coeffs -= 0.03 * d_tile * np.ones_like(tile.coeffs)

        # Update output projection
        d_out = np.einsum('bsd,bse->de', x, d_final_x) / (batch * seq)
        self.out_proj -= 0.03 * d_out

        return float(loss), {'emb': d_emb_head + d_emb_input}

    def update(self, grads: Dict, lr: float):
        self.token_emb -= lr * grads['emb']

        # Keep embeddings bounded
        norms = np.linalg.norm(self.token_emb, axis=1, keepdims=True)
        scale = np.minimum(1.0, 3.0 / (norms + 1e-8))
        self.token_emb = self.token_emb * scale

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        tokenizer = ByteTokenizer()
        prompt_ids = list(tokenizer.encode(prompt))
        generated = prompt_ids.copy()
        recent = []

        for _ in range(max_new_tokens):
            context = generated[-self.seq_len:]
            if len(context) < self.seq_len:
                context = [ord(' ')] * (self.seq_len - len(context)) + context

            context = np.array([context], dtype=np.int32)
            logits = self.forward(context)
            last_logits = logits[0, -1].copy()

            # Repetition penalty
            for tok in recent[-15:]:
                last_logits[tok] /= 1.2

            last_logits = last_logits / max(temperature, 0.01)

            probs = np.exp(last_logits - last_logits.max())
            probs = probs / probs.sum()

            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            cutoff = max(cutoff, 3)

            top_idx = sorted_idx[:cutoff]
            top_probs = probs[top_idx]
            top_probs = top_probs / top_probs.sum()

            next_token = int(np.random.choice(top_idx, p=top_probs))
            generated.append(next_token)
            recent.append(next_token)

        return tokenizer.decode(np.array(generated[len(prompt_ids):]))

    def close(self):
        self.shapes.close()


def train(config):
    print("=" * 70)
    print("TriX LLM with CUBIC TILE LAYERS")
    print("4x4x4 = 64 tiles per layer, 3D neighbor interactions")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    loader = SlimPajamaLoader(
        path=config.get('data_path'),
        seq_len=config['seq_len'],
        batch_size=config['batch_size']
    )

    # Create model
    print("\nInitializing model...")
    model = TriXCubeLM(
        vocab_size=256,
        d_model=config['d_model'],
        seq_len=config['seq_len'],
        n_layers=config['n_layers'],
        cube_size=config['cube_size']
    )

    n_tiles = config['cube_size'] ** 3
    n_params = (
        256 * config['d_model'] +  # token_emb
        config['seq_len'] * config['d_model'] +  # pos_emb
        config['d_model'] * config['d_model'] +  # out_proj
        config['n_layers'] * (
            n_tiles * 8 * 6 +  # tile coeffs (approx)
            config['d_model'] * config['d_model'] +  # layer proj
            6  # neighbor weights
        )
    )

    print(f"Parameters: ~{n_params:,}")
    print(f"Layers: {config['n_layers']} cubic layers")
    print(f"Cube: {config['cube_size']}x{config['cube_size']}x{config['cube_size']} = {n_tiles} tiles/layer")

    # Training
    print("\nTraining...")
    print(f"{'Step':>6} {'Loss':>10} {'Tokens/s':>12} {'Time':>10}")
    print("-" * 44)

    start_time = time.perf_counter()
    tokens_processed = 0
    losses = []

    for step in range(1, config['max_steps'] + 1):
        input_ids, target_ids = loader.get_batch()

        # Learning rate schedule
        warmup = 100
        if step < warmup:
            lr = config['learning_rate'] * step / warmup
        else:
            lr = config['learning_rate'] * 0.5 ** ((step - warmup) / 2000)

        loss, grads = model.compute_loss(input_ids, target_ids)
        model.update(grads, lr)

        tokens_processed += config['batch_size'] * config['seq_len']
        losses.append(loss)

        if step % 100 == 0:
            elapsed = time.perf_counter() - start_time
            tps = tokens_processed / elapsed
            avg_loss = np.mean(losses[-100:])
            print(f"{step:>6} {avg_loss:>10.4f} {tps:>12,.0f} {elapsed:>10.1f}s")

        if step % 500 == 0:
            print("\n--- Sample Generation ---")
            for prompt in ["The ", "In the "]:
                output = model.generate(prompt, max_new_tokens=50, temperature=0.6)
                print(f"'{prompt}' -> '{output[:60]}...'")
            print()

    # Final stats
    elapsed = time.perf_counter() - start_time
    print()
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Final loss: {np.mean(losses[-100:]):.4f}")
    print(f"Throughput: {tokens_processed / elapsed:,.0f} tokens/sec")

    # Final generation
    print("\n--- Final Generation ---")
    for prompt in ["The quick ", "Once upon ", "Machine learning "]:
        output = model.generate(prompt, max_new_tokens=60, temperature=0.7)
        print(f"\n'{prompt}' -> '{output}'")

    model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--cube", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()

    config = {
        'data_path': args.data,
        'max_steps': args.steps,
        'batch_size': args.batch,
        'seq_len': args.seq,
        'n_layers': args.layers,
        'cube_size': args.cube,
        'd_model': args.dim,
        'learning_rate': 0.528,
    }

    train(config)
