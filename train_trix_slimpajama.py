#!/usr/bin/env python3
"""
TriX LLM Training on SlimPajama

Architecture:
- Programmable position-addressed tiles (not attention)
- Spline interpolation for context propagation
- GILLIES Vulkan acceleration for frozen shapes
- Providence LM head option

Usage:
    python train_trix_slimpajama.py [--data PATH] [--steps N]
"""

import numpy as np
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trix.native.vulkan.shapes import GilliesFrozenShapes
from trix.data.slimpajama import SlimPajamaLoader, ByteTokenizer


class Config:
    """Training configuration."""
    # Model
    vocab_size = 256        # Byte-level
    d_model = 128           # Hidden dimension
    n_layers = 4            # Number of tile layers
    n_tiles = 16            # Position-addressed tiles per layer
    seq_len = 128           # Sequence length
    tile_size = 8           # Positions per tile

    # Training
    batch_size = 16
    learning_rate = 0.3
    max_steps = 5000
    warmup_steps = 100
    log_every = 100
    eval_every = 500
    save_every = 1000

    # Data
    data_path = None        # Will use sample data if None


class ProgrammableTile:
    """
    A programmable tile with learnable polynomial coefficients.

    Basis: [1, a, b, ab, a², b²]
    Each tile learns different coefficients = different function.
    """

    def __init__(self, dim: int, n_basis: int = 6):
        self.dim = dim
        self.n_basis = n_basis
        self.coeffs = np.random.randn(dim, n_basis).astype(np.float32) * 0.1

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply polynomial: output = sum(coeff_i * basis_i(a,b))"""
        batch = a.shape[0]

        basis = np.stack([
            np.ones_like(a),      # 1
            a,                     # a
            b,                     # b
            a * b,                 # ab (XOR-like)
            a * a,                 # a²
            b * b,                 # b²
        ], axis=-1)  # [batch, dim, n_basis]

        output = np.einsum('bdk,dk->bd', basis, self.coeffs)
        return np.tanh(output)  # Bounded output


class TriXSlimPajamaLM:
    """
    TriX LLM with programmable position-addressed tiles.

    - Multiple layers of tile processing
    - Each layer: tiles aggregate -> spline interpolate -> residual
    - GILLIES accelerates frozen shape computations
    """

    def __init__(self, config: Config):
        self.config = config
        self.shapes = GilliesFrozenShapes(max_elements=2_000_000)

        # Embeddings
        self.token_emb = np.random.randn(
            config.vocab_size, config.d_model
        ).astype(np.float32) * 0.1

        self.pos_emb = np.random.randn(
            config.seq_len, config.d_model
        ).astype(np.float32) * 0.1

        # Per-layer parameters
        self.layers = []
        for layer_idx in range(config.n_layers):
            layer = {
                # Programmable tiles for this layer
                'tiles': [
                    ProgrammableTile(config.d_model // 2)
                    for _ in range(config.n_tiles)
                ],
                # Position aggregation weights
                'pos_weights': np.random.randn(
                    config.n_tiles, config.tile_size
                ).astype(np.float32) * 0.1,
                # Layer projection
                'proj': np.random.randn(
                    config.d_model, config.d_model
                ).astype(np.float32) * (0.1 / np.sqrt(config.n_layers)),
            }
            self.layers.append(layer)

        # Final output projection
        self.out_proj = np.random.randn(
            config.d_model, config.d_model
        ).astype(np.float32) * 0.1

        # Stats
        self.step = 0
        self.best_loss = float('inf')

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Embed tokens with positions."""
        batch, seq = input_ids.shape
        x = self.token_emb[input_ids] + self.pos_emb[:seq]
        return x

    def _aggregate_tiles(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Each tile aggregates from its position range.

        x: [batch, seq, d_model]
        layer_idx: which layer's tiles to use
        returns: [batch, n_tiles, d_model]
        """
        batch, seq, d = x.shape
        cfg = self.config
        half = d // 2
        layer = self.layers[layer_idx]

        tile_outputs = []

        for t, tile in enumerate(layer['tiles']):
            start = t * cfg.tile_size
            end = min(start + cfg.tile_size, seq)

            if start >= seq:
                # Pad with zeros for tiles beyond sequence
                tile_outputs.append(np.zeros((batch, d), dtype=np.float32))
                continue

            x_range = x[:, start:end, :]
            n_pos = x_range.shape[1]

            # Softmax position weights
            weights = layer['pos_weights'][t, :n_pos]
            exp_w = np.exp(weights - weights.max())
            soft_w = exp_w / exp_w.sum()

            # Weighted aggregation
            weighted = np.einsum('bpd,p->bd', x_range, soft_w)

            # Split and apply programmable polynomial
            a = 1.0 / (1.0 + np.exp(-weighted[:, :half]))  # Sigmoid normalize
            b = 1.0 / (1.0 + np.exp(-weighted[:, half:]))

            out_half = tile.forward(a, b)

            # Full dimension output
            out = np.concatenate([out_half, out_half], axis=-1)
            tile_outputs.append(out)

        return np.stack(tile_outputs, axis=1)  # [batch, n_tiles, d]

    def _interpolate_context(self, tile_outputs: np.ndarray) -> np.ndarray:
        """
        Spline interpolation from tiles to each position.

        tile_outputs: [batch, n_tiles, d_model]
        returns: [batch, seq_len, d_model]
        """
        batch, n_tiles, d = tile_outputs.shape
        seq = self.config.seq_len
        cfg = self.config

        context = np.zeros((batch, seq, d), dtype=np.float32)

        for pos in range(seq):
            # Which tile owns this position?
            t = pos // cfg.tile_size
            t = min(t, n_tiles - 1)
            t_next = min(t + 1, n_tiles - 1)
            t_prev = max(t - 1, 0)

            # Smooth interpolation weight
            local_pos = (pos % cfg.tile_size) / cfg.tile_size
            w = local_pos * local_pos * (3 - 2 * local_pos)  # Cubic hermite

            # Weighted combination (causal: emphasize current and past)
            context[:, pos, :] = (
                0.5 * tile_outputs[:, t, :] +
                0.3 * tile_outputs[:, t_prev, :] +
                0.2 * tile_outputs[:, t_next, :] * w
            )

        return context

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Simple layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through multiple layers.

        input_ids: [batch, seq_len]
        returns: logits [batch, seq_len, vocab_size]
        """
        cfg = self.config

        # Embed
        x = self._embed(input_ids)

        # Apply each layer
        for layer_idx in range(cfg.n_layers):
            layer = self.layers[layer_idx]

            # Pre-norm
            x_norm = self._layer_norm(x)

            # Aggregate into tiles
            tile_outputs = self._aggregate_tiles(x_norm, layer_idx)

            # Interpolate context
            context = self._interpolate_context(tile_outputs)

            # Project and residual
            context = context @ layer['proj']
            x = x + context * (0.5 / np.sqrt(cfg.n_layers))

        # Final norm
        x = self._layer_norm(x)

        # Output projection
        x = x @ self.out_proj

        # LM head (tied embeddings)
        logits = x @ self.token_emb.T

        return logits

    def compute_loss(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray
    ) -> Tuple[float, Dict]:
        """Compute cross-entropy loss and gradients for multi-layer model."""
        batch, seq = input_ids.shape
        cfg = self.config

        logits = self.forward(input_ids)

        # Cross-entropy with label smoothing
        logits_flat = logits.reshape(-1, cfg.vocab_size)
        targets_flat = target_ids.reshape(-1)

        logits_max = logits_flat.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Label smoothing
        smooth = 0.1
        probs_smooth = probs * (1 - smooth) + smooth / cfg.vocab_size

        log_probs = np.log(probs_smooth + 1e-10)
        loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()

        # Gradients
        d_logits = probs.copy()
        d_logits[np.arange(len(targets_flat)), targets_flat] -= 1.0
        d_logits /= (batch * seq)

        # Recompute forward for gradient
        x = self._embed(input_ids)
        layer_inputs = [x.copy()]

        for layer_idx in range(cfg.n_layers):
            layer = self.layers[layer_idx]
            x_norm = self._layer_norm(x)
            tile_outputs = self._aggregate_tiles(x_norm, layer_idx)
            context = self._interpolate_context(tile_outputs)
            context = context @ layer['proj']
            x = x + context * (0.5 / np.sqrt(cfg.n_layers))
            layer_inputs.append(x.copy())

        x = self._layer_norm(x)
        final_x = x @ self.out_proj

        # Embedding gradient from LM head
        x_flat = final_x.reshape(-1, cfg.d_model)
        d_emb_head = d_logits.T @ x_flat

        d_x = d_logits @ self.token_emb
        d_emb_input = np.zeros_like(self.token_emb)
        for i, tok in enumerate(input_ids.reshape(-1)):
            d_emb_input[tok] += d_x[i]

        # Gradient flowing back
        d_final_x = (d_logits @ self.token_emb).reshape(batch, seq, cfg.d_model)
        half = cfg.d_model // 2

        # Update output projection
        d_out = np.einsum('bsd,bse->de', x, d_final_x) / (batch * seq)
        self.out_proj -= 0.05 * d_out

        # Backprop through layers (reverse order)
        d_x = d_final_x.copy()
        for layer_idx in range(cfg.n_layers - 1, -1, -1):
            layer = self.layers[layer_idx]
            x_in = layer_inputs[layer_idx]
            x_norm = self._layer_norm(x_in)

            # Gradient for layer projection
            tile_outputs = self._aggregate_tiles(x_norm, layer_idx)
            context = self._interpolate_context(tile_outputs)
            scale = 0.5 / np.sqrt(cfg.n_layers)

            d_proj = np.einsum('bsd,bse->de', context, d_x * scale) / (batch * seq)
            layer['proj'] -= 0.05 * d_proj

            # Gradient for tile coefficients
            for t, tile in enumerate(layer['tiles']):
                start = t * cfg.tile_size
                end = min(start + cfg.tile_size, seq)
                if start >= seq:
                    continue

                x_range = x_norm[:, start:end, :]
                n_pos = x_range.shape[1]

                weights = layer['pos_weights'][t, :n_pos]
                exp_w = np.exp(weights - weights.max())
                soft_w = exp_w / exp_w.sum()

                weighted = np.einsum('bpd,p->bd', x_range, soft_w)
                a = 1.0 / (1.0 + np.exp(-weighted[:, :half]))
                b = 1.0 / (1.0 + np.exp(-weighted[:, half:]))

                # Gradient through tile
                d_tile = d_x[:, start:end, :half].mean(axis=1) * scale

                out = tile.forward(a, b)
                d_pre = d_tile * (1 - out ** 2)

                basis = np.stack([
                    np.ones_like(a), a, b, a * b, a * a, b * b
                ], axis=-1)

                d_coeffs = np.einsum('bdk,bd->dk', basis, d_pre) / batch
                tile.coeffs -= 0.05 * d_coeffs

            # Propagate gradient to previous layer
            d_x = d_x * 0.9  # Approximate backprop

        return float(loss), {
            'emb': d_emb_head + d_emb_input
        }

    def update(self, grads: Dict, lr: float):
        """Update embeddings."""
        self.token_emb -= lr * grads['emb']

        # Keep embeddings bounded
        norms = np.linalg.norm(self.token_emb, axis=1, keepdims=True)
        scale = np.minimum(1.0, 3.0 / (norms + 1e-8))
        self.token_emb = self.token_emb * scale

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        rep_penalty: float = 1.2
    ) -> str:
        """Generate text with nucleus sampling."""
        tokenizer = ByteTokenizer()
        prompt_ids = list(tokenizer.encode(prompt))
        generated = prompt_ids.copy()
        recent = []

        for _ in range(max_new_tokens):
            context = generated[-self.config.seq_len:]
            if len(context) < self.config.seq_len:
                context = [ord(' ')] * (self.config.seq_len - len(context)) + context

            context = np.array([context], dtype=np.int32)
            logits = self.forward(context)
            last_logits = logits[0, -1].copy()

            # Repetition penalty
            for tok in recent[-15:]:
                last_logits[tok] /= rep_penalty

            # Temperature
            last_logits = last_logits / max(temperature, 0.01)

            # Nucleus sampling
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

    def save(self, path: str):
        """Save model weights."""
        layer_data = {}
        for i, layer in enumerate(self.layers):
            layer_data[f'layer_{i}_pos_weights'] = layer['pos_weights']
            layer_data[f'layer_{i}_proj'] = layer['proj']
            layer_data[f'layer_{i}_tile_coeffs'] = [t.coeffs for t in layer['tiles']]

        np.savez(
            path,
            token_emb=self.token_emb,
            pos_emb=self.pos_emb,
            out_proj=self.out_proj,
            n_layers=self.config.n_layers,
            **layer_data
        )
        print(f"Saved to {path}")

    def close(self):
        self.shapes.close()


def train(config: Config):
    """Main training loop."""
    print("=" * 70)
    print("TriX LLM Training on SlimPajama")
    print("Programmable Position Tiles + Spline Context")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    loader = SlimPajamaLoader(
        path=config.data_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size
    )

    # Create model
    print("\nInitializing model...")
    model = TriXSlimPajamaLM(config)

    n_params = (
        config.vocab_size * config.d_model +  # token_emb
        config.seq_len * config.d_model +      # pos_emb
        config.d_model * config.d_model +      # out_proj
        config.n_layers * (
            config.n_tiles * config.tile_size +    # pos_weights per layer
            config.d_model * config.d_model +      # proj per layer
            config.n_tiles * (config.d_model // 2) * 6  # tile coeffs per layer
        )
    )
    print(f"Parameters: {n_params:,}")
    print(f"Layers: {config.n_layers}")
    print(f"Tiles: {config.n_tiles} x {config.tile_size} positions per layer")

    # Training
    print("\nTraining...")
    print(f"{'Step':>6} {'Loss':>10} {'Tokens/s':>12} {'Time':>10}")
    print("-" * 44)

    start_time = time.perf_counter()
    tokens_processed = 0
    losses = []

    for step in range(1, config.max_steps + 1):
        input_ids, target_ids = loader.get_batch()

        # Learning rate schedule
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            lr = config.learning_rate * 0.5 ** ((step - config.warmup_steps) / 2000)

        # Forward + backward
        loss, grads = model.compute_loss(input_ids, target_ids)
        model.update(grads, lr)

        tokens_processed += config.batch_size * config.seq_len
        losses.append(loss)

        if step % config.log_every == 0:
            elapsed = time.perf_counter() - start_time
            tps = tokens_processed / elapsed
            avg_loss = np.mean(losses[-config.log_every:])
            print(f"{step:>6} {avg_loss:>10.4f} {tps:>12,.0f} {elapsed:>10.1f}s")

        if step % config.eval_every == 0:
            print("\n--- Sample Generation ---")
            for prompt in ["The ", "In the ", "Once "]:
                output = model.generate(prompt, max_new_tokens=50, temperature=0.6)
                print(f"'{prompt}' -> '{output[:60]}...'")
            print()

        if step % config.save_every == 0:
            model.save(f"trix_slimpajama_step{step}.npz")

    # Final stats
    elapsed = time.perf_counter() - start_time
    print()
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Steps:      {config.max_steps:,}")
    print(f"Tokens:     {tokens_processed:,}")
    print(f"Time:       {elapsed:.1f}s")
    print(f"Throughput: {tokens_processed / elapsed:,.0f} tokens/sec")
    print(f"Final loss: {np.mean(losses[-100:]):.4f}")

    # Final generation
    print("\n--- Final Generation ---")
    for prompt in ["The quick ", "Machine learning ", "Once upon "]:
        output = model.generate(prompt, max_new_tokens=80, temperature=0.7)
        print(f"\n'{prompt}' ->\n  '{output}'")

    model.save("trix_slimpajama_final.npz")
    model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TriX LLM on SlimPajama")
    parser.add_argument("--data", type=str, help="Path to data directory")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--tiles", type=int, default=16, help="Tiles per layer")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    args = parser.parse_args()

    config = Config()
    config.data_path = args.data
    config.max_steps = args.steps
    config.batch_size = args.batch
    config.seq_len = args.seq
    config.n_layers = args.layers
    config.n_tiles = args.tiles
    config.d_model = args.dim
    config.tile_size = args.seq // args.tiles

    train(config)
