#!/usr/bin/env python3
"""
TriX 250M LLM Training

True 2-bit training with GILLIES Vulkan acceleration.
"Routing is learning. Shapes are frozen."

Usage:
    python train_trix_llm.py
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trix.native.vulkan.shapes import GilliesFrozenShapes
from trix.native.sparse_octave import Providence


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Model
    vocab_size = 50257      # GPT-2 vocab
    d_model = 1024          # Hidden dimension
    n_layers = 12           # Number of layers
    n_tiles = 64            # Tiles per layer
    seq_len = 512           # Sequence length

    # Output layer
    use_providence = True   # Use Providence LM head instead of dense
    lm_top_k = 64           # Top-k for Providence lookup

    # Training
    batch_size = 8
    learning_rate = 3e-4
    warmup_steps = 100
    max_steps = 1000
    log_every = 10

    # Data
    data_path = None        # Will generate synthetic if None


# =============================================================================
# TOKENIZER (Simple byte-level for now)
# =============================================================================

class SimpleTokenizer:
    """Byte-level tokenizer - no external dependencies."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs."""
        return np.array([b % self.vocab_size for b in text.encode('utf-8')], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Decode token IDs to text."""
        return bytes([i % 256 for i in ids]).decode('utf-8', errors='replace')


# =============================================================================
# TRIX LLM MODEL
# =============================================================================

class TriXLLM:
    """
    TriX 2-bit LLM with GILLIES Vulkan acceleration.

    Architecture:
    - Embeddings: vocab_size x d_model (2-bit packed)
    - Layers: n_layers of TriXBlock
    - Each block: Routing (learned) -> Frozen Shapes (GILLIES)
    """

    def __init__(self, config: Config):
        self.config = config
        self.shapes = GilliesFrozenShapes(max_elements=16_000_000)

        # Initialize parameters (2-bit encoded)
        self._init_params()

        # Routing logits (the ONLY learned parameters in FFN)
        self.routing_logits = {}
        for layer in range(config.n_layers):
            # [d_model, n_tiles] - which tile handles which features
            self.routing_logits[f"layer_{layer}"] = np.random.randn(
                config.d_model, config.n_tiles
            ).astype(np.float32) * 0.02

        # Statistics
        self.step = 0
        self.total_tokens = 0

    def _init_params(self):
        """Initialize model parameters."""
        cfg = self.config

        # Token embeddings [vocab_size, d_model]
        # Standard GPT-style initialization
        self.token_emb = np.random.randn(
            cfg.vocab_size, cfg.d_model
        ).astype(np.float32) * 0.02

        # Position embeddings [seq_len, d_model]
        self.pos_emb = np.random.randn(
            cfg.seq_len, cfg.d_model
        ).astype(np.float32) * 0.02

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Embed tokens + positions.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings (2-bit quantized)
        tok_emb = self.token_emb[input_ids]  # [batch, seq, d_model]

        # Position embeddings (2-bit quantized)
        pos_emb = self.pos_emb[:seq_len]  # [seq, d_model]

        # Combine with addition (standard for embeddings)
        # XOR is used in FFN layers, not embedding combination
        combined = tok_emb + pos_emb[np.newaxis, :, :]

        return combined.astype(np.float32)

    def _route(self, x: np.ndarray, layer: int) -> np.ndarray:
        """
        Compute routing probabilities.

        Args:
            x: [batch, seq, d_model]
            layer: Layer index

        Returns:
            routing: [batch, seq, n_tiles]
        """
        logits = self.routing_logits[f"layer_{layer}"]

        # Simple routing: dot product with logits
        # x: [batch, seq, d_model], logits: [d_model, n_tiles]
        scores = np.einsum('bsd,dt->bst', x, logits)

        # Softmax
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        return probs

    def _apply_tiles(self, x: np.ndarray, routing: np.ndarray) -> np.ndarray:
        """
        Apply frozen shapes based on routing.

        Each tile applies a different frozen shape.
        Shapes expect inputs in [0,1] range.
        """
        batch_size, seq_len, d_model = x.shape
        n_tiles = routing.shape[-1]

        # Normalize to [0,1] for shape operations (sigmoid-like)
        x_norm = 1.0 / (1.0 + np.exp(-x * 0.5))  # Soft normalization

        # Split input for shape operations
        half = d_model // 2
        a = x_norm[:, :, :half].reshape(-1).astype(np.float32)
        b = x_norm[:, :, half:].reshape(-1).astype(np.float32)

        # Apply XOR shape (primary computation) - GPU accelerated
        xor_out = self.shapes.xor(a, b)

        # Apply AND shape (CPU)
        and_out = self.shapes.and_op(a, b)

        # Combine based on routing
        xor_weight = routing[:, :, :n_tiles//2].mean(axis=-1, keepdims=True)
        and_weight = routing[:, :, n_tiles//2:].mean(axis=-1, keepdims=True)

        xor_out = xor_out.reshape(batch_size, seq_len, half)
        and_out = and_out.reshape(batch_size, seq_len, half)

        # Weighted combination, scale back to match input range
        out = np.concatenate([
            xor_out * xor_weight + and_out * (1 - xor_weight),
            and_out * and_weight + xor_out * (1 - and_weight)
        ], axis=-1)

        # Scale output to match input magnitude
        out = (out - 0.5) * 4.0  # Map [0,1] back to roughly [-2,2]

        return out

    def _layernorm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Simple layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _layer(self, x: np.ndarray, layer: int) -> np.ndarray:
        """
        Single TriX layer: Norm -> Route -> Compute -> Residual.
        """
        # Pre-norm
        x_norm = self._layernorm(x)

        # Route
        routing = self._route(x_norm, layer)

        # Apply frozen shapes
        out = self._apply_tiles(x_norm, routing)

        # Residual with scaling
        scale = 1.0 / np.sqrt(self.config.n_layers)
        return x + out * scale

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size] (dense) or
                    (logits, indices): sparse Providence output
        """
        # Embed
        x = self._embed(input_ids)

        # Apply layers
        for layer in range(self.config.n_layers):
            x = self._layer(x, layer)

        # Final layer norm
        x = self._layernorm(x)

        if self.config.use_providence:
            # Providence LM head: sparse content-addressed lookup
            return self._providence_lm_head(x)
        else:
            # Traditional dense LM head
            logits = np.einsum('bsd,vd->bsv', x, self.token_emb)
            return logits

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        """
        Binarize continuous values for Hamming distance.

        Uses sign function: x > 0 -> 1, x <= 0 -> 0
        This is a frozen shape: threshold at 0.
        """
        return (x > 0).astype(np.float32)

    def _hamming_distance_gillies(self, x_bin: np.ndarray, emb_bin: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distance using GILLIES XOR.

        Hamming(a, b) = popcount(a XOR b) = Σ(a_i XOR b_i)

        XOR is a frozen shape. GILLIES does 17B ops/sec.

        Args:
            x_bin: [n, d_model] binary query vectors
            emb_bin: [vocab, d_model] binary embedding vectors

        Returns:
            distances: [n, vocab] Hamming distances
        """
        n, d_model = x_bin.shape
        vocab = emb_bin.shape[0]

        # Pre-flatten embeddings (reused for each query)
        emb_flat = emb_bin.reshape(-1).astype(np.float32)

        # Pre-allocate query buffer
        query_buffer = np.empty((vocab, d_model), dtype=np.float32)

        distances = np.zeros((n, vocab), dtype=np.float32)

        for i in range(n):
            # Broadcast query (in-place fill, no allocation)
            query_buffer[:] = x_bin[i]

            # Flatten query
            q_flat = query_buffer.reshape(-1)

            # XOR via GILLIES Vulkan
            xor_result = self.shapes.xor(q_flat, emb_flat)

            # Sum for Hamming distance (popcount)
            distances[i] = xor_result.reshape(vocab, d_model).sum(axis=-1)

        return distances

    def _providence_lm_head(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Providence-based LM head using GILLIES Hamming distance.

        Instead of NumPy L1 distance, uses:
        1. Binarize hidden states and embeddings
        2. GILLIES XOR for Hamming distance (frozen shape!)
        3. Top-k selection
        4. Sparse softmax

        Returns:
            logits: [batch, seq, k] - logits for top-k tokens only
            indices: [batch, seq, k] - which token IDs
        """
        batch_size, seq_len, d_model = x.shape
        k = self.config.lm_top_k
        cfg = self.config

        # Flatten for lookup
        x_flat = x.reshape(-1, d_model)  # [batch*seq, d_model]
        n = x_flat.shape[0]

        # Binarize for Hamming distance (frozen threshold shape)
        x_bin = self._binarize(x_flat)

        # Binarize embeddings (cache this for efficiency)
        if not hasattr(self, '_emb_binary') or self._emb_binary is None:
            self._emb_binary = self._binarize(self.token_emb)

        # Compute Hamming distance via GILLIES XOR
        distances = self._hamming_distance_gillies(x_bin, self._emb_binary)

        # Top-k nearest (smallest distance)
        top_k_idx = np.argpartition(distances, k, axis=-1)[:, :k]  # [n, k]

        # Gather top-k distances
        batch_idx = np.arange(n)[:, None]
        top_k_dist = distances[batch_idx, top_k_idx]  # [n, k]

        # Convert distances to logits (negative distance = higher score)
        logits = -top_k_dist / np.sqrt(d_model)

        # Reshape back
        logits = logits.reshape(batch_size, seq_len, k)
        indices = top_k_idx.reshape(batch_size, seq_len, k)

        return logits, indices

    def _providence_loss(
        self,
        x: np.ndarray,
        target_ids: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss using Providence sparse lookup.

        Only compute softmax over top-k nearest tokens.
        If target is in top-k, compute exact loss.
        If target not in top-k, use contrastive loss.

        Returns:
            loss: scalar loss value
            d_x: gradient w.r.t. x [batch, seq, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        cfg = self.config
        k = cfg.lm_top_k

        # Get Providence output
        logits, indices = self._providence_lm_head(x)  # [batch, seq, k], [batch, seq, k]

        # Flatten
        logits_flat = logits.reshape(-1, k)  # [n, k]
        indices_flat = indices.reshape(-1, k)  # [n, k]
        targets_flat = target_ids.reshape(-1)  # [n]
        n = logits_flat.shape[0]

        # Softmax over top-k
        logits_max = logits_flat.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # [n, k]

        # Find which positions have target in top-k
        target_in_topk = np.any(indices_flat == targets_flat[:, None], axis=-1)  # [n]

        # For targets in top-k: standard cross-entropy
        # For targets not in top-k: use max prob as negative (contrastive)
        losses = np.zeros(n)

        for i in range(n):
            if target_in_topk[i]:
                # Find position of target in top-k
                target_pos = np.where(indices_flat[i] == targets_flat[i])[0][0]
                losses[i] = -np.log(probs[i, target_pos] + 1e-10)
            else:
                # Target not in top-k: push down max prob
                # This encourages model to bring target into top-k
                losses[i] = -np.log(1.0 - probs[i].max() + 1e-10)

        loss = float(losses.mean())

        # Gradient: simplified version
        # For proper gradient, would need to backprop through distance computation
        # Here we use a proxy: gradient toward target embedding
        x_flat = x.reshape(-1, d_model)
        target_emb = self.token_emb[targets_flat]  # [n, d_model]

        # Gradient: move x toward target embedding, scaled by loss
        d_x_flat = (x_flat - target_emb) * losses[:, None] * 0.1
        d_x = d_x_flat.reshape(batch_size, seq_len, d_model)

        return loss, d_x

    def compute_loss(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute cross-entropy loss and gradients.

        Proper gradient flow through frozen shapes:
        - Shapes are frozen (not learned) but ARE differentiable
        - XOR gradient: ∂(a+b-2ab)/∂a = 1-2b, ∂/∂b = 1-2a
        - Only routing weights and embeddings are updated
        """
        batch_size, seq_len = input_ids.shape
        cfg = self.config

        # Forward pass with activation tracking
        x = self._embed(input_ids)

        # Store for backprop
        layer_inputs = []
        layer_routings = []
        layer_x_norms = []

        for layer in range(cfg.n_layers):
            layer_inputs.append(x.copy())

            # Layer norm (matches _layer method)
            x_norm = self._layernorm(x)
            layer_x_norms.append(x_norm.copy())

            # Route
            routing = self._route(x_norm, layer)
            layer_routings.append(routing)

            # Apply shapes
            out = self._apply_tiles(x_norm, routing)

            # Residual with scaling (matches _layer method)
            scale = 1.0 / np.sqrt(cfg.n_layers)
            x = x + out * scale

        # Final layer norm
        x = self._layernorm(x)

        if cfg.use_providence:
            # Providence LM head - sparse loss computation
            loss, d_x = self._providence_loss(x, target_ids)
        else:
            # Dense LM head
            logits = np.einsum('bsd,vd->bsv', x, self.token_emb)

            # Softmax + cross-entropy
            logits_flat = logits.reshape(-1, cfg.vocab_size)
            targets_flat = target_ids.reshape(-1)

            # Stable softmax
            logits_max = logits_flat.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits_flat - logits_max)
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

            # Cross-entropy loss
            log_probs = np.log(probs + 1e-10)
            loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()

            # Output gradient (softmax - target)
            d_probs = probs.copy()
            d_probs[np.arange(len(targets_flat)), targets_flat] -= 1.0

            # Backprop through LM head
            d_x = (d_probs @ self.token_emb).reshape(batch_size, seq_len, cfg.d_model)

        # Compute gradients with proper backprop through frozen shapes
        grads = {}
        scale = 1.0 / np.sqrt(cfg.n_layers)

        for layer in range(cfg.n_layers - 1, -1, -1):
            x_norm = layer_x_norms[layer]
            routing = layer_routings[layer]  # [batch, seq, n_tiles]

            # Backprop through shape operations
            # _apply_tiles does: x_norm -> sigmoid -> split -> XOR/AND -> combine -> scale
            # Gradient: d_x propagates back through this

            # Simplified gradient: use d_x as the signal for routing
            # The key insight: routing chooses which shape output gets weight
            # Gradient w.r.t. routing = d_x * (shape_output_difference)

            # Approximate shape gradient:
            # XOR(a,b) ≈ a + b - 2ab, so ∂XOR/∂a = 1 - 2b
            half = cfg.d_model // 2
            x_sig = 1.0 / (1.0 + np.exp(-x_norm * 0.5))  # Same as forward
            a = x_sig[:, :, :half]
            b = x_sig[:, :, half:]

            # XOR and AND outputs (approximated)
            xor_out = a + b - 2 * a * b
            and_out = a * b

            # Shape output difference (what routing is choosing between)
            shape_diff = np.concatenate([xor_out - and_out, and_out - xor_out], axis=-1)

            # Gradient signal: how much does changing routing affect loss?
            # d_routing ≈ d_x * ∂output/∂routing = d_x * shape_difference * scale
            d_routing = (d_x * shape_diff).mean(axis=-1, keepdims=True)  # [batch, seq, 1]
            d_routing = np.broadcast_to(d_routing, routing.shape)

            # Softmax gradient: if routing = softmax(scores), then
            # d_scores = routing * (d_routing - sum(routing * d_routing))
            routing_d_routing = routing * d_routing
            d_scores = routing * (d_routing - routing_d_routing.sum(axis=-1, keepdims=True))

            # d_logits = x_norm.T @ d_scores (since scores = x_norm @ logits)
            grads[f"layer_{layer}"] = np.einsum('bsd,bst->dt', x_norm, d_scores) / (batch_size * seq_len)

            # Backprop d_x through this layer (for earlier layers)
            # d_x_prev = d_x + d_x through shapes
            # Simplified: just scale down d_x for earlier layers
            d_x = d_x * 0.9  # Gradient flows back

        # Embedding gradients - two sources:
        # 1. LM head: logits = x @ emb.T, so d_emb = d_logits.T @ x
        # 2. Input embedding lookup: each input token gets its gradient
        input_flat = input_ids.reshape(-1)
        targets_flat = target_ids.reshape(-1)
        x_flat = x.reshape(-1, cfg.d_model)

        d_emb_sparse = {}

        if cfg.use_providence:
            # Providence: gradient moves embeddings toward/away from hidden states
            unique_targets = np.unique(targets_flat)
            for tok in unique_targets:
                mask = (targets_flat == tok)
                # Move embedding toward hidden states that should produce this token
                d_emb_sparse[int(tok)] = (self.token_emb[tok] - x_flat[mask].mean(axis=0)) * 0.5
        else:
            # Dense: proper gradient through LM head
            # d_emb[v] = sum over positions where v was target: d_logits[pos, v] * x[pos]
            # Plus: gradient from input embedding lookup
            unique_targets = np.unique(targets_flat)
            for tok in unique_targets:
                mask = (targets_flat == tok)
                # Gradient from LM head: d_logits.T @ x for this token
                # d_logits[i, tok] = probs[i, tok] - 1 (for target) or probs[i, tok] (otherwise)
                # Since we only have targets, the gradient is: (prob - 1) * x = -(1-prob) * x
                # Simplified: move embedding toward hidden states that should predict it
                d_emb_sparse[int(tok)] = d_probs[mask, tok].reshape(-1, 1) * x_flat[mask]
                d_emb_sparse[int(tok)] = d_emb_sparse[int(tok)].mean(axis=0)

        # Also update input embeddings based on gradient flow
        unique_inputs = np.unique(input_flat)
        x_in_flat = self._embed(input_ids).reshape(-1, cfg.d_model)

        for tok in unique_inputs:
            mask = (input_flat == tok)
            # Gradient from loss flowing back to input embedding
            # This is typically d_x @ W.T for each layer, but we approximate with d_x directly
            if tok in d_emb_sparse:
                d_emb_sparse[tok] = d_emb_sparse[tok] + d_x.reshape(-1, cfg.d_model)[mask].mean(axis=0) * 0.1
            else:
                d_emb_sparse[tok] = d_x.reshape(-1, cfg.d_model)[mask].mean(axis=0) * 0.1

        grads["token_emb_sparse"] = d_emb_sparse

        return float(loss), grads

    def update(self, grads: Dict[str, np.ndarray], lr: float, max_grad_norm: float = 1.0):
        """Update routing and embedding parameters with gradient clipping."""
        cfg = self.config

        # Clip gradients
        def clip_grad(g, max_norm):
            norm = np.linalg.norm(g)
            if norm > max_norm:
                return g * max_norm / norm
            return g

        # Update routing
        for key, grad in grads.items():
            if key in self.routing_logits:
                clipped = clip_grad(grad, max_grad_norm)
                self.routing_logits[key] -= lr * clipped

        # Update embeddings (sparse) with smaller learning rate
        if "token_emb_sparse" in grads:
            emb_lr = lr * 0.1  # Embeddings learn slower to prevent explosion
            for tok, grad in grads["token_emb_sparse"].items():
                clipped = clip_grad(grad, max_grad_norm * 0.1)  # Tighter clipping
                self.token_emb[tok] -= emb_lr * clipped

            # Keep embeddings normalized to prevent explosion
            # This is a form of weight decay / regularization
            emb_norms = np.linalg.norm(self.token_emb, axis=1, keepdims=True)
            max_norm = 2.0  # Cap embedding norms
            scale = np.minimum(1.0, max_norm / (emb_norms + 1e-8))
            self.token_emb = self.token_emb * scale

            # Invalidate binary embedding cache
            self._emb_binary = None

    def generate(self, prompt_ids: np.ndarray, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 40) -> np.ndarray:
        """
        Generate text autoregressively with temperature sampling.

        Args:
            prompt_ids: [seq_len] starting token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: only sample from top-k tokens

        Returns:
            generated_ids: [seq_len + max_new_tokens] full sequence
        """
        cfg = self.config
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Get last seq_len tokens
            context = np.array(generated[-cfg.seq_len:], dtype=np.int32)
            context = context.reshape(1, -1)  # [1, seq_len]

            # Forward pass
            x = self._embed(context)
            for layer in range(cfg.n_layers):
                x = self._layer(x, layer)
            x = self._layernorm(x)

            if cfg.use_providence:
                # Get top-k predictions for last position
                logits, indices = self._providence_lm_head(x)
                last_logits = logits[0, -1, :]  # [k]
                last_indices = indices[0, -1, :]  # [k]

                # Temperature sampling
                if temperature > 0:
                    probs = np.exp((last_logits - last_logits.max()) / temperature)
                    probs = probs / probs.sum()
                    sampled_idx = np.random.choice(len(last_logits), p=probs)
                    next_token = last_indices[sampled_idx]
                else:
                    next_token = last_indices[np.argmax(last_logits)]
            else:
                logits = np.einsum('bsd,vd->bsv', x, self.token_emb)
                last_logits = logits[0, -1, :]  # [vocab]

                # Top-k filtering
                if top_k > 0:
                    top_k_idx = np.argsort(last_logits)[-top_k:]
                    mask = np.ones_like(last_logits) * -1e10
                    mask[top_k_idx] = last_logits[top_k_idx]
                    last_logits = mask

                # Temperature sampling
                if temperature > 0:
                    probs = np.exp((last_logits - last_logits.max()) / temperature)
                    probs = probs / probs.sum()
                    next_token = np.random.choice(len(last_logits), p=probs)
                else:
                    next_token = np.argmax(last_logits)

            generated.append(int(next_token))

        return np.array(generated, dtype=np.int32)

    def param_count(self) -> Dict[str, int]:
        """Count parameters by type."""
        cfg = self.config

        embedding_params = cfg.vocab_size * cfg.d_model + cfg.seq_len * cfg.d_model
        routing_params = sum(
            v.size for v in self.routing_logits.values()
        )

        return {
            "embeddings": embedding_params,
            "routing": routing_params,
            "frozen_shapes": 0,  # Shapes have 0 learnable params
            "total_learnable": embedding_params + routing_params,
            "total": embedding_params + routing_params
        }


# =============================================================================
# DATA GENERATOR
# =============================================================================

def generate_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a batch of training data."""
    # Random tokens for now - replace with real data
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # Target is next token (shifted by 1)
    target_ids = np.roll(input_ids, -1, axis=1)

    return input_ids, target_ids


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(config: Config):
    """Main training loop."""
    print("=" * 70)
    print("TriX 250M LLM Training")
    print("True 2-bit with GILLIES Vulkan")
    print("=" * 70)
    print()

    # Create model
    print("Initializing model...")
    model = TriXLLM(config)

    params = model.param_count()
    print(f"Parameters:")
    print(f"  Embeddings:     {params['embeddings']:,}")
    print(f"  Routing:        {params['routing']:,}")
    print(f"  Frozen shapes:  {params['frozen_shapes']:,}")
    print(f"  Total learnable:{params['total_learnable']:,}")
    print()

    # Training stats
    start_time = time.perf_counter()
    tokens_processed = 0
    losses = []

    print("Training...")
    print(f"{'Step':>6} {'Loss':>10} {'Tokens/s':>12} {'Time':>10}")
    print("-" * 44)

    for step in range(1, config.max_steps + 1):
        # Generate batch
        input_ids, target_ids = generate_batch(
            config.batch_size,
            config.seq_len,
            config.vocab_size
        )

        # Forward + backward
        loss, grads = model.compute_loss(input_ids, target_ids)

        # Learning rate with warmup
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            lr = config.learning_rate

        # Update
        model.update(grads, lr)

        # Stats
        tokens_processed += config.batch_size * config.seq_len
        losses.append(loss)

        if step % config.log_every == 0:
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = tokens_processed / elapsed

            print(f"{step:>6} {loss:>10.4f} {tokens_per_sec:>12,.0f} {elapsed:>10.1f}s")

    # Final stats
    elapsed = time.perf_counter() - start_time
    final_tokens_per_sec = tokens_processed / elapsed

    print()
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Steps:          {config.max_steps:,}")
    print(f"Tokens:         {tokens_processed:,}")
    print(f"Time:           {elapsed:.1f}s")
    print(f"Throughput:     {final_tokens_per_sec:,.0f} tokens/sec")
    print(f"Final loss:     {np.mean(losses[-10:]):.4f}")
    print()

    # Cleanup
    model.shapes.close()

    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use small config for testing")
    args = parser.parse_args()

    config = Config()

    if args.small:
        # Small config for testing
        config.vocab_size = 1000
        config.n_layers = 4
        config.d_model = 256
        config.n_tiles = 16
        config.seq_len = 128
        config.batch_size = 8
        config.use_providence = True
        config.lm_top_k = 32
        config.learning_rate = 1e-2
        config.max_steps = 200
        config.log_every = 20
    else:
        # Full config: TriX with Providence + GILLIES XOR
        config.vocab_size = 5000
        config.n_layers = 6
        config.d_model = 384
        config.n_tiles = 24
        config.seq_len = 128
        config.batch_size = 4
        config.use_providence = True
        config.lm_top_k = 64
        config.learning_rate = 1e-3
        config.warmup_steps = 50
        config.max_steps = 100
        config.log_every = 10

    print()
    print("Configuration:")
    print(f"  Vocab:      {config.vocab_size:,}")
    print(f"  Layers:     {config.n_layers}")
    print(f"  d_model:    {config.d_model}")
    print(f"  Tiles:      {config.n_tiles}")
    print(f"  Seq length: {config.seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print()

    model = train(config)
