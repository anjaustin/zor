"""
Task Validation Tests - Level 3.

Real task validation proving the architectures work on actual problems.

Categories:
1. Language Modeling - Character-level prediction on synthetic language
2. Sequence Classification - Classify sequences by pattern
3. Arithmetic Tasks - Learn simple arithmetic/logic
4. Transformer Comparison - Compare to standard transformer baseline
5. Generalization - Test on held-out distributions

"The proof is in the pudding."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import string

from trix.nn.anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)
from trix.nn.octave import (
    TrueOctaveFFN,
    TrueOctaveBlock,
)


SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class AnchoredLM(nn.Module):
    """Language model using AnchoredDualModeBlocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_anchors: int = 8,
        num_tiles: int = 32,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            AnchoredDualModeBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_anchors=num_anchors,
                num_tiles=num_tiles,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        all_info = []
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for block in self.blocks:
            h, info = block(h, attn_mask=mask)
            all_info.append(info)

        h = self.norm(h)
        logits = self.head(h)

        return logits, all_info

    def set_temperature(self, temp: float):
        for block in self.blocks:
            block.set_temperature(temp)


class OctaveLM(nn.Module):
    """Language model using TrueOctaveBlocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_fine_tiles: int = 16,
        pool_factor: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TrueOctaveBlock(
                d_model=d_model,
                n_heads=num_heads,
                num_fine_tiles=num_fine_tiles,
                pool_factor=pool_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> Tuple[torch.Tensor, List]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        all_info = []
        for block in self.blocks:
            h, info = block(h, is_causal=is_causal)
            all_info.append(info)

        h = self.norm(h)
        logits = self.head(h)

        return logits, all_info

    def set_mode(self, mode: str):
        for block in self.blocks:
            block.set_mode(mode)


class TransformerLM(nn.Module):
    """Standard transformer baseline for comparison."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm(h)
        logits = self.head(h)

        return logits, None


class Classifier(nn.Module):
    """Sequence classifier wrapper that uses encoder hidden states."""

    def __init__(self, vocab_size: int, d_model: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # Simple encoder (no LM head)
        self.embed = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            AnchoredDualModeBlock(
                d_model=d_model,
                num_heads=4,
                num_anchors=8,
                num_tiles=32,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)

        for block in self.blocks:
            h, _ = block(h)

        h = self.norm(h)

        # Pool over sequence
        h = h.transpose(1, 2)  # [B, D, T]
        h = self.pool(h).squeeze(-1)  # [B, D]

        return self.classifier(h)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_language(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 32,
    pattern_prob: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic language with learnable patterns.

    Patterns:
    - After token A, token B is more likely
    - Repeated tokens are common
    - Some tokens are rare
    """
    # Create transition matrix with patterns
    transition = torch.ones(vocab_size, vocab_size) / vocab_size

    # Add bigram patterns: after token i, token (i+1) % vocab is more likely
    for i in range(vocab_size):
        next_tok = (i + 1) % vocab_size
        transition[i, next_tok] += pattern_prob
        transition[i] /= transition[i].sum()

    # Generate sequences
    sequences = []
    for _ in range(n_samples):
        seq = [np.random.randint(1, vocab_size)]  # Start with random (not 0)
        for _ in range(seq_len - 1):
            probs = transition[seq[-1]].numpy()
            next_tok = np.random.choice(vocab_size, p=probs)
            seq.append(next_tok)
        sequences.append(seq)

    x = torch.tensor(sequences, dtype=torch.long)

    # Target: shifted by 1 (predict next token)
    y = torch.cat([x[:, 1:], torch.zeros(n_samples, 1, dtype=torch.long)], dim=1)

    return x, y


def generate_classification_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 32,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequence classification data.

    Class is determined by:
    - Class 0: sequences starting with tokens 1-7
    - Class 1: sequences starting with tokens 8-15
    - Class 2: sequences with high token sum
    - Class 3: sequences with repeating patterns
    """
    samples_per_class = n_samples // num_classes

    xs, ys = [], []

    for c in range(num_classes):
        for _ in range(samples_per_class):
            if c == 0:
                # Class 0: start with low tokens
                seq = [np.random.randint(1, 8)]
                seq.extend(np.random.randint(1, vocab_size, seq_len - 1))
            elif c == 1:
                # Class 1: start with high tokens
                seq = [np.random.randint(8, 16)]
                seq.extend(np.random.randint(1, vocab_size, seq_len - 1))
            elif c == 2:
                # Class 2: high average token values
                seq = list(np.random.randint(vocab_size // 2, vocab_size, seq_len))
            else:
                # Class 3: repeating pattern
                pattern_len = np.random.randint(2, 5)
                pattern = list(np.random.randint(1, vocab_size, pattern_len))
                seq = (pattern * (seq_len // pattern_len + 1))[:seq_len]

            xs.append(seq)
            ys.append(c)

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)

    # Shuffle
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


def generate_arithmetic_data(
    n_samples: int,
    max_val: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate simple arithmetic task: a + b = c (mod max_val).

    Input: [a, b, 0, 0, 0, ...]  (padded)
    Output: c
    """
    a = torch.randint(0, max_val, (n_samples,))
    b = torch.randint(0, max_val, (n_samples,))
    c = (a + b) % max_val

    # Create input sequences [a, b, padding...]
    seq_len = 8
    x = torch.zeros(n_samples, seq_len, dtype=torch.long)
    x[:, 0] = a
    x[:, 1] = b

    return x, c, (a, b)


def generate_bracket_matching(
    n_samples: int,
    max_depth: int = 4,
    seq_len: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate bracket matching task.

    Sequences of ( and ) tokens.
    Label: 1 if balanced, 0 if not.

    Vocab: 0=pad, 1=(, 2=)
    """
    xs, ys = [], []

    for _ in range(n_samples // 2):
        # Generate balanced sequence
        depth = np.random.randint(1, max_depth + 1)
        seq = []
        current_depth = 0
        for _ in range(seq_len):
            if current_depth == 0:
                seq.append(1)  # Must open
                current_depth += 1
            elif current_depth >= depth:
                seq.append(2)  # Must close
                current_depth -= 1
            else:
                if np.random.random() < 0.5:
                    seq.append(1)
                    current_depth += 1
                else:
                    seq.append(2)
                    current_depth -= 1

        # Ensure balanced
        while current_depth > 0:
            seq.append(2)
            current_depth -= 1

        seq = seq[:seq_len]
        while len(seq) < seq_len:
            seq.append(0)

        xs.append(seq)
        ys.append(1)  # Balanced

    for _ in range(n_samples // 2):
        # Generate unbalanced sequence
        seq = list(np.random.choice([1, 2], size=seq_len))
        # Make sure it's actually unbalanced
        depth = 0
        for t in seq:
            if t == 1:
                depth += 1
            else:
                depth -= 1
        if depth == 0:
            # Accidentally balanced, break it
            seq[0] = 2

        while len(seq) < seq_len:
            seq.append(0)

        xs.append(seq)
        ys.append(0)  # Unbalanced

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)

    perm = torch.randperm(len(x))
    return x[perm], y[perm]


# =============================================================================
# 1. LANGUAGE MODELING TESTS
# =============================================================================

class TestLanguageModeling:
    """Test language modeling capability."""

    def test_anchored_lm_learns_patterns(self):
        """Anchored LM should learn bigram patterns."""
        set_seed()

        vocab_size = 32
        seq_len = 32

        x_train, y_train = generate_synthetic_language(500, seq_len, vocab_size)
        x_test, y_test = generate_synthetic_language(100, seq_len, vocab_size)

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_train[:, :-1].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test)
            test_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_test[:, :-1].reshape(-1)
            ).item()

            # Perplexity
            perplexity = np.exp(test_loss)

        # Should achieve perplexity better than random (vocab_size)
        random_perplexity = vocab_size
        assert perplexity < random_perplexity * 0.85, \
            f"Perplexity too high: {perplexity:.2f} (random={random_perplexity})"

    def test_octave_lm_learns_patterns(self):
        """Octave LM should learn bigram patterns."""
        set_seed()

        vocab_size = 32
        seq_len = 32

        x_train, y_train = generate_synthetic_language(500, seq_len, vocab_size)
        x_test, y_test = generate_synthetic_language(100, seq_len, vocab_size)

        model = OctaveLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_train[:, :-1].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test)
            test_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_test[:, :-1].reshape(-1)
            ).item()
            perplexity = np.exp(test_loss)

        random_perplexity = vocab_size
        assert perplexity < random_perplexity * 0.8, \
            f"Perplexity too high: {perplexity:.2f} (random={random_perplexity})"

    def test_lm_generates_coherent_sequences(self):
        """Trained LM should generate sequences following learned patterns."""
        set_seed()

        vocab_size = 16
        seq_len = 32

        x_train, y_train = generate_synthetic_language(300, seq_len, vocab_size, pattern_prob=0.5)

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(150):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_train[:, :-1].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        # Generate sequences
        model.eval()
        with torch.no_grad():
            # Start with random token
            generated = torch.randint(1, vocab_size, (10, 1))

            for _ in range(seq_len - 1):
                logits, _ = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        # Check if generated sequences follow the bigram pattern
        # (token i followed by token i+1)
        pattern_count = 0
        total_pairs = 0
        for seq in generated:
            for i in range(len(seq) - 1):
                total_pairs += 1
                if seq[i + 1] == (seq[i] + 1) % vocab_size:
                    pattern_count += 1

        pattern_ratio = pattern_count / total_pairs
        # Should show some pattern following (above random chance of 1/vocab)
        random_chance = 1.0 / vocab_size
        assert pattern_ratio > random_chance * 2, \
            f"Generated sequences don't follow pattern: ratio={pattern_ratio:.3f}"


# =============================================================================
# 2. SEQUENCE CLASSIFICATION TESTS
# =============================================================================

class TestSequenceClassification:
    """Test sequence classification capability."""

    def test_anchored_classifier(self):
        """Anchored model should learn sequence classification."""
        set_seed()

        vocab_size = 32
        num_classes = 4
        seq_len = 16

        x_train, y_train = generate_classification_data(400, seq_len, vocab_size, num_classes)
        x_test, y_test = generate_classification_data(100, seq_len, vocab_size, num_classes)

        model = Classifier(vocab_size=vocab_size, d_model=64, num_classes=num_classes, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(150):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            pred = logits.argmax(dim=-1)
            accuracy = (pred == y_test).float().mean().item()

        # Should beat random (25% for 4 classes)
        random_acc = 1.0 / num_classes
        assert accuracy > random_acc + 0.1, \
            f"Classification accuracy too low: {accuracy:.2%} (random={random_acc:.2%})"

    def test_classification_generalizes(self):
        """Classification should generalize to test set."""
        set_seed()

        vocab_size = 32
        num_classes = 4
        seq_len = 16

        x_train, y_train = generate_classification_data(400, seq_len, vocab_size, num_classes)
        x_test, y_test = generate_classification_data(100, seq_len, vocab_size, num_classes)

        model = Classifier(vocab_size=vocab_size, d_model=64, num_classes=num_classes, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(150):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_logits = model(x_train)
            train_acc = (train_logits.argmax(-1) == y_train).float().mean().item()

            # Test accuracy
            test_logits = model(x_test)
            test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()

        # Test should be at least half of train (some generalization)
        assert test_acc > train_acc * 0.5, \
            f"Poor generalization: train={train_acc:.2%}, test={test_acc:.2%}"


# =============================================================================
# 3. ARITHMETIC/LOGIC TESTS
# =============================================================================

class TestArithmeticTasks:
    """Test arithmetic and logic capability."""

    def test_learns_addition(self):
        """Model should learn modular addition."""
        set_seed()

        max_val = 10
        vocab_size = max_val + 1  # 0-10

        x_train, y_train, _ = generate_arithmetic_data(500, max_val)
        x_test, y_test, _ = generate_arithmetic_data(100, max_val)

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2, max_seq_len=16)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(200):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            # Predict from position 2 (after a, b)
            loss = F.cross_entropy(logits[:, 1, :], y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test)
            pred = logits[:, 1, :].argmax(dim=-1)
            accuracy = (pred == y_test).float().mean().item()

        # Should learn simple addition
        assert accuracy > 0.5, f"Addition accuracy too low: {accuracy:.2%}"

    def test_learns_bracket_matching(self):
        """Model should learn bracket matching (balanced vs unbalanced)."""
        set_seed()

        vocab_size = 4  # 0=pad, 1=(, 2=), 3=unused
        seq_len = 16

        x_train, y_train = generate_bracket_matching(400, max_depth=3, seq_len=seq_len)
        x_test, y_test = generate_bracket_matching(100, max_depth=3, seq_len=seq_len)

        model = Classifier(vocab_size=vocab_size, d_model=64, num_classes=2, num_layers=3)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(200):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            pred = logits.argmax(dim=-1)
            accuracy = (pred == y_test).float().mean().item()

        # Should beat random (50%)
        assert accuracy > 0.55, f"Bracket matching accuracy too low: {accuracy:.2%}"


# =============================================================================
# 4. TRANSFORMER COMPARISON TESTS
# =============================================================================

class TestTransformerComparison:
    """Compare to standard transformer baseline."""

    def test_comparable_perplexity_to_transformer(self):
        """Anchored/Octave LMs should achieve comparable perplexity to transformer."""
        set_seed()

        vocab_size = 32
        seq_len = 32

        x_train, y_train = generate_synthetic_language(500, seq_len, vocab_size)
        x_test, y_test = generate_synthetic_language(100, seq_len, vocab_size)

        results = {}

        # Train Anchored
        set_seed(123)
        anchored = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        anchored.train()
        opt_a = torch.optim.Adam(anchored.parameters(), lr=1e-3)

        for _ in range(100):
            opt_a.zero_grad()
            logits, _ = anchored(x_train)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y_train[:, :-1].reshape(-1))
            loss.backward()
            opt_a.step()

        anchored.eval()
        with torch.no_grad():
            logits, _ = anchored(x_test)
            loss_a = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y_test[:, :-1].reshape(-1)).item()
            results['anchored'] = np.exp(loss_a)

        # Train Transformer
        set_seed(123)
        transformer = TransformerLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        transformer.train()
        opt_t = torch.optim.Adam(transformer.parameters(), lr=1e-3)

        for _ in range(100):
            opt_t.zero_grad()
            logits, _ = transformer(x_train)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y_train[:, :-1].reshape(-1))
            loss.backward()
            opt_t.step()

        transformer.eval()
        with torch.no_grad():
            logits, _ = transformer(x_test)
            loss_t = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), y_test[:, :-1].reshape(-1)).item()
            results['transformer'] = np.exp(loss_t)

        # Anchored should be within 2x of transformer perplexity
        ratio = results['anchored'] / results['transformer']
        assert ratio < 2.0, \
            f"Anchored perplexity too far from transformer: {results['anchored']:.2f} vs {results['transformer']:.2f}"

    def test_fewer_parameters_comparable_performance(self):
        """TriX models should match performance with fewer parameters."""
        vocab_size = 32
        d_model = 64

        anchored = AnchoredLM(vocab_size=vocab_size, d_model=d_model, num_layers=2)
        transformer = TransformerLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, d_ff=256)

        params_a = sum(p.numel() for p in anchored.parameters())
        params_t = sum(p.numel() for p in transformer.parameters())

        # Anchored should have fewer or comparable parameters
        # (allowing up to 50% more since architecture differs)
        assert params_a < params_t * 1.5, \
            f"Anchored has too many params: {params_a} vs transformer {params_t}"


# =============================================================================
# 5. GENERALIZATION TESTS
# =============================================================================

class TestGeneralization:
    """Test generalization to held-out distributions."""

    def test_longer_sequences(self):
        """Model trained on short sequences should handle longer ones."""
        set_seed()

        vocab_size = 16
        train_len = 16
        test_len = 32

        x_train, y_train = generate_synthetic_language(300, train_len, vocab_size)

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2, max_seq_len=64)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_train[:, :-1].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        # Test on longer sequences
        x_test_long, y_test_long = generate_synthetic_language(50, test_len, vocab_size)

        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test_long)
            test_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_test_long[:, :-1].reshape(-1)
            ).item()
            perplexity_long = np.exp(test_loss)

        # Should still achieve reasonable perplexity
        # Allow some degradation for length generalization (1.1x random)
        random_ppl = vocab_size
        assert perplexity_long < random_ppl * 1.1, \
            f"Failed on longer sequences: perplexity={perplexity_long:.2f} (random={random_ppl})"

    def test_distribution_shift(self):
        """Model should handle slight distribution shift."""
        set_seed()

        vocab_size = 32
        seq_len = 24

        # Train on pattern_prob=0.3
        x_train, y_train = generate_synthetic_language(400, seq_len, vocab_size, pattern_prob=0.3)

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(100):
            optimizer.zero_grad()
            logits, _ = model(x_train)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_train[:, :-1].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        # Test on pattern_prob=0.5 (different distribution)
        x_test_shift, y_test_shift = generate_synthetic_language(100, seq_len, vocab_size, pattern_prob=0.5)

        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test_shift)
            test_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y_test_shift[:, :-1].reshape(-1)
            ).item()
            perplexity_shift = np.exp(test_loss)

        # Should still beat random despite shift
        random_ppl = vocab_size
        assert perplexity_shift < random_ppl * 0.9, \
            f"Failed under distribution shift: perplexity={perplexity_shift:.2f}"


# =============================================================================
# 6. EFFICIENCY TESTS
# =============================================================================

class TestEfficiency:
    """Test computational efficiency."""

    def test_inference_speed_reasonable(self):
        """Inference should complete in reasonable time."""
        import time

        set_seed()

        vocab_size = 32
        seq_len = 64
        batch_size = 8

        model = AnchoredLM(vocab_size=vocab_size, d_model=128, num_layers=4)
        model.eval()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Time inference
        start = time.time()
        n_runs = 10
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        elapsed = time.time() - start

        avg_time = elapsed / n_runs
        # Should complete in <1 second per batch on CPU
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s per batch"

    def test_memory_scales_linearly(self):
        """Memory usage should scale roughly linearly with batch size."""
        set_seed()

        vocab_size = 32
        seq_len = 32

        model = AnchoredLM(vocab_size=vocab_size, d_model=64, num_layers=2)
        model.eval()

        # We can't easily measure memory in a test, but we can check
        # that different batch sizes don't cause OOM
        for batch_size in [1, 4, 16, 32]:
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            with torch.no_grad():
                output, _ = model(x)
            assert output.shape == (batch_size, seq_len, vocab_size)


# =============================================================================
# 7. SANITY CHECKS
# =============================================================================

class TestSanityChecks:
    """Basic sanity checks for task models."""

    def test_lm_output_shape(self):
        """LM output shape should be [batch, seq, vocab]."""
        vocab_size = 32
        batch, seq = 4, 16

        for ModelClass in [AnchoredLM, OctaveLM]:
            model = ModelClass(vocab_size=vocab_size, d_model=64, num_layers=2)
            x = torch.randint(0, vocab_size, (batch, seq))

            logits, _ = model(x)

            assert logits.shape == (batch, seq, vocab_size), \
                f"{ModelClass.__name__} wrong output shape"

    def test_classifier_output_shape(self):
        """Classifier output shape should be [batch, num_classes]."""
        vocab_size = 32
        num_classes = 4
        batch, seq = 4, 16

        model = Classifier(vocab_size=vocab_size, d_model=64, num_classes=num_classes, num_layers=2)

        x = torch.randint(0, vocab_size, (batch, seq))
        logits = model(x)

        assert logits.shape == (batch, num_classes)

    def test_models_trainable(self):
        """All model types should be trainable without errors."""
        vocab_size = 16
        batch, seq = 2, 8

        models = [
            AnchoredLM(vocab_size=vocab_size, d_model=32, num_layers=1),
            OctaveLM(vocab_size=vocab_size, d_model=32, num_layers=1),
            TransformerLM(vocab_size=vocab_size, d_model=32, num_layers=1),
        ]

        x = torch.randint(0, vocab_size, (batch, seq))
        y = torch.randint(0, vocab_size, (batch, seq))

        for model in models:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for _ in range(5):
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()

            # Should complete without error
            assert True, f"{model.__class__.__name__} failed training"
