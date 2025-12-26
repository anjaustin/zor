"""
SlimPajama-style data loading for TriX LLM.

Supports:
1. Local text files
2. Streaming from HuggingFace (if available)
3. Synthetic data for testing

Usage:
    loader = SlimPajamaLoader(path="./data", seq_len=512, batch_size=8)
    for batch in loader:
        input_ids, target_ids = batch
        # train...
"""

import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import json
import os


class ByteTokenizer:
    """Simple byte-level tokenizer - no dependencies."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> np.ndarray:
        return np.array([b % self.vocab_size for b in text.encode('utf-8')], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        return bytes([i % 256 for i in ids]).decode('utf-8', errors='replace')


class SlimPajamaLoader:
    """
    Data loader for SlimPajama-style training.

    Loads text from:
    1. Local .txt files
    2. Local .jsonl files (with 'text' field)
    3. Falls back to synthetic data if nothing found
    """

    def __init__(
        self,
        path: Optional[str] = None,
        seq_len: int = 512,
        batch_size: int = 8,
        vocab_size: int = 256,
        shuffle: bool = True,
        max_tokens: Optional[int] = None
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.max_tokens = max_tokens

        self.tokenizer = ByteTokenizer(vocab_size)
        self.tokens = self._load_data(path)

        print(f"SlimPajamaLoader: {len(self.tokens):,} tokens loaded")

    def _load_data(self, path: Optional[str]) -> np.ndarray:
        """Load and tokenize data from path."""
        texts = []

        if path is None:
            # Use built-in sample data
            return self._get_sample_data()

        path = Path(path)

        if path.is_file():
            texts.extend(self._load_file(path))
        elif path.is_dir():
            # Load all text/jsonl files in directory
            for ext in ['*.txt', '*.jsonl', '*.json']:
                for file in sorted(path.glob(ext)):
                    texts.extend(self._load_file(file))

        if not texts:
            print(f"Warning: No data found at {path}, using sample data")
            return self._get_sample_data()

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.append(tokens)

            if self.max_tokens and sum(len(t) for t in all_tokens) >= self.max_tokens:
                break

        return np.concatenate(all_tokens)

    def _load_file(self, path: Path) -> List[str]:
        """Load text from a single file."""
        texts = []

        try:
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'text' in data:
                                texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                            elif isinstance(item, str):
                                texts.append(item)
                    elif isinstance(data, dict) and 'text' in data:
                        texts.append(data['text'])
            else:  # .txt or other
                with open(path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())

            print(f"  Loaded {len(texts)} texts from {path.name}")

        except Exception as e:
            print(f"  Error loading {path}: {e}")

        return texts

    def _get_sample_data(self) -> np.ndarray:
        """Generate sample data for testing."""
        sample_texts = [
            # Classic pangram
            "The quick brown fox jumps over the lazy dog. " * 100,

            # Common phrases
            "To be or not to be, that is the question. " * 50,
            "All that glitters is not gold. " * 50,
            "A journey of a thousand miles begins with a single step. " * 50,

            # Technical text
            "The neural network processes input data through multiple layers. " * 50,
            "Machine learning models learn patterns from training data. " * 50,

            # Stories
            "Once upon a time, in a land far away, there lived a wise old wizard. " * 30,
            "The stars twinkled in the night sky as the travelers continued their journey. " * 30,

            # Code-like
            "def hello_world(): print('Hello, World!') " * 50,
            "for i in range(10): x = x + i " * 50,
        ]

        all_text = "\n".join(sample_texts)
        return self.tokenizer.encode(all_text)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.tokens) - self.seq_len - 1) // (self.batch_size * self.seq_len)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        n_tokens = len(self.tokens)
        n_batches = len(self)

        if self.shuffle:
            # Random batch starting positions
            max_start = n_tokens - self.seq_len - 1
            starts = np.random.randint(0, max_start, size=n_batches * self.batch_size)
        else:
            # Sequential
            starts = np.arange(0, n_tokens - self.seq_len - 1, self.seq_len)
            starts = starts[:n_batches * self.batch_size]

        for batch_idx in range(n_batches):
            batch_starts = starts[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

            input_ids = np.array([
                self.tokens[s:s + self.seq_len]
                for s in batch_starts
            ], dtype=np.int32)

            target_ids = np.array([
                self.tokens[s + 1:s + self.seq_len + 1]
                for s in batch_starts
            ], dtype=np.int32)

            yield input_ids, target_ids

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single random batch."""
        max_start = len(self.tokens) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)

        input_ids = np.array([
            self.tokens[s:s + self.seq_len]
            for s in starts
        ], dtype=np.int32)

        target_ids = np.array([
            self.tokens[s + 1:s + self.seq_len + 1]
            for s in starts
        ], dtype=np.int32)

        return input_ids, target_ids


def download_slimpajama_sample(output_dir: str = "./data", n_samples: int = 1000):
    """
    Download a sample from SlimPajama (requires datasets library).

    Falls back to creating synthetic data if download fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_file = output_path / "slimpajama_sample.jsonl"

    if sample_file.exists():
        print(f"Sample already exists at {sample_file}")
        return str(sample_file)

    try:
        from datasets import load_dataset

        print(f"Downloading {n_samples} samples from SlimPajama...")
        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True
        )

        with open(sample_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(ds):
                if i >= n_samples:
                    break
                f.write(json.dumps({'text': example['text']}) + '\n')
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{n_samples}")

        print(f"Saved to {sample_file}")
        return str(sample_file)

    except Exception as e:
        print(f"Could not download SlimPajama: {e}")
        print("Creating synthetic sample instead...")

        # Create synthetic data
        synthetic_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we process information.",
            "Python is a versatile programming language used in many domains.",
            "Neural networks consist of layers of interconnected nodes.",
            "Data science combines statistics, programming, and domain expertise.",
        ] * (n_samples // 5)

        with open(sample_file, 'w', encoding='utf-8') as f:
            for text in synthetic_texts[:n_samples]:
                f.write(json.dumps({'text': text}) + '\n')

        print(f"Created synthetic sample at {sample_file}")
        return str(sample_file)


if __name__ == "__main__":
    # Test the loader
    print("Testing SlimPajamaLoader...")

    # Test with sample data
    loader = SlimPajamaLoader(seq_len=64, batch_size=4)

    print(f"\nTotal batches: {len(loader)}")

    for i, (input_ids, target_ids) in enumerate(loader):
        if i >= 3:
            break
        print(f"\nBatch {i}:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  target_ids shape: {target_ids.shape}")
        print(f"  Sample text: '{loader.tokenizer.decode(input_ids[0, :30])}...'")

    print("\n✓ SlimPajamaLoader working!")
