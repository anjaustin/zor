"""
Tests for Chip → Backend integration.
"""

import pytest
from trix.forge import Chip
from trix.forge.backend import CPUBackend, CUDABackend


# =============================================================================
# CHIP EXECUTE
# =============================================================================

class TestChipExecute:
    """Tests for Chip.execute() direct backend execution."""

    def test_execute_xor(self):
        """Test direct XOR execution."""
        chip = Chip.alu(["xor"], bits=8)
        assert chip.execute(0xAA, 0x55, "xor") == 0xFF
        assert chip.execute(0xFF, 0xFF, "xor") == 0x00
        assert chip.execute(0x00, 0x00, "xor") == 0x00

    def test_execute_and(self):
        """Test direct AND execution."""
        chip = Chip.alu(["and"], bits=8)
        assert chip.execute(0xFF, 0xF0, "and") == 0xF0
        assert chip.execute(0xAA, 0x55, "and") == 0x00
        assert chip.execute(0xFF, 0xFF, "and") == 0xFF

    def test_execute_or(self):
        """Test direct OR execution."""
        chip = Chip.alu(["or"], bits=8)
        assert chip.execute(0xF0, 0x0F, "or") == 0xFF
        assert chip.execute(0x00, 0x00, "or") == 0x00
        assert chip.execute(0xAA, 0x55, "or") == 0xFF

    def test_execute_not(self):
        """Test direct NOT execution."""
        chip = Chip.alu(["not"], bits=8)
        assert chip.execute(0x00, 0, "not") == 0xFF
        assert chip.execute(0xFF, 0, "not") == 0x00
        assert chip.execute(0xAA, 0, "not") == 0x55

    def test_execute_by_opcode(self):
        """Test execution by opcode number."""
        chip = Chip.alu(["xor", "and", "or"], bits=8)
        # opcode 0 = xor
        assert chip.execute(0xAA, 0x55, 0) == 0xFF
        # opcode 1 = and
        assert chip.execute(0xFF, 0xF0, 1) == 0xF0
        # opcode 2 = or
        assert chip.execute(0xF0, 0x0F, 2) == 0xFF

    def test_execute_invalid_op(self):
        """Test error on invalid operation."""
        chip = Chip.alu(["xor"], bits=8)
        with pytest.raises(ValueError, match="Unknown operation"):
            chip.execute(1, 2, "and")

    def test_execute_invalid_opcode(self):
        """Test error on invalid opcode."""
        chip = Chip.alu(["xor"], bits=8)
        with pytest.raises(ValueError, match="out of range"):
            chip.execute(1, 2, 5)

    def test_execute_no_training_needed(self):
        """Test that execute works without compile/train."""
        chip = Chip.alu(["xor", "and", "or", "not"], bits=8)
        # No compile(), no train_router() - just execute
        assert chip.execute(0x12, 0x34, "xor") == 0x12 ^ 0x34
        assert chip.execute(0xFF, 0x0F, "and") == 0x0F


# =============================================================================
# CHIP EXECUTE BATCH
# =============================================================================

class TestChipExecuteBatch:
    """Tests for Chip.execute_batch() batched execution."""

    def test_execute_batch_xor(self):
        """Test batched XOR execution."""
        chip = Chip.alu(["xor"], bits=8)
        a_batch = [1, 2, 3, 4, 5]
        b_batch = [5, 4, 3, 2, 1]
        results = chip.execute_batch(a_batch, b_batch, "xor")
        expected = [a ^ b for a, b in zip(a_batch, b_batch)]
        assert results == expected

    def test_execute_batch_and(self):
        """Test batched AND execution."""
        chip = Chip.alu(["and"], bits=8)
        a_batch = [0xFF, 0xF0, 0x0F]
        b_batch = [0x0F, 0x0F, 0x0F]
        results = chip.execute_batch(a_batch, b_batch, "and")
        expected = [a & b for a, b in zip(a_batch, b_batch)]
        assert results == expected

    def test_execute_batch_empty(self):
        """Test batched execution with empty input."""
        chip = Chip.alu(["xor"], bits=8)
        results = chip.execute_batch([], [], "xor")
        assert results == []

    def test_execute_batch_single(self):
        """Test batched execution with single element."""
        chip = Chip.alu(["xor"], bits=8)
        results = chip.execute_batch([5], [3], "xor")
        assert results == [6]


# =============================================================================
# WITH BACKEND
# =============================================================================

class TestWithBackend:
    """Tests for Chip.with_backend() configuration."""

    def test_with_backend_auto(self):
        """Test auto backend selection."""
        chip = Chip.alu(["xor"], bits=8).with_backend()
        assert chip._backend is not None
        # Should work
        assert chip.execute(1, 2, "xor") == 3

    def test_with_backend_cpu(self):
        """Test explicit CPU backend."""
        chip = Chip.alu(["xor"], bits=8).with_backend("cpu")
        assert isinstance(chip._backend, CPUBackend)
        assert chip.execute(1, 2, "xor") == 3

    def test_with_backend_cuda(self):
        """Test explicit CUDA backend."""
        chip = Chip.alu(["xor"], bits=8).with_backend("cuda")
        assert isinstance(chip._backend, CUDABackend)
        assert chip.execute(1, 2, "xor") == 3

    def test_with_backend_instance(self):
        """Test backend instance."""
        backend = CPUBackend(bits=8)
        chip = Chip.alu(["xor"], bits=8).with_backend(backend)
        assert chip._backend is backend

    def test_with_backend_chaining(self):
        """Test method chaining."""
        chip = (Chip("test", bits=8)
                .input("a", 8)
                .input("b", 8)
                .operation(0, "xor")
                .output("result", 8)
                .with_backend("cpu"))
        assert chip._backend is not None
        assert chip.execute(5, 3, "xor") == 6


# =============================================================================
# BENCHMARK
# =============================================================================

class TestBenchmark:
    """Tests for Chip.benchmark()."""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dict."""
        chip = Chip.alu(["xor"], bits=8)
        stats = chip.benchmark("xor", n_samples=1000)

        assert "operation" in stats
        assert "samples" in stats
        assert "total_time" in stats
        assert "ops_per_sec" in stats
        assert "backend" in stats

    def test_benchmark_operation(self):
        """Test benchmark reports correct operation."""
        chip = Chip.alu(["xor", "and"], bits=8)
        stats = chip.benchmark("and", n_samples=100)
        assert stats["operation"] == "and"

    def test_benchmark_samples(self):
        """Test benchmark reports correct samples."""
        chip = Chip.alu(["xor"], bits=8)
        stats = chip.benchmark("xor", n_samples=5000)
        assert stats["samples"] == 5000

    def test_benchmark_reasonable_throughput(self):
        """Test benchmark reports reasonable throughput."""
        chip = Chip.alu(["xor"], bits=8)
        stats = chip.benchmark("xor", n_samples=10000)
        # Should be at least 100K ops/sec on any system
        assert stats["ops_per_sec"] > 100000


# =============================================================================
# DIFFERENT BIT WIDTHS
# =============================================================================

class TestBitWidths:
    """Tests for different bit widths."""

    def test_8bit(self):
        """Test 8-bit chip."""
        chip = Chip.alu(["xor", "not"], bits=8)
        assert chip.execute(0xFF, 0x00, "xor") == 0xFF
        assert chip.execute(0x00, 0, "not") == 0xFF

    def test_16bit(self):
        """Test 16-bit chip."""
        chip = Chip.alu(["xor", "not"], bits=16)
        assert chip.execute(0xFFFF, 0x0000, "xor") == 0xFFFF
        assert chip.execute(0x0000, 0, "not") == 0xFFFF

    def test_32bit(self):
        """Test 32-bit chip."""
        chip = Chip.alu(["xor", "not"], bits=32)
        assert chip.execute(0xFFFFFFFF, 0x00000000, "xor") == 0xFFFFFFFF
        assert chip.execute(0x00000000, 0, "not") == 0xFFFFFFFF


# =============================================================================
# SUMMARY
# =============================================================================

class TestSummary:
    """Tests for Chip.summary() with backend info."""

    def test_summary_shows_backend(self):
        """Test summary includes backend info."""
        chip = Chip.alu(["xor"], bits=8).with_backend("cpu")
        summary = chip.summary()
        assert "Backend: CPU" in summary

    def test_summary_shows_no_backend(self):
        """Test summary shows no backend when not set."""
        chip = Chip.alu(["xor"], bits=8)
        chip._backend = None  # Force no backend
        summary = chip.summary()
        assert "Backend: None" in summary
