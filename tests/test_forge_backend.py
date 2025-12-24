"""
Tests for trix.forge.backend - Hardware backend layer.
"""

import pytest
from trix.forge.backend import (
    Backend,
    BackendInfo,
    CPUBackend,
    CUDABackend,
    get_backend,
    list_backends,
    register_backend,
)


# =============================================================================
# BACKEND INFO
# =============================================================================

class TestBackendInfo:
    """Tests for BackendInfo dataclass."""

    def test_create_info(self):
        """Test creating backend info."""
        info = BackendInfo(
            name="test",
            device="test device",
            available=True,
            compute_units=4,
            memory_bytes=1024,
        )
        assert info.name == "test"
        assert info.device == "test device"
        assert info.available is True
        assert info.compute_units == 4
        assert info.memory_bytes == 1024

    def test_default_values(self):
        """Test default values."""
        info = BackendInfo(name="test", device="dev", available=False)
        assert info.compute_units == 0
        assert info.memory_bytes == 0
        assert info.driver_version == ""


# =============================================================================
# CPU BACKEND
# =============================================================================

class TestCPUBackend:
    """Tests for CPU backend."""

    def test_info(self):
        """Test CPU backend info."""
        backend = CPUBackend()
        info = backend.info()
        assert info.name == "CPU"
        assert info.available is True

    def test_available_shapes(self):
        """Test available shapes list."""
        backend = CPUBackend()
        shapes = backend.available_shapes()
        assert "xor" in shapes
        assert "and" in shapes
        assert "or" in shapes
        assert "not" in shapes
        assert "add" in shapes
        assert len(shapes) >= 15

    def test_supports_shape(self):
        """Test shape support check."""
        backend = CPUBackend()
        assert backend.supports_shape("xor")
        assert backend.supports_shape("XOR")  # Case insensitive
        assert backend.supports_shape("and")
        assert not backend.supports_shape("nonexistent")

    def test_execute_xor(self):
        """Test XOR execution."""
        backend = CPUBackend()
        assert backend.execute("xor", 0xAAAA, 0x5555, bits=16) == 0xFFFF
        assert backend.execute("xor", 0xFF, 0xFF, bits=8) == 0
        assert backend.execute("xor", 0, 0x12345678, bits=32) == 0x12345678

    def test_execute_and(self):
        """Test AND execution."""
        backend = CPUBackend()
        assert backend.execute("and", 0xFF00, 0x0FF0, bits=16) == 0x0F00
        assert backend.execute("and", 0xFFFF, 0x0000, bits=16) == 0
        assert backend.execute("and", 0xF0F0, 0xF0F0, bits=16) == 0xF0F0

    def test_execute_or(self):
        """Test OR execution."""
        backend = CPUBackend()
        assert backend.execute("or", 0xF000, 0x00F0, bits=16) == 0xF0F0
        assert backend.execute("or", 0, 0, bits=16) == 0
        assert backend.execute("or", 0xAAAA, 0x5555, bits=16) == 0xFFFF

    def test_execute_not(self):
        """Test NOT execution."""
        backend = CPUBackend()
        assert backend.execute("not", 0, bits=8) == 0xFF
        assert backend.execute("not", 0xFF, bits=8) == 0
        assert backend.execute("not", 0, bits=32) == 0xFFFFFFFF

    def test_execute_nand(self):
        """Test NAND execution."""
        backend = CPUBackend()
        result = backend.execute("nand", 0xFFFF, 0xFFFF, bits=32) & 0xFFFFFFFF
        assert result == 0xFFFF0000

    def test_execute_nor(self):
        """Test NOR execution."""
        backend = CPUBackend()
        result = backend.execute("nor", 0, 0, bits=8)
        assert result == 0xFF

    def test_execute_xnor(self):
        """Test XNOR execution."""
        backend = CPUBackend()
        result = backend.execute("xnor", 0xAAAA, 0xAAAA, bits=16)
        assert result == 0xFFFF

    def test_execute_batch_xor(self):
        """Test batch XOR execution."""
        backend = CPUBackend()
        a_batch = [1, 2, 3, 4, 5]
        b_batch = [5, 4, 3, 2, 1]
        results = backend.execute_batch("xor", a_batch, b_batch)
        expected = [a ^ b for a, b in zip(a_batch, b_batch)]
        assert results == expected

    def test_execute_batch_and(self):
        """Test batch AND execution."""
        backend = CPUBackend()
        a_batch = [0xFF, 0xF0, 0x0F]
        b_batch = [0xF0, 0xF0, 0xF0]
        results = backend.execute_batch("and", a_batch, b_batch)
        expected = [a & b for a, b in zip(a_batch, b_batch)]
        assert results == expected

    def test_execute_batch_unary(self):
        """Test batch execution with unary shape (NOT)."""
        backend = CPUBackend()
        a_batch = [0, 0xFF, 0xAA]
        results = backend.execute_batch("not", a_batch, bits=8)
        expected = [(~a) & 0xFF for a in a_batch]
        assert results == expected

    def test_execute_complex_shape(self):
        """Test execution of non-fast-path shape."""
        backend = CPUBackend()
        # ADD is not in fast path (uses polynomial)
        result = backend.execute("add", 5, 3, bits=32)
        # ADD is XOR-based in our implementation
        assert result == 6  # 5 ^ 3 = 6

    def test_compile_shapes_noop(self):
        """Test that compile_shapes is a no-op for CPU."""
        backend = CPUBackend()
        assert backend.compile_shapes(["xor", "and"]) is True


# =============================================================================
# CUDA BACKEND
# =============================================================================

class TestCUDABackend:
    """Tests for CUDA backend."""

    def test_info(self):
        """Test CUDA backend info."""
        backend = CUDABackend()
        info = backend.info()
        assert info.name == "CUDA"
        # May or may not be available depending on system

    def test_available_shapes(self):
        """Test available shapes list."""
        backend = CUDABackend()
        shapes = backend.available_shapes()
        assert "xor" in shapes
        assert "and" in shapes
        assert len(shapes) >= 15

    def test_execute_falls_back_to_cpu(self):
        """Test that single execution uses CPU for efficiency."""
        backend = CUDABackend()
        # Single ops should work even without CUDA
        result = backend.execute("xor", 0xAAAA, 0x5555, bits=16)
        assert result == 0xFFFF

    def test_batch_small_falls_back_to_cpu(self):
        """Test that small batches fall back to CPU."""
        backend = CUDABackend()
        a_batch = [1, 2, 3]
        b_batch = [3, 2, 1]
        results = backend.execute_batch("xor", a_batch, b_batch)
        expected = [a ^ b for a, b in zip(a_batch, b_batch)]
        assert results == expected


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends(self):
        """Test listing backends."""
        backends = list_backends()
        assert "cpu" in backends
        assert "cuda" in backends
        assert backends["cpu"].available is True

    def test_get_backend_by_name(self):
        """Test getting backend by name."""
        cpu = get_backend("cpu")
        assert isinstance(cpu, CPUBackend)

        cuda = get_backend("cuda")
        assert isinstance(cuda, CUDABackend)

    def test_get_backend_auto(self):
        """Test auto-detection."""
        backend = get_backend()
        assert isinstance(backend, (CPUBackend, CUDABackend))

    def test_get_backend_invalid(self):
        """Test invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_backend(self):
        """Test registering custom backend."""
        class CustomBackend(Backend):
            def info(self):
                return BackendInfo(name="Custom", device="test", available=True)
            def available_shapes(self):
                return ["xor"]
            def execute(self, shape, a, b=0, bits=32):
                return a ^ b
            def execute_batch(self, shape, a_batch, b_batch=None, bits=32):
                return [a ^ b for a, b in zip(a_batch, b_batch or [0]*len(a_batch))]

        register_backend("custom", CustomBackend)
        backend = get_backend("custom")
        assert isinstance(backend, CustomBackend)
        assert backend.execute("xor", 5, 3) == 6


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_execute_zero_inputs(self):
        """Test with zero inputs."""
        backend = CPUBackend()
        assert backend.execute("xor", 0, 0) == 0
        assert backend.execute("and", 0, 0) == 0
        assert backend.execute("or", 0, 0) == 0

    def test_execute_max_32bit(self):
        """Test with max 32-bit values."""
        backend = CPUBackend()
        max_val = 0xFFFFFFFF
        assert backend.execute("xor", max_val, max_val, bits=32) == 0
        assert backend.execute("and", max_val, max_val, bits=32) == max_val
        assert backend.execute("or", max_val, 0, bits=32) == max_val

    def test_execute_batch_empty(self):
        """Test batch with empty input."""
        backend = CPUBackend()
        results = backend.execute_batch("xor", [], [])
        assert results == []

    def test_execute_batch_single(self):
        """Test batch with single element."""
        backend = CPUBackend()
        results = backend.execute_batch("xor", [5], [3])
        assert results == [6]

    def test_different_bit_widths(self):
        """Test different bit widths."""
        backend = CPUBackend()

        # 8-bit
        assert backend.execute("not", 0, bits=8) == 0xFF

        # 16-bit
        assert backend.execute("not", 0, bits=16) == 0xFFFF

        # 32-bit
        assert backend.execute("not", 0, bits=32) == 0xFFFFFFFF
