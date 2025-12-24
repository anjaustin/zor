"""
CUDA Driver API - Frozen Shapes

Direct GPU execution via CUDA driver API.
Minimal stack: Python → CUDA Driver → GPU Hardware

No CuPy. No PyTorch. Just the driver.
"""

import ctypes
import numpy as np
import time
from pathlib import Path

# =============================================================================
# CUDA Driver API Bindings (via ctypes)
# =============================================================================

# Load CUDA driver
try:
    cuda = ctypes.CDLL("libcuda.so")
except OSError:
    cuda = ctypes.CDLL("libcuda.so.1")

# CUDA types
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUstream = ctypes.c_void_p

# Error checking
def check_cuda(result, func_name=""):
    if result != 0:
        raise RuntimeError(f"CUDA error in {func_name}: {result}")
    return result

# Initialize
check_cuda(cuda.cuInit(0), "cuInit")

# Get device
device = CUdevice()
check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

# Get device name
name = ctypes.create_string_buffer(256)
check_cuda(cuda.cuDeviceGetName(name, 256, device), "cuDeviceGetName")
print(f"Device: {name.value.decode()}")

# Create context
context = CUcontext()
check_cuda(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")

# =============================================================================
# Load PTX Module
# =============================================================================

PTX_PATH = Path(__file__).parent / "cuda_shapes.ptx"

# Load module from PTX
module = CUmodule()
ptx_data = PTX_PATH.read_bytes()
check_cuda(
    cuda.cuModuleLoadData(ctypes.byref(module), ptx_data),
    "cuModuleLoadData"
)
print(f"Loaded PTX module: {PTX_PATH.name}")

# Get kernel functions
kernel_alu = CUfunction()
kernel_direct = CUfunction()
kernel_batch = CUfunction()

check_cuda(
    cuda.cuModuleGetFunction(ctypes.byref(kernel_alu), module, b"execute_6502_alu"),
    "cuModuleGetFunction(execute_6502_alu)"
)
check_cuda(
    cuda.cuModuleGetFunction(ctypes.byref(kernel_direct), module, b"execute_shape_direct"),
    "cuModuleGetFunction(execute_shape_direct)"
)
check_cuda(
    cuda.cuModuleGetFunction(ctypes.byref(kernel_batch), module, b"execute_shape_batch"),
    "cuModuleGetFunction(execute_shape_batch)"
)
print("Loaded kernel functions")

# Get constant memory pointer for routing table
routing_table_ptr = CUdeviceptr()
routing_table_size = ctypes.c_size_t()
check_cuda(
    cuda.cuModuleGetGlobal_v2(
        ctypes.byref(routing_table_ptr),
        ctypes.byref(routing_table_size),
        module,
        b"d_routing_table"
    ),
    "cuModuleGetGlobal(d_routing_table)"
)
print(f"Routing table: {routing_table_size.value} bytes at 0x{routing_table_ptr.value:x}")

# =============================================================================
# Memory Management (Unified Memory for Thor)
# =============================================================================

def alloc_unified(size):
    """Allocate unified memory (accessible from CPU and GPU)."""
    ptr = CUdeviceptr()
    # CU_MEM_ATTACH_GLOBAL = 1
    check_cuda(
        cuda.cuMemAllocManaged(ctypes.byref(ptr), size, 1),
        "cuMemAllocManaged"
    )
    return ptr

def free_mem(ptr):
    """Free GPU memory."""
    check_cuda(cuda.cuMemFree_v2(ptr), "cuMemFree")

def copy_to_device(dst_ptr, src_array):
    """Copy numpy array to device memory."""
    src = src_array.ctypes.data_as(ctypes.c_void_p)
    size = src_array.nbytes
    check_cuda(
        cuda.cuMemcpyHtoD_v2(dst_ptr, src, size),
        "cuMemcpyHtoD"
    )

def copy_from_device(dst_array, src_ptr):
    """Copy from device memory to numpy array."""
    dst = dst_array.ctypes.data_as(ctypes.c_void_p)
    size = dst_array.nbytes
    check_cuda(
        cuda.cuMemcpyDtoH_v2(dst, src_ptr, size),
        "cuMemcpyDtoH"
    )

# =============================================================================
# Set Routing Table
# =============================================================================

def set_routing_table(table):
    """
    Set the opcode -> shape_id routing table.

    table: numpy array of shape [256] with dtype uint8
    """
    assert table.shape == (256,) and table.dtype == np.uint8
    src = table.ctypes.data_as(ctypes.c_void_p)
    check_cuda(
        cuda.cuMemcpyHtoD_v2(routing_table_ptr, src, 256),
        "cuMemcpyHtoD(routing_table)"
    )

# =============================================================================
# Kernel Launch
# =============================================================================

def launch_kernel(kernel, grid, block, args):
    """
    Launch a CUDA kernel.

    kernel: CUfunction
    grid: (gridDimX, gridDimY, gridDimZ)
    block: (blockDimX, blockDimY, blockDimZ)
    args: list of kernel arguments (ctypes values)
    """
    # Build argument array
    arg_ptrs = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

    check_cuda(
        cuda.cuLaunchKernel(
            kernel,
            grid[0], grid[1], grid[2],      # Grid dimensions
            block[0], block[1], block[2],   # Block dimensions
            0,                               # Shared memory bytes
            None,                            # Stream (0 = default)
            arg_ptrs,                        # Kernel arguments
            None                             # Extra (unused)
        ),
        "cuLaunchKernel"
    )

def sync():
    """Synchronize (wait for GPU)."""
    check_cuda(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

# =============================================================================
# High-Level API
# =============================================================================

class CUDAShapes:
    """
    CUDA Driver-level frozen shapes executor.

    Usage:
        shapes = CUDAShapes()
        shapes.set_routing([0, 1, 2, 3, ...])  # opcode -> shape mapping
        result, carry = shapes.execute(opcodes, a, b, c)
    """

    # Shape names for reference
    SHAPE_NAMES = [
        'RIPPLE_ADD', 'RIPPLE_SUB', 'PARALLEL_AND', 'PARALLEL_OR',
        'PARALLEL_XOR', 'SHIFT_LEFT', 'SHIFT_RIGHT', 'ROTATE_LEFT',
        'ROTATE_RIGHT', 'INCREMENT', 'DECREMENT', 'TRANSFER',
        'LOAD', 'STORE', 'BIT_TEST', 'IDENTITY'
    ]

    def __init__(self, max_batch=1024*1024):
        """Initialize with pre-allocated buffers."""
        self.max_batch = max_batch

        # Pre-allocate unified memory buffers
        self.d_opcodes = alloc_unified(max_batch)
        self.d_a = alloc_unified(max_batch)
        self.d_b = alloc_unified(max_batch)
        self.d_c = alloc_unified(max_batch)
        self.d_result = alloc_unified(max_batch)
        self.d_carry = alloc_unified(max_batch)

        # Default routing: identity (opcode i -> shape i % 16)
        default_routing = np.array([i % 16 for i in range(256)], dtype=np.uint8)
        set_routing_table(default_routing)

    def __del__(self):
        """Free GPU memory."""
        try:
            free_mem(self.d_opcodes)
            free_mem(self.d_a)
            free_mem(self.d_b)
            free_mem(self.d_c)
            free_mem(self.d_result)
            free_mem(self.d_carry)
        except:
            pass

    def set_routing(self, routing):
        """Set opcode -> shape routing table."""
        table = np.zeros(256, dtype=np.uint8)
        for i, shape_id in enumerate(routing):
            if i < 256:
                table[i] = shape_id
        set_routing_table(table)

    def execute(self, opcodes, a, b, c):
        """
        Execute ALU operations via CUDA.

        Args:
            opcodes: [N] uint8 opcode indices
            a, b: [N] uint8 operands
            c: [N] uint8 carry input

        Returns:
            result: [N] uint8 results
            carry: [N] uint8 carry outputs
        """
        n = len(opcodes)
        assert n <= self.max_batch

        # Ensure numpy arrays
        opcodes = np.asarray(opcodes, dtype=np.uint8)
        a = np.asarray(a, dtype=np.uint8)
        b = np.asarray(b, dtype=np.uint8)
        c = np.asarray(c, dtype=np.uint8)

        # Copy to device
        copy_to_device(self.d_opcodes, opcodes)
        copy_to_device(self.d_a, a)
        copy_to_device(self.d_b, b)
        copy_to_device(self.d_c, c)

        # Launch kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(n)
        ]

        launch_kernel(kernel_alu, (grid_size, 1, 1), (block_size, 1, 1), args)
        sync()

        # Copy results back
        result = np.zeros(n, dtype=np.uint8)
        carry = np.zeros(n, dtype=np.uint8)
        copy_from_device(result, self.d_result)
        copy_from_device(carry, self.d_carry)

        return result, carry

    def execute_direct(self, shape_ids, a, b, c):
        """Execute with explicit shape IDs (no routing table)."""
        n = len(shape_ids)
        assert n <= self.max_batch

        shape_ids = np.asarray(shape_ids, dtype=np.uint8)
        a = np.asarray(a, dtype=np.uint8)
        b = np.asarray(b, dtype=np.uint8)
        c = np.asarray(c, dtype=np.uint8)

        copy_to_device(self.d_opcodes, shape_ids)  # Reuse buffer
        copy_to_device(self.d_a, a)
        copy_to_device(self.d_b, b)
        copy_to_device(self.d_c, c)

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(n)
        ]

        launch_kernel(kernel_direct, (grid_size, 1, 1), (block_size, 1, 1), args)
        sync()

        result = np.zeros(n, dtype=np.uint8)
        carry = np.zeros(n, dtype=np.uint8)
        copy_from_device(result, self.d_result)
        copy_from_device(carry, self.d_carry)

        return result, carry


# =============================================================================
# Verification
# =============================================================================

def verify_all_shapes():
    """Verify all shapes compute correctly."""
    print("\n" + "=" * 70)
    print("VERIFYING CUDA DRIVER SHAPES")
    print("=" * 70)

    shapes = CUDAShapes(max_batch=1024)

    # Test cases
    test_cases = [
        # (shape_id, a, b, c, expected_result, expected_carry)
        (0, 0x10, 0x20, 0, 0x30, 0),  # ADC
        (0, 0xFF, 0x01, 0, 0x00, 1),  # ADC with carry
        (1, 0x50, 0x20, 1, 0x30, 1),  # SBC
        (2, 0xF0, 0x0F, 0, 0x00, 0),  # AND
        (3, 0xF0, 0x0F, 0, 0xFF, 0),  # OR
        (4, 0xFF, 0xAA, 0, 0x55, 0),  # XOR
        (5, 0x40, 0x00, 0, 0x80, 0),  # ASL
        (5, 0x80, 0x00, 0, 0x00, 1),  # ASL with carry
        (6, 0x02, 0x00, 0, 0x01, 0),  # LSR
        (7, 0x40, 0x00, 1, 0x81, 0),  # ROL with carry in
        (8, 0x02, 0x00, 1, 0x81, 0),  # ROR with carry in
        (9, 0xFF, 0x00, 0, 0x00, 0),  # INC (wraps)
        (10, 0x00, 0x00, 0, 0xFF, 0), # DEC (wraps)
    ]

    passed = 0
    failed = 0

    for shape_id, a, b, c, expected_result, expected_carry in test_cases:
        result, carry = shapes.execute_direct(
            np.array([shape_id], dtype=np.uint8),
            np.array([a], dtype=np.uint8),
            np.array([b], dtype=np.uint8),
            np.array([c], dtype=np.uint8)
        )

        ok = (result[0] == expected_result) and (carry[0] == expected_carry)

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        name = CUDAShapes.SHAPE_NAMES[shape_id]
        print(f"  {name:15s}: {a:02X} op {b:02X} (c={c}) -> "
              f"got {result[0]:02X} (c={carry[0]}), "
              f"expected {expected_result:02X} (c={expected_carry}) [{status}]")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


class CUDAShapesZeroCopy:
    """
    Zero-copy version using unified memory.

    Data is written directly to GPU-accessible memory.
    No explicit copies needed on unified memory systems (Thor).
    """

    def __init__(self, max_batch=1024*1024):
        self.max_batch = max_batch

        # Allocate unified memory
        self.d_opcodes = alloc_unified(max_batch)
        self.d_a = alloc_unified(max_batch)
        self.d_b = alloc_unified(max_batch)
        self.d_c = alloc_unified(max_batch)
        self.d_result = alloc_unified(max_batch)
        self.d_carry = alloc_unified(max_batch)

        # Create numpy arrays that directly map to unified memory
        # This avoids copies - CPU writes go directly to GPU-visible memory
        self.opcodes = np.zeros(max_batch, dtype=np.uint8)
        self.a = np.zeros(max_batch, dtype=np.uint8)
        self.b = np.zeros(max_batch, dtype=np.uint8)
        self.c = np.zeros(max_batch, dtype=np.uint8)
        self.result = np.zeros(max_batch, dtype=np.uint8)
        self.carry = np.zeros(max_batch, dtype=np.uint8)

        # Default routing
        default_routing = np.array([i % 16 for i in range(256)], dtype=np.uint8)
        set_routing_table(default_routing)

    def set_routing(self, routing):
        table = np.zeros(256, dtype=np.uint8)
        for i, shape_id in enumerate(routing):
            if i < 256:
                table[i] = shape_id
        set_routing_table(table)

    def execute_preallocated(self, n):
        """
        Execute on pre-filled buffers.

        Caller must fill self.opcodes, self.a, self.b, self.c first.
        Results are in self.result, self.carry after call.
        """
        # Copy to unified memory (still needed, but fast on Thor)
        copy_to_device(self.d_opcodes, self.opcodes[:n])
        copy_to_device(self.d_a, self.a[:n])
        copy_to_device(self.d_b, self.b[:n])
        copy_to_device(self.d_c, self.c[:n])

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(n)
        ]

        launch_kernel(kernel_alu, (grid_size, 1, 1), (block_size, 1, 1), args)
        sync()

        copy_from_device(self.result[:n], self.d_result)
        copy_from_device(self.carry[:n], self.d_carry)

    def execute_kernel_only(self, n):
        """
        Execute kernel only - assume data already in unified memory.

        This is the minimal latency path.
        """
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(n)
        ]

        launch_kernel(kernel_alu, (grid_size, 1, 1), (block_size, 1, 1), args)
        sync()


def benchmark():
    """Benchmark throughput."""
    print("\n" + "=" * 70)
    print("BENCHMARKING CUDA DRIVER SHAPES")
    print("=" * 70)

    shapes = CUDAShapes(max_batch=1024*1024)

    # Set 6502 routing
    routing = list(range(11)) + [15] * 245  # 11 ops + identity for rest
    shapes.set_routing(routing)

    print("\n[With memory copies]")
    for n in [1024, 10*1024, 100*1024, 1024*1024]:
        # Random data
        opcodes = np.random.randint(0, 11, n, dtype=np.uint8)
        a = np.random.randint(0, 256, n, dtype=np.uint8)
        b = np.random.randint(0, 256, n, dtype=np.uint8)
        c = np.random.randint(0, 2, n, dtype=np.uint8)

        # Warm up
        shapes.execute(opcodes, a, b, c)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            shapes.execute(opcodes, a, b, c)
        elapsed = time.perf_counter() - start

        ops_per_sec = (n * 10) / elapsed
        print(f"  N={n:>8,}: {ops_per_sec:>12,.0f} ops/sec ({elapsed*1000/10:.3f} ms/batch)")

    # Kernel-only benchmark (no copies)
    print("\n[Kernel only - no copies]")
    zc = CUDAShapesZeroCopy(max_batch=1024*1024)
    zc.set_routing(routing)

    for n in [1024, 10*1024, 100*1024, 1024*1024]:
        # Pre-fill data once
        zc.opcodes[:n] = np.random.randint(0, 11, n, dtype=np.uint8)
        zc.a[:n] = np.random.randint(0, 256, n, dtype=np.uint8)
        zc.b[:n] = np.random.randint(0, 256, n, dtype=np.uint8)
        zc.c[:n] = np.random.randint(0, 2, n, dtype=np.uint8)

        # Copy once
        copy_to_device(zc.d_opcodes, zc.opcodes[:n])
        copy_to_device(zc.d_a, zc.a[:n])
        copy_to_device(zc.d_b, zc.b[:n])
        copy_to_device(zc.d_c, zc.c[:n])

        # Warm up
        zc.execute_kernel_only(n)

        # Benchmark kernel only
        start = time.perf_counter()
        for _ in range(100):
            zc.execute_kernel_only(n)
        elapsed = time.perf_counter() - start

        ops_per_sec = (n * 100) / elapsed
        latency_us = elapsed * 1000000 / 100
        print(f"  N={n:>8,}: {ops_per_sec:>12,.0f} ops/sec ({latency_us:.1f} μs/kernel)")


# =============================================================================
# CUDA Graphs - Frozen Execution Shape
# =============================================================================

# CUDA Graph types
CUgraph = ctypes.c_void_p
CUgraphExec = ctypes.c_void_p
CUgraphNode = ctypes.c_void_p

class CUDAGraphShapes:
    """
    CUDA Graphs: The kernel launch itself becomes a frozen shape.

    Record once, replay infinitely without Python overhead.
    """

    def __init__(self, n, block_size=256):
        self.n = n
        self.block_size = block_size
        self.grid_size = (n + block_size - 1) // block_size

        # Allocate unified memory
        self.d_opcodes = alloc_unified(n)
        self.d_a = alloc_unified(n)
        self.d_b = alloc_unified(n)
        self.d_c = alloc_unified(n)
        self.d_result = alloc_unified(n)
        self.d_carry = alloc_unified(n)

        # Pre-build kernel arguments (never changes)
        self._args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(n)
        ]
        self._arg_ptrs = (ctypes.c_void_p * len(self._args))()
        for i, arg in enumerate(self._args):
            self._arg_ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

        # Create stream for graph execution
        self.stream = CUstream()
        check_cuda(cuda.cuStreamCreate(ctypes.byref(self.stream), 0), "cuStreamCreate")

        # Record the graph
        self._record_graph()

    def _record_graph(self):
        """Record the kernel launch as a CUDA graph."""
        # Begin capture
        check_cuda(
            cuda.cuStreamBeginCapture_v2(self.stream, 0),  # 0 = global mode
            "cuStreamBeginCapture"
        )

        # Launch kernel (captured, not executed)
        check_cuda(
            cuda.cuLaunchKernel(
                kernel_alu,
                self.grid_size, 1, 1,
                self.block_size, 1, 1,
                0, self.stream,
                self._arg_ptrs, None
            ),
            "cuLaunchKernel (capture)"
        )

        # End capture, get graph
        self.graph = CUgraph()
        check_cuda(
            cuda.cuStreamEndCapture(self.stream, ctypes.byref(self.graph)),
            "cuStreamEndCapture"
        )

        # Instantiate executable graph
        self.graph_exec = CUgraphExec()
        error_node = CUgraphNode()
        log_buffer = ctypes.create_string_buffer(1024)
        check_cuda(
            cuda.cuGraphInstantiate_v2(
                ctypes.byref(self.graph_exec),
                self.graph,
                ctypes.byref(error_node),
                log_buffer,
                1024
            ),
            "cuGraphInstantiate"
        )

    def execute(self):
        """Execute the frozen graph - minimal Python overhead."""
        check_cuda(
            cuda.cuGraphLaunch(self.graph_exec, self.stream),
            "cuGraphLaunch"
        )
        check_cuda(cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def execute_async(self):
        """Launch without sync - caller must sync later."""
        check_cuda(
            cuda.cuGraphLaunch(self.graph_exec, self.stream),
            "cuGraphLaunch"
        )

    def sync(self):
        """Wait for completion."""
        check_cuda(cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def __del__(self):
        try:
            cuda.cuGraphExecDestroy(self.graph_exec)
            cuda.cuGraphDestroy(self.graph)
            cuda.cuStreamDestroy_v2(self.stream)
            free_mem(self.d_opcodes)
            free_mem(self.d_a)
            free_mem(self.d_b)
            free_mem(self.d_c)
            free_mem(self.d_result)
            free_mem(self.d_carry)
        except:
            pass


# =============================================================================
# Prebatch Executor - Process Multiple Batches Without Python
# =============================================================================

class CUDAPrebatchShapes:
    """
    Prebatch execution: Load N batches, execute all, read results once.

    Python only touches setup and final result collection.
    """

    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.total_ops = batch_size * num_batches

        # Allocate for all batches at once
        self.d_opcodes = alloc_unified(self.total_ops)
        self.d_a = alloc_unified(self.total_ops)
        self.d_b = alloc_unified(self.total_ops)
        self.d_c = alloc_unified(self.total_ops)
        self.d_result = alloc_unified(self.total_ops)
        self.d_carry = alloc_unified(self.total_ops)

        # Host arrays for prebatch loading
        self.h_opcodes = np.zeros(self.total_ops, dtype=np.uint8)
        self.h_a = np.zeros(self.total_ops, dtype=np.uint8)
        self.h_b = np.zeros(self.total_ops, dtype=np.uint8)
        self.h_c = np.zeros(self.total_ops, dtype=np.uint8)
        self.h_result = np.zeros(self.total_ops, dtype=np.uint8)
        self.h_carry = np.zeros(self.total_ops, dtype=np.uint8)

        # Create graph for single mega-kernel launch
        block_size = 256
        grid_size = (self.total_ops + block_size - 1) // block_size

        self._args = [
            CUdeviceptr(self.d_opcodes.value),
            CUdeviceptr(self.d_a.value),
            CUdeviceptr(self.d_b.value),
            CUdeviceptr(self.d_c.value),
            CUdeviceptr(self.d_result.value),
            CUdeviceptr(self.d_carry.value),
            ctypes.c_uint32(self.total_ops)
        ]
        self._arg_ptrs = (ctypes.c_void_p * len(self._args))()
        for i, arg in enumerate(self._args):
            self._arg_ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

        self.stream = CUstream()
        check_cuda(cuda.cuStreamCreate(ctypes.byref(self.stream), 0), "cuStreamCreate")

        self.block_size = block_size
        self.grid_size = grid_size

        # Record graph for mega-batch
        self._record_graph()

    def _record_graph(self):
        """Record mega-batch kernel launch."""
        check_cuda(cuda.cuStreamBeginCapture_v2(self.stream, 0), "cuStreamBeginCapture")

        check_cuda(
            cuda.cuLaunchKernel(
                kernel_alu,
                self.grid_size, 1, 1,
                self.block_size, 1, 1,
                0, self.stream,
                self._arg_ptrs, None
            ),
            "cuLaunchKernel (capture)"
        )

        self.graph = CUgraph()
        check_cuda(cuda.cuStreamEndCapture(self.stream, ctypes.byref(self.graph)), "cuStreamEndCapture")

        self.graph_exec = CUgraphExec()
        error_node = CUgraphNode()
        log_buffer = ctypes.create_string_buffer(1024)
        check_cuda(
            cuda.cuGraphInstantiate_v2(
                ctypes.byref(self.graph_exec),
                self.graph,
                ctypes.byref(error_node),
                log_buffer,
                1024
            ),
            "cuGraphInstantiate"
        )

    def load_batch(self, batch_idx, opcodes, a, b, c):
        """Load a single batch into the prebatch buffer."""
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        self.h_opcodes[start:end] = opcodes
        self.h_a[start:end] = a
        self.h_b[start:end] = b
        self.h_c[start:end] = c

    def upload(self):
        """Upload all batches to GPU (one copy)."""
        copy_to_device(self.d_opcodes, self.h_opcodes)
        copy_to_device(self.d_a, self.h_a)
        copy_to_device(self.d_b, self.h_b)
        copy_to_device(self.d_c, self.h_c)

    def execute_all(self):
        """Execute all batches with single graph launch."""
        check_cuda(cuda.cuGraphLaunch(self.graph_exec, self.stream), "cuGraphLaunch")
        check_cuda(cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def download(self):
        """Download all results (one copy)."""
        copy_from_device(self.h_result, self.d_result)
        copy_from_device(self.h_carry, self.d_carry)

    def get_batch_result(self, batch_idx):
        """Get results for a specific batch."""
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        return self.h_result[start:end], self.h_carry[start:end]


def benchmark_graphs():
    """Benchmark CUDA Graphs vs raw kernel launches."""
    print("\n" + "=" * 70)
    print("BENCHMARKING CUDA GRAPHS (FROZEN EXECUTION SHAPES)")
    print("=" * 70)

    n = 1024 * 1024  # 1M ops
    iterations = 100

    # Set routing
    routing = list(range(11)) + [15] * 245
    table = np.array([routing[i] if i < len(routing) else 15 for i in range(256)], dtype=np.uint8)
    set_routing_table(table)

    # Create graph executor
    graph = CUDAGraphShapes(n)

    # Fill with random data (once)
    opcodes = np.random.randint(0, 11, n, dtype=np.uint8)
    a = np.random.randint(0, 256, n, dtype=np.uint8)
    b = np.random.randint(0, 256, n, dtype=np.uint8)
    c = np.random.randint(0, 2, n, dtype=np.uint8)

    copy_to_device(graph.d_opcodes, opcodes)
    copy_to_device(graph.d_a, a)
    copy_to_device(graph.d_b, b)
    copy_to_device(graph.d_c, c)

    # Warm up
    for _ in range(5):
        graph.execute()

    # Benchmark graph execution
    start = time.perf_counter()
    for _ in range(iterations):
        graph.execute()
    elapsed = time.perf_counter() - start

    ops_per_sec = (n * iterations) / elapsed
    latency_us = elapsed * 1000000 / iterations
    print(f"\n[CUDA Graph - Frozen Launch]")
    print(f"  N={n:>8,}: {ops_per_sec:>12,.0f} ops/sec ({latency_us:.1f} μs/launch)")

    # Compare to raw kernel (for reference)
    zc = CUDAShapesZeroCopy(max_batch=n)
    copy_to_device(zc.d_opcodes, opcodes)
    copy_to_device(zc.d_a, a)
    copy_to_device(zc.d_b, b)
    copy_to_device(zc.d_c, c)

    # Warm up
    for _ in range(5):
        zc.execute_kernel_only(n)

    start = time.perf_counter()
    for _ in range(iterations):
        zc.execute_kernel_only(n)
    elapsed = time.perf_counter() - start

    ops_per_sec_raw = (n * iterations) / elapsed
    latency_us_raw = elapsed * 1000000 / iterations
    print(f"\n[Raw Kernel Launch]")
    print(f"  N={n:>8,}: {ops_per_sec_raw:>12,.0f} ops/sec ({latency_us_raw:.1f} μs/launch)")

    speedup = ops_per_sec / ops_per_sec_raw
    print(f"\n  Graph speedup: {speedup:.2f}x")


def benchmark_prebatch():
    """Benchmark prebatch execution."""
    print("\n" + "=" * 70)
    print("BENCHMARKING PREBATCH (MULTIPLE BATCHES, SINGLE PYTHON TOUCH)")
    print("=" * 70)

    batch_size = 1024 * 1024  # 1M ops per batch
    num_batches = 10

    # Set routing
    routing = list(range(11)) + [15] * 245
    table = np.array([routing[i] if i < len(routing) else 15 for i in range(256)], dtype=np.uint8)
    set_routing_table(table)

    # Create prebatch executor
    prebatch = CUDAPrebatchShapes(batch_size, num_batches)

    # Load all batches
    print(f"\nLoading {num_batches} batches of {batch_size:,} ops each...")
    load_start = time.perf_counter()
    for i in range(num_batches):
        opcodes = np.random.randint(0, 11, batch_size, dtype=np.uint8)
        a = np.random.randint(0, 256, batch_size, dtype=np.uint8)
        b = np.random.randint(0, 256, batch_size, dtype=np.uint8)
        c = np.random.randint(0, 2, batch_size, dtype=np.uint8)
        prebatch.load_batch(i, opcodes, a, b, c)
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time*1000:.1f} ms")

    # Upload once
    upload_start = time.perf_counter()
    prebatch.upload()
    upload_time = time.perf_counter() - upload_start
    print(f"  Upload time: {upload_time*1000:.1f} ms")

    # Execute all (single graph launch for 10M ops)
    exec_start = time.perf_counter()
    prebatch.execute_all()
    exec_time = time.perf_counter() - exec_start
    print(f"  Execute time: {exec_time*1000:.3f} ms")

    # Download results
    download_start = time.perf_counter()
    prebatch.download()
    download_time = time.perf_counter() - download_start
    print(f"  Download time: {download_time*1000:.1f} ms")

    total_ops = batch_size * num_batches
    total_time = exec_time  # Just execution time
    ops_per_sec = total_ops / total_time

    print(f"\n[Prebatch Results]")
    print(f"  Total ops: {total_ops:,}")
    print(f"  Kernel time: {exec_time*1000:.3f} ms")
    print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")
    print(f"  Per-batch latency: {exec_time*1000000/num_batches:.1f} μs (amortized)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if verify_all_shapes():
        benchmark()
        benchmark_graphs()
        benchmark_prebatch()

        print("\n" + "=" * 70)
        print("CUDA DRIVER SHAPES - COMPLETE")
        print("=" * 70)
        print()
        print("Execution modes:")
        print("  1. Raw kernel:  Python builds args each call")
        print("  2. CUDA Graph:  Launch pattern frozen, replay without Python")
        print("  3. Prebatch:    Single graph launch for N batches")
        print()
        print("Stack:")
        print("  Python (setup only) → CUDA Graph → GPU Hardware")
        print()
        print("The kernel launch IS a frozen shape.")
        print()
