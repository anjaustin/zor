"""
Chip: Declarative chip specification for frozen neural-geometric compilation.

Usage:
    from trix.forge import Chip

    # Define chip
    chip = Chip("alu_4op")
    chip.input("a", 8).input("b", 8).input("op", 2)
    chip.operation(0, "add")
    chip.operation(1, "sub")
    chip.operation(2, "xor")
    chip.operation(3, "and")
    chip.output("result", 8)

    # Neural router mode (differentiable, requires training)
    model = chip.compile()
    chip.train_router()
    chip.validate()  # 100%

    # Direct backend mode (exact, no training needed)
    result = chip.execute(17, 38, "xor")  # Uses CUDA/CPU backend
    results = chip.execute_batch([1,2,3], [4,5,6], "and")  # Batched

Two execution modes:
    1. Neural Router: compile() + train_router() + compute()
       - Differentiable soft routing
       - Requires training to learn opcode → operation mapping
       - Good for: learning, integration with neural systems

    2. Direct Backend: execute() / execute_batch()
       - Exact polynomial computation
       - No training needed - works immediately
       - Good for: production, validation, hardware simulation
       - Uses CUDA on GPU, falls back to CPU
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union, TYPE_CHECKING

import torch
import torch.nn as nn

from trix.routing.shapes import BUILTIN_SHAPES, get_builtin
from trix.routing.primitives import xor, and_, or_, not_

if TYPE_CHECKING:
    from .backend import Backend


@dataclass
class Port:
    """Input or output port specification."""
    name: str
    bits: int
    is_input: bool


@dataclass
class Operation:
    """Operation mapping."""
    opcode: int
    name: str
    shape: Callable
    truth: Callable


class CompiledChip(nn.Module):
    """Compiled chip as PyTorch module."""

    def __init__(self, bits: int, operations: List[Operation], mode: str = "soft"):
        super().__init__()
        self.bits = bits
        self.operations = operations
        self.mode = mode
        self.num_ops = len(operations)

        # Router: input bits → operation weights
        input_bits = bits * 2 + self._opcode_bits()
        self.router = nn.Linear(input_bits, self.num_ops)

    def _opcode_bits(self) -> int:
        if self.num_ops <= 1:
            return 1
        return math.ceil(math.log2(self.num_ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: route to frozen shapes."""
        a_bits = x[:, :self.bits]
        b_bits = x[:, self.bits:self.bits*2]

        # Compute all frozen shapes
        outputs = []
        for op in self.operations:
            out = op.shape(a_bits, b_bits)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=1)  # [batch, num_ops, bits]

        # Route
        if self.mode == "soft":
            weights = torch.softmax(self.router(x), dim=1)  # [batch, num_ops]
            result = torch.einsum('bn,bno->bo', weights, stacked)
        else:
            idx = self.router(x).argmax(dim=1)
            result = stacked[torch.arange(len(x)), idx]

        return result


class Chip:
    """
    Declarative chip specification.

    Build chips by declaring inputs, outputs, and operations,
    then compile to frozen neural-geometric models.
    """

    def __init__(self, name: str, bits: int = 8):
        """
        Create a new chip specification.

        Args:
            name: Chip name (for documentation/export)
            bits: Default bit width for ports
        """
        self.name = name
        self.bits = bits
        self._inputs: List[Port] = []
        self._outputs: List[Port] = []
        self._operations: Dict[int, Operation] = {}
        self._compiled: Optional[CompiledChip] = None
        self._backend: Optional["Backend"] = None

    def input(self, name: str, bits: int = None) -> "Chip":
        """
        Declare an input port.

        Args:
            name: Port name
            bits: Bit width (default: chip's default)

        Returns:
            self (for chaining)
        """
        self._inputs.append(Port(name, bits or self.bits, is_input=True))
        self._compiled = None
        return self

    def output(self, name: str, bits: int = None) -> "Chip":
        """
        Declare an output port.

        Args:
            name: Port name
            bits: Bit width (default: chip's default)

        Returns:
            self (for chaining)
        """
        self._outputs.append(Port(name, bits or self.bits, is_input=False))
        self._compiled = None
        return self

    def operation(self, opcode: int, op_name: str) -> "Chip":
        """
        Map an opcode to an operation.

        Args:
            opcode: Integer opcode value
            op_name: Name of built-in operation (add, sub, xor, etc.)

        Returns:
            self (for chaining)
        """
        shape, truth = get_builtin(op_name)
        self._operations[opcode] = Operation(opcode, op_name, shape, truth)
        self._compiled = None
        return self

    @property
    def num_operations(self) -> int:
        """Number of registered operations."""
        return len(self._operations)

    @property
    def opcode_bits(self) -> int:
        """Bits needed to encode opcode."""
        if self.num_operations <= 1:
            return 1
        return math.ceil(math.log2(self.num_operations))

    @property
    def input_bits(self) -> int:
        """Total input bits."""
        return self.bits * 2 + self.opcode_bits

    def compile(self, mode: str = "soft") -> CompiledChip:
        """
        Compile chip specification to PyTorch module.

        Args:
            mode: "soft" for differentiable, "hard" for deterministic

        Returns:
            CompiledChip module
        """
        if not self._operations:
            raise ValueError("No operations defined. Use operation() first.")

        # Sort operations by opcode
        sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]

        self._compiled = CompiledChip(self.bits, sorted_ops, mode)
        return self._compiled

    def _int_to_bits(self, x: int, width: int) -> torch.Tensor:
        """Convert integer to bit tensor."""
        return torch.tensor(
            [(x >> i) & 1 for i in range(width)],
            dtype=torch.float32
        )

    def _bits_to_int(self, bits: torch.Tensor) -> int:
        """Convert bit tensor to integer."""
        result = 0
        for i, b in enumerate(bits):
            result += int(round(float(b))) << i
        return result

    def _generate_data(self):
        """Generate exhaustive test data."""
        inputs = []
        targets = []

        sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]
        max_val = 2 ** self.bits

        for a in range(max_val):
            for b in range(max_val):
                for op_idx, op in enumerate(sorted_ops):
                    a_bits = self._int_to_bits(a, self.bits)
                    b_bits = self._int_to_bits(b, self.bits)
                    op_bits = self._int_to_bits(op_idx, self.opcode_bits)
                    inputs.append(torch.cat([a_bits, b_bits, op_bits]))

                    result = op.truth(a, b)
                    targets.append(self._int_to_bits(result, self.bits))

        return torch.stack(inputs), torch.stack(targets)

    def validate(self, exhaustive: bool = True, n_samples: int = 100000) -> float:
        """
        Validate compiled chip against truth functions.

        Args:
            exhaustive: Test all possible inputs
            n_samples: Sample size if not exhaustive

        Returns:
            Accuracy (0.0 to 1.0)
        """
        if self._compiled is None:
            self.compile()

        X, Y = self._generate_data()

        self._compiled.eval()
        with torch.no_grad():
            pred = self._compiled(X)
            pred_bits = (pred > 0.5).float()
            correct = (pred_bits == Y).all(dim=1).sum().item()

        return correct / len(X)

    def train_router(self, max_epochs: int = 100, lr: float = 0.001,
                     verbose: bool = True) -> dict:
        """
        Train the router to select correct operations.

        Args:
            max_epochs: Maximum training epochs
            lr: Learning rate
            verbose: Print progress

        Returns:
            Dict with accuracy, epochs, time
        """
        import time

        if self._compiled is None:
            self.compile()

        X, Y = self._generate_data()

        optimizer = torch.optim.Adam(self._compiled.parameters(), lr=lr)
        criterion = nn.MSELoss()
        batch_size = 256

        if verbose:
            print(f"Training {self.name} router...")
            print(f"  Operations: {[op.name for op in self._compiled.operations]}")
            print(f"  Dataset: {len(X)} cases")

        start = time.time()

        for epoch in range(max_epochs):
            perm = torch.randperm(len(X))
            X_shuf, Y_shuf = X[perm], Y[perm]

            for i in range(len(X) // batch_size):
                batch_X = X_shuf[i*batch_size:(i+1)*batch_size]
                batch_Y = Y_shuf[i*batch_size:(i+1)*batch_size]

                optimizer.zero_grad()
                pred = self._compiled(batch_X)
                loss = criterion(pred, batch_Y)
                loss.backward()
                optimizer.step()

            acc = self.validate()

            if verbose and (epoch < 3 or epoch % 10 == 0):
                print(f"  Epoch {epoch}: {acc*100:.2f}%")

            if acc >= 0.9999:
                if verbose:
                    print(f"  Epoch {epoch}: 100%!")
                break

        elapsed = time.time() - start
        final_acc = self.validate()

        if verbose:
            print(f"  Done in {elapsed:.1f}s")

        return {"accuracy": final_acc, "epochs": epoch + 1, "time": elapsed}

    def compute(self, a: int, b: int, op: Union[int, str]) -> int:
        """
        Compute a single operation.

        Args:
            a: First operand
            b: Second operand
            op: Opcode (int) or operation name (str)

        Returns:
            Result as integer
        """
        if self._compiled is None:
            self.compile()

        # Resolve op to index
        if isinstance(op, str):
            op_idx = None
            sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]
            for i, operation in enumerate(sorted_ops):
                if operation.name == op:
                    op_idx = i
                    break
            if op_idx is None:
                names = [o.name for o in sorted_ops]
                raise ValueError(f"Unknown operation '{op}'. Available: {names}")
        else:
            op_idx = op

        # Build input
        a_bits = self._int_to_bits(a, self.bits)
        b_bits = self._int_to_bits(b, self.bits)
        op_bits = self._int_to_bits(op_idx, self.opcode_bits)
        x = torch.cat([a_bits, b_bits, op_bits]).unsqueeze(0)

        # Forward
        self._compiled.eval()
        with torch.no_grad():
            result = self._compiled(x)
            result_bits = (result > 0.5).float().squeeze(0)

        return self._bits_to_int(result_bits)

    def save(self, path: str) -> None:
        """Save chip to PyTorch format."""
        if self._compiled is None:
            self.compile()

        torch.save({
            'name': self.name,
            'bits': self.bits,
            'operations': [(op.opcode, op.name) for op in self._compiled.operations],
            'router_state': self._compiled.router.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> "Chip":
        """Load chip from PyTorch format."""
        data = torch.load(path, weights_only=False)

        chip = cls(data['name'], data['bits'])
        for opcode, name in data['operations']:
            chip.operation(opcode, name)

        chip.compile()
        chip._compiled.router.load_state_dict(data['router_state'])

        return chip

    def to_onnx(self, path: str) -> None:
        """Export to ONNX format."""
        if self._compiled is None:
            self.compile()

        example = torch.zeros(1, self.input_bits)
        torch.onnx.export(
            self._compiled,
            example,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )

    @classmethod
    def alu(cls, ops: List[str], bits: int = 8, name: str = "alu") -> "Chip":
        """
        Create a standard ALU with given operations.

        Args:
            ops: List of operation names
            bits: Bit width
            name: Chip name

        Returns:
            Configured Chip
        """
        chip = cls(name, bits)
        chip.input("a", bits).input("b", bits).input("op", math.ceil(math.log2(len(ops))) if len(ops) > 1 else 1)

        for i, op in enumerate(ops):
            chip.operation(i, op)

        chip.output("result", bits)
        return chip

    def summary(self) -> str:
        """Return chip summary."""
        sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]
        op_names = [op.name for op in sorted_ops]

        lines = [
            f"Chip: {self.name}",
            f"  Bits: {self.bits}",
            f"  Operations: {self.num_operations}",
            f"  Opcodes: {op_names}",
            f"  Input bits: {self.input_bits}",
            f"  Compiled: {self._compiled is not None}",
            f"  Backend: {self._backend.info().name if self._backend else 'None (use with_backend())'}",
        ]
        return "\n".join(lines)

    # =========================================================================
    # DIRECT BACKEND EXECUTION
    # =========================================================================

    def _get_backend(self) -> "Backend":
        """Get or create backend."""
        if self._backend is None:
            from .backend import get_backend
            self._backend = get_backend(bits=self.bits)
        return self._backend

    def with_backend(self, backend: Union[str, "Backend"] = None) -> "Chip":
        """
        Set the execution backend.

        Args:
            backend: Backend name ("cpu", "cuda"), Backend instance, or None for auto

        Returns:
            self (for chaining)
        """
        from .backend import get_backend, Backend as BackendClass

        if backend is None:
            self._backend = get_backend(bits=self.bits)
        elif isinstance(backend, str):
            self._backend = get_backend(backend, bits=self.bits)
        elif isinstance(backend, BackendClass):
            self._backend = backend
        else:
            raise TypeError(f"Expected str or Backend, got {type(backend)}")

        return self

    def execute(self, a: int, b: int, op: Union[int, str]) -> int:
        """
        Execute operation directly on backend (no neural router).

        This is the production execution path - exact computation using
        polynomial shapes on CUDA/CPU. No training required.

        Args:
            a: First operand
            b: Second operand
            op: Operation name (str) or opcode (int)

        Returns:
            Result as integer

        Example:
            chip = Chip.alu(["xor", "and", "or"])
            result = chip.execute(0xAA, 0x55, "xor")  # 0xFF
        """
        # Resolve operation name
        op_name = self._resolve_op_name(op)

        # Execute on backend
        backend = self._get_backend()
        return backend.execute(op_name, a, b, bits=self.bits)

    def execute_batch(
        self,
        a_batch: List[int],
        b_batch: List[int],
        op: Union[int, str]
    ) -> List[int]:
        """
        Execute operation on batched inputs.

        High-throughput execution path for processing many inputs.
        Uses GPU parallelism when CUDA backend is available.

        Args:
            a_batch: List of first operands
            b_batch: List of second operands
            op: Operation name (str) or opcode (int)

        Returns:
            List of results

        Example:
            chip = Chip.alu(["xor", "and"])
            results = chip.execute_batch([1,2,3], [4,5,6], "xor")
            # [5, 7, 5]
        """
        op_name = self._resolve_op_name(op)
        backend = self._get_backend()
        return backend.execute_batch(op_name, a_batch, b_batch, bits=self.bits)

    def _resolve_op_name(self, op: Union[int, str]) -> str:
        """Resolve opcode/name to operation name."""
        if isinstance(op, str):
            # Verify operation exists
            sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]
            names = [o.name for o in sorted_ops]
            if op not in names:
                raise ValueError(f"Unknown operation '{op}'. Available: {names}")
            return op
        else:
            # Opcode to name
            sorted_ops = [self._operations[i] for i in sorted(self._operations.keys())]
            if op < 0 or op >= len(sorted_ops):
                raise ValueError(f"Opcode {op} out of range [0, {len(sorted_ops)})")
            return sorted_ops[op].name

    def benchmark(self, op: Union[int, str], n_samples: int = 100000) -> dict:
        """
        Benchmark operation throughput.

        Args:
            op: Operation to benchmark
            n_samples: Number of samples

        Returns:
            Dict with ops_per_sec, total_time, samples
        """
        import time
        import random

        max_val = (1 << self.bits) - 1
        a_batch = [random.randint(0, max_val) for _ in range(n_samples)]
        b_batch = [random.randint(0, max_val) for _ in range(n_samples)]

        backend = self._get_backend()
        op_name = self._resolve_op_name(op)

        start = time.perf_counter()
        results = backend.execute_batch(op_name, a_batch, b_batch, bits=self.bits)
        elapsed = time.perf_counter() - start

        return {
            "operation": op_name,
            "samples": n_samples,
            "total_time": elapsed,
            "ops_per_sec": n_samples / elapsed,
            "backend": backend.info().name,
        }

    def __repr__(self) -> str:
        return f"Chip('{self.name}', bits={self.bits}, ops={self.num_operations})"
