"""
RoutingPipeline: Learn routing, freeze execution.

The core insight: separate WHAT to compute (learned routing) from
HOW to compute it (frozen polynomial shapes).
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .shapes import get_builtin


@dataclass
class TrainResult:
    """Results from training the router."""
    accuracy: float      # Final accuracy (0.0 to 1.0)
    epochs: int          # Epochs to convergence
    time: float          # Training time in seconds
    router_params: int   # Number of trainable parameters


class Router(nn.Module):
    """Minimal router: linear layer → softmax → shape weights."""

    def __init__(self, input_bits: int, num_shapes: int):
        super().__init__()
        self.linear = nn.Linear(input_bits, num_shapes)

    def forward(self, x: Tensor) -> Tensor:
        """Returns shape selection weights [batch, num_shapes]."""
        return torch.softmax(self.linear(x), dim=1)


class RoutingPipeline:
    """
    A complete pipeline for routing-based neural computation.

    Usage:
        pipe = RoutingPipeline(bit_width=8)
        pipe.add("add", shapes.add_8bit, truth=lambda a,b: (a+b) & 0xFF)
        pipe.add("xor", shapes.xor_8bit, truth=lambda a,b: a ^ b)
        result = pipe.train()
        acc = pipe.validate(exhaustive=True)
        pipe.save("model.pt")
    """

    def __init__(self, bit_width: int = 8, device: str = "cpu"):
        """
        Create a routing pipeline.

        Args:
            bit_width: Number of bits per operand (default 8)
            device: PyTorch device ("cpu" or "cuda")
        """
        self.bit_width = bit_width
        self.device = device
        self.shapes: list[tuple[str, Callable, Callable]] = []
        self.router: Optional[Router] = None
        self._trained = False

    def add(self, name: str, shape: Callable, truth: Callable) -> "RoutingPipeline":
        """
        Register a shape with its truth function.

        Args:
            name: Name of the operation (e.g., "add", "xor")
            shape: Frozen shape function (a_bits, b_bits) -> result_bits
            truth: Truth function (a_int, b_int) -> result_int

        Returns:
            self (for chaining)
        """
        self.shapes.append((name, shape, truth))
        self._trained = False
        return self

    def add_builtin(self, name: str) -> "RoutingPipeline":
        """
        Add a built-in shape by name.

        Available: add, sub, inc, dec, xor, and, or, not, shl, shr, rol, ror
        """
        shape, truth = get_builtin(name)
        return self.add(name, shape, truth)

    def add_builtins(self, names: list[str]) -> "RoutingPipeline":
        """Add multiple built-in shapes."""
        for name in names:
            self.add_builtin(name)
        return self

    @property
    def num_shapes(self) -> int:
        """Number of registered shapes."""
        return len(self.shapes)

    @property
    def opcode_bits(self) -> int:
        """Number of bits needed to encode opcode."""
        if self.num_shapes <= 1:
            return 1
        return math.ceil(math.log2(self.num_shapes))

    @property
    def input_bits(self) -> int:
        """Total input bits: a + b + opcode."""
        return self.bit_width * 2 + self.opcode_bits

    @property
    def router_params(self) -> int:
        """Number of trainable parameters in router."""
        if self.router is None:
            return self.input_bits * self.num_shapes + self.num_shapes
        return sum(p.numel() for p in self.router.parameters())

    def _int_to_bits(self, x: int, width: int) -> Tensor:
        """Convert integer to bit tensor (LSB first)."""
        return torch.tensor(
            [(x >> i) & 1 for i in range(width)],
            dtype=torch.float32,
            device=self.device
        )

    def _bits_to_int(self, bits: Tensor) -> int:
        """Convert bit tensor to integer."""
        result = 0
        for i, b in enumerate(bits):
            result += int(round(float(b))) << i
        return result

    def _generate_data(self, exhaustive: bool = True, n_samples: int = 100000):
        """Generate training data from truth functions."""
        inputs = []
        targets = []

        max_val = 2 ** self.bit_width

        if exhaustive:
            # All possible inputs
            for a in range(max_val):
                for b in range(max_val):
                    for op in range(self.num_shapes):
                        a_bits = self._int_to_bits(a, self.bit_width)
                        b_bits = self._int_to_bits(b, self.bit_width)
                        op_bits = self._int_to_bits(op, self.opcode_bits)
                        inputs.append(torch.cat([a_bits, b_bits, op_bits]))

                        _, _, truth = self.shapes[op]
                        result = truth(a, b)
                        targets.append(self._int_to_bits(result, self.bit_width))
        else:
            # Random sample
            for _ in range(n_samples):
                a = torch.randint(0, max_val, (1,)).item()
                b = torch.randint(0, max_val, (1,)).item()
                op = torch.randint(0, self.num_shapes, (1,)).item()

                a_bits = self._int_to_bits(a, self.bit_width)
                b_bits = self._int_to_bits(b, self.bit_width)
                op_bits = self._int_to_bits(op, self.opcode_bits)
                inputs.append(torch.cat([a_bits, b_bits, op_bits]))

                _, _, truth = self.shapes[op]
                result = truth(a, b)
                targets.append(self._int_to_bits(result, self.bit_width))

        X = torch.stack(inputs).to(self.device)
        Y = torch.stack(targets).to(self.device)
        return X, Y

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass: router selects shapes, shapes compute."""
        batch_size = x.shape[0]
        a_bits = x[:, :self.bit_width]
        b_bits = x[:, self.bit_width:self.bit_width*2]

        # Compute all shape outputs
        shape_outputs = []
        for name, shape, _ in self.shapes:
            out = shape(a_bits, b_bits)
            shape_outputs.append(out)

        # Stack: [batch, num_shapes, output_bits]
        stacked = torch.stack(shape_outputs, dim=1)

        # Router produces weights: [batch, num_shapes]
        weights = self.router(x)

        # Weighted combination: [batch, output_bits]
        result = torch.einsum('bn,bno->bo', weights, stacked)
        return result

    def train(
        self,
        max_epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 256,
        verbose: bool = True,
        exhaustive_data: bool = True
    ) -> TrainResult:
        """
        Train the router.

        Args:
            max_epochs: Maximum training epochs
            lr: Learning rate
            batch_size: Batch size
            verbose: Print progress
            exhaustive_data: Use all possible inputs (vs random sample)

        Returns:
            TrainResult with accuracy, epochs, time, params
        """
        if self.num_shapes == 0:
            raise ValueError("No shapes registered. Use add() or add_builtin() first.")

        # Create router
        self.router = Router(self.input_bits, self.num_shapes).to(self.device)

        if verbose:
            print(f"Training router ({self.router_params} params) for {self.num_shapes} shapes...")

        # Generate data
        X, Y = self._generate_data(exhaustive=exhaustive_data)
        n_batches = len(X) // batch_size

        if verbose:
            print(f"  Dataset: {len(X)} cases")

        # Training setup
        optimizer = optim.Adam(self.router.parameters(), lr=lr)
        criterion = nn.MSELoss()

        start_time = time.time()
        final_epoch = 0

        for epoch in range(max_epochs):
            # Shuffle
            perm = torch.randperm(len(X), device=self.device)
            X_shuf = X[perm]
            Y_shuf = Y[perm]

            # Train batches
            for i in range(n_batches):
                batch_X = X_shuf[i*batch_size:(i+1)*batch_size]
                batch_Y = Y_shuf[i*batch_size:(i+1)*batch_size]

                optimizer.zero_grad()
                pred = self._forward(batch_X)
                loss = criterion(pred, batch_Y)
                loss.backward()
                optimizer.step()

            # Check accuracy
            acc = self._calculate_accuracy(X, Y)
            final_epoch = epoch + 1

            if verbose and (epoch < 3 or epoch % 10 == 0):
                print(f"  Epoch {epoch}: {acc*100:.2f}%")

            if acc >= 0.9999:
                if verbose:
                    print(f"  Epoch {epoch}: 100%!")
                break

        elapsed = time.time() - start_time
        final_acc = self._calculate_accuracy(X, Y)

        if verbose:
            print(f"  Done in {elapsed:.1f}s")

        self._trained = True

        return TrainResult(
            accuracy=final_acc,
            epochs=final_epoch,
            time=elapsed,
            router_params=self.router_params
        )

    def _calculate_accuracy(self, X: Tensor, Y: Tensor) -> float:
        """Calculate exact match accuracy."""
        self.router.eval()
        with torch.no_grad():
            pred = self._forward(X)
            pred_bits = (pred > 0.5).float()
            correct = (pred_bits == Y).all(dim=1).sum().item()
        self.router.train()
        return correct / len(X)

    def validate(self, exhaustive: bool = True, n_samples: int = 100000) -> float:
        """
        Validate accuracy on test data.

        Args:
            exhaustive: Test all possible inputs
            n_samples: Number of random samples (if not exhaustive)

        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        if not self._trained:
            raise ValueError("Pipeline not trained. Call train() first.")

        X, Y = self._generate_data(exhaustive=exhaustive, n_samples=n_samples)
        return self._calculate_accuracy(X, Y)

    def compute(self, a: int, b: int, op: Union[str, int]) -> int:
        """
        Compute a single operation.

        Args:
            a: First operand (integer)
            b: Second operand (integer)
            op: Operation name (str) or index (int)

        Returns:
            Result as integer
        """
        if not self._trained:
            raise ValueError("Pipeline not trained. Call train() first.")

        # Resolve op to index
        if isinstance(op, str):
            op_idx = None
            for i, (name, _, _) in enumerate(self.shapes):
                if name == op:
                    op_idx = i
                    break
            if op_idx is None:
                available = [name for name, _, _ in self.shapes]
                raise ValueError(f"Unknown operation '{op}'. Available: {available}")
        else:
            op_idx = op

        # Build input
        a_bits = self._int_to_bits(a, self.bit_width)
        b_bits = self._int_to_bits(b, self.bit_width)
        op_bits = self._int_to_bits(op_idx, self.opcode_bits)
        x = torch.cat([a_bits, b_bits, op_bits]).unsqueeze(0)

        # Forward
        self.router.eval()
        with torch.no_grad():
            result = self._forward(x)
            result_bits = (result > 0.5).float().squeeze(0)

        return self._bits_to_int(result_bits)

    def save(self, path: str) -> None:
        """Save pipeline to PyTorch format."""
        if not self._trained:
            raise ValueError("Pipeline not trained. Call train() first.")

        torch.save({
            'bit_width': self.bit_width,
            'shape_names': [name for name, _, _ in self.shapes],
            'router_state': self.router.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, shapes_module=None) -> "RoutingPipeline":
        """
        Load pipeline from PyTorch format.

        Args:
            path: Path to saved model
            shapes_module: Module with shape definitions (default: built-ins)
        """
        data = torch.load(path, weights_only=False)

        pipe = cls(bit_width=data['bit_width'])

        # Re-register shapes
        for name in data['shape_names']:
            pipe.add_builtin(name)

        # Create and load router
        pipe.router = Router(pipe.input_bits, pipe.num_shapes)
        pipe.router.load_state_dict(data['router_state'])
        pipe._trained = True

        return pipe

    def to_onnx(self, path: str) -> None:
        """Export to ONNX format."""
        if not self._trained:
            raise ValueError("Pipeline not trained. Call train() first.")

        # Create wrapper module for export
        class ExportModel(nn.Module):
            def __init__(self, pipeline):
                super().__init__()
                self.pipeline = pipeline

            def forward(self, x):
                return self.pipeline._forward(x)

        model = ExportModel(self)
        model.eval()

        # Example input
        example = torch.zeros(1, self.input_bits)

        torch.onnx.export(
            model,
            example,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"RoutingPipeline:",
            f"  Bit width: {self.bit_width}",
            f"  Shapes: {self.num_shapes}",
            f"  Router params: {self.router_params}",
            f"  Trained: {self._trained}",
            f"  Operations: {[name for name, _, _ in self.shapes]}"
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RoutingPipeline(bit_width={self.bit_width}, shapes={self.num_shapes}, trained={self._trained})"
