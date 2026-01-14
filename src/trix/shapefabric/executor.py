"""
Dataflow Executor - Executes shape graphs via data-driven evaluation.

The executor processes nodes as their inputs become available,
automatically extracting parallelism from the graph structure.

Key features:
- Dataflow execution (no program counter)
- Automatic parallelism
- Lazy and eager evaluation modes
- Memory for stateful operations
"""

from typing import Dict, List, Set, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time

from .shapes import SHAPES, get_shape, COMPOSED_SHAPES
from .graph import ShapeGraph, Node, NodeType, Edge


@dataclass
class ExecutionStats:
    """Statistics from graph execution."""
    total_nodes: int = 0
    executed_nodes: int = 0
    execution_time_ms: float = 0.0
    parallel_levels: int = 0
    max_parallelism: int = 0
    shape_counts: Dict[str, int] = field(default_factory=dict)


class DataflowExecutor:
    """
    Executes shape graphs using dataflow semantics.

    Values flow through the graph as they become available.
    Nodes execute when all their inputs are ready.
    """

    def __init__(self, parallel: bool = False, max_workers: int = 4):
        """
        Initialize executor.

        Args:
            parallel: Enable parallel execution
            max_workers: Maximum worker threads for parallel mode
        """
        self.parallel = parallel
        self.max_workers = max_workers
        self.memory: Dict[int, int] = {}  # Shared memory for LOAD/STORE
        self.stats = ExecutionStats()

    def execute(self, graph: ShapeGraph, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a shape graph.

        Args:
            graph: The ShapeGraph to execute
            inputs: Dictionary mapping input names to values

        Returns:
            Dictionary mapping output names to values
        """
        start_time = time.perf_counter()

        # Initialize value storage
        values: Dict[str, Any] = {}  # node_id -> value

        # Set input values
        for name, value in inputs.items():
            if name in graph.input_nodes:
                values[graph.input_nodes[name]] = value

        # Set constant values
        for node_id, node in graph.nodes.items():
            if node.node_type == NodeType.CONST:
                values[node_id] = node.value

        # Get execution levels
        levels = graph.parallel_levels()

        self.stats = ExecutionStats(
            total_nodes=len(graph.nodes),
            parallel_levels=len(levels),
            max_parallelism=max(len(level) for level in levels) if levels else 0,
        )

        if self.parallel:
            self._execute_parallel(graph, values, levels)
        else:
            self._execute_sequential(graph, values, levels)

        # Collect outputs
        outputs = {}
        for name, node_id in graph.output_nodes.items():
            node = graph.nodes[node_id]
            if node.inputs:
                source_id = node.inputs[0].source
                outputs[name] = values.get(source_id)
            else:
                outputs[name] = None

        self.stats.execution_time_ms = (time.perf_counter() - start_time) * 1000
        self.stats.executed_nodes = sum(self.stats.shape_counts.values())

        return outputs

    def _execute_sequential(self, graph: ShapeGraph, values: Dict[str, Any],
                            levels: List[List[str]]) -> None:
        """Execute graph sequentially level by level."""
        for level in levels:
            for node_id in level:
                self._execute_node(graph.nodes[node_id], values)

    def _execute_parallel(self, graph: ShapeGraph, values: Dict[str, Any],
                          levels: List[List[str]]) -> None:
        """Execute graph in parallel within each level."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for level in levels:
                # Submit all nodes in this level
                futures = {
                    executor.submit(self._execute_node, graph.nodes[node_id], values): node_id
                    for node_id in level
                    if graph.nodes[node_id].node_type == NodeType.SHAPE
                }

                # Wait for all to complete
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            values[node_id] = result
                    except Exception as e:
                        print(f"Error executing {node_id}: {e}")

    def _execute_node(self, node: Node, values: Dict[str, Any]) -> Optional[Any]:
        """
        Execute a single node.

        Args:
            node: Node to execute
            values: Current value storage

        Returns:
            Computed value (or None for non-compute nodes)
        """
        if node.node_type != NodeType.SHAPE:
            return None

        shape_name = node.shape

        # Track stats
        self.stats.shape_counts[shape_name] = self.stats.shape_counts.get(shape_name, 0) + 1

        # Gather inputs
        input_values = []
        for edge in node.inputs:
            if edge.source in values:
                input_values.append(values[edge.source])
            else:
                # Input not ready - shouldn't happen with proper level ordering
                raise ValueError(f"Input {edge.source} not ready for node {node.id}")

        # Get shape and execute
        shape = get_shape(shape_name)

        if shape is None:
            # Check composed shapes
            if shape_name in COMPOSED_SHAPES:
                result = COMPOSED_SHAPES[shape_name](*input_values)
            else:
                raise ValueError(f"Unknown shape: {shape_name}")
        else:
            # Handle special shapes
            if shape_name == "LOAD":
                result = self.memory.get(input_values[0], 0)
            elif shape_name == "STORE":
                self.memory[input_values[0]] = input_values[1]
                result = None
            elif shape_name == "SELECT":
                idx = input_values[0]
                if 0 <= idx < len(input_values) - 1:
                    result = input_values[idx + 1]
                else:
                    result = 0
            else:
                result = shape(*input_values)

        values[node.id] = result
        return result

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self.stats


def execute(graph: ShapeGraph, inputs: Dict[str, Any],
            parallel: bool = False) -> Dict[str, Any]:
    """
    Convenience function to execute a shape graph.

    Args:
        graph: The ShapeGraph to execute
        inputs: Dictionary mapping input names to values
        parallel: Enable parallel execution

    Returns:
        Dictionary mapping output names to values
    """
    executor = DataflowExecutor(parallel=parallel)
    return executor.execute(graph, inputs)


# =============================================================================
# Recursive/Iterative Execution Support
# =============================================================================

class RecursiveExecutor:
    """
    Executor that supports recursive graph patterns.

    For algorithms like Fibonacci that need recursion,
    this executor handles self-referential graph structures.
    """

    def __init__(self):
        self.cache: Dict[tuple, Any] = {}  # Memoization

    def execute_fibonacci(self, n: int) -> int:
        """
        Compute Fibonacci using shape operations.

        This demonstrates how a recursive algorithm works
        in the shape fabric model.
        """
        if n in self.cache:
            return self.cache[n]

        # Base cases
        lt = SHAPES["LT"]
        add = SHAPES["ADD"]
        sub = SHAPES["SUB"]

        if lt(n, 2):
            result = n
        else:
            # Recursive cases
            n1 = sub(n, 1)
            n2 = sub(n, 2)
            fib1 = self.execute_fibonacci(n1)
            fib2 = self.execute_fibonacci(n2)
            result = add(fib1, fib2)

        self.cache[n] = result
        return result

    def execute_factorial(self, n: int) -> int:
        """Compute factorial using shape operations."""
        if n in self.cache:
            return self.cache[n]

        lt = SHAPES["LT"]
        mul = SHAPES["MUL"]
        sub = SHAPES["SUB"]

        if lt(n, 2):
            result = 1
        else:
            n1 = sub(n, 1)
            fact_n1 = self.execute_factorial(n1)
            result = mul(n, fact_n1)

        self.cache[n] = result
        return result


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_executor():
    """Benchmark the executor on various operations."""
    from .graph import build_full_adder_graph, build_alu_graph

    print("Executor Benchmark")
    print("=" * 60)

    # Full adder benchmark
    fa_graph = build_full_adder_graph()
    executor = DataflowExecutor()

    iterations = 10000
    start = time.perf_counter()
    for i in range(iterations):
        result = executor.execute(fa_graph, {"a": i % 2, "b": (i // 2) % 2, "cin": (i // 4) % 2})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nFull Adder ({iterations} iterations):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Per iteration: {elapsed/iterations*1000:.2f} µs")

    # ALU benchmark
    alu_graph = build_alu_graph()

    start = time.perf_counter()
    for i in range(iterations):
        result = executor.execute(alu_graph, {"a": i, "b": i + 1, "op": i % 5})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nALU ({iterations} iterations):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Per iteration: {elapsed/iterations*1000:.2f} µs")

    # Fibonacci benchmark
    rec_executor = RecursiveExecutor()

    start = time.perf_counter()
    for i in range(35):
        rec_executor.cache.clear()
        result = rec_executor.execute_fibonacci(i)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nFibonacci (0-34):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Fib(34) = {rec_executor.execute_fibonacci(34)}")

    # Compare with native Python
    def native_fib(n):
        if n < 2:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    start = time.perf_counter()
    for i in range(35):
        native_fib(i)
    native_elapsed = (time.perf_counter() - start) * 1000

    print(f"\nNative Python Fibonacci (0-34):")
    print(f"  Total time: {native_elapsed:.2f} ms")
    print(f"  Shape/Native ratio: {elapsed/native_elapsed:.2f}x")


if __name__ == "__main__":
    print("Dataflow Executor - Self Test")
    print("=" * 50)

    from .graph import build_full_adder_graph, build_alu_graph

    # Test full adder
    fa_graph = build_full_adder_graph()
    executor = DataflowExecutor()

    test_cases = [
        ({"a": 0, "b": 0, "cin": 0}, {"sum": 0, "cout": 0}),
        ({"a": 1, "b": 0, "cin": 0}, {"sum": 1, "cout": 0}),
        ({"a": 0, "b": 1, "cin": 0}, {"sum": 1, "cout": 0}),
        ({"a": 1, "b": 1, "cin": 0}, {"sum": 0, "cout": 1}),
        ({"a": 1, "b": 1, "cin": 1}, {"sum": 1, "cout": 1}),
    ]

    print("\nFull Adder Tests:")
    for inputs, expected in test_cases:
        result = executor.execute(fa_graph, inputs)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {inputs} -> {result} : {status}")

    # Test ALU
    alu_graph = build_alu_graph()

    alu_tests = [
        ({"a": 5, "b": 3, "op": 0}, 8),   # ADD
        ({"a": 5, "b": 3, "op": 1}, 2),   # SUB
        ({"a": 5, "b": 3, "op": 2}, 1),   # AND
        ({"a": 5, "b": 3, "op": 3}, 7),   # OR
        ({"a": 5, "b": 3, "op": 4}, 6),   # XOR
    ]

    print("\nALU Tests:")
    for inputs, expected in alu_tests:
        result = executor.execute(alu_graph, inputs)
        status = "PASS" if result["result"] == expected else f"FAIL (got {result['result']})"
        print(f"  {inputs} -> {result['result']} : {status}")

    # Test Fibonacci
    rec_executor = RecursiveExecutor()
    fib_expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    print("\nFibonacci Tests:")
    for i, expected in enumerate(fib_expected):
        result = rec_executor.execute_fibonacci(i)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  fib({i}) = {result} : {status}")

    print("\n" + "=" * 50)
    print("Running benchmark...")
    benchmark_executor()
