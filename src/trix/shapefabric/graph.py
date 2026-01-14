"""
Shape Graph - Dataflow graph representation for shape-routed computation.

A ShapeGraph represents a computation as a directed acyclic graph where:
- Nodes are shape activations (operations)
- Edges are data dependencies
- Execution order is determined by data flow

This replaces traditional instruction sequences with a graph that
explicitly captures parallelism.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from enum import Enum
import json


class NodeType(Enum):
    """Types of nodes in a shape graph."""
    INPUT = "input"       # External input to the graph
    OUTPUT = "output"     # External output from the graph
    SHAPE = "shape"       # Shape activation (computation)
    CONST = "const"       # Constant value


@dataclass
class Edge:
    """
    An edge in the shape graph representing data flow.

    Attributes:
        source: Source node ID
        source_port: Output port on source node
        target: Target node ID
        target_port: Input port on target node
    """
    source: str
    source_port: int
    target: str
    target_port: int

    def __hash__(self):
        return hash((self.source, self.source_port, self.target, self.target_port))


@dataclass
class Node:
    """
    A node in the shape graph.

    Attributes:
        id: Unique identifier
        node_type: Type of node (INPUT, OUTPUT, SHAPE, CONST)
        shape: Shape name (for SHAPE nodes)
        value: Constant value (for CONST nodes)
        name: Human-readable name (for INPUT/OUTPUT nodes)
        inputs: List of input edge IDs
        outputs: List of output edge IDs
    """
    id: str
    node_type: NodeType
    shape: Optional[str] = None
    value: Optional[Any] = None
    name: Optional[str] = None
    inputs: List[Edge] = field(default_factory=list)
    outputs: List[Edge] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


class ShapeGraph:
    """
    A dataflow graph of shape compositions.

    The graph is built incrementally by adding nodes and connecting them.
    Execution is performed by the DataflowExecutor.

    Example:
        graph = ShapeGraph("add_example")
        a = graph.input("a")
        b = graph.input("b")
        sum_node = graph.shape("ADD", [a, b])
        graph.output("result", sum_node)
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.input_nodes: Dict[str, str] = {}   # name -> node_id
        self.output_nodes: Dict[str, str] = {}  # name -> node_id
        self._node_counter = 0

    def _next_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"n{self._node_counter}"

    def input(self, name: str) -> str:
        """
        Create an input node.

        Args:
            name: Name of the input

        Returns:
            Node ID
        """
        node_id = self._next_id()
        node = Node(
            id=node_id,
            node_type=NodeType.INPUT,
            name=name
        )
        self.nodes[node_id] = node
        self.input_nodes[name] = node_id
        return node_id

    def const(self, value: Any) -> str:
        """
        Create a constant node.

        Args:
            value: Constant value

        Returns:
            Node ID
        """
        node_id = self._next_id()
        node = Node(
            id=node_id,
            node_type=NodeType.CONST,
            value=value
        )
        self.nodes[node_id] = node
        return node_id

    def shape(self, shape_name: str, inputs: List[str], output_ports: int = 1) -> Union[str, List[str]]:
        """
        Create a shape node (computation).

        Args:
            shape_name: Name of the shape (e.g., "ADD", "XOR")
            inputs: List of input node IDs
            output_ports: Number of output ports (default 1)

        Returns:
            Node ID (or list of node IDs for multi-output shapes)
        """
        node_id = self._next_id()
        node = Node(
            id=node_id,
            node_type=NodeType.SHAPE,
            shape=shape_name.upper()
        )

        # Create input edges
        for port, source_id in enumerate(inputs):
            edge = Edge(
                source=source_id,
                source_port=0,  # Assume single output from source
                target=node_id,
                target_port=port
            )
            node.inputs.append(edge)
            self.edges.append(edge)

            # Add to source's outputs
            if source_id in self.nodes:
                self.nodes[source_id].outputs.append(edge)

        self.nodes[node_id] = node
        return node_id

    def output(self, name: str, source: str, source_port: int = 0) -> str:
        """
        Create an output node.

        Args:
            name: Name of the output
            source: Source node ID
            source_port: Output port on source node

        Returns:
            Node ID
        """
        node_id = self._next_id()
        node = Node(
            id=node_id,
            node_type=NodeType.OUTPUT,
            name=name
        )

        edge = Edge(
            source=source,
            source_port=source_port,
            target=node_id,
            target_port=0
        )
        node.inputs.append(edge)
        self.edges.append(edge)

        if source in self.nodes:
            self.nodes[source].outputs.append(edge)

        self.nodes[node_id] = node
        self.output_nodes[name] = node_id
        return node_id

    def connect(self, source: str, target: str,
                source_port: int = 0, target_port: int = 0) -> Edge:
        """
        Explicitly connect two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            source_port: Output port on source
            target_port: Input port on target

        Returns:
            Created edge
        """
        edge = Edge(source, source_port, target, target_port)
        self.edges.append(edge)

        if source in self.nodes:
            self.nodes[source].outputs.append(edge)
        if target in self.nodes:
            self.nodes[target].inputs.append(edge)

        return edge

    def topological_order(self) -> List[str]:
        """
        Return nodes in topological order (respecting dependencies).

        Returns:
            List of node IDs in execution order
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}

        for edge in self.edges:
            in_degree[edge.target] += 1

        # Start with nodes that have no inputs
        queue = [nid for nid, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            node = self.nodes[node_id]
            for edge in node.outputs:
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles!")

        return result

    def parallel_levels(self) -> List[List[str]]:
        """
        Group nodes into parallel execution levels.

        Nodes in the same level have no dependencies on each other
        and can execute in parallel.

        Returns:
            List of levels, each containing node IDs
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}

        for edge in self.edges:
            in_degree[edge.target] += 1

        levels = []
        remaining = set(self.nodes.keys())

        while remaining:
            # Find all nodes with in_degree 0 among remaining
            level = [nid for nid in remaining if in_degree[nid] == 0]

            if not level:
                raise ValueError("Graph contains cycles!")

            levels.append(level)

            # Remove these nodes and update in_degrees
            for node_id in level:
                remaining.remove(node_id)
                node = self.nodes[node_id]
                for edge in node.outputs:
                    if edge.target in remaining:
                        in_degree[edge.target] -= 1

        return levels

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the graph."""
        shape_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            if node.node_type == NodeType.SHAPE:
                shape_counts[node.shape] = shape_counts.get(node.shape, 0) + 1

        levels = self.parallel_levels()

        return {
            "name": self.name,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "inputs": len(self.input_nodes),
            "outputs": len(self.output_nodes),
            "shapes": sum(1 for n in self.nodes.values() if n.node_type == NodeType.SHAPE),
            "shape_distribution": shape_counts,
            "depth": len(levels),
            "max_parallelism": max(len(level) for level in levels) if levels else 0,
        }

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type.value,
                    "shape": n.shape,
                    "value": n.value,
                    "name": n.name,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "source_port": e.source_port,
                    "target": e.target,
                    "target_port": e.target_port,
                }
                for e in self.edges
            ],
        }

    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "ShapeGraph":
        """Deserialize graph from dictionary."""
        graph = cls(data.get("name", "unnamed"))

        # First pass: create all nodes
        for node_data in data["nodes"]:
            node_type = NodeType(node_data["type"])
            node = Node(
                id=node_data["id"],
                node_type=node_type,
                shape=node_data.get("shape"),
                value=node_data.get("value"),
                name=node_data.get("name"),
            )
            graph.nodes[node.id] = node

            if node_type == NodeType.INPUT:
                graph.input_nodes[node.name] = node.id
            elif node_type == NodeType.OUTPUT:
                graph.output_nodes[node.name] = node.id

        # Second pass: create edges
        for edge_data in data["edges"]:
            edge = Edge(
                source=edge_data["source"],
                source_port=edge_data["source_port"],
                target=edge_data["target"],
                target_port=edge_data["target_port"],
            )
            graph.edges.append(edge)

            if edge.source in graph.nodes:
                graph.nodes[edge.source].outputs.append(edge)
            if edge.target in graph.nodes:
                graph.nodes[edge.target].inputs.append(edge)

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> "ShapeGraph":
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        stats = self.stats()
        return (f"ShapeGraph({self.name!r}, "
                f"nodes={stats['nodes']}, edges={stats['edges']}, "
                f"depth={stats['depth']}, parallelism={stats['max_parallelism']})")


# =============================================================================
# Graph Building Helpers
# =============================================================================

def build_fibonacci_graph() -> ShapeGraph:
    """
    Build a shape graph for computing Fibonacci iteratively.

    This demonstrates the graph-building API.
    """
    graph = ShapeGraph("fibonacci")

    # Inputs
    n = graph.input("n")

    # Constants
    zero = graph.const(0)
    one = graph.const(1)
    two = graph.const(2)

    # Base case check: n < 2
    lt_2 = graph.shape("LT", [n, two])

    # For iterative Fib, we'd need loops which require recursion
    # For now, this is a simplified version
    # Real implementation would use the executor's recursion support

    # n=0 -> 0, n=1 -> 1, else computed
    result = graph.shape("MUX", [lt_2, n, n])  # Simplified

    graph.output("result", result)

    return graph


def build_full_adder_graph() -> ShapeGraph:
    """Build a shape graph for a full adder."""
    graph = ShapeGraph("full_adder")

    # Inputs
    a = graph.input("a")
    b = graph.input("b")
    cin = graph.input("cin")

    # sum = a XOR b XOR cin
    xor1 = graph.shape("XOR", [a, b])
    sum_out = graph.shape("XOR", [xor1, cin])

    # cout = (a AND b) OR ((a XOR b) AND cin)
    and1 = graph.shape("AND", [a, b])
    and2 = graph.shape("AND", [xor1, cin])
    cout = graph.shape("OR", [and1, and2])

    graph.output("sum", sum_out)
    graph.output("cout", cout)

    return graph


def build_alu_graph() -> ShapeGraph:
    """Build a shape graph for a simple ALU."""
    graph = ShapeGraph("alu")

    # Inputs
    a = graph.input("a")
    b = graph.input("b")
    op = graph.input("op")

    # Compute all operations
    add_result = graph.shape("ADD", [a, b])
    sub_result = graph.shape("SUB", [a, b])
    and_result = graph.shape("AND", [a, b])
    or_result = graph.shape("OR", [a, b])
    xor_result = graph.shape("XOR", [a, b])

    # Select based on opcode
    # op=0: ADD, op=1: SUB, op=2: AND, op=3: OR, op=4: XOR
    result = graph.shape("SELECT", [op, add_result, sub_result,
                                     and_result, or_result, xor_result])

    graph.output("result", result)

    return graph


if __name__ == "__main__":
    print("Shape Graph - Self Test")
    print("=" * 50)

    # Test full adder graph
    fa = build_full_adder_graph()
    print(f"\nFull Adder: {fa}")
    print(f"Stats: {fa.stats()}")
    print(f"Levels: {fa.parallel_levels()}")

    # Test ALU graph
    alu = build_alu_graph()
    print(f"\nALU: {alu}")
    print(f"Stats: {alu.stats()}")
    print(f"Levels: {alu.parallel_levels()}")

    # Test serialization
    json_str = fa.to_json()
    fa2 = ShapeGraph.from_json(json_str)
    assert fa2.stats() == fa.stats(), "Serialization failed"
    print("\nSerialization: PASS")

    print("\nAll tests passed!")
