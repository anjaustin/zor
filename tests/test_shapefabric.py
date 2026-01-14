"""
Tests for the TriX Shape Fabric.

Tests the atomic shapes, graph construction, and dataflow execution.
"""

import pytest
import sys
import os
import importlib.util

# Direct import to avoid trix package permission issues
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'trix', 'shapefabric')
shapes_module = load_module('shapefabric_shapes', os.path.join(base_path, 'shapes.py'))
graph_module = load_module('shapefabric_graph', os.path.join(base_path, 'graph.py'))

# Get references to key items
SHAPES = shapes_module.SHAPES
shape_stats = shapes_module.shape_stats
compose_full_adder = shapes_module.compose_full_adder
compose_adder_n = shapes_module.compose_adder_n
ShapeGraph = graph_module.ShapeGraph
NodeType = graph_module.NodeType
build_full_adder_graph = graph_module.build_full_adder_graph
build_alu_graph = graph_module.build_alu_graph


class TestAtomicShapes:
    """Tests for the 32 atomic shapes."""

    def test_shape_count(self):
        """Verify we have exactly 32 atomic shapes."""
        stats = shape_stats()
        assert stats['total'] == 32
        assert stats['logic'] == 7
        assert stats['arithmetic'] == 7
        assert stats['shift'] == 7
        assert stats['compare'] == 6
        assert stats['memory'] == 2
        assert stats['routing'] == 3

    def test_logic_shapes(self):
        """Test logic shapes."""
        # XOR
        assert SHAPES['XOR'](0b1010, 0b1100) == 0b0110
        assert SHAPES['XOR'](0, 0) == 0
        assert SHAPES['XOR'](1, 1) == 0

        # AND
        assert SHAPES['AND'](0b1010, 0b1100) == 0b1000
        assert SHAPES['AND'](0, 1) == 0
        assert SHAPES['AND'](1, 1) == 1

        # OR
        assert SHAPES['OR'](0b1010, 0b1100) == 0b1110
        assert SHAPES['OR'](0, 0) == 0
        assert SHAPES['OR'](0, 1) == 1

        # NAND
        assert SHAPES['NAND'](1, 1) & 1 == 0
        assert SHAPES['NAND'](0, 1) & 1 == 1

        # NOR
        assert SHAPES['NOR'](0, 0) & 1 == 1
        assert SHAPES['NOR'](1, 0) & 1 == 0

        # XNOR
        assert SHAPES['XNOR'](1, 1) & 1 == 1
        assert SHAPES['XNOR'](1, 0) & 1 == 0

    def test_arithmetic_shapes(self):
        """Test arithmetic shapes."""
        assert SHAPES['ADD'](5, 3) == 8
        assert SHAPES['ADD'](-1, 1) == 0

        assert SHAPES['SUB'](5, 3) == 2
        assert SHAPES['SUB'](3, 5) == -2

        assert SHAPES['MUL'](5, 3) == 15
        assert SHAPES['MUL'](0, 100) == 0

        assert SHAPES['DIV'](10, 3) == 3
        assert SHAPES['DIV'](10, 0) == 0  # Division by zero returns 0

        assert SHAPES['MOD'](10, 3) == 1
        assert SHAPES['MOD'](10, 0) == 0

        assert SHAPES['NEG'](5) == -5
        assert SHAPES['NEG'](-5) == 5

        assert SHAPES['ABS'](-5) == 5
        assert SHAPES['ABS'](5) == 5

    def test_compare_shapes(self):
        """Test comparison shapes."""
        assert SHAPES['EQ'](5, 5) == 1
        assert SHAPES['EQ'](5, 3) == 0

        assert SHAPES['NE'](5, 3) == 1
        assert SHAPES['NE'](5, 5) == 0

        assert SHAPES['LT'](3, 5) == 1
        assert SHAPES['LT'](5, 3) == 0
        assert SHAPES['LT'](5, 5) == 0

        assert SHAPES['LE'](3, 5) == 1
        assert SHAPES['LE'](5, 5) == 1
        assert SHAPES['LE'](5, 3) == 0

        assert SHAPES['GT'](5, 3) == 1
        assert SHAPES['GT'](3, 5) == 0

        assert SHAPES['GE'](5, 3) == 1
        assert SHAPES['GE'](5, 5) == 1

    def test_routing_shapes(self):
        """Test routing shapes."""
        # MUX
        assert SHAPES['MUX'](0, 10, 20) == 10
        assert SHAPES['MUX'](1, 10, 20) == 20

        # DEMUX
        a, b = SHAPES['DEMUX'](0, 42)
        assert a == 42 and b == 0

        a, b = SHAPES['DEMUX'](1, 42)
        assert a == 0 and b == 42

        # SELECT
        assert SHAPES['SELECT'](0, 10, 20, 30) == 10
        assert SHAPES['SELECT'](1, 10, 20, 30) == 20
        assert SHAPES['SELECT'](2, 10, 20, 30) == 30


class TestComposedShapes:
    """Tests for composed shapes."""

    def test_full_adder(self):
        """Test full adder composition."""
        # All 8 input combinations
        test_cases = [
            ((0, 0, 0), (0, 0)),  # sum=0, cout=0
            ((0, 0, 1), (1, 0)),  # sum=1, cout=0
            ((0, 1, 0), (1, 0)),  # sum=1, cout=0
            ((0, 1, 1), (0, 1)),  # sum=0, cout=1
            ((1, 0, 0), (1, 0)),  # sum=1, cout=0
            ((1, 0, 1), (0, 1)),  # sum=0, cout=1
            ((1, 1, 0), (0, 1)),  # sum=0, cout=1
            ((1, 1, 1), (1, 1)),  # sum=1, cout=1
        ]

        for (a, b, cin), (exp_sum, exp_cout) in test_cases:
            s, c = compose_full_adder(a, b, cin)
            assert s == exp_sum, f"FullAdder({a},{b},{cin}) sum failed"
            assert c == exp_cout, f"FullAdder({a},{b},{cin}) cout failed"

    def test_adder_n(self):
        """Test N-bit adder."""
        # 8-bit tests
        assert compose_adder_n(15, 1, 8) == 16
        assert compose_adder_n(255, 1, 8) == 0  # Overflow wraps
        assert compose_adder_n(100, 50, 8) == 150

        # 16-bit tests
        assert compose_adder_n(1000, 2000, 16) == 3000
        assert compose_adder_n(65535, 1, 16) == 0  # Overflow wraps


class TestShapeGraph:
    """Tests for shape graph construction and analysis."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = ShapeGraph("test")
        a = graph.input("a")
        b = graph.input("b")
        result = graph.shape("ADD", [a, b])
        graph.output("result", result)

        assert graph.name == "test"
        assert len(graph.nodes) == 4  # 2 inputs + 1 shape + 1 output
        assert len(graph.input_nodes) == 2
        assert len(graph.output_nodes) == 1

    def test_full_adder_graph(self):
        """Test full adder graph structure."""
        graph = build_full_adder_graph()
        stats = graph.stats()

        assert stats['inputs'] == 3  # a, b, cin
        assert stats['outputs'] == 2  # sum, cout
        assert stats['shapes'] == 5  # 2 XOR + 2 AND + 1 OR
        assert stats['shape_distribution']['XOR'] == 2
        assert stats['shape_distribution']['AND'] == 2
        assert stats['shape_distribution']['OR'] == 1

    def test_parallel_levels(self):
        """Test parallel level extraction."""
        graph = build_full_adder_graph()
        levels = graph.parallel_levels()

        # Should have multiple levels
        assert len(levels) >= 3

        # First level should contain inputs
        assert len(levels[0]) >= 2

    def test_serialization(self):
        """Test graph serialization."""
        original = build_full_adder_graph()

        # Serialize and deserialize
        json_str = original.to_json()
        restored = ShapeGraph.from_json(json_str)

        # Compare
        assert original.stats()['nodes'] == restored.stats()['nodes']
        assert original.stats()['edges'] == restored.stats()['edges']


class TestExecution:
    """Tests for graph execution."""

    def test_full_adder_execution(self):
        """Test full adder graph execution."""
        graph = build_full_adder_graph()

        # Simple executor
        def execute(graph, inputs):
            values = {}
            for name, value in inputs.items():
                if name in graph.input_nodes:
                    values[graph.input_nodes[name]] = value
            for node_id, node in graph.nodes.items():
                if node.node_type == NodeType.CONST:
                    values[node_id] = node.value

            for level in graph.parallel_levels():
                for node_id in level:
                    node = graph.nodes[node_id]
                    if node.node_type != NodeType.SHAPE:
                        continue
                    input_vals = [values[e.source] for e in node.inputs]
                    shape = SHAPES.get(node.shape)
                    if shape:
                        values[node_id] = shape(*input_vals)

            outputs = {}
            for name, node_id in graph.output_nodes.items():
                node = graph.nodes[node_id]
                if node.inputs:
                    outputs[name] = values.get(node.inputs[0].source)
            return outputs

        test_cases = [
            ({'a': 0, 'b': 0, 'cin': 0}, {'sum': 0, 'cout': 0}),
            ({'a': 1, 'b': 0, 'cin': 0}, {'sum': 1, 'cout': 0}),
            ({'a': 1, 'b': 1, 'cin': 0}, {'sum': 0, 'cout': 1}),
            ({'a': 1, 'b': 1, 'cin': 1}, {'sum': 1, 'cout': 1}),
        ]

        for inputs, expected in test_cases:
            result = execute(graph, inputs)
            assert result == expected, f"Failed for {inputs}"

    def test_alu_execution(self):
        """Test ALU graph execution."""
        graph = build_alu_graph()

        def execute(graph, inputs):
            values = {}
            for name, value in inputs.items():
                if name in graph.input_nodes:
                    values[graph.input_nodes[name]] = value

            for level in graph.parallel_levels():
                for node_id in level:
                    node = graph.nodes[node_id]
                    if node.node_type != NodeType.SHAPE:
                        continue
                    input_vals = [values[e.source] for e in node.inputs]
                    if node.shape == 'SELECT':
                        idx = input_vals[0]
                        result = input_vals[idx + 1] if 0 <= idx < len(input_vals) - 1 else 0
                    else:
                        shape = SHAPES.get(node.shape)
                        result = shape(*input_vals) if shape else 0
                    values[node_id] = result

            outputs = {}
            for name, node_id in graph.output_nodes.items():
                node = graph.nodes[node_id]
                if node.inputs:
                    outputs[name] = values.get(node.inputs[0].source)
            return outputs

        # Test each ALU operation
        assert execute(graph, {'a': 5, 'b': 3, 'op': 0})['result'] == 8   # ADD
        assert execute(graph, {'a': 5, 'b': 3, 'op': 1})['result'] == 2   # SUB
        assert execute(graph, {'a': 5, 'b': 3, 'op': 2})['result'] == 1   # AND
        assert execute(graph, {'a': 5, 'b': 3, 'op': 3})['result'] == 7   # OR
        assert execute(graph, {'a': 5, 'b': 3, 'op': 4})['result'] == 6   # XOR


class TestFibonacci:
    """Test Fibonacci using shape operations."""

    def test_fibonacci_shapes(self):
        """Test Fibonacci computed with shapes."""
        def shape_fib(n, cache=None):
            if cache is None:
                cache = {}
            if n in cache:
                return cache[n]
            if n < 2:
                result = n
            else:
                result = SHAPES['ADD'](
                    shape_fib(n - 1, cache),
                    shape_fib(n - 2, cache)
                )
            cache[n] = result
            return result

        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for i, exp in enumerate(expected):
            assert shape_fib(i) == exp, f"fib({i}) failed"


class TestRoutingFabric:
    """Tests for the routing fabric."""

    @pytest.fixture
    def fabric_module(self):
        """Load the fabric module."""
        import importlib.util
        base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'trix', 'shapefabric')

        # Load shapes first
        shapes_spec = importlib.util.spec_from_file_location('sf_shapes', os.path.join(base_path, 'shapes.py'))
        shapes_mod = importlib.util.module_from_spec(shapes_spec)
        sys.modules['sf_shapes'] = shapes_mod
        shapes_spec.loader.exec_module(shapes_mod)

        # Read and patch fabric code
        with open(os.path.join(base_path, 'fabric.py')) as f:
            code = f.read()
        code = code.replace('from .shapes import', 'from sf_shapes import')

        # Execute in a namespace
        namespace = {'__name__': 'fabric'}
        exec(compile(code, 'fabric.py', 'exec'), namespace)

        return namespace

    def test_fabric_creation(self, fabric_module):
        """Test fabric creation with tile pools."""
        fabric, router = fabric_module['create_fabric'](tiles_per_shape=4)
        stats = fabric.stats()

        # Should have 32 shapes * 4 tiles = 128 tiles
        assert stats['total_tiles'] == 128
        assert stats['tiles_per_shape'] == 4
        assert stats['idle'] == 128

    def test_tile_allocation(self, fabric_module):
        """Test tile allocation and freeing."""
        fabric, _ = fabric_module['create_fabric'](tiles_per_shape=2)

        # Allocate XOR tiles
        tile1 = fabric.allocate_tile("XOR")
        tile2 = fabric.allocate_tile("XOR")
        tile3 = fabric.allocate_tile("XOR")  # Should be None (only 2 tiles)

        assert tile1 is not None
        assert tile2 is not None
        assert tile3 is None

        # Free and reallocate
        fabric.free_tile(tile1)
        tile4 = fabric.allocate_tile("XOR")
        assert tile4 is not None

    def test_full_adder_routing(self, fabric_module):
        """Test full adder through fabric."""
        fabric, router = fabric_module['create_fabric']()

        test_cases = [
            ((0, 0, 0), (0, 0)),
            ((0, 0, 1), (1, 0)),
            ((1, 1, 0), (0, 1)),
            ((1, 1, 1), (1, 1)),
        ]

        for (a, b, cin), expected in test_cases:
            result = router.route("full_adder", a, b, cin)
            assert result == expected, f"FA({a},{b},{cin}) failed"

    def test_alu_routing(self, fabric_module):
        """Test ALU operations through fabric."""
        fabric, router = fabric_module['create_fabric']()

        tests = [
            ((5, 3, 0), 8),   # ADD
            ((5, 3, 1), 2),   # SUB
            ((5, 3, 2), 1),   # AND
            ((5, 3, 3), 7),   # OR
            ((5, 3, 4), 6),   # XOR
        ]

        for (a, b, op), expected in tests:
            result = router.route("alu", a, b, op)
            assert result == expected, f"ALU({a},{b},op={op}) failed"

    def test_ripple_adder_routing(self, fabric_module):
        """Test N-bit ripple adder through fabric."""
        fabric, router = fabric_module['create_fabric']()

        tests = [
            (15, 1, 16),
            (100, 50, 150),
            (255, 1, 0),  # Overflow wraps
        ]

        for a, b, expected in tests:
            result = router.route("ripple_adder", a, b, 8)
            assert result == expected, f"ADD8({a},{b}) failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
