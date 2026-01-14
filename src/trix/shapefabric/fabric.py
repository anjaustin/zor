"""
Routing Fabric - Dynamic shape routing for dataflow execution.

The fabric is the "magick" that routes data to shapes during operation.
No instruction decode needed - the data patterns themselves determine routing.

Key concepts:
- Shape Tiles: Instances of atomic shapes that can be allocated
- Routing Mesh: Connects tiles via crossbar/mesh topology
- Pattern Router: Routes data based on structural patterns
- Dynamic Allocation: Shapes allocated/freed as computation flows
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import time

from .shapes import SHAPES, Shape, get_shape


class TileState(Enum):
    """State of a shape tile."""
    IDLE = "idle"           # Available for allocation
    ALLOCATED = "allocated"  # Reserved for an operation
    EXECUTING = "executing"  # Currently computing
    DONE = "done"           # Result ready


@dataclass
class ShapeTile:
    """
    A single instantiation of a shape in the fabric.

    Tiles are the physical units that execute shapes.
    Multiple tiles can exist for the same shape type.
    """
    id: str
    shape_name: str
    state: TileState = TileState.IDLE
    inputs: List[Any] = field(default_factory=list)
    output: Any = None

    def execute(self) -> Any:
        """Execute the shape with current inputs."""
        shape = get_shape(self.shape_name)
        if shape is None:
            raise ValueError(f"Unknown shape: {self.shape_name}")

        self.state = TileState.EXECUTING
        self.output = shape(*self.inputs)
        self.state = TileState.DONE
        return self.output

    def reset(self):
        """Reset tile for reuse."""
        self.state = TileState.IDLE
        self.inputs = []
        self.output = None


@dataclass
class RouteEntry:
    """A routing table entry."""
    source_tile: str
    source_port: int
    target_tile: str
    target_port: int


class RoutingMesh:
    """
    Interconnect mesh for routing data between tiles.

    Implements a crossbar-style routing where any tile
    can connect to any other tile.
    """

    def __init__(self):
        self.routes: List[RouteEntry] = []
        self.pending_data: Dict[str, Dict[int, Any]] = {}  # tile_id -> port -> value

    def add_route(self, source: str, source_port: int,
                  target: str, target_port: int):
        """Add a route between tiles."""
        self.routes.append(RouteEntry(source, source_port, target, target_port))

    def clear_routes(self):
        """Clear all routes."""
        self.routes = []
        self.pending_data = {}

    def propagate(self, source_tile: str, source_port: int, value: Any):
        """Propagate a value through the mesh."""
        for route in self.routes:
            if route.source_tile == source_tile and route.source_port == source_port:
                if route.target_tile not in self.pending_data:
                    self.pending_data[route.target_tile] = {}
                self.pending_data[route.target_tile][route.target_port] = value

    def get_inputs(self, tile_id: str) -> Dict[int, Any]:
        """Get pending inputs for a tile."""
        return self.pending_data.get(tile_id, {})

    def clear_tile_inputs(self, tile_id: str):
        """Clear pending inputs for a tile."""
        if tile_id in self.pending_data:
            del self.pending_data[tile_id]


class ShapeFabric:
    """
    The main routing fabric for shape-based computation.

    The fabric manages a pool of shape tiles and routes
    data between them based on computation graphs.

    This is the "magick" layer - data patterns determine
    which shapes activate, not instruction fetch/decode.
    """

    def __init__(self, tiles_per_shape: int = 4):
        """
        Initialize the fabric.

        Args:
            tiles_per_shape: Number of tile instances per shape type
        """
        self.tiles: Dict[str, ShapeTile] = {}
        self.mesh = RoutingMesh()
        self.tile_pools: Dict[str, List[str]] = {}  # shape_name -> [tile_ids]

        # Create tile pool for each shape
        self._tile_counter = 0
        for shape_name in SHAPES:
            self.tile_pools[shape_name] = []
            for _ in range(tiles_per_shape):
                tile_id = f"tile_{self._tile_counter}"
                self._tile_counter += 1
                tile = ShapeTile(id=tile_id, shape_name=shape_name)
                self.tiles[tile_id] = tile
                self.tile_pools[shape_name].append(tile_id)

    def allocate_tile(self, shape_name: str) -> Optional[str]:
        """
        Allocate an idle tile for a shape.

        Returns:
            Tile ID or None if no tiles available
        """
        shape_name = shape_name.upper()
        if shape_name not in self.tile_pools:
            return None

        for tile_id in self.tile_pools[shape_name]:
            tile = self.tiles[tile_id]
            if tile.state == TileState.IDLE:
                tile.state = TileState.ALLOCATED
                return tile_id

        return None

    def free_tile(self, tile_id: str):
        """Free a tile for reuse."""
        if tile_id in self.tiles:
            self.tiles[tile_id].reset()

    def execute_tile(self, tile_id: str, inputs: List[Any]) -> Any:
        """
        Execute a tile with given inputs.

        Args:
            tile_id: The tile to execute
            inputs: Input values

        Returns:
            Output value
        """
        tile = self.tiles[tile_id]
        tile.inputs = inputs
        return tile.execute()

    def stats(self) -> Dict[str, Any]:
        """Get fabric statistics."""
        total_tiles = len(self.tiles)
        idle = sum(1 for t in self.tiles.values() if t.state == TileState.IDLE)
        allocated = sum(1 for t in self.tiles.values() if t.state == TileState.ALLOCATED)
        executing = sum(1 for t in self.tiles.values() if t.state == TileState.EXECUTING)
        done = sum(1 for t in self.tiles.values() if t.state == TileState.DONE)

        return {
            "total_tiles": total_tiles,
            "tiles_per_shape": len(self.tiles) // len(SHAPES),
            "idle": idle,
            "allocated": allocated,
            "executing": executing,
            "done": done,
            "route_count": len(self.mesh.routes),
        }


class PatternRouter:
    """
    Routes computation based on data patterns.

    This implements the key insight: routing IS computation.
    The pattern of the data determines which shapes activate.
    """

    def __init__(self, fabric: ShapeFabric):
        self.fabric = fabric
        self.pattern_table: Dict[str, Callable] = {}

    def register_pattern(self, name: str, router: Callable):
        """
        Register a pattern router.

        The router is a function that takes inputs and returns
        a list of (shape_name, inputs, output_handler) tuples.
        """
        self.pattern_table[name] = router

    def route(self, pattern: str, *args) -> Any:
        """
        Route computation based on pattern.

        Args:
            pattern: Pattern name
            *args: Input arguments

        Returns:
            Computation result
        """
        if pattern not in self.pattern_table:
            raise ValueError(f"Unknown pattern: {pattern}")

        router = self.pattern_table[pattern]
        return router(self.fabric, *args)


# =============================================================================
# Built-in Pattern Routers
# =============================================================================

def route_full_adder(fabric: ShapeFabric, a: int, b: int, cin: int) -> Tuple[int, int]:
    """Route a full adder computation through the fabric."""
    # Allocate tiles
    xor1_tile = fabric.allocate_tile("XOR")
    xor2_tile = fabric.allocate_tile("XOR")
    and1_tile = fabric.allocate_tile("AND")
    and2_tile = fabric.allocate_tile("AND")
    or_tile = fabric.allocate_tile("OR")

    try:
        # Execute in dataflow order
        # Level 1: xor1 = a ^ b, and1 = a & b
        xor1_result = fabric.execute_tile(xor1_tile, [a, b])
        and1_result = fabric.execute_tile(and1_tile, [a, b])

        # Level 2: sum = xor1 ^ cin, and2 = xor1 & cin
        sum_result = fabric.execute_tile(xor2_tile, [xor1_result, cin])
        and2_result = fabric.execute_tile(and2_tile, [xor1_result, cin])

        # Level 3: cout = and1 | and2
        cout_result = fabric.execute_tile(or_tile, [and1_result, and2_result])

        return sum_result, cout_result

    finally:
        # Free tiles
        for tile_id in [xor1_tile, xor2_tile, and1_tile, and2_tile, or_tile]:
            if tile_id:
                fabric.free_tile(tile_id)


def route_alu(fabric: ShapeFabric, a: int, b: int, op: int) -> int:
    """Route an ALU operation through the fabric."""
    op_map = {
        0: "ADD",
        1: "SUB",
        2: "AND",
        3: "OR",
        4: "XOR",
        5: "SHL",
        6: "SHR",
        7: "MUL",
    }

    shape_name = op_map.get(op, "ADD")
    tile_id = fabric.allocate_tile(shape_name)

    if tile_id is None:
        raise RuntimeError(f"No available tiles for {shape_name}")

    try:
        return fabric.execute_tile(tile_id, [a, b])
    finally:
        fabric.free_tile(tile_id)


def route_ripple_adder(fabric: ShapeFabric, a: int, b: int, n_bits: int = 8) -> int:
    """Route an N-bit ripple carry adder through the fabric."""
    result = 0
    carry = 0

    for i in range(n_bits):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        sum_bit, carry = route_full_adder(fabric, a_bit, b_bit, carry)
        result |= (sum_bit << i)

    return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_fabric(tiles_per_shape: int = 4) -> Tuple[ShapeFabric, PatternRouter]:
    """
    Create a fabric with built-in pattern routers.

    Returns:
        (fabric, router) tuple
    """
    fabric = ShapeFabric(tiles_per_shape=tiles_per_shape)
    router = PatternRouter(fabric)

    # Register built-in patterns
    router.register_pattern("full_adder", route_full_adder)
    router.register_pattern("alu", route_alu)
    router.register_pattern("ripple_adder", route_ripple_adder)

    return fabric, router


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_fabric():
    """Benchmark the routing fabric."""
    print("Routing Fabric Benchmark")
    print("=" * 60)

    fabric, router = create_fabric(tiles_per_shape=8)

    print(f"\nFabric Stats: {fabric.stats()}")

    # Benchmark full adder
    iterations = 10000
    start = time.perf_counter()
    for i in range(iterations):
        router.route("full_adder", i % 2, (i // 2) % 2, (i // 4) % 2)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nFull Adder ({iterations} iterations):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Per operation: {elapsed/iterations*1000:.2f} µs")
    print(f"  Ops/sec: {iterations/(elapsed/1000):,.0f}")

    # Benchmark ALU
    start = time.perf_counter()
    for i in range(iterations):
        router.route("alu", i, i + 1, i % 8)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nALU ({iterations} iterations):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Per operation: {elapsed/iterations*1000:.2f} µs")
    print(f"  Ops/sec: {iterations/(elapsed/1000):,.0f}")

    # Benchmark 8-bit adder
    iterations = 1000
    start = time.perf_counter()
    for i in range(iterations):
        router.route("ripple_adder", i, i + 1, 8)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\n8-bit Ripple Adder ({iterations} iterations):")
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Per operation: {elapsed/iterations*1000:.2f} µs")
    print(f"  Ops/sec: {iterations/(elapsed/1000):,.0f}")


if __name__ == "__main__":
    print("Routing Fabric - Self Test")
    print("=" * 50)

    fabric, router = create_fabric()

    # Test full adder
    print("\nFull Adder Tests:")
    test_cases = [
        ((0, 0, 0), (0, 0)),
        ((0, 0, 1), (1, 0)),
        ((0, 1, 0), (1, 0)),
        ((0, 1, 1), (0, 1)),
        ((1, 0, 0), (1, 0)),
        ((1, 0, 1), (0, 1)),
        ((1, 1, 0), (0, 1)),
        ((1, 1, 1), (1, 1)),
    ]

    for (a, b, cin), (exp_sum, exp_cout) in test_cases:
        result = router.route("full_adder", a, b, cin)
        status = "PASS" if result == (exp_sum, exp_cout) else f"FAIL (got {result})"
        print(f"  FA({a},{b},{cin}) = {result} : {status}")

    # Test ALU
    print("\nALU Tests:")
    alu_tests = [
        ((5, 3, 0), 8),   # ADD
        ((5, 3, 1), 2),   # SUB
        ((5, 3, 2), 1),   # AND
        ((5, 3, 3), 7),   # OR
        ((5, 3, 4), 6),   # XOR
    ]

    for (a, b, op), expected in alu_tests:
        result = router.route("alu", a, b, op)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  ALU({a},{b},op={op}) = {result} : {status}")

    # Test ripple adder
    print("\nRipple Adder Tests:")
    for a, b in [(15, 1), (100, 50), (255, 1)]:
        result = router.route("ripple_adder", a, b, 8)
        expected = (a + b) & 0xFF
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  ADD8({a},{b}) = {result} : {status}")

    print(f"\nFabric Stats: {fabric.stats()}")

    print("\n" + "=" * 50)
    print("Running benchmark...")
    benchmark_fabric()
