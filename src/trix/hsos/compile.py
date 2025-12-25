"""
HSOS → GILLIES Compiler

Translates HSOS Networks to GILLIES C/CUDA code.

"The compiler isn't transforming semantics. It's lowering abstraction."

Usage:
    from trix.hsos import Network, Processor
    from trix.hsos.compile import GilliesCompiler

    net = Network()
    net.add(Processor(signature=0, shape='xor'))
    # ...

    compiler = GilliesCompiler(net)
    c_code = compiler.compile()
    print(c_code)
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import textwrap

from .network import Network
from .processor import Processor


# =============================================================================
# SHAPE MAPPING - HSOS names to GILLIES enum
# =============================================================================

SHAPE_TO_GILLIES = {
    'xor': 'GILLIES_SHAPE_XOR',
    'and': 'GILLIES_SHAPE_AND',
    'or': 'GILLIES_SHAPE_OR',
    'not': 'GILLIES_SHAPE_NOT',
    'nand': 'GILLIES_SHAPE_NAND',
    'nor': 'GILLIES_SHAPE_NOR',
    'xnor': 'GILLIES_SHAPE_XNOR',
    'identity': 'GILLIES_SHAPE_IDENTITY',
    'full_adder': 'GILLIES_SHAPE_FULL_ADDER',
    'half_adder': 'GILLIES_SHAPE_HALF_ADDER',
    'mux': 'GILLIES_SHAPE_MUX',
    'compare': 'GILLIES_SHAPE_COMPARE',
    'accumulator': 'GILLIES_SHAPE_ACCUMULATOR',
    'counter': 'GILLIES_SHAPE_COUNTER',
    'latch': 'GILLIES_SHAPE_LATCH',
}

# Port counts per shape (inputs, outputs, state)
SHAPE_PORTS = {
    'xor': (2, 1, 0),
    'and': (2, 1, 0),
    'or': (2, 1, 0),
    'not': (1, 1, 0),
    'nand': (2, 1, 0),
    'nor': (2, 1, 0),
    'xnor': (2, 1, 0),
    'identity': (1, 1, 0),
    'full_adder': (3, 2, 0),      # a, b, c_in → sum, carry
    'half_adder': (2, 2, 0),      # a, b → sum, carry
    'mux': (3, 1, 0),             # a, b, sel → result
    'compare': (2, 3, 0),         # a, b → lt, eq, gt
    'accumulator': (1, 1, 1),     # value → result, state: acc
    'counter': (0, 1, 1),         # (none) → count, state: count
    'latch': (2, 1, 1),           # data, set → value, state: value
}


# =============================================================================
# PORT ALLOCATION
# =============================================================================

@dataclass
class PortAllocation:
    """Port assignment for a single processor."""
    signature: int
    shape_name: str
    base_port: int
    input_ports: List[int] = field(default_factory=list)
    output_ports: List[int] = field(default_factory=list)
    state_ports: List[int] = field(default_factory=list)

    @property
    def total_ports(self) -> int:
        return len(self.input_ports) + len(self.output_ports) + len(self.state_ports)


@dataclass
class WireConnection:
    """A connection from one processor's output to another's input."""
    source_sig: int
    source_port: int
    dest_sig: int
    dest_port: int
    payload_key: str


# =============================================================================
# COMPILER
# =============================================================================

class GilliesCompiler:
    """
    Compile HSOS Network to GILLIES C code.

    Three-pass translation:
    1. Analyze: Enumerate processors, infer routing
    2. Allocate: Assign port ranges
    3. Emit: Generate C code
    """

    def __init__(self, network: Network, name: str = "hsos_network"):
        self.network = network
        self.name = name

        # Analysis results
        self.processors: List[Processor] = []
        self.port_map: Dict[int, PortAllocation] = {}
        self.wires: List[WireConnection] = []

        # Port allocation state
        self.next_port = 0
        self.state_port_base = 200  # State starts at port 200

    # =========================================================================
    # PASS 1: ANALYZE
    # =========================================================================

    def analyze(self):
        """Pass 1: Enumerate processors and build routing table."""
        self.processors = list(self.network.processors.values())
        self.processors.sort(key=lambda p: p.signature)

        # Build wires from output_target relationships
        for proc in self.processors:
            if proc.output_target is not None:
                dest_proc = self.network.get(proc.output_target)
                if dest_proc is None:
                    # Find nearest
                    dest_proc = self.network.find_nearest(proc.output_target)

                if dest_proc:
                    # Wire will be created after port allocation
                    pass

    # =========================================================================
    # PASS 2: ALLOCATE
    # =========================================================================

    def allocate(self):
        """Pass 2: Assign port ranges to each processor."""
        self.next_port = 0
        state_port = self.state_port_base

        for proc in self.processors:
            shape = proc.shape_name
            n_inputs, n_outputs, n_state = SHAPE_PORTS.get(shape, (2, 1, 0))

            alloc = PortAllocation(
                signature=proc.signature,
                shape_name=shape,
                base_port=self.next_port,
                input_ports=[self.next_port + i for i in range(n_inputs)],
                output_ports=[self.next_port + n_inputs + i for i in range(n_outputs)],
                state_ports=[state_port + i for i in range(n_state)],
            )

            self.port_map[proc.signature] = alloc
            self.next_port += n_inputs + n_outputs
            state_port += n_state

        # Build wires based on output_target
        self._build_wires()

    def _build_wires(self):
        """Build wire connections from processor output_targets."""
        self.wires = []

        for proc in self.processors:
            if proc.output_target is None:
                continue

            src_alloc = self.port_map[proc.signature]

            # Find destination
            dest_sig = proc.output_target
            if dest_sig not in self.port_map:
                # Find nearest
                dest_proc = self.network.find_nearest(dest_sig)
                if dest_proc:
                    dest_sig = dest_proc.signature

            if dest_sig not in self.port_map:
                continue

            dest_alloc = self.port_map[dest_sig]

            # Wire first output to first input of destination
            if src_alloc.output_ports and dest_alloc.input_ports:
                self.wires.append(WireConnection(
                    source_sig=proc.signature,
                    source_port=src_alloc.output_ports[0],
                    dest_sig=dest_sig,
                    dest_port=dest_alloc.input_ports[0],
                    payload_key='result',
                ))

    # =========================================================================
    # PASS 3: EMIT
    # =========================================================================

    def emit(self, initial_values: Optional[Dict[int, Dict[str, float]]] = None) -> str:
        """Pass 3: Generate C code."""
        lines = []

        # Header
        lines.append(self._emit_header())

        # Main function
        lines.append('int main() {')
        lines.append('    gillies_context_t* ctx = gillies_create();')
        lines.append('    gillies_set_tracing(ctx, true);')
        lines.append('')

        # Port layout comment
        lines.append('    // === Port Layout ===')
        for sig, alloc in sorted(self.port_map.items()):
            lines.append(f'    // Processor 0x{sig:04x} ({alloc.shape_name}): '
                        f'in={alloc.input_ports}, out={alloc.output_ports}')
        lines.append('')

        # Initial values
        if initial_values:
            lines.append('    // === Initial Values ===')
            for sig, values in initial_values.items():
                if sig in self.port_map:
                    alloc = self.port_map[sig]
                    for i, (key, val) in enumerate(values.items()):
                        if i < len(alloc.input_ports):
                            lines.append(f'    gillies_set_port(ctx, {alloc.input_ports[i]}, {val}f);  // {key}')
            lines.append('')

        # Wire carry chain (for ripple adders)
        lines.append('    // === Wire Connections ===')
        for wire in self.wires:
            lines.append(f'    // Port {wire.source_port} → Port {wire.dest_port}')
        lines.append('')

        # Invocations
        lines.append('    // === Invocations ===')
        for proc in self.processors:
            lines.append(self._emit_invocation(proc))
        lines.append('')

        # Execute
        lines.append('    // === Execute ===')
        lines.append('    gillies_execute(ctx);')
        lines.append('')

        # Results
        lines.append('    // === Read Results ===')
        for sig, alloc in sorted(self.port_map.items()):
            for i, port in enumerate(alloc.output_ports):
                lines.append(f'    printf("Processor 0x{sig:04x} output[{i}]: %f\\n", gillies_get_port(ctx, {port}));')
        lines.append('')

        # Stats
        lines.append('    // === Statistics ===')
        lines.append('    gillies_print_stats(ctx);')
        lines.append('')

        # Cleanup
        lines.append('    gillies_destroy(ctx);')
        lines.append('    return 0;')
        lines.append('}')

        return '\n'.join(lines)

    def _emit_header(self) -> str:
        """Emit file header and includes."""
        return textwrap.dedent(f'''\
            /*
             * {self.name} - Generated by HSOS → GILLIES Compiler
             *
             * Processors: {len(self.processors)}
             * Ports used: {self.next_port}
             *
             * Generated from HSOS Network
             */

            #include "gillies.h"
            #include <stdio.h>

        ''')

    def _emit_invocation(self, proc: Processor) -> str:
        """Emit a single invocation."""
        alloc = self.port_map[proc.signature]
        shape_enum = SHAPE_TO_GILLIES.get(proc.shape_name, 'GILLIES_SHAPE_IDENTITY')

        # Get ports
        in0 = alloc.input_ports[0] if len(alloc.input_ports) > 0 else 0
        in1 = alloc.input_ports[1] if len(alloc.input_ports) > 1 else 0
        out0 = alloc.output_ports[0] if len(alloc.output_ports) > 0 else 0

        return f'    gillies_invoke(ctx, {shape_enum}, {in0}, {in1}, {out0});'

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def compile(self, initial_values: Optional[Dict[int, Dict[str, float]]] = None) -> str:
        """Run all passes and return generated C code."""
        self.analyze()
        self.allocate()
        return self.emit(initial_values)

    def get_port_map(self) -> Dict[int, PortAllocation]:
        """Get the port allocation map (for testing/debugging)."""
        return self.port_map

    def get_wires(self) -> List[WireConnection]:
        """Get the wire connections (for testing/debugging)."""
        return self.wires


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def compile_to_gillies(network: Network, name: str = "hsos_network") -> str:
    """Compile an HSOS Network to GILLIES C code."""
    compiler = GilliesCompiler(network, name)
    return compiler.compile()


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == '__main__':
    from .network import Network
    from .processor import Processor

    # Create a simple 4-XOR network
    net = Network()
    for i in range(4):
        net.add(Processor(signature=i, shape='xor'))

    compiler = GilliesCompiler(net, "four_xor")
    code = compiler.compile()
    print(code)
