"""
Network for HSOS.

The network:
- Contains processors
- Routes messages by Hamming distance
- Executes until quiescent
- Logs everything for replay

"The topology encodes the algorithm."
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque
from dataclasses import dataclass

from .processor import Processor, ProcessorBank
from .message import Message, MessageLog


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two signatures."""
    return bin(a ^ b).count('1')


@dataclass
class NetworkStats:
    """Statistics from a network run."""
    cycles: int
    messages_total: int
    messages_per_cycle: float
    processors_active: int
    converged: bool


class Network:
    """
    A network of processors connected by content-addressable routing.

    Messages are routed to the processor with the closest signature
    (minimum Hamming distance). Ties are broken by lowest signature.

    The network runs in cycles:
    1. All processors with messages execute in parallel
    2. Output messages are collected
    3. Messages are routed to destinations
    4. Repeat until quiescent (no messages in flight)

    This is Hollywood Squares: processors are tiles,
    signatures are addresses, routing is message passing.
    """

    def __init__(self, signature_bits: int = 16):
        """
        Create a network.

        Args:
            signature_bits: Number of bits in signatures (for display)
        """
        self.processors: Dict[int, Processor] = {}
        self.signature_bits = signature_bits

        # Message handling
        self.pending: deque[Message] = deque()
        self.log = MessageLog()
        self.seq_counter = 0

        # Execution state
        self.cycle = 0
        self.external_source = -1  # Signature for external messages

    def add(self, processor: Processor):
        """Add a processor to the network."""
        if processor.signature in self.processors:
            raise ValueError(f"Duplicate signature: 0x{processor.signature:04x}")
        self.processors[processor.signature] = processor

    def add_bank(self, bank: ProcessorBank):
        """Add all processors from a bank."""
        for p in bank:
            self.add(p)

    def remove(self, signature: int):
        """Remove a processor from the network."""
        if signature in self.processors:
            del self.processors[signature]

    def get(self, signature: int) -> Optional[Processor]:
        """Get a processor by exact signature."""
        return self.processors.get(signature)

    def find_nearest(self, target: int) -> Optional[Processor]:
        """
        Find the processor with minimum Hamming distance to target.

        This is content-addressable routing.
        """
        if not self.processors:
            return None

        best_proc = None
        best_dist = float('inf')

        for sig, proc in self.processors.items():
            dist = hamming_distance(sig, target)
            if dist < best_dist or (dist == best_dist and sig < best_proc.signature):
                best_dist = dist
                best_proc = proc

        return best_proc

    def send(self, msg: Message, source: Optional[int] = None):
        """
        Queue a message for delivery.

        Args:
            msg: Message to send
            source: Source signature (None = external)
        """
        src = source if source is not None else self.external_source
        self.seq_counter += 1
        msg_with_meta = msg.with_metadata(src, self.seq_counter)
        self.pending.append(msg_with_meta)

    def _deliver_pending(self):
        """Deliver all pending messages to their destinations."""
        while self.pending:
            msg = self.pending.popleft()

            # Find destination by Hamming distance
            dest = self.find_nearest(msg.target)
            if dest is None:
                continue  # No processors - drop message

            # Deliver
            dest.receive(msg)
            self.log.record(msg, dest.signature, self.cycle)

    def _execute_cycle(self) -> int:
        """
        Execute one cycle: all processors with work run in parallel.

        Returns:
            Number of output messages generated.
        """
        self.cycle += 1

        # Collect all processors with work
        active = [p for p in self.processors.values() if p.has_work()]

        if not active:
            return 0

        # Execute all in "parallel" (deterministic order by signature)
        active.sort(key=lambda p: p.signature)

        all_outputs: List[Tuple[int, Message]] = []
        for proc in active:
            outputs = proc.run_to_empty()
            for msg in outputs:
                all_outputs.append((proc.signature, msg))

        # Queue outputs for next cycle
        for source_sig, msg in all_outputs:
            self.send(msg, source=source_sig)

        return len(all_outputs)

    def run(self, max_cycles: int = 1000) -> NetworkStats:
        """
        Run network until quiescent or max_cycles.

        Returns:
            Statistics from the run.
        """
        start_cycle = self.cycle
        messages_total = 0

        while self.cycle - start_cycle < max_cycles:
            # Deliver pending messages
            self._deliver_pending()

            # Check if any processor has work
            if not any(p.has_work() for p in self.processors.values()):
                break

            # Execute cycle
            outputs = self._execute_cycle()
            messages_total += outputs

        cycles_run = self.cycle - start_cycle
        converged = not any(p.has_work() for p in self.processors.values())
        converged = converged and len(self.pending) == 0

        return NetworkStats(
            cycles=cycles_run,
            messages_total=messages_total,
            messages_per_cycle=messages_total / cycles_run if cycles_run > 0 else 0,
            processors_active=len([p for p in self.processors.values() if p.processed_count > 0]),
            converged=converged,
        )

    def step(self) -> bool:
        """
        Execute one cycle.

        Returns:
            True if there is more work to do.
        """
        self._deliver_pending()

        has_work = any(p.has_work() for p in self.processors.values())
        if has_work:
            self._execute_cycle()

        return has_work or len(self.pending) > 0

    def inject(self, target: int, payload: Dict[str, Any]):
        """Inject a message from external source."""
        self.send(Message(target=target, payload=payload))

    def broadcast(self, payload: Dict[str, Any]):
        """Send message to all processors."""
        for sig in self.processors:
            self.inject(sig, payload)

    def collect_states(self) -> Dict[int, Dict[str, Any]]:
        """Collect local state from all processors."""
        return {sig: p.state.copy() for sig, p in self.processors.items()}

    def collect_outputs(self) -> Dict[int, List[Dict]]:
        """Collect outputs from processor outboxes."""
        return {
            sig: [m.payload for m in p.outbox]
            for sig, p in self.processors.items()
        }

    def reset(self):
        """Reset network to initial state."""
        self.pending.clear()
        self.log = MessageLog()
        self.seq_counter = 0
        self.cycle = 0
        for p in self.processors.values():
            p.inbox.clear()
            p.outbox.clear()
            p.processed_count = 0
            p.cycle_count = 0

    def status(self) -> str:
        """Human-readable status."""
        lines = [
            f"Network: {len(self.processors)} processors, cycle {self.cycle}",
            f"  Pending: {len(self.pending)} messages",
            f"  Log: {len(self.log)} entries",
        ]

        active = [(sig, len(p.inbox)) for sig, p in self.processors.items() if p.has_work()]
        if active:
            lines.append(f"  Active: {active[:5]}{'...' if len(active) > 5 else ''}")

        return "\n".join(lines)

    def __repr__(self):
        return f"Network({len(self.processors)} processors, cycle={self.cycle})"


# =============================================================================
# TOPOLOGY BUILDERS
# =============================================================================

def create_ring(shape: str, n: int, signature_base: int = 0) -> Network:
    """
    Create a ring topology.

    Each processor's output_target is the next processor in the ring.
    """
    net = Network()
    sigs = [signature_base + i for i in range(n)]

    for i, sig in enumerate(sigs):
        next_sig = sigs[(i + 1) % n]
        net.add(Processor(sig, shape, output_target=next_sig))

    return net


def create_pipeline(shapes: List[str], signature_base: int = 0) -> Network:
    """
    Create a pipeline topology.

    Each processor feeds the next. Last processor has no output target.
    """
    net = Network()
    n = len(shapes)

    for i, shape in enumerate(shapes):
        sig = signature_base + i
        next_sig = signature_base + i + 1 if i < n - 1 else None
        net.add(Processor(sig, shape, output_target=next_sig))

    return net


def create_star(center_shape: str, leaf_shape: str, n_leaves: int, center_sig: int = 0) -> Network:
    """
    Create a star topology.

    Center processor receives from all leaves.
    Leaves receive from center.
    """
    net = Network()

    # Center
    net.add(Processor(center_sig, center_shape))

    # Leaves
    for i in range(n_leaves):
        leaf_sig = center_sig + 1 + i
        net.add(Processor(leaf_sig, leaf_shape, output_target=center_sig))

    return net


def create_mesh(shape: str, width: int, height: int, signature_base: int = 0) -> Network:
    """
    Create a 2D mesh topology.

    Processors are connected to their 4-neighbors (no output_target set -
    use signatures for content-addressable routing).
    """
    net = Network()

    for y in range(height):
        for x in range(width):
            sig = signature_base + y * width + x
            net.add(Processor(sig, shape))

    return net
