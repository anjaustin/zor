"""
Fabric: The core TriX Fabric class.

The fabric IS the program.
The fabric IS the database.
The fabric IS the knowledge.

A Fabric is:
- A recursive topology of processors
- Hierarchically addressed (tree of hypercubes)
- Stateless (messages carry context)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from .cluster import Cluster
from .processor import FabricProcessor
from .router import Router
from .message import Message


@dataclass
class ExecutionResult:
    """Result of executing a message through the fabric."""
    outputs: Dict[str, Any]          # Final outputs by signature
    messages_processed: int           # Total messages processed
    trace: List[str]                  # Execution trace
    cycles: int                       # Propagation cycles


class Fabric:
    """
    TriX Fabric: Unified Database-Computer.

    Attributes:
        name: Human-readable name
        root: Root cluster of the hierarchy
        router: Hierarchical router
        exports: Export points (where to collect outputs)
        entry_points: Entry points (where to inject messages)
    """

    def __init__(self, name: str = "fabric"):
        """
        Create a new fabric.

        Args:
            name: Human-readable name
        """
        self.name = name
        self.root = Cluster(path='/')
        self.router = Router(self.root)
        self.exports: Set[str] = set()       # Signatures of export processors
        self.entry_points: Set[str] = set()  # Signatures of entry processors

    def add_processor(self, processor: FabricProcessor) -> None:
        """
        Add a processor to the fabric.

        Creates intermediate clusters as needed.

        Args:
            processor: The processor to add
        """
        # Parse the signature to get path
        path_parts, local_sig = self.router.parse_signature(processor.signature)

        # Navigate/create path
        current = self.root
        for part in path_parts[1:]:  # Skip root
            child = current.get_child(part)
            if child is None:
                child = current.add_child(part)
            current = child

        # Add processor to the target cluster
        current.add_processor(processor)

    def get_processor(self, signature: str) -> Optional[FabricProcessor]:
        """Get a processor by exact signature."""
        return self.router.get_processor(signature)

    def add_cluster(self, path: str, name: Optional[str] = None) -> Cluster:
        """
        Add a cluster at the given path.

        Creates intermediate clusters as needed.

        Args:
            path: Cluster path (e.g., "/0/1")
            name: Optional human-readable name

        Returns:
            The created or existing cluster
        """
        if path == '/':
            return self.root

        parts = path.split('/')
        if parts[0] == '':
            parts = parts[1:]

        current = self.root
        current_path = ''
        for part in parts:
            if not part:
                continue
            current_path = f"{current_path}/{part}"
            child = current.get_child(part)
            if child is None:
                child = current.add_child(part)
            current = child

        if name:
            current.name = name

        return current

    def connect(self, source_sig: str, target_sig: str,
                source_output: str = 'result', target_input: str = 'a') -> None:
        """
        Connect two processors.

        Args:
            source_sig: Source processor signature
            target_sig: Target processor signature
            source_output: Output name on source (default: 'result')
            target_input: Input name on target (default: 'a')
        """
        source = self.get_processor(source_sig)
        if source is None:
            raise ValueError(f"Source processor not found: {source_sig}")

        source.connect_to(target_sig, target_input, source_output)

        # Also update target's input map
        target = self.get_processor(target_sig)
        if target:
            target.input_map[target_input] = f"{source_sig}.{source_output}"

    def set_export(self, signature: str) -> None:
        """Mark a processor as an export point."""
        self.exports.add(signature)

    def set_entry_point(self, signature: str) -> None:
        """Mark a processor as an entry point."""
        self.entry_points.add(signature)

    def inject(self, message: Message) -> ExecutionResult:
        """
        Inject a message and execute until quiescent.

        This is the core execution loop.

        Args:
            message: The message to inject

        Returns:
            ExecutionResult with outputs and trace
        """
        # Message queue
        queue: List[Message] = [message]
        processed = 0
        cycles = 0
        trace: List[str] = []
        outputs: Dict[str, Any] = {}
        max_cycles = 1000  # Safety limit

        while queue and cycles < max_cycles:
            cycles += 1
            next_queue: List[Message] = []

            for msg in queue:
                # Route the message
                result = self.router.route(msg.target)
                processor = result.processor

                if processor is None:
                    trace.append(f"MISS: {msg.target}")
                    continue

                # Evaluate the processor
                trace.append(f"EVAL: {processor.signature} <- {msg.payload}")
                output = processor.evaluate(msg.payload)
                processed += 1

                # Check if this is an export point
                if processor.signature in self.exports:
                    outputs[processor.signature] = output

                # Propagate to connected processors
                for conn in processor.connections:
                    new_payload = {conn.target_input: output.get(conn.output_name, 0.0)}
                    new_msg = msg.derive(
                        new_target=conn.target_signature,
                        new_payload=new_payload,
                        source=processor.signature
                    )
                    next_queue.append(new_msg)

            queue = next_queue

        return ExecutionResult(
            outputs=outputs,
            messages_processed=processed,
            trace=trace,
            cycles=cycles
        )

    def execute(self, inputs: Dict[str, Dict[str, Any]]) -> ExecutionResult:
        """
        Execute the fabric with inputs at entry points.

        Args:
            inputs: Dict mapping entry point signatures to their inputs

        Returns:
            ExecutionResult
        """
        # Create messages for each entry point
        all_outputs: Dict[str, Any] = {}
        total_processed = 0
        total_trace: List[str] = []
        max_cycles = 0

        for sig, payload in inputs.items():
            msg = Message(target=sig, payload=payload)
            result = self.inject(msg)
            all_outputs.update(result.outputs)
            total_processed += result.messages_processed
            total_trace.extend(result.trace)
            max_cycles = max(max_cycles, result.cycles)

        return ExecutionResult(
            outputs=all_outputs,
            messages_processed=total_processed,
            trace=total_trace,
            cycles=max_cycles
        )

    def iter_processors(self) -> Iterator[FabricProcessor]:
        """Iterate over all processors in the fabric."""
        return self.root.iter_all_processors()

    @property
    def processor_count(self) -> int:
        """Total number of processors."""
        return self.root.total_processor_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'root': self.root.to_dict(),
            'exports': list(self.exports),
            'entry_points': list(self.entry_points)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Fabric':
        """Deserialize from dictionary."""
        fabric = cls(name=d['name'])
        fabric.root = Cluster.from_dict(d['root'])
        fabric.router = Router(fabric.root)
        fabric.exports = set(d.get('exports', []))
        fabric.entry_points = set(d.get('entry_points', []))
        return fabric

    def __repr__(self):
        return f"Fabric({self.name!r}, processors={self.processor_count})"
