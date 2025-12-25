"""
Processor for HSOS.

A processor is:
- A signature (its address)
- A shape (its computation)
- Local state (its memory)
- A message queue (its inbox)

Processors execute independently. Their concurrent execution IS parallelism.
No scheduler needed - just run all ready processors.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

from .message import Message
from .shapes import Shape, get_shape, SHAPES


@dataclass
class ProcessorState:
    """
    Complete state of a processor.

    This is everything needed to checkpoint and restore a processor.
    """
    signature: int
    shape_name: str
    local_state: Dict[str, Any]
    inbox: List[Message]
    processed_count: int = 0

    def copy(self) -> 'ProcessorState':
        """Deep copy of state."""
        return ProcessorState(
            signature=self.signature,
            shape_name=self.shape_name,
            local_state=dict(self.local_state),
            inbox=list(self.inbox),
            processed_count=self.processed_count,
        )


class Processor:
    """
    An addressable processor with a frozen shape.

    The processor:
    - Has a signature (content-addressable identity)
    - Executes a frozen shape (deterministic computation)
    - Maintains local state (its memory)
    - Receives messages (its inputs)
    - Emits messages (its outputs)

    Processors are the fundamental unit of computation in HSOS.
    """

    def __init__(
        self,
        signature: int,
        shape: str | Shape,
        initial_state: Optional[Dict[str, Any]] = None,
        output_target: Optional[int] = None,
    ):
        """
        Create a processor.

        Args:
            signature: The processor's address (for routing)
            shape: Name of frozen shape or Shape instance
            initial_state: Initial local state
            output_target: Default target for output messages (None = broadcast)
        """
        self.signature = signature

        if isinstance(shape, str):
            self.shape = get_shape(shape)
            self.shape_name = shape
        else:
            self.shape = shape
            self.shape_name = shape.name

        self.state = initial_state or {}
        self.inbox: deque[Message] = deque()
        self.outbox: List[Message] = []
        self.output_target = output_target

        self.processed_count = 0
        self.cycle_count = 0

        # Optional hooks for observability
        self.on_receive: Optional[Callable[[Message], None]] = None
        self.on_execute: Optional[Callable[[Dict, Dict], None]] = None
        self.on_emit: Optional[Callable[[Message], None]] = None

    def receive(self, msg: Message):
        """Receive a message into inbox."""
        self.inbox.append(msg)
        if self.on_receive:
            self.on_receive(msg)

    def has_work(self) -> bool:
        """Check if processor has messages to process."""
        return len(self.inbox) > 0

    def step(self) -> List[Message]:
        """
        Execute one step: process one message, emit outputs.

        Returns:
            List of output messages to be routed.
        """
        if not self.inbox:
            return []

        # Get next message
        msg = self.inbox.popleft()

        # Execute shape
        new_state, outputs = self.shape(self.state, msg.payload)

        if self.on_execute:
            self.on_execute(msg.payload, outputs)

        # Update state
        self.state = new_state
        self.processed_count += 1

        # Create output messages
        self.outbox.clear()
        if outputs and self.output_target is not None:
            # Only emit if we have an explicit output target
            out_msg = Message(target=self.output_target, payload=outputs)
            self.outbox.append(out_msg)
            if self.on_emit:
                self.on_emit(out_msg)

        return list(self.outbox)

    def run_to_empty(self) -> List[Message]:
        """Process all messages in inbox."""
        all_outputs = []
        while self.has_work():
            outputs = self.step()
            all_outputs.extend(outputs)
        self.cycle_count += 1
        return all_outputs

    def get_state(self) -> ProcessorState:
        """Get complete processor state for checkpointing."""
        return ProcessorState(
            signature=self.signature,
            shape_name=self.shape_name,
            local_state=dict(self.state),
            inbox=list(self.inbox),
            processed_count=self.processed_count,
        )

    def set_state(self, state: ProcessorState):
        """Restore processor from checkpoint."""
        assert state.signature == self.signature
        self.state = dict(state.local_state)
        self.inbox = deque(state.inbox)
        self.processed_count = state.processed_count

    def __repr__(self):
        return f"Processor(0x{self.signature:04x}, {self.shape_name}, inbox={len(self.inbox)})"


class ProcessorBank:
    """
    A collection of identical processors with different signatures.

    Useful for creating tile-like structures where many processors
    share the same shape but have unique addresses.
    """

    def __init__(
        self,
        shape: str | Shape,
        signatures: List[int],
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.processors = [
            Processor(sig, shape, initial_state.copy() if initial_state else None)
            for sig in signatures
        ]
        self.by_signature = {p.signature: p for p in self.processors}

    def __len__(self):
        return len(self.processors)

    def __iter__(self):
        return iter(self.processors)

    def get(self, signature: int) -> Optional[Processor]:
        return self.by_signature.get(signature)

    def total_inbox_size(self) -> int:
        return sum(len(p.inbox) for p in self.processors)
