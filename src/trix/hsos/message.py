"""
Messages for HSOS.

A message carries data between processors.
Routing is by signature matching (Hamming distance).

Messages are immutable. The log provides full traceability.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class Message:
    """
    An immutable message between processors.

    Attributes:
        target: Target signature (routes to closest match)
        payload: Data carried by the message
        source: Source processor signature (set by network)
        seq: Sequence number (set by network)
        timestamp: Creation time (set by network)
    """
    target: int
    payload: Dict[str, Any]
    source: Optional[int] = None
    seq: Optional[int] = None
    timestamp: Optional[float] = None

    def with_metadata(self, source: int, seq: int) -> 'Message':
        """Create a new message with metadata added."""
        return Message(
            target=self.target,
            payload=self.payload,
            source=source,
            seq=seq,
            timestamp=time.time(),
        )

    def __repr__(self):
        src = f"0x{self.source:04x}" if self.source is not None else "?"
        tgt = f"0x{self.target:04x}"
        return f"Message({src} -> {tgt}, {self.payload})"


@dataclass
class MessageLog:
    """
    Complete log of all messages for replay and verification.

    "Enforced observability" - every message is recorded.
    """
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, msg: Message, delivered_to: int, cycle: int):
        """Record a message delivery."""
        self.entries.append({
            'cycle': cycle,
            'seq': msg.seq,
            'source': msg.source,
            'target': msg.target,
            'delivered_to': delivered_to,
            'payload': msg.payload,
            'timestamp': msg.timestamp,
        })

    def __len__(self):
        return len(self.entries)

    def replay(self):
        """Iterate through messages in order."""
        return iter(sorted(self.entries, key=lambda e: (e['cycle'], e['seq'])))

    def filter_by_processor(self, sig: int):
        """Get all messages involving a processor."""
        return [
            e for e in self.entries
            if e['source'] == sig or e['delivered_to'] == sig
        ]

    def filter_by_cycle(self, cycle: int):
        """Get all messages in a cycle."""
        return [e for e in self.entries if e['cycle'] == cycle]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        if not self.entries:
            return {'total': 0}

        cycles = set(e['cycle'] for e in self.entries)
        sources = set(e['source'] for e in self.entries if e['source'] is not None)
        targets = set(e['delivered_to'] for e in self.entries)

        return {
            'total': len(self.entries),
            'cycles': len(cycles),
            'sources': len(sources),
            'targets': len(targets),
            'first_cycle': min(cycles),
            'last_cycle': max(cycles),
        }

    def print_trace(self, limit: int = 20):
        """Print human-readable trace."""
        print(f"Message Trace ({len(self.entries)} total)")
        print("-" * 60)

        for i, e in enumerate(self.entries[:limit]):
            src = f"0x{e['source']:04x}" if e['source'] is not None else "ext"
            tgt = f"0x{e['delivered_to']:04x}"
            print(f"  [{e['cycle']:3d}] {src} -> {tgt}: {e['payload']}")

        if len(self.entries) > limit:
            print(f"  ... ({len(self.entries) - limit} more)")
