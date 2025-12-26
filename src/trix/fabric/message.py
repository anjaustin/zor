"""
Message: The unit of communication in TriX Fabric.

Messages carry state THROUGH the fabric, not stored IN processors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """
    A message flowing through the fabric.

    Attributes:
        target: Hierarchical signature (e.g., "/0/1/0b1010")
        payload: Input data for the processor
        context: Accumulated state (trace, depth, metadata)
        source: Where this message came from (optional)
    """
    target: str
    payload: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self):
        # Ensure context has standard fields
        if 'trace' not in self.context:
            self.context['trace'] = []
        if 'depth' not in self.context:
            self.context['depth'] = 0

    def derive(self, new_target: str, new_payload: Dict[str, Any], source: str) -> 'Message':
        """Create a derived message with updated trace."""
        new_context = {
            'trace': self.context['trace'] + [source],
            'depth': self.context['depth'] + 1,
            **{k: v for k, v in self.context.items() if k not in ('trace', 'depth')}
        }
        return Message(
            target=new_target,
            payload=new_payload,
            context=new_context,
            source=source
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'target': self.target,
            'payload': self.payload,
            'context': self.context,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Message':
        """Deserialize from dictionary."""
        return cls(
            target=d['target'],
            payload=d['payload'],
            context=d.get('context', {}),
            source=d.get('source')
        )

    def __repr__(self):
        return f"Message(target={self.target!r}, payload={self.payload}, depth={self.context['depth']})"
