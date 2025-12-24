"""
Event Stream - Glassbox protocol implementation for CLI.

Handles event subscription, filtering, and streaming.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import time
import json


@dataclass
class Event:
    """A single Glassbox event."""
    id: int
    type: str
    source: str
    message: str
    frame: int = 0
    timestamp: float = field(default_factory=time.time)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "message": self.message,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(**d)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class EventStream:
    """
    Event stream manager.

    Provides:
    - Event buffering
    - Filtering by type/source/frame
    - Subscription callbacks
    - Export to various formats
    """

    def __init__(self, max_events: int = 10000):
        self.events: List[Event] = []
        self.max_events = max_events
        self._next_id = 1
        self._subscribers: List[Callable[[Event], None]] = []
        self._filters: Dict[str, Any] = {}

    def emit(self, type: str, source: str, message: str,
             frame: int = 0, data: Optional[Dict[str, Any]] = None) -> Event:
        """Emit a new event."""
        event = Event(
            id=self._next_id,
            type=type,
            source=source,
            message=message,
            frame=frame,
            data=data
        )
        self._next_id += 1

        self.events.append(event)

        # Trim if over limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Notify subscribers
        for callback in self._subscribers:
            if self._matches_filter(event):
                callback(event)

        return event

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def set_filter(self, type_prefix: Optional[str] = None,
                   source: Optional[str] = None,
                   min_frame: Optional[int] = None,
                   max_frame: Optional[int] = None) -> None:
        """Set event filter."""
        self._filters = {
            "type_prefix": type_prefix,
            "source": source,
            "min_frame": min_frame,
            "max_frame": max_frame
        }

    def clear_filter(self) -> None:
        """Clear event filter."""
        self._filters = {}

    def _matches_filter(self, event: Event) -> bool:
        """Check if event matches current filter."""
        if not self._filters:
            return True

        if self._filters.get("type_prefix"):
            if not event.type.startswith(self._filters["type_prefix"]):
                return False

        if self._filters.get("source"):
            if event.source != self._filters["source"]:
                return False

        if self._filters.get("min_frame") is not None:
            if event.frame < self._filters["min_frame"]:
                return False

        if self._filters.get("max_frame") is not None:
            if event.frame > self._filters["max_frame"]:
                return False

        return True

    def query(self, type_prefix: Optional[str] = None,
              source: Optional[str] = None,
              frame: Optional[int] = None,
              limit: int = 100) -> List[Event]:
        """Query events with filters."""
        results = []
        for event in reversed(self.events):
            if type_prefix and not event.type.startswith(type_prefix):
                continue
            if source and event.source != source:
                continue
            if frame is not None and event.frame != frame:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return list(reversed(results))

    def recent(self, n: int = 10) -> List[Event]:
        """Get n most recent events."""
        return self.events[-n:]

    def clear(self) -> None:
        """Clear all events."""
        self.events = []
        self._next_id = 1

    def export_json(self) -> str:
        """Export all events as JSON."""
        return json.dumps([e.to_dict() for e in self.events], indent=2)

    def export_jsonl(self) -> str:
        """Export events as JSON Lines."""
        return "\n".join(e.to_json() for e in self.events)

    def stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        by_type: Dict[str, int] = {}
        by_source: Dict[str, int] = {}

        for event in self.events:
            by_type[event.type] = by_type.get(event.type, 0) + 1
            by_source[event.source] = by_source.get(event.source, 0) + 1

        return {
            "total": len(self.events),
            "by_type": by_type,
            "by_source": by_source,
            "first_frame": self.events[0].frame if self.events else None,
            "last_frame": self.events[-1].frame if self.events else None
        }
