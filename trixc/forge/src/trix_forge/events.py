"""
Event Spine - Glassbox protocol implementation.

The Event Spine is the nervous system of TRIX Forge:
- Every significant action emits an event
- Events are frame-stamped for replay
- Subscribers receive events in real-time
- Events can be filtered, queried, and exported

The Glassbox Principle: The data structures that compute
are the same ones that report.
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import threading


class EventType(Enum):
    """Standard event types."""
    # Lifecycle
    LOAD_START = "load.start"
    LOAD_COMPLETE = "load.complete"
    VALIDATE_START = "validate.start"
    VALIDATE_PASS = "validate.pass"
    VALIDATE_FAIL = "validate.fail"

    # Compilation
    COMPILE_START = "compile.start"
    COMPILE_COMPLETE = "compile.complete"
    OPTIMIZE_START = "optimize.start"
    OPTIMIZE_FUSE = "optimize.fuse"
    OPTIMIZE_QUANTIZE = "optimize.quantize"
    EMIT_START = "emit.start"
    EMIT_COMPLETE = "emit.complete"
    PACKAGE_START = "package.start"
    PACKAGE_COMPLETE = "package.complete"

    # Inference
    FORWARD_START = "forward.start"
    FORWARD_COMPLETE = "forward.complete"
    LAYER_ACTIVATION = "layer.activation"
    OUTPUT_DECISION = "output.decision"

    # Training
    BACKWARD_START = "backward.start"
    BACKWARD_COMPLETE = "backward.complete"
    WEIGHT_UPDATE = "weight.update"

    # Levers
    LEVER_SET = "lever.set"
    LEVER_RESET = "lever.reset"

    # Errors
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Event:
    """
    A single Glassbox event.

    Attributes:
        id: Unique event ID
        type: Event type (string or EventType)
        source: Component that emitted the event
        message: Human-readable message
        frame: Frame number (for inference/training)
        timestamp: Unix timestamp
        data: Optional structured data
    """
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

    def __repr__(self) -> str:
        return f"Event({self.id}, {self.type}, '{self.message[:30]}...')"


class EventSpine:
    """
    The central event bus for TRIX Forge.

    Thread-safe event emission and subscription.

    Usage:
        spine = EventSpine()

        # Subscribe to events
        spine.subscribe(lambda e: print(e))

        # Emit events
        spine.emit("load.complete", "loader", "Model loaded successfully")

        # Query events
        recent = spine.query(type_prefix="load", limit=10)

        # Export
        json_data = spine.export_json()
    """

    def __init__(self, max_events: int = 10000):
        self.events: List[Event] = []
        self.max_events = max_events
        self._next_id = 1
        self._subscribers: List[Callable[[Event], None]] = []
        self._filters: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._current_frame = 0

    def emit(self, type: str, source: str, message: str,
             frame: Optional[int] = None,
             data: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a new event.

        Args:
            type: Event type string
            source: Component emitting the event
            message: Human-readable description
            frame: Frame number (uses current if not specified)
            data: Optional structured data

        Returns:
            The created Event
        """
        with self._lock:
            event = Event(
                id=self._next_id,
                type=type,
                source=source,
                message=message,
                frame=frame if frame is not None else self._current_frame,
                data=data
            )
            self._next_id += 1

            self.events.append(event)

            # Trim if over limit
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

        # Notify subscribers (outside lock)
        for callback in self._subscribers:
            if self._matches_filter(event):
                try:
                    callback(event)
                except Exception:
                    pass  # Don't let subscriber errors break the spine

        return event

    def set_frame(self, frame: int) -> None:
        """Set the current frame number."""
        self._current_frame = frame

    def next_frame(self) -> int:
        """Increment and return the frame number."""
        self._current_frame += 1
        return self._current_frame

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
        """Set filter for subscriber notifications."""
        self._filters = {
            "type_prefix": type_prefix,
            "source": source,
            "min_frame": min_frame,
            "max_frame": max_frame
        }

    def clear_filter(self) -> None:
        """Clear subscriber filter."""
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
        """
        Query events with filters.

        Args:
            type_prefix: Filter by type prefix
            source: Filter by source
            frame: Filter by exact frame
            limit: Maximum results

        Returns:
            List of matching events (most recent first)
        """
        results = []
        with self._lock:
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
        with self._lock:
            return self.events[-n:]

    def by_frame(self, frame: int) -> List[Event]:
        """Get all events for a specific frame."""
        return self.query(frame=frame, limit=1000)

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self.events = []
            self._next_id = 1
            self._current_frame = 0

    def export_json(self, indent: int = 2) -> str:
        """Export all events as JSON."""
        with self._lock:
            return json.dumps([e.to_dict() for e in self.events], indent=indent)

    def export_jsonl(self) -> str:
        """Export events as JSON Lines (one event per line)."""
        with self._lock:
            return "\n".join(e.to_json() for e in self.events)

    def stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        with self._lock:
            by_type: Dict[str, int] = {}
            by_source: Dict[str, int] = {}

            for event in self.events:
                by_type[event.type] = by_type.get(event.type, 0) + 1
                by_source[event.source] = by_source.get(event.source, 0) + 1

            return {
                "total": len(self.events),
                "by_type": by_type,
                "by_source": by_source,
                "frames": self._current_frame,
                "first_id": self.events[0].id if self.events else None,
                "last_id": self.events[-1].id if self.events else None
            }

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"EventSpine(events={len(self.events)}, frame={self._current_frame})"
