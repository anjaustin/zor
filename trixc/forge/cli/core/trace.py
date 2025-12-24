"""
Trace File - Binary trace format for frame-by-frame replay.

Format:
- Header: magic (4) + version (2) + flags (2) + frame_count (4) + metadata_len (4)
- Metadata: JSON
- Frames: [frame_size (4) + frame_data (var)] * frame_count
"""

import struct
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, BinaryIO
import time


TRACE_MAGIC = b"TRXT"  # TRIX Trace
TRACE_VERSION = 1


@dataclass
class TraceFrame:
    """A single trace frame."""
    frame_id: int
    timestamp: float
    input_values: List[float]
    layer_activations: Dict[str, List[float]]
    output_values: List[float]
    output_class: int
    output_confidence: float
    ground_truth: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "input": self.input_values,
            "layers": self.layer_activations,
            "output": self.output_values,
            "class": self.output_class,
            "confidence": self.output_confidence,
            "ground_truth": self.ground_truth,
            "meta": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceFrame":
        """Create from dictionary."""
        return cls(
            frame_id=d["frame_id"],
            timestamp=d.get("timestamp", 0),
            input_values=d["input"],
            layer_activations=d.get("layers", {}),
            output_values=d["output"],
            output_class=d["class"],
            output_confidence=d["confidence"],
            ground_truth=d.get("ground_truth"),
            metadata=d.get("meta", {})
        )


@dataclass
class TraceMetadata:
    """Trace file metadata."""
    model_name: str
    model_version: str
    target: str
    created: float
    input_shape: List[int]
    output_classes: int
    layer_names: List[str]
    levers: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class TraceFile:
    """
    Binary trace file for frame-by-frame replay.

    Supports:
    - Streaming writes (append frames during inference)
    - Random access reads (jump to any frame)
    - Compression (optional)
    - Metadata header
    """

    def __init__(self):
        self.metadata: Optional[TraceMetadata] = None
        self.frames: List[TraceFrame] = []
        self._file: Optional[BinaryIO] = None

    def create(self, path: str, metadata: TraceMetadata) -> None:
        """Create a new trace file for writing."""
        self.metadata = metadata
        self._file = open(path, "wb")

        # Write header placeholder (will update on close)
        self._write_header(0)

        # Write metadata
        meta_json = json.dumps({
            "model": metadata.model_name,
            "version": metadata.model_version,
            "target": metadata.target,
            "created": metadata.created,
            "input_shape": metadata.input_shape,
            "output_classes": metadata.output_classes,
            "layers": metadata.layer_names,
            "levers": metadata.levers,
            "extra": metadata.extra
        }).encode("utf-8")

        self._file.write(struct.pack("<I", len(meta_json)))
        self._file.write(meta_json)

    def _write_header(self, frame_count: int) -> None:
        """Write file header."""
        if self._file:
            self._file.seek(0)
            self._file.write(TRACE_MAGIC)
            self._file.write(struct.pack("<H", TRACE_VERSION))
            self._file.write(struct.pack("<H", 0))  # flags
            self._file.write(struct.pack("<I", frame_count))

    def append_frame(self, frame: TraceFrame) -> None:
        """Append a frame to the trace."""
        if not self._file:
            raise RuntimeError("Trace file not open for writing")

        self.frames.append(frame)

        # Serialize frame
        frame_json = json.dumps(frame.to_dict()).encode("utf-8")
        self._file.write(struct.pack("<I", len(frame_json)))
        self._file.write(frame_json)

    def close(self) -> None:
        """Close and finalize the trace file."""
        if self._file:
            # Update header with frame count
            self._write_header(len(self.frames))
            self._file.close()
            self._file = None

    def load(self, path: str) -> None:
        """Load a trace file for reading."""
        with open(path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != TRACE_MAGIC:
                raise ValueError(f"Invalid trace file: bad magic {magic}")

            version = struct.unpack("<H", f.read(2))[0]
            flags = struct.unpack("<H", f.read(2))[0]
            frame_count = struct.unpack("<I", f.read(4))[0]

            # Read metadata
            meta_len = struct.unpack("<I", f.read(4))[0]
            meta_json = json.loads(f.read(meta_len).decode("utf-8"))

            self.metadata = TraceMetadata(
                model_name=meta_json.get("model", "unknown"),
                model_version=meta_json.get("version", "1.0"),
                target=meta_json.get("target", "unknown"),
                created=meta_json.get("created", 0),
                input_shape=meta_json.get("input_shape", []),
                output_classes=meta_json.get("output_classes", 2),
                layer_names=meta_json.get("layers", []),
                levers=meta_json.get("levers", {}),
                extra=meta_json.get("extra", {})
            )

            # Read frames
            self.frames = []
            for _ in range(frame_count):
                frame_len = struct.unpack("<I", f.read(4))[0]
                frame_json = json.loads(f.read(frame_len).decode("utf-8"))
                self.frames.append(TraceFrame.from_dict(frame_json))

    def get_frame(self, index: int) -> Optional[TraceFrame]:
        """Get frame by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def frame_count(self) -> int:
        """Get total frame count."""
        return len(self.frames)

    def summary(self) -> Dict[str, Any]:
        """Get trace summary."""
        if not self.metadata:
            return {"error": "No trace loaded"}

        correct = sum(1 for f in self.frames
                      if f.ground_truth is not None and f.output_class == f.ground_truth)
        total_with_labels = sum(1 for f in self.frames if f.ground_truth is not None)

        return {
            "model": self.metadata.model_name,
            "target": self.metadata.target,
            "frames": len(self.frames),
            "accuracy": correct / total_with_labels if total_with_labels > 0 else None,
            "input_shape": self.metadata.input_shape,
            "output_classes": self.metadata.output_classes,
            "layers": self.metadata.layer_names
        }
