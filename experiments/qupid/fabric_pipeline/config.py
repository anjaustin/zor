"""
Configuration for TriX → XOR Fabric Pipeline

Defines all configurable parameters for:
- Fabric architecture (tiles, signatures, block size)
- Training parameters
- Compilation options
- Verification settings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path


class OptimizationLevel(Enum):
    """CUDA compilation optimization level"""
    DEBUG = 0      # No optimization, debug symbols
    BALANCED = 1   # Moderate optimization
    AGGRESSIVE = 2 # Maximum optimization, longer compile
    EXTREME = 3    # All optimizations + loop unrolling


class FabricTopology(Enum):
    """Fabric interconnection pattern"""
    BUTTERFLY = "butterfly"   # Classic FFT-like pattern
    HYPERCUBE = "hypercube"   # N-dimensional cube
    TORUS = "torus"           # Wrap-around mesh
    CUSTOM = "custom"         # User-defined


@dataclass
class FabricConfig:
    """
    Configuration for XOR Fabric architecture.

    Attributes:
        num_tiles: Number of computation tiles (default: 64)
        signature_bits: Bits per tile signature (default: 8)
        block_size_bits: Size of data blocks in bits (default: 512)
        topology: Interconnection pattern
        workers_per_tile: Workers within each tile
        optimization: CUDA optimization level
    """
    num_tiles: int = 64
    signature_bits: int = 8
    block_size_bits: int = 512
    topology: FabricTopology = FabricTopology.BUTTERFLY
    workers_per_tile: int = 16
    optimization: OptimizationLevel = OptimizationLevel.AGGRESSIVE

    # CUDA-specific
    threads_per_block: int = 256
    max_registers: int = 64
    shared_memory_kb: int = 48

    # Derived properties
    @property
    def block_size_bytes(self) -> int:
        return self.block_size_bits // 8

    @property
    def block_size_words(self) -> int:
        return self.block_size_bits // 32

    @property
    def total_workers(self) -> int:
        return self.num_tiles * self.workers_per_tile

    def validate(self) -> List[str]:
        """Validate configuration, return list of issues."""
        issues = []
        if self.num_tiles < 1 or self.num_tiles > 1024:
            issues.append(f"num_tiles must be 1-1024, got {self.num_tiles}")
        if self.signature_bits < 4 or self.signature_bits > 32:
            issues.append(f"signature_bits must be 4-32, got {self.signature_bits}")
        if self.block_size_bits not in [128, 256, 512, 1024]:
            issues.append(f"block_size_bits must be 128/256/512/1024, got {self.block_size_bits}")
        if self.threads_per_block not in [64, 128, 256, 512, 1024]:
            issues.append(f"threads_per_block must be power of 2 in [64,1024]")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_tiles": self.num_tiles,
            "signature_bits": self.signature_bits,
            "block_size_bits": self.block_size_bits,
            "topology": self.topology.value,
            "workers_per_tile": self.workers_per_tile,
            "optimization": self.optimization.name,
            "threads_per_block": self.threads_per_block,
            "max_registers": self.max_registers,
            "shared_memory_kb": self.shared_memory_kb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FabricConfig":
        """Deserialize from dictionary."""
        return cls(
            num_tiles=d.get("num_tiles", 64),
            signature_bits=d.get("signature_bits", 8),
            block_size_bits=d.get("block_size_bits", 512),
            topology=FabricTopology(d.get("topology", "butterfly")),
            workers_per_tile=d.get("workers_per_tile", 16),
            optimization=OptimizationLevel[d.get("optimization", "AGGRESSIVE")],
            threads_per_block=d.get("threads_per_block", 256),
            max_registers=d.get("max_registers", 64),
            shared_memory_kb=d.get("shared_memory_kb", 48),
        )

    def save(self, path: Path):
        """Save configuration to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "FabricConfig":
        """Load configuration from JSON file."""
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class TaskConfig:
    """
    Configuration for a computation task.

    Defines what the fabric should compute.
    """
    name: str = "cipher"
    rounds: int = 20
    shape_pattern: List[str] = field(default_factory=lambda: [
        "xor", "and", "xor", "or"
    ])

    # Training parameters (for TriX)
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 256

    # Verification
    verify_samples: int = 10000
    tolerance: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rounds": self.rounds,
            "shape_pattern": self.shape_pattern,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verify_samples": self.verify_samples,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskConfig":
        return cls(**d)


# Preset configurations
PRESETS = {
    "tiny": FabricConfig(
        num_tiles=16,
        signature_bits=4,
        block_size_bits=128,
    ),
    "small": FabricConfig(
        num_tiles=32,
        signature_bits=6,
        block_size_bits=256,
    ),
    "default": FabricConfig(
        num_tiles=64,
        signature_bits=8,
        block_size_bits=512,
    ),
    "large": FabricConfig(
        num_tiles=128,
        signature_bits=10,
        block_size_bits=512,
    ),
    "huge": FabricConfig(
        num_tiles=256,
        signature_bits=12,
        block_size_bits=1024,
    ),
}


def get_preset(name: str) -> FabricConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
