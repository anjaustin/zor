"""
FabricProcessor: A processor in the TriX Fabric.

Processor = Shape + Signature (stateless by default)

The processor evaluates its shape when a message arrives,
then emits output messages to connected targets.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..hsos.shapes import get_shape


@dataclass
class Connection:
    """A connection from this processor's output to another processor's input."""
    target_signature: str  # Where to send output
    target_input: str      # Which input port on target
    output_name: str       # Which output of this processor


@dataclass
class FabricProcessor:
    """
    A processor in the fabric.

    Attributes:
        signature: Hierarchical address (e.g., "/0/0b1010")
        shape: Name of the shape to evaluate (e.g., "xor")
        name: Human-readable name (optional)
        connections: Output connections to other processors
        input_map: Map of input names to source signatures
    """
    signature: str
    shape: str
    name: Optional[str] = None
    connections: List[Connection] = field(default_factory=list)
    input_map: Dict[str, str] = field(default_factory=dict)  # input_name -> source_signature.output

    def __post_init__(self):
        # Validate shape exists
        self._shape_fn = get_shape(self.shape)
        if self.name is None:
            self.name = f"proc_{self.signature.replace('/', '_')}"

    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the shape with given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary of output values
        """
        # Call the shape function
        state, outputs = self._shape_fn({}, inputs)
        return outputs

    @property
    def cluster_path(self) -> str:
        """Get the cluster path (everything except last component)."""
        parts = self.signature.rsplit('/', 1)
        if len(parts) == 1:
            return '/'
        return parts[0] or '/'

    @property
    def local_signature(self) -> str:
        """Get the local signature within the cluster."""
        return self.signature.rsplit('/', 1)[-1]

    def connect_to(self, target_signature: str, target_input: str, output_name: str = 'result'):
        """Add a connection to another processor."""
        self.connections.append(Connection(
            target_signature=target_signature,
            target_input=target_input,
            output_name=output_name
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'signature': self.signature,
            'shape': self.shape,
            'name': self.name,
            'connections': [
                {'target': c.target_signature, 'input': c.target_input, 'output': c.output_name}
                for c in self.connections
            ],
            'input_map': self.input_map
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FabricProcessor':
        """Deserialize from dictionary."""
        proc = cls(
            signature=d['signature'],
            shape=d['shape'],
            name=d.get('name'),
            input_map=d.get('input_map', {})
        )
        for conn in d.get('connections', []):
            proc.connect_to(
                target_signature=conn['target'],
                target_input=conn['input'],
                output_name=conn.get('output', 'result')
            )
        return proc

    def __repr__(self):
        return f"FabricProcessor({self.signature!r}, shape={self.shape!r})"
