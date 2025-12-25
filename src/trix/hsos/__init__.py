"""
trix.hsos - Hollywood Squares Operating System

Distributed microkernel for addressable processor networks.
No arrays. No batches. No memory allocation.
Just processors and messages.

"Deterministic message passing + bounded local semantics + enforced observability
 => global convergence with inherited correctness"

The topology encodes the algorithm.
Different wiring = different computation.
Same frozen shapes everywhere.

Usage:
    from trix.hsos import Processor, Network, Message

    # Create processors with frozen shapes
    net = Network()
    net.add(Processor(signature=0b1010, shape='xor'))
    net.add(Processor(signature=0b0101, shape='and'))

    # Send message - routes by Hamming distance
    net.send(Message(target=0b1010, payload={'a': 1, 'b': 0}))

    # Run until quiescent
    net.run()
"""

from .processor import Processor, ProcessorState
from .message import Message, MessageLog
from .network import Network
from .shapes import Shape, SHAPES
from .compile import GilliesCompiler, compile_to_gillies

__all__ = [
    'Processor',
    'ProcessorState',
    'Message',
    'MessageLog',
    'Network',
    'Shape',
    'SHAPES',
    'GilliesCompiler',
    'compile_to_gillies',
]
