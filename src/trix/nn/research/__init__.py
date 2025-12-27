"""
TriX Research Branches

Experimental architectures kept separate from production code.
These represent active research directions that may or may not
be integrated into the main codebase.

Subdirectories:
    kan/   - Track B: Kolmogorov-Arnold Networks
    sdpmx/ - Vi's Synthesis: Hilbert Space Operators
"""

from . import kan
from . import sdpmx

__all__ = ["kan", "sdpmx"]
