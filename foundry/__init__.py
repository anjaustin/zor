"""
THE FOUNDRY
===========

A chip catalog with instant fabrication.

Browse → Select → Export → Use.

Watch computation become geometry.
"""

__version__ = "0.1.0"
__author__ = "TriX Project"

from .library.catalog import Catalog
from .export.trixc import TriXcFormat
from .export.c_export import CExporter
from .export.trix_6502 import TriX6502Exporter
