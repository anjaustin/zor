"""
FDL: Fabric Definition Language

One format for:
- Definition (what the fabric is)
- Persistence (how it's stored)
- Query target (what you search over)

Syntax:
    fabric "name" {
        processor name {
            shape: shape_name
            signature: /path/0b1010
            input.a: other_proc.result
        }
        export: proc_name
    }
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from .fabric import Fabric
from .processor import FabricProcessor


class FDLParseError(Exception):
    """Error parsing FDL."""
    pass


def parse_fdl(source: str) -> Fabric:
    """
    Parse FDL source into a Fabric.

    Args:
        source: FDL source code

    Returns:
        Parsed Fabric
    """
    # Remove comments
    source = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)

    # Parse fabric header
    fabric_match = re.search(r'fabric\s+"([^"]+)"\s*\{', source)
    if not fabric_match:
        raise FDLParseError("No fabric definition found")

    fabric_name = fabric_match.group(1)
    fabric = Fabric(name=fabric_name)

    # Find the fabric body
    start = fabric_match.end()
    depth = 1
    end = start
    for i, c in enumerate(source[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    body = source[start:end]

    # Parse processors
    proc_pattern = r'processor\s+(\w+)\s*\{([^}]*)\}'
    for proc_match in re.finditer(proc_pattern, body):
        proc_name = proc_match.group(1)
        proc_body = proc_match.group(2)

        # Parse processor fields
        shape = None
        signature = None
        input_map = {}

        for line in proc_body.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().rstrip(',')

                if key == 'shape':
                    shape = value
                elif key == 'signature':
                    signature = value
                elif key.startswith('input.'):
                    input_name = key[6:]  # Remove 'input.'
                    input_map[input_name] = value

        if shape is None:
            raise FDLParseError(f"Processor {proc_name} missing shape")
        if signature is None:
            raise FDLParseError(f"Processor {proc_name} missing signature")

        processor = FabricProcessor(
            signature=signature,
            shape=shape,
            name=proc_name,
            input_map=input_map
        )
        fabric.add_processor(processor)

    # Parse exports
    for match in re.finditer(r'export:\s*(\w+)', body):
        proc_name = match.group(1)
        # Find processor by name and mark as export
        for proc in fabric.iter_processors():
            if proc.name == proc_name:
                fabric.set_export(proc.signature)
                break

    # Parse entry points
    for match in re.finditer(r'entry:\s*(\w+)', body):
        proc_name = match.group(1)
        for proc in fabric.iter_processors():
            if proc.name == proc_name:
                fabric.set_entry_point(proc.signature)
                break

    # Resolve connections from input maps
    _resolve_connections(fabric)

    return fabric


def _resolve_connections(fabric: Fabric) -> None:
    """Resolve input_map references into actual connections."""
    # Build name -> signature lookup
    name_to_sig = {}
    for proc in fabric.iter_processors():
        if proc.name:
            name_to_sig[proc.name] = proc.signature

    # Resolve each processor's input map
    for proc in fabric.iter_processors():
        for input_name, source_ref in proc.input_map.items():
            # Parse source_ref: "proc_name.output_name" or "signature.output_name"
            if '.' in source_ref:
                parts = source_ref.rsplit('.', 1)
                source_name = parts[0]
                output_name = parts[1] if len(parts) > 1 else 'result'
            else:
                source_name = source_ref
                output_name = 'result'

            # Resolve source name to signature
            if source_name in name_to_sig:
                source_sig = name_to_sig[source_name]
            elif source_name.startswith('/'):
                source_sig = source_name
            else:
                continue  # Can't resolve

            # Add connection from source to this processor
            source_proc = fabric.get_processor(source_sig)
            if source_proc:
                source_proc.connect_to(
                    target_signature=proc.signature,
                    target_input=input_name,
                    output_name=output_name
                )


def emit_fdl(fabric: Fabric) -> str:
    """
    Emit FDL source from a Fabric.

    Args:
        fabric: The fabric to serialize

    Returns:
        FDL source code
    """
    lines = []
    lines.append(f'fabric "{fabric.name}" {{')

    # Emit processors
    for proc in fabric.iter_processors():
        lines.append(f'    processor {proc.name} {{')
        lines.append(f'        shape: {proc.shape}')
        lines.append(f'        signature: {proc.signature}')

        for input_name, source_ref in proc.input_map.items():
            lines.append(f'        input.{input_name}: {source_ref}')

        lines.append('    }')

    # Emit exports
    for sig in fabric.exports:
        proc = fabric.get_processor(sig)
        if proc and proc.name:
            lines.append(f'    export: {proc.name}')

    # Emit entry points
    for sig in fabric.entry_points:
        proc = fabric.get_processor(sig)
        if proc and proc.name:
            lines.append(f'    entry: {proc.name}')

    lines.append('}')

    return '\n'.join(lines)


def load_fdl(filepath: str) -> Fabric:
    """Load a Fabric from an FDL file."""
    with open(filepath, 'r') as f:
        return parse_fdl(f.read())


def save_fdl(fabric: Fabric, filepath: str) -> None:
    """Save a Fabric to an FDL file."""
    with open(filepath, 'w') as f:
        f.write(emit_fdl(fabric))


# Also support JSON persistence
def save_json(fabric: Fabric, filepath: str) -> None:
    """Save a Fabric to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(fabric.to_dict(), f, indent=2)


def load_json(filepath: str) -> Fabric:
    """Load a Fabric from a JSON file."""
    with open(filepath, 'r') as f:
        return Fabric.from_dict(json.load(f))
