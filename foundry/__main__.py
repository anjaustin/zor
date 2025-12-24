"""
THE FOUNDRY
===========

Entry point: python -m foundry

A chip catalog with instant fabrication.
"""

import sys
from pathlib import Path

from .library.catalog import Catalog, Category
from .export.trixc import TriXcFormat
from .export.c_export import CExporter


def demo():
    """
    Demonstrate the export pipeline.

    SEE → SELECT → EXPORT → DONE
    """
    print()
    print("═" * 60)
    print("  T R I X   F O U N D R Y")
    print("  Watch computation become geometry.")
    print("═" * 60)
    print()

    # Load catalog
    catalog = Catalog()
    print(f"  Library loaded: {len(catalog)} chips")
    print()

    # List by category
    for category in catalog.categories:
        entries = catalog.list_by_category(category)
        if entries:
            print(f"  ┌─ {category.value.upper()} {'─' * (50 - len(category.value))}┐")
            for entry in entries:
                status = "✓ ready" if entry.ready else "soon"
                size = f"{entry.spec.size} B"
                print(f"  │   {entry.spec.name:<20} {size:>8}  {status:<10} │")
            print(f"  └{'─' * 54}┘")
            print()

    # Export XOR as demo
    print("  FABRICATING: XOR → C")
    print()

    xor_entry = catalog.get("xor")
    if xor_entry:
        output_dir = Path("/tmp/foundry_export")
        header_path, source_path = CExporter.export(xor_entry.spec, output_dir)

        print(f"  ✓ {header_path.name}")
        print(f"  ✓ {source_path.name}")
        print()

        # Show generated code
        print("  ┌─ trix_xor.h ─────────────────────────────────────┐")
        for line in header_path.read_text().split('\n')[:15]:
            print(f"  │ {line[:52]:<52} │")
        print("  │ ...                                                │")
        print("  └──────────────────────────────────────────────────────┘")
        print()

        # Also save as .trixc
        trixc_path = output_dir / "xor.trixc"
        size = TriXcFormat.save(xor_entry.spec, trixc_path)
        print(f"  ✓ xor.trixc ({size} bytes)")
        print()

    print("═" * 60)
    print("  The wood cuts itself.")
    print("═" * 60)
    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo()
    else:
        # TODO: Launch Textual UI
        print("THE FOUNDRY")
        print()
        print("Usage:")
        print("  python -m foundry --demo    Run export demo")
        print()
        print("Full UI coming soon...")


if __name__ == "__main__":
    main()
