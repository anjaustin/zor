"""
CLI entry point - the single door.
"""
import sys
import argparse

from .display import print_banner


def main():
    """Main entry point for the trix CLI."""
    parser = argparse.ArgumentParser(
        prog='trix',
        description='TriX - Frozen Shape Foundry',
        epilog='Type a function. Get hardware.'
    )

    parser.add_argument(
        'command',
        nargs='?',
        choices=['demo', 'shell', 'build', 'help'],
        help='Command to run (default: interactive shell)'
    )

    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments'
    )

    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version'
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"trix {__version__}")
        return

    if args.command is None or args.command == 'shell':
        # Default: launch REPL
        from .repl import run_repl
        run_repl()

    elif args.command == 'demo':
        from .demo import run_demo
        run_demo()

    elif args.command == 'help':
        parser.print_help()

    elif args.command == 'build':
        # Build from file or expression
        if not args.args:
            print("Usage: trix build <file.py> or trix build \"lambda a, b: ...\"")
            sys.exit(1)

        expr = args.args[0]
        from .foundry import get_foundry

        foundry = get_foundry()

        if expr.startswith("lambda"):
            # Direct lambda expression
            try:
                shape = foundry.custom(expr)
                print(f"Created shape: {shape.name}")
                passed, total = shape.validate()
                print(f"Validated: {passed}/{total}")
                if passed == total:
                    c_code = shape.export_c()
                    print(c_code)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            # File path
            print(f"Building from file not yet implemented: {expr}")
            sys.exit(1)


if __name__ == "__main__":
    main()
