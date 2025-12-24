#!/usr/bin/env python3
"""
CLI Interface for TriX → XOR Fabric Pipeline

Usage:
    python -m fabric_pipeline.cli compile --tiles 64 --output fabric.cu
    python -m fabric_pipeline.cli benchmark --binary ./fabric
    python -m fabric_pipeline.cli run --preset default --output fabric.cu
    python -m fabric_pipeline.cli verify --binary ./fabric
"""

import argparse
import sys
from pathlib import Path

from .config import FabricConfig, TaskConfig, get_preset, PRESETS
from .pipeline import Pipeline
from .shapes import get_registry


def cmd_compile(args):
    """Compile frozen shapes to CUDA."""
    config = FabricConfig(
        num_tiles=args.tiles,
        signature_bits=args.signature_bits,
        block_size_bits=args.block_size,
        threads_per_block=args.threads,
    )

    task = TaskConfig(
        rounds=args.rounds,
        shape_pattern=args.pattern.split(",") if args.pattern else ["xor", "and", "xor", "or"],
    )

    pipeline = Pipeline(config, task)
    pipeline.freeze()
    pipeline.compile(args.output, compile_binary=not args.source_only)

    if not args.source_only:
        print(f"\nTo run: {Path(args.output).with_suffix('')}")


def cmd_benchmark(args):
    """Run benchmark on compiled binary."""
    import subprocess

    result = subprocess.run(
        [args.binary, str(args.blocks), str(args.iterations)],
        capture_output=False,
    )

    if result.returncode != 0:
        print("Benchmark failed", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    """Run complete pipeline."""
    if args.preset:
        pipeline = Pipeline(preset=args.preset)
    else:
        config = FabricConfig(
            num_tiles=args.tiles,
            signature_bits=args.signature_bits,
            block_size_bits=args.block_size,
        )
        task = TaskConfig(
            rounds=args.rounds,
            shape_pattern=args.pattern.split(",") if args.pattern else None,
        )
        pipeline = Pipeline(config, task)

    pipeline.run(
        output_path=args.output,
        num_blocks=args.blocks,
        iterations=args.iterations,
    )


def cmd_verify(args):
    """Verify shapes."""
    registry = get_registry()

    print("Shape Verification")
    print("=" * 50)
    print()

    verification = registry.verify_all()

    passed = 0
    failed = 0

    for name, verified in sorted(verification.items()):
        status = "✓" if verified else "✗"
        print(f"  {status} {name}")
        if verified:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Verified: {passed}/{passed + failed}")

    if failed > 0:
        sys.exit(1)


def cmd_list(args):
    """List available shapes and presets."""
    if args.what == "shapes":
        registry = get_registry()
        print("Available Shapes")
        print("=" * 50)

        for shape in sorted(registry, key=lambda s: (s.category.value, s.name)):
            verified = "✓" if shape.verified else " "
            print(f"  [{verified}] {shape.name:12} ({shape.category.value})")
            print(f"       Polynomial → CUDA: {shape.cuda_expr}")

    elif args.what == "presets":
        print("Available Presets")
        print("=" * 50)

        for name, config in PRESETS.items():
            print(f"\n  {name}:")
            print(f"    Tiles: {config.num_tiles}")
            print(f"    Signature bits: {config.signature_bits}")
            print(f"    Block size: {config.block_size_bits} bits")


def cmd_info(args):
    """Show pipeline info."""
    print("TriX → XOR Fabric Pipeline")
    print("=" * 50)
    print()
    print("The Mathematical Identity:")
    print("  Polynomial (TriX):  XOR(a,b) = a + b - 2ab")
    print("  Hardware (CUDA):    XOR(a,b) = a ^ b")
    print("  On binary {0,1}:    IDENTICAL")
    print()
    print("Train with gradients. Execute with geometry.")
    print("Shape IS compute.")
    print()
    print("Usage:")
    print("  fabric run --preset default --output fabric.cu")
    print("  fabric compile --tiles 64 --output fabric.cu")
    print("  fabric benchmark --binary ./fabric")
    print("  fabric verify")
    print("  fabric list shapes")
    print("  fabric list presets")


def main():
    parser = argparse.ArgumentParser(
        prog="fabric",
        description="TriX → XOR Fabric Pipeline",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile to CUDA")
    compile_parser.add_argument("--tiles", type=int, default=64, help="Number of tiles")
    compile_parser.add_argument("--signature-bits", type=int, default=8, help="Signature bits")
    compile_parser.add_argument("--block-size", type=int, default=512, help="Block size in bits")
    compile_parser.add_argument("--rounds", type=int, default=20, help="Computation rounds")
    compile_parser.add_argument("--threads", type=int, default=256, help="Threads per block")
    compile_parser.add_argument("--pattern", type=str, help="Shape pattern (comma-separated)")
    compile_parser.add_argument("--output", "-o", required=True, help="Output path")
    compile_parser.add_argument("--source-only", action="store_true", help="Don't compile binary")
    compile_parser.set_defaults(func=cmd_compile)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--binary", "-b", required=True, help="Compiled binary path")
    bench_parser.add_argument("--blocks", type=int, default=1 << 20, help="Number of blocks")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use preset")
    run_parser.add_argument("--tiles", type=int, default=64)
    run_parser.add_argument("--signature-bits", type=int, default=8)
    run_parser.add_argument("--block-size", type=int, default=512)
    run_parser.add_argument("--rounds", type=int, default=20)
    run_parser.add_argument("--pattern", type=str)
    run_parser.add_argument("--output", "-o", default="fabric.cu")
    run_parser.add_argument("--blocks", type=int, default=1 << 20)
    run_parser.add_argument("--iterations", type=int, default=100)
    run_parser.set_defaults(func=cmd_run)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify shapes")
    verify_parser.set_defaults(func=cmd_verify)

    # List command
    list_parser = subparsers.add_parser("list", help="List available items")
    list_parser.add_argument("what", choices=["shapes", "presets"])
    list_parser.set_defaults(func=cmd_list)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show pipeline info")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
