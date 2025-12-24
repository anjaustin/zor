"""
Export All CPU Models to ONNX

Builds and exports neural network versions of:
- MOS 6502 (8-bit, 1975)
- WDC 65816 (16-bit, 1983)
- Intel 80286 (16-bit, 1982)
- Intel 80486 (32-bit, 1989)

All achieve 100% accuracy with 0 training steps.
"""

import os
import time
from pathlib import Path

# Import the CPU builders
from trix.foundry.mos6502 import build_6502
from trix.foundry.wdc65816 import build_65816
from trix.foundry.intel286 import build_286
from trix.foundry.intel386 import build_486


def export_all_cpus(output_dir: str = "exports"):
    """Build and export all CPU models to ONNX."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FROZEN FOUNDRY - CPU EXPORT PIPELINE")
    print("=" * 70)
    print()
    print("Building and exporting neural network versions of classic CPUs...")
    print()

    results = []

    # =========================================================================
    # MOS 6502
    # =========================================================================
    print("-" * 70)
    print("Building MOS 6502...")
    print("-" * 70)

    start = time.time()
    foundry_6502, result_6502 = build_6502()
    elapsed_6502 = time.time() - start

    onnx_path = output_path / "mos_6502.onnx"
    pt_path = output_path / "mos_6502.pt"
    foundry_6502.export_onnx(str(onnx_path))
    foundry_6502.export(str(pt_path))

    results.append({
        "name": "MOS 6502",
        "year": 1975,
        "bits": 8,
        "ops": len(foundry_6502.operations),
        "shapes": len(foundry_6502.library),
        "params": result_6502.num_params,
        "time_ms": elapsed_6502 * 1000,
        "accuracy": result_6502.accuracy,
        "onnx_path": str(onnx_path),
        "pt_path": str(pt_path),
    })

    print(f"  Exported: {onnx_path}")
    print()

    # =========================================================================
    # WDC 65816
    # =========================================================================
    print("-" * 70)
    print("Building WDC 65816...")
    print("-" * 70)

    start = time.time()
    foundry_65816, result_65816 = build_65816()
    elapsed_65816 = time.time() - start

    onnx_path = output_path / "wdc_65816.onnx"
    pt_path = output_path / "wdc_65816.pt"
    foundry_65816.export_onnx(str(onnx_path))
    foundry_65816.export(str(pt_path))

    results.append({
        "name": "WDC 65816",
        "year": 1983,
        "bits": 16,
        "ops": len(foundry_65816.operations),
        "shapes": len(foundry_65816.library),
        "params": result_65816.num_params,
        "time_ms": elapsed_65816 * 1000,
        "accuracy": result_65816.accuracy,
        "onnx_path": str(onnx_path),
        "pt_path": str(pt_path),
    })

    print(f"  Exported: {onnx_path}")
    print()

    # =========================================================================
    # Intel 80286
    # =========================================================================
    print("-" * 70)
    print("Building Intel 80286...")
    print("-" * 70)

    start = time.time()
    foundry_286, result_286 = build_286()
    elapsed_286 = time.time() - start

    onnx_path = output_path / "intel_80286.onnx"
    pt_path = output_path / "intel_80286.pt"
    foundry_286.export_onnx(str(onnx_path))
    foundry_286.export(str(pt_path))

    results.append({
        "name": "Intel 80286",
        "year": 1982,
        "bits": 16,
        "ops": len(foundry_286.operations),
        "shapes": len(foundry_286.library),
        "params": result_286.num_params,
        "time_ms": elapsed_286 * 1000,
        "accuracy": result_286.accuracy,
        "onnx_path": str(onnx_path),
        "pt_path": str(pt_path),
    })

    print(f"  Exported: {onnx_path}")
    print()

    # =========================================================================
    # Intel 80486
    # =========================================================================
    print("-" * 70)
    print("Building Intel 80486...")
    print("-" * 70)

    start = time.time()
    foundry_486, result_486 = build_486()
    elapsed_486 = time.time() - start

    onnx_path = output_path / "intel_80486.onnx"
    pt_path = output_path / "intel_80486.pt"
    foundry_486.export_onnx(str(onnx_path))
    foundry_486.export(str(pt_path))

    results.append({
        "name": "Intel 80486",
        "year": 1989,
        "bits": 32,
        "ops": len(foundry_486.operations),
        "shapes": len(foundry_486.library),
        "params": result_486.num_params,
        "time_ms": elapsed_486 * 1000,
        "accuracy": result_486.accuracy,
        "onnx_path": str(onnx_path),
        "pt_path": str(pt_path),
    })

    print(f"  Exported: {onnx_path}")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print()
    print(f"{'CPU':<15} {'Year':>6} {'Bits':>6} {'Ops':>6} {'Shapes':>8} {'Params':>8} {'Time':>10} {'Accuracy':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<15} {r['year']:>6} {r['bits']:>6} {r['ops']:>6} {r['shapes']:>8} {r['params']:>8} {r['time_ms']:>8.0f}ms {r['accuracy']*100:>9.2f}%")

    print("-" * 70)
    print()
    print("Files exported:")
    for r in results:
        print(f"  {r['onnx_path']}")
        print(f"  {r['pt_path']}")
    print()
    print("=" * 70)
    print()
    print("  \"Computation is topology. Learning is routing.\"")
    print()

    return results


if __name__ == "__main__":
    results = export_all_cpus("exports")
