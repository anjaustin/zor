# Shape Substrate Experiments

**Universal Substrate for Growing Deterministic Neural-Geometric Shapes of Compute**

---

## New Here? Start Here.

Don't read the theory. Just run these, in order:

```bash
python 01_wait_what.py       # XOR without XOR
python 02_so_what.py         # Why polynomial matters
python 03_build_something.py # Build an 8-bit adder
python 04_why_fast.py        # Training vs inference speed
python 05_the_trick.py       # Shapes compose like LEGO
python 06_what_is_a_shape.py # Visualize the shape
python 07_shapes_compose.py  # Molecules from atoms
python 08_routing.py         # How inputs find shapes
python 09_put_it_together.py # Train a tiny router
python 10_protein_version.py # The biological analogy
```

Each file is < 100 lines. Each one shows you something. No lectures.

---

## Overview

This directory contains CUDA implementations demonstrating the Shape Substrate thesis:
that LFSR fabrics can serve as high-speed execution substrates for trained neural-geometric shapes.

## Files

| File | Description | Throughput |
|------|-------------|------------|
| `fabric_4x16.cu` | 4 LFSRs in series × 16 clusters parallel | 23.27 Tbits/sec |
| `fabric_4x4.cu` | 4 LFSRs in series × 4 clusters parallel | 6.71 Tbits/sec |
| `fabric_4x4x4.cu` | 4 parallel layers × (4×4) = 64 LFSRs | 25.79 Tbits/sec |
| `molecular_shapes.cu` | Atomic → Molecular composition | 51.93 Tbits/sec |
| `protein_compute.cu` | Protein-like conformational computation | 21.85B reactions/sec |

## Quick Start

```bash
# Build all
nvcc -O3 -arch=native -o fabric_4x16 fabric_4x16.cu
nvcc -O3 -arch=native -o fabric_4x4 fabric_4x4.cu
nvcc -O3 -arch=native -o fabric_4x4x4 fabric_4x4x4.cu
nvcc -O3 -arch=native -o molecular_shapes molecular_shapes.cu
nvcc -O3 -arch=native -o protein_compute protein_compute.cu

# Run
./fabric_4x16
./molecular_shapes
./protein_compute
```

## Concepts

### Atomic Shapes

Pre-trained LFSR tap patterns:

```
Atom A: Fast mixing      (taps: 511, 509, 495, 483)
Atom B: Diffusion        (taps: 503, 490, 479, 465)
Atom C: Long period      (taps: 510, 497, 486, 471)
Atom D: Decorrelation    (taps: 507, 492, 477, 459)
```

### Molecular Composition

```
Serial:   A → B → C → D     (4-stage mixing, 41.51 Tb/s)
Parallel: A ⊕ B ⊕ C ⊕ D     (max throughput, 51.93 Tb/s)
Hybrid:   (A→B) ⊕ (C→D)     (balanced, 51.13 Tb/s)
```

### Protein-like Computation

```
Input → [Binding Site] → [Folding] → [Active Site] → Output
             match          evolve        emit
```

## Results Summary

| Experiment | Throughput | Memory | Efficiency |
|------------|------------|--------|------------|
| Flat LFSR (262K) | 35.57 Tb/s | 16.78 MB | 2.12 Tb/s/MB |
| 4×16 Fabric | 23.27 Tb/s | 4.19 MB | 5.55 Tb/s/MB |
| Molecular Parallel | 51.93 Tb/s | 16.78 MB | 3.09 Tb/s/MB |
| Protein Reactions | 21.85B/sec | 13.63 MB | 1.60B/sec/MB |

## The Insight

```
Train atoms once.
Compose molecules forever.
Let proteins fold.
Watch computation emerge.
```

---

*Hardware: NVIDIA Thor (Blackwell)*
*Date: 2025-12-22*
