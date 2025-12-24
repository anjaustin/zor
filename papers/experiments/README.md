# ZIT-1 Experiment Suite

## Quick Start

```bash
# Run standard test suite
./run_tests.sh

# Or using make
make test
```

## Test Modes

| Command | Description | Time |
|---------|-------------|------|
| `./run_tests.sh --quick` | Minimal tests (64, 512 nodes) | ~10s |
| `./run_tests.sh --full` | Standard suite (up to 262K nodes) | ~1min |
| `./run_tests.sh --stress` | Full + large scales (up to 16.7M nodes) | ~2min |

Add `-v` for verbose output with timing and rewire counts.

## What Gets Tested

### Convergence Tests
- **4x4x4** (64 nodes) - Must converge in 50-250 cycles
- **8x8x8** (512 nodes) - Must converge in 50-180 cycles
- **16x16x16** (4K nodes) - Must converge in 100-350 cycles
- **32x32x32** (32K nodes) - Must converge in 100-350 cycles
- **64x64x64** (262K nodes) - Must converge in 80-250 cycles
- **128x128x128** (2M nodes) - Must converge in 300-700 cycles
- **256x256x256** (16.7M nodes) - Must converge in 600-1400 cycles

### Topology Invariant Tests
- All neighbor indices valid (< num_nodes)
- No self-loops in initial topology
- All LFSR seeds unique
- Initial states follow (i*3)&0xFF pattern

### Determinism Tests
- Same seed produces identical convergence cycle
- Same seed produces identical rewire count

## Expected Output

```
╔═══════════════════════════════════════════════════════════════════╗
║             ZIT-1 RIGOROUS TEST HARNESS                           ║
║             Second Star Constant: 1122911624                     ║
╚═══════════════════════════════════════════════════════════════════╝

GPU: NVIDIA Thor (20 SMs)
Memory: 3.2 GB free / 131.9 GB total

=== Quick Convergence Tests ===
  [PASS] 4x4x4 (64 nodes)
  [PASS] 8x8x8 (512 nodes)

=== Topology Invariant Tests ===
  [PASS] All neighbor indices valid
  [PASS] No self-loops in initial topology
  [PASS] All LFSR seeds unique
  [PASS] Initial states follow (i*3)&0xFF pattern

=== Determinism Tests ===
  [PASS] Convergence cycle deterministic: 102
  [PASS] Rewire count deterministic: 1391

═══════════════════════════════════════════════════════════════════
  ALL 8 TESTS PASSED
═══════════════════════════════════════════════════════════════════
```

## Requirements

- CUDA Toolkit (nvcc)
- NVIDIA GPU with compute capability 5.0+
- ~100MB GPU memory for quick tests
- ~800MB GPU memory for full tests
- ~6GB GPU memory for stress tests

## Building Individual Experiments

```bash
# Build specific experiment
make fabric_128

# Build all experiments
make all

# Clean binaries
make clean
```

## The Second Star Constant

All experiments use seed **1122911624** for reproducibility.

```c
#define SECOND_STAR 1122911624u

// LFSR initialization
n[i].lfsr = SECOND_STAR
          ^ (i * 0x9E3779B9)      // Golden ratio
          ^ ((i >> 8) * 0x85EBCA6B)   // MurmurHash3
          ^ ((i >> 16) * 0xC2B2AE35); // MurmurHash3
```

## Experiment Files

| File | Nodes | Description |
|------|-------|-------------|
| `test_harness.cu` | All | Rigorous test suite |
| `fabric_64_v2.cu` | 262K | 64³ with 32-bit LFSR |
| `fabric_128.cu` | 2M | 128³ flagship |
| `fabric_256.cu` | 16.7M | 256³ stress test |
| `fabric_384.cu` | 56.6M | Maximum tested scale |
| `sequential_fabric*.cu` | Various | Development versions |
| `turbo_fabric*.cu` | 64 | Fused kernel experiments |

## Interpreting Results

- **[PASS]** - Test passed within expected bounds
- **[WARN]** - Converged faster than expected (not a failure)
- **[FAIL]** - Did not converge or exceeded cycle limit
- **[SKIP]** - Insufficient GPU memory

## CI/CD Integration

```yaml
# Example GitHub Actions step
- name: Run ZIT-1 Tests
  run: |
    cd papers/experiments
    ./run_tests.sh --full
```

Exit codes:
- `0` - All tests passed
- `1` - One or more tests failed
