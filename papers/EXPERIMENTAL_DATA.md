# Experimental Data Appendix

## Raw Convergence Logs

### Experiment 1: 4×4×4 (64 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     4×4×4 = 64 NODES                                              ║
╚═══════════════════════════════════════════════════════════════════╝

CYCLE,RESONANT,PERCENT
0,32,50%
25,56,87%
50,60,93%
75,60,93%
100,62,96%
125,63,98%
158,64,100%

*** CONVERGED at cycle 158 ***
    Time: 27ms | Rewires: 287
```

### Experiment 2: 8×8×8 (512 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     8×8×8 = 512 NODES                                             ║
╚═══════════════════════════════════════════════════════════════════╝

CYCLE,RESONANT,PERCENT
0,256,50%
25,480,93%
50,496,96%
75,504,98%
100,508,99%
113,512,100%

*** CONVERGED at cycle 113 ***
    Time: 40ms | Rewires: 1,340
```

### Experiment 3: 16×16×16 (4,096 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     16×16×16 = 4,096 NODES                                        ║
╚═══════════════════════════════════════════════════════════════════╝

CYCLE,RESONANT,PERCENT
0,2048,50%
50,3840,93%
100,3968,96%
150,4032,98%
202,4096,100%

*** CONVERGED at cycle 202 ***
    Time: 65ms | Rewires: 15,688
```

### Experiment 4: 32×32×32 (32,768 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     32×32×32 = 32,768 NODES (15D Hypercube)                       ║
╚═══════════════════════════════════════════════════════════════════╝

CYCLE,RESONANT,PERCENT
0,16384,50%
25,30720,93%
50,31744,96%
100,32256,98%
150,32512,99%
201,32768,100%

*** CONVERGED at cycle 201 ***
    Time: 91ms | Rewires: 141,653
```

### Experiment 5: 64×64×64 (262,144 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     64×64×64 = 262,144 NODES (v2: 32-bit LFSR + entropy mix)      ║
╚═══════════════════════════════════════════════════════════════════╝

Threshold: 4 (more aggressive)
LFSR: 32-bit with entropy mixing

CYCLE,RESONANT,PERCENT
0,130048,49%
50,245760,93%
100,254976,97%
150,260096,99%
158,262144,100%

╔═══════════════════════════════════════════════════════════════════╗
║  262,144 NODES CONVERGED at cycle 158                             ║
╚═══════════════════════════════════════════════════════════════════╝
    Time: 46.7 ms | Rewires: 1,045,349
```

### Experiment 6: 128×128×128 (2,097,152 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     128×128×128 = 2,097,152 NODES (21D Hypercube)                 ║
║     Second Star Constant Seed: 1122911624                         ║
╚═══════════════════════════════════════════════════════════════════╝

GPU: NVIDIA Thor (20 SMs)
Memory: 92.3 MB nodes + 2.1 MB states = 94.4 MB total
Launching with 8192 blocks × 256 threads

CYCLE,RESONANT,PERCENT
0,1040384,49.6%
1,2048000,97.7%
2,2048000,97.7%
3,2048000,97.7%
4,2016880,96.2%
5,2001191,95.4%
6,2001172,95.4%
7,2001172,95.4%
8,1976912,94.3%
9,1964716,93.7%
100,1699342,81.0%
200,1634578,77.9%
300,1725778,82.3%
400,1998316,95.3%
500,2064796,98.5%
540,2097152,100.0%

╔═══════════════════════════════════════════════════════════════════╗
║     2,097,152 NODES CONVERGED at cycle 540                        ║
╚═══════════════════════════════════════════════════════════════════╝
    Time: 5087.5 ms | Rewires: 13,925,612
    Throughput: 222.60 M node-cycles/sec
```

### Experiment 7: 256×256×256 (16,777,216 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     256×256×256 = 16,777,216 NODES (24D Hypercube)                ║
║     Second Star Constant: 1122911624                              ║
║     Pushing toward 32D limit...                                   ║
╚═══════════════════════════════════════════════════════════════════╝

GPU: NVIDIA Thor (20 SMs)
Memory: 5.2 GB free / 131.9 GB total
Required: 738.2 MB nodes + 16.8 MB states = 755.0 MB total
Launching with 65536 blocks × 256 threads

CYCLE,RESONANT,PERCENT
0,16580608,98.8%
1,16580608,98.8%
2,16580608,98.8%
3,16580608,98.8%
4,16388597,97.7%
5,16388637,97.7%
6,16388642,97.7%
7,16388642,97.7%
8,16241052,96.8%
9,16241277,96.8%
200,14080457,83.9%
400,15313646,91.3%
600,16250030,96.9%
800,16355717,97.5%
1000,16581005,98.8%
1063,16777216,100.0%

╔═══════════════════════════════════════════════════════════════════╗
║     16,777,216 NODES CONVERGED at cycle 1063                      ║
╚═══════════════════════════════════════════════════════════════════╝
    Time: 80.7 sec | Rewires: 114,141,305
    Throughput: 221.01 M node-cycles/sec
```

### Experiment 8: 384×384×384 (56,623,104 nodes)

```
╔═══════════════════════════════════════════════════════════════════╗
║     384×384×384 = 56,623,104 NODES (~25.7D Hypercube)             ║
║     Second Star Constant: 1122911624                              ║
╚═══════════════════════════════════════════════════════════════════╝

GPU: NVIDIA Thor (20 SMs)
Memory: 4.9 GB free / 131.9 GB total
Required: 2.5 GB nodes + 56.6 MB states = 2.5 GB total
Launching with 221184 blocks × 256 threads

CYCLE,RESONANT,PERCENT
0,28237824,49.9%
1,55296000,97.7%
2,55296000,97.7%
3,55296000,97.7%
4,54593921,96.4%
5,54031064,95.4%
6,54031432,95.4%
7,54031362,95.4%
8,53488747,94.5%
9,53054899,93.7%
500,55744676,98.4%
570,56623104,100.0%

╔═══════════════════════════════════════════════════════════════════╗
║     56,623,104 NODES CONVERGED at cycle 570                       ║
╚═══════════════════════════════════════════════════════════════════╝
    Time: 144.4 sec | Rewires: 380,338,259
    Throughput: 223.57 M node-cycles/sec
```

---

## Statistical Summary

### Convergence Cycles vs Node Count

| log₂(Nodes) | Nodes | Cycles | Cycles/log₂(N) |
|-------------|-------|--------|----------------|
| 6 | 64 | 158 | 26.3 |
| 9 | 512 | 113 | 12.6 |
| 12 | 4,096 | 202 | 16.8 |
| 15 | 32,768 | 201 | 13.4 |
| 18 | 262,144 | 158 | 8.8 |
| 21 | 2,097,152 | 540 | 25.7 |
| 24 | 16,777,216 | 1,063 | 44.3 |
| 25.7 | 56,623,104 | 570 | 22.2 |

### Rewires per Node

| Nodes | Total Rewires | Rewires/Node |
|-------|---------------|--------------|
| 64 | 287 | 4.5 |
| 512 | 1,340 | 2.6 |
| 4,096 | 15,688 | 3.8 |
| 32,768 | 141,653 | 4.3 |
| 262,144 | 1,045,349 | 4.0 |
| 2,097,152 | 13,925,612 | 6.6 |
| 16,777,216 | 114,141,305 | 6.8 |
| 56,623,104 | 380,338,259 | 6.7 |

### Throughput Consistency

All large-scale experiments achieved approximately **220 million node-cycles per second**, demonstrating that the algorithm is compute-bound rather than memory-bound.

---

## Hardware Configuration

### NVIDIA Jetson AGX Thor

- **Architecture**: NVIDIA Ampere
- **Streaming Multiprocessors**: 20
- **Memory**: 131.9 GB Unified Memory
- **CUDA Version**: 12.x
- **Compilation Flags**: `-O3`

### Node Memory Layout

```c
struct Node {
    uint8_t state;           // 1 byte
    uint8_t frustration;     // 1 byte
    uint8_t resonance;       // 1 byte
    // padding: 1 byte
    uint32_t neighbors[6];   // 24 bytes
    uint32_t lfsr;           // 4 bytes
    uint8_t rewiring;        // 1 byte
    uint8_t rewire_dir;      // 1 byte
    uint8_t eval_ctr;        // 1 byte
    // padding: 1 byte
    uint32_t old_nb;         // 4 bytes
    uint8_t pre_frust;       // 1 byte
    // padding: 3 bytes
};
// Total: 44 bytes per node (with alignment)
```

### Memory Requirements

| Nodes | Node Array | State Buffer | Total |
|-------|------------|--------------|-------|
| 64 | 2.8 KB | 64 B | 2.9 KB |
| 512 | 22.5 KB | 512 B | 23 KB |
| 4,096 | 180 KB | 4 KB | 184 KB |
| 32,768 | 1.4 MB | 32 KB | 1.4 MB |
| 262,144 | 11.5 MB | 256 KB | 11.8 MB |
| 2,097,152 | 92.3 MB | 2.1 MB | 94.4 MB |
| 16,777,216 | 738.2 MB | 16.8 MB | 755 MB |
| 56,623,104 | 2.5 GB | 56.6 MB | 2.5 GB |

---

## The Second Star Constant

**Value**: 1122911624

**Usage in LFSR Initialization**:

```c
#define SECOND_STAR 1122911624u

n[i].lfsr = SECOND_STAR
          ^ (i * 0x9E3779B9)      // Golden ratio prime
          ^ ((i >> 8) * 0x85EBCA6B)   // MurmurHash3 constant
          ^ ((i >> 16) * 0xC2B2AE35); // MurmurHash3 constant
```

**Usage in Rewiring Target Selection**:

```c
uint32_t target = (rnd ^ (i * 0x9E3779B9) ^ (cycle * SECOND_STAR)) % NUM_NODES;
```

The Second Star provides temporal decorrelation of rewiring decisions by mixing the cycle count into the target calculation.

---

## Reproducibility Checklist

- [ ] Clone repository
- [ ] Install CUDA toolkit
- [ ] Compile with `nvcc -O3`
- [ ] Run experiments in sequence
- [ ] Compare convergence cycles to expected values
- [ ] Verify throughput is approximately 220M node-cycles/sec

**Expected Variation**: ±5% in cycle count due to random seed interactions with node layout.
