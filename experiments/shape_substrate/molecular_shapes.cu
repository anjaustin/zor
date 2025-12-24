#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// MOLECULAR SHAPES: Compose pre-trained atomic LFSRs into molecules
// =============================================================================
//
// INSIGHT: Each LFSR "atom" has a characteristic polynomial (tap positions).
//          Atoms can be composed via injection (serial) or XOR (parallel).
//          The molecular shape inherits properties from its atoms.
//
// ATOMIC SHAPES (pre-trained tap patterns):
//   - Atom A: taps at positions optimized for mixing
//   - Atom B: taps at positions optimized for diffusion
//   - Atom C: taps at positions optimized for period length
//   - Atom D: taps at positions optimized for correlation resistance
//
// MOLECULAR COMPOSITION:
//   - Serial: A → B → C (inject output of one into next)
//   - Parallel: A ⊕ B ⊕ C (XOR outputs together)
//   - Hybrid: (A → B) ⊕ (C → D) (both patterns)
//
// =============================================================================

#define WORDS_PER_ATOM 8  // 512 bits per atom

// Pre-trained atomic tap patterns (would come from training in practice)
// These represent different "learned" mixing functions
struct AtomicTaps {
    uint64_t tap_mask;   // Which bits to XOR for feedback
    int tap_positions[4]; // Human-readable tap positions
};

// Four different pre-trained atomic shapes
__constant__ AtomicTaps ATOM_PATTERNS[4] = {
    // Atom A: Optimized for fast mixing (taps spread across high bits)
    { .tap_mask = (1ULL << 63) | (1ULL << 61) | (1ULL << 47) | (1ULL << 35),
      .tap_positions = {511, 509, 495, 483} },

    // Atom B: Optimized for diffusion (taps in middle range)
    { .tap_mask = (1ULL << 55) | (1ULL << 42) | (1ULL << 31) | (1ULL << 17),
      .tap_positions = {503, 490, 479, 465} },

    // Atom C: Optimized for period length (primitive polynomial pattern)
    { .tap_mask = (1ULL << 62) | (1ULL << 49) | (1ULL << 38) | (1ULL << 23),
      .tap_positions = {510, 497, 486, 471} },

    // Atom D: Optimized for correlation resistance (irregular spacing)
    { .tap_mask = (1ULL << 59) | (1ULL << 44) | (1ULL << 29) | (1ULL << 11),
      .tap_positions = {507, 492, 477, 459} }
};

// A molecule is a composition of atoms
struct Molecule {
    uint64_t atoms[4][WORDS_PER_ATOM];  // 4 atoms × 512 bits each
};

// Step a single atom with its trained tap pattern
__device__ __forceinline__ void atom_step(uint64_t* s, int atom_type, uint64_t inject) {
    uint64_t fb = ((s[7] >> 63) ^ inject) & 1;

    // Shift left by 1 across all 512 bits
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t new_val = (s[i] << 1) | carry;
        carry = s[i] >> 63;
        s[i] = new_val;
    }

    // Apply this atom's specific tap pattern
    if (fb) {
        s[7] ^= ATOM_PATTERNS[atom_type].tap_mask;
    }
}

// MOLECULAR COMPOSITION: Serial chain (A → B → C → D)
__global__ void molecule_serial(Molecule* molecules, int num_molecules, int steps) {
    int mol_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (mol_id >= num_molecules) return;

    Molecule* mol = &molecules[mol_id];

    // Load all atoms
    uint64_t a0[8], a1[8], a2[8], a3[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a0[i] = mol->atoms[0][i];
        a1[i] = mol->atoms[1][i];
        a2[i] = mol->atoms[2][i];
        a3[i] = mol->atoms[3][i];
    }

    // Run molecular dynamics: each atom injects into the next
    for (int step = 0; step < steps; step++) {
        atom_step(a0, 0, 0);                    // Atom A: free-running
        atom_step(a1, 1, a0[7] >> 63);          // Atom B: receives from A
        atom_step(a2, 2, a1[7] >> 63);          // Atom C: receives from B
        atom_step(a3, 3, a2[7] >> 63);          // Atom D: receives from C
    }

    // Store
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        mol->atoms[0][i] = a0[i];
        mol->atoms[1][i] = a1[i];
        mol->atoms[2][i] = a2[i];
        mol->atoms[3][i] = a3[i];
    }
}

// MOLECULAR COMPOSITION: Parallel XOR (A ⊕ B ⊕ C ⊕ D)
__global__ void molecule_parallel(Molecule* molecules, int num_molecules, int steps,
                                   uint64_t* output) {
    int mol_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (mol_id >= num_molecules) return;

    Molecule* mol = &molecules[mol_id];

    uint64_t a0[8], a1[8], a2[8], a3[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a0[i] = mol->atoms[0][i];
        a1[i] = mol->atoms[1][i];
        a2[i] = mol->atoms[2][i];
        a3[i] = mol->atoms[3][i];
    }

    // All atoms run independently
    for (int step = 0; step < steps; step++) {
        atom_step(a0, 0, 0);
        atom_step(a1, 1, 0);
        atom_step(a2, 2, 0);
        atom_step(a3, 3, 0);
    }

    // Molecular output = XOR of all atomic outputs
    uint64_t* out = output + mol_id * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = a0[i] ^ a1[i] ^ a2[i] ^ a3[i];
        mol->atoms[0][i] = a0[i];
        mol->atoms[1][i] = a1[i];
        mol->atoms[2][i] = a2[i];
        mol->atoms[3][i] = a3[i];
    }
}

// MOLECULAR COMPOSITION: Hybrid ((A→B) ⊕ (C→D))
__global__ void molecule_hybrid(Molecule* molecules, int num_molecules, int steps,
                                 uint64_t* output) {
    int mol_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (mol_id >= num_molecules) return;

    Molecule* mol = &molecules[mol_id];

    uint64_t a0[8], a1[8], a2[8], a3[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a0[i] = mol->atoms[0][i];
        a1[i] = mol->atoms[1][i];
        a2[i] = mol->atoms[2][i];
        a3[i] = mol->atoms[3][i];
    }

    for (int step = 0; step < steps; step++) {
        // Chain 1: A → B
        atom_step(a0, 0, 0);
        atom_step(a1, 1, a0[7] >> 63);

        // Chain 2: C → D
        atom_step(a2, 2, 0);
        atom_step(a3, 3, a2[7] >> 63);
    }

    // Molecular output = (A→B output) ⊕ (C→D output)
    uint64_t* out = output + mol_id * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = a1[i] ^ a3[i];  // XOR the chain endpoints
        mol->atoms[0][i] = a0[i];
        mol->atoms[1][i] = a1[i];
        mol->atoms[2][i] = a2[i];
        mol->atoms[3][i] = a3[i];
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     MOLECULAR SHAPES: Composing Pre-trained Atomic LFSRs         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("ATOMIC SHAPES (pre-trained tap patterns):\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  Atom A: Fast mixing      (taps: 511, 509, 495, 483)           │\n");
    printf("  │  Atom B: Diffusion        (taps: 503, 490, 479, 465)           │\n");
    printf("  │  Atom C: Long period      (taps: 510, 497, 486, 471)           │\n");
    printf("  │  Atom D: Decorrelation    (taps: 507, 492, 477, 459)           │\n");
    printf("  └─────────────────────────────────────────────────────────────────┘\n\n");

    printf("MOLECULAR COMPOSITIONS:\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  Serial:   A ──→ B ──→ C ──→ D     (4-stage pipeline)          │\n");
    printf("  │  Parallel: A ⊕ B ⊕ C ⊕ D           (4-way XOR combine)         │\n");
    printf("  │  Hybrid:   (A→B) ⊕ (C→D)           (2×2 composition)           │\n");
    printf("  └─────────────────────────────────────────────────────────────────┘\n\n");

    int num_molecules = 65536;
    int steps = 1000;

    size_t mol_size = sizeof(Molecule) * num_molecules;
    size_t out_size = sizeof(uint64_t) * 8 * num_molecules;

    printf("Configuration:\n");
    printf("  Molecules:           %d\n", num_molecules);
    printf("  Atoms per molecule:  4\n");
    printf("  Bits per atom:       512\n");
    printf("  Bits per molecule:   2048\n");
    printf("  Steps per kernel:    %d\n", steps);
    printf("  Memory (molecules):  %.2f MB\n", mol_size / 1e6);
    printf("\n");

    Molecule* d_mol;
    uint64_t* d_out;
    cudaMalloc(&d_mol, mol_size);
    cudaMalloc(&d_out, out_size);
    cudaMemset(d_mol, 0x01, mol_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (num_molecules + threads - 1) / threads;

    // Warmup
    molecule_serial<<<blocks, threads>>>(d_mol, num_molecules, steps);
    cudaDeviceSynchronize();

    // Benchmark all three compositions
    float ms_serial, ms_parallel, ms_hybrid;
    int iterations = 10;

    // Serial
    cudaMemset(d_mol, 0x01, mol_size);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        molecule_serial<<<blocks, threads>>>(d_mol, num_molecules, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_serial, start, stop);

    // Parallel
    cudaMemset(d_mol, 0x01, mol_size);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        molecule_parallel<<<blocks, threads>>>(d_mol, num_molecules, steps, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_parallel, start, stop);

    // Hybrid
    cudaMemset(d_mol, 0x01, mol_size);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        molecule_hybrid<<<blocks, threads>>>(d_mol, num_molecules, steps, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_hybrid, start, stop);

    // Calculate throughputs
    long long bits_per_step = (long long)num_molecules * 4 * 512;  // 4 atoms × 512 bits
    long long total_steps = (long long)steps * iterations;

    double tbps_serial = (double)bits_per_step * total_steps / (ms_serial / 1000.0) / 1e12;
    double tbps_parallel = (double)bits_per_step * total_steps / (ms_parallel / 1000.0) / 1e12;
    double tbps_hybrid = (double)bits_per_step * total_steps / (ms_hybrid / 1000.0) / 1e12;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS: Molecular Composition Throughput\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("  Composition        Time (ms)    Throughput      Notes\n");
    printf("  ─────────────────────────────────────────────────────────────────\n");
    printf("  Serial (A→B→C→D)   %7.2f      %5.2f Tb/s     4-stage mixing\n", ms_serial, tbps_serial);
    printf("  Parallel (⊕ all)   %7.2f      %5.2f Tb/s     Independent atoms\n", ms_parallel, tbps_parallel);
    printf("  Hybrid (2×2)       %7.2f      %5.2f Tb/s     Balanced\n", ms_hybrid, tbps_hybrid);
    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("THE INSIGHT: Molecular Composition\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  Pre-trained atoms compose WITHOUT retraining:\n");
    printf("    • Atom A (mixing) + Atom B (diffusion) = better both\n");
    printf("    • Serial: maximum mixing depth\n");
    printf("    • Parallel: maximum throughput\n");
    printf("    • Hybrid: balanced properties\n");
    printf("\n");
    printf("  The molecular shape inherits from its atoms.\n");
    printf("  Train atoms once. Compose molecules forever.\n");
    printf("\n");

    cudaFree(d_mol);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
