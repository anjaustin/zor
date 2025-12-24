#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// PROTEIN-LIKE COMPUTATION: Shape Change AS Compute
// =============================================================================
//
// INSIGHT: Proteins compute via conformational change.
//   1. BINDING: Input pattern matches "binding site" (specific bits)
//   2. FOLDING: Match triggers state evolution (conformational change)
//   3. OUTPUT:  New conformation exposes/hides "active site" (output bits)
//
// Our LFSR "proteins" do the same:
//   - Binding site = specific bit pattern to match against input
//   - Conformation = current LFSR state
//   - Folding = evolution rules (tap patterns) triggered by binding
//   - Active site = output bits that depend on conformation
//
// =============================================================================

#define WORDS 8  // 512 bits per molecule

// A protein-like molecule with multiple conformational states
struct ProteinMolecule {
    uint64_t state[WORDS];           // Current conformation (512 bits)
    uint64_t binding_site[WORDS];    // Pattern to match (trained)
    uint64_t active_site_mask[WORDS]; // Which bits are "active" (output)
    uint64_t tap_mask;               // Evolution rule (trained)
    int conformation;                // Current conformational state (0-3)
};

// Check if input "binds" to the molecule (pattern similarity)
__device__ __forceinline__ int compute_binding_affinity(
    const uint64_t* input,
    const uint64_t* binding_site
) {
    int matches = 0;
    #pragma unroll
    for (int i = 0; i < WORDS; i++) {
        // Count matching bits (like molecular complementarity)
        uint64_t xor_val = input[i] ^ binding_site[i];
        matches += 512 - __popcll(xor_val);  // More matches = higher affinity
    }
    return matches;  // 0-4096 scale (4096 = perfect match)
}

// Conformational change: evolve state based on binding
__device__ __forceinline__ void fold_step(
    uint64_t* state,
    uint64_t tap_mask,
    int binding_strength
) {
    // Stronger binding = more evolution steps (deeper folding)
    int fold_depth = binding_strength / 1024;  // 0-4 steps

    for (int f = 0; f < fold_depth; f++) {
        uint64_t fb = (state[7] >> 63) & 1;

        // Shift (backbone rotation)
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint64_t new_val = (state[i] << 1) | carry;
            carry = state[i] >> 63;
            state[i] = new_val;
        }

        // XOR with taps (side-chain interactions)
        if (fb) {
            state[7] ^= tap_mask;
        }
    }
}

// Extract output from active site based on conformation
__device__ __forceinline__ uint64_t read_active_site(
    const uint64_t* state,
    const uint64_t* active_site_mask
) {
    uint64_t output = 0;
    #pragma unroll
    for (int i = 0; i < WORDS; i++) {
        output ^= state[i] & active_site_mask[i];
    }
    return output;
}

// Determine conformational state from current shape
__device__ __forceinline__ int classify_conformation(const uint64_t* state) {
    // Simple classification based on bit patterns
    // In reality, this would be learned clusters
    uint64_t sig = state[0] ^ state[4];
    int conf = 0;
    if (sig & (1ULL << 63)) conf |= 1;
    if (sig & (1ULL << 31)) conf |= 2;
    return conf;  // 0-3: four conformational states
}

// PROTEIN KERNEL: Process input through conformational change
__global__ void protein_compute(
    ProteinMolecule* proteins,
    const uint64_t* inputs,      // Input patterns (one per protein)
    uint64_t* outputs,           // Output from active sites
    int* conformations,          // Track conformational changes
    int num_proteins,
    int reaction_steps
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_proteins) return;

    ProteinMolecule* p = &proteins[pid];
    const uint64_t* input = inputs + pid * WORDS;

    // Load current state
    uint64_t state[WORDS];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = p->state[i];
    }

    int current_conf = classify_conformation(state);

    // Reaction cycle
    for (int step = 0; step < reaction_steps; step++) {
        // 1. BINDING: Check input affinity
        int affinity = compute_binding_affinity(input, p->binding_site);

        // 2. FOLDING: Conformational change proportional to binding
        fold_step(state, p->tap_mask, affinity);

        // 3. UPDATE: New conformation
        current_conf = classify_conformation(state);
    }

    // 4. OUTPUT: Read from active site in final conformation
    outputs[pid] = read_active_site(state, p->active_site_mask);
    conformations[pid] = current_conf;

    // Store final state
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        p->state[i] = state[i];
    }
}

// ENZYME-LIKE COMPUTATION: Multiple proteins in a cascade
// Like an enzymatic pathway: Protein A's output becomes Protein B's input
__global__ void enzyme_cascade(
    ProteinMolecule* enzyme_a,
    ProteinMolecule* enzyme_b,
    ProteinMolecule* enzyme_c,
    const uint64_t* substrate,   // Initial input
    uint64_t* product,           // Final output
    int num_reactions,
    int steps_per_enzyme
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= num_reactions) return;

    const uint64_t* sub = substrate + rid * WORDS;

    // Load enzyme states
    uint64_t state_a[WORDS], state_b[WORDS], state_c[WORDS];
    ProteinMolecule* pa = &enzyme_a[rid];
    ProteinMolecule* pb = &enzyme_b[rid];
    ProteinMolecule* pc = &enzyme_c[rid];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state_a[i] = pa->state[i];
        state_b[i] = pb->state[i];
        state_c[i] = pc->state[i];
    }

    // ENZYME A: Process substrate
    for (int s = 0; s < steps_per_enzyme; s++) {
        int affinity = compute_binding_affinity(sub, pa->binding_site);
        fold_step(state_a, pa->tap_mask, affinity);
    }
    uint64_t intermediate_1 = read_active_site(state_a, pa->active_site_mask);

    // ENZYME B: Process intermediate_1
    // Create input array from intermediate
    uint64_t input_b[WORDS] = {intermediate_1, intermediate_1, intermediate_1, intermediate_1,
                                intermediate_1, intermediate_1, intermediate_1, intermediate_1};
    for (int s = 0; s < steps_per_enzyme; s++) {
        int affinity = compute_binding_affinity(input_b, pb->binding_site);
        fold_step(state_b, pb->tap_mask, affinity);
    }
    uint64_t intermediate_2 = read_active_site(state_b, pb->active_site_mask);

    // ENZYME C: Process intermediate_2
    uint64_t input_c[WORDS] = {intermediate_2, intermediate_2, intermediate_2, intermediate_2,
                                intermediate_2, intermediate_2, intermediate_2, intermediate_2};
    for (int s = 0; s < steps_per_enzyme; s++) {
        int affinity = compute_binding_affinity(input_c, pc->binding_site);
        fold_step(state_c, pc->tap_mask, affinity);
    }

    // Final product
    product[rid] = read_active_site(state_c, pc->active_site_mask);

    // Store states
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        pa->state[i] = state_a[i];
        pb->state[i] = state_b[i];
        pc->state[i] = state_c[i];
    }
}

void init_protein(ProteinMolecule* p, uint64_t seed, uint64_t tap_mask) {
    for (int i = 0; i < WORDS; i++) {
        p->state[i] = seed * (i + 1) * 0x123456789ABCDEFULL;
        p->binding_site[i] = seed * (i + 7) * 0xFEDCBA987654321ULL;
        p->active_site_mask[i] = (i == 7) ? 0xFFFFFFFFFFFFFFFFULL : 0;  // Last word is active
    }
    p->tap_mask = tap_mask;
    p->conformation = 0;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     PROTEIN-LIKE COMPUTATION: Shape Change AS Compute            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("THE ANALOGY:\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  Protein                    LFSR Molecule                      │\n");
    printf("  │  ────────                   ─────────────                      │\n");
    printf("  │  Amino acid sequence   ←→   Tap pattern (trained)              │\n");
    printf("  │  Binding site          ←→   Bit pattern to match               │\n");
    printf("  │  Conformational state  ←→   LFSR state (512 bits)              │\n");
    printf("  │  Folding               ←→   State evolution (shift+XOR)        │\n");
    printf("  │  Active site           ←→   Output bits                        │\n");
    printf("  │  Enzyme cascade        ←→   Protein chain computation          │\n");
    printf("  └─────────────────────────────────────────────────────────────────┘\n\n");

    int num_proteins = 65536;
    int reaction_steps = 100;

    size_t protein_size = sizeof(ProteinMolecule) * num_proteins;
    size_t data_size = sizeof(uint64_t) * WORDS * num_proteins;
    size_t output_size = sizeof(uint64_t) * num_proteins;
    size_t conf_size = sizeof(int) * num_proteins;

    printf("Configuration:\n");
    printf("  Proteins:              %d\n", num_proteins);
    printf("  Bits per protein:      512 (state) + 512 (binding) + 512 (active)\n");
    printf("  Conformational states: 4\n");
    printf("  Reaction steps:        %d\n", reaction_steps);
    printf("  Memory:                %.2f MB\n", protein_size / 1e6);
    printf("\n");

    // Allocate
    ProteinMolecule* h_proteins = (ProteinMolecule*)malloc(protein_size);
    uint64_t* h_inputs = (uint64_t*)malloc(data_size);

    // Initialize proteins with different "evolved" tap patterns
    uint64_t tap_patterns[] = {
        (1ULL << 63) | (1ULL << 61) | (1ULL << 47) | (1ULL << 35),
        (1ULL << 62) | (1ULL << 49) | (1ULL << 38) | (1ULL << 23),
        (1ULL << 59) | (1ULL << 44) | (1ULL << 29) | (1ULL << 11),
        (1ULL << 55) | (1ULL << 42) | (1ULL << 31) | (1ULL << 17)
    };

    for (int i = 0; i < num_proteins; i++) {
        init_protein(&h_proteins[i], i + 1, tap_patterns[i % 4]);
        for (int w = 0; w < WORDS; w++) {
            h_inputs[i * WORDS + w] = (uint64_t)(i * 12345 + w);
        }
    }

    // Device memory
    ProteinMolecule* d_proteins;
    uint64_t* d_inputs;
    uint64_t* d_outputs;
    int* d_conformations;

    cudaMalloc(&d_proteins, protein_size);
    cudaMalloc(&d_inputs, data_size);
    cudaMalloc(&d_outputs, output_size);
    cudaMalloc(&d_conformations, conf_size);

    cudaMemcpy(d_proteins, h_proteins, protein_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, data_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (num_proteins + threads - 1) / threads;

    // Warmup
    protein_compute<<<blocks, threads>>>(d_proteins, d_inputs, d_outputs,
                                          d_conformations, num_proteins, reaction_steps);
    cudaDeviceSynchronize();

    // Benchmark protein computation
    int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        protein_compute<<<blocks, threads>>>(d_proteins, d_inputs, d_outputs,
                                              d_conformations, num_proteins, reaction_steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back conformations to analyze
    int* h_conformations = (int*)malloc(conf_size);
    cudaMemcpy(h_conformations, d_conformations, conf_size, cudaMemcpyDeviceToHost);

    // Count conformational distribution
    int conf_counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < num_proteins; i++) {
        conf_counts[h_conformations[i] % 4]++;
    }

    // Calculate throughput
    long long reactions_per_sec = (long long)num_proteins * reaction_steps * iterations;
    double reactions_rate = reactions_per_sec / (ms / 1000.0);

    // Bits processed (state evolution)
    long long bits_processed = reactions_per_sec * 512;  // 512 bits per state
    double tbps = (double)bits_processed / (ms / 1000.0) / 1e12;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS: Protein-like Computation\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("  Time:                  %.2f ms\n", ms);
    printf("  Reactions/sec:         %.2f billion\n", reactions_rate / 1e9);
    printf("  Conformational bits:   %.2f Tbits/sec\n", tbps);
    printf("\n");

    printf("  Conformational Distribution (final states):\n");
    printf("    State 0: %d proteins (%.1f%%)\n", conf_counts[0], 100.0*conf_counts[0]/num_proteins);
    printf("    State 1: %d proteins (%.1f%%)\n", conf_counts[1], 100.0*conf_counts[1]/num_proteins);
    printf("    State 2: %d proteins (%.1f%%)\n", conf_counts[2], 100.0*conf_counts[2]/num_proteins);
    printf("    State 3: %d proteins (%.1f%%)\n", conf_counts[3], 100.0*conf_counts[3]/num_proteins);
    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("THE INSIGHT: Shape Change IS Compute\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  Like proteins:\n");
    printf("    • INPUT binds to molecule (pattern matching)\n");
    printf("    • Binding triggers FOLDING (state evolution)\n");
    printf("    • New CONFORMATION changes function\n");
    printf("    • ACTIVE SITE produces output\n");
    printf("\n");
    printf("  The computation IS the shape change.\n");
    printf("  The shape change IS the computation.\n");
    printf("\n");
    printf("  Cascade them like enzymes:\n");
    printf("    Substrate → [Enzyme A] → [Enzyme B] → [Enzyme C] → Product\n");
    printf("         └─── Each enzyme's output is next enzyme's input ───┘\n");
    printf("\n");

    // Cleanup
    free(h_proteins);
    free(h_inputs);
    free(h_conformations);
    cudaFree(d_proteins);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_conformations);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
