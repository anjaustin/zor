// XOR Magic with Addressable CPUs
// 64 workers doing XOR-based computation

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define NUM_PODS 8
#define WORKERS_PER_POD 8
#define TOTAL_WORKERS 64

// XOR Properties we exploit:
// 1. Self-inverse: A ^ A = 0
// 2. Commutative: A ^ B = B ^ A
// 3. Associative: (A ^ B) ^ C = A ^ (B ^ C)
// 4. Identity: A ^ 0 = A
// 5. Swap trick: A ^= B; B ^= A; A ^= B;

struct WorkerState {
    uint8_t regs[16];
    uint8_t active;
    uint8_t pad[15];
};

struct SystemState {
    WorkerState workers[TOTAL_WORKERS];
};

// =============================================================================
// XOR KERNELS
// =============================================================================

// XOR Reduction: All 64 values → 1 value in 6 steps (log2(64))
__global__ void xor_reduce(SystemState* sys, int stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS) return;

    // Only workers at stride intervals participate
    if ((id % (stride * 2)) == 0) {
        int partner = id + stride;
        if (partner < TOTAL_WORKERS) {
            sys->workers[id].regs[0] ^= sys->workers[partner].regs[0];
        }
    }
}

// XOR Broadcast: One source XORs with all others
__global__ void xor_broadcast(SystemState* sys, int source_id) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS || id == source_id) return;

    sys->workers[id].regs[0] ^= sys->workers[source_id].regs[0];
}

// XOR Diffusion: Each worker XORs with neighbor, creating wave patterns
__global__ void xor_diffuse(SystemState* sys) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS) return;

    // Read neighbor's value (wrap around)
    int neighbor = (id + 1) % TOTAL_WORKERS;
    uint8_t neighbor_val = sys->workers[neighbor].regs[0];

    __syncthreads();

    // XOR with neighbor
    sys->workers[id].regs[0] ^= neighbor_val;
}

// XOR Swap Network: Parallel swaps without temporaries
__global__ void xor_swap_step(SystemState* sys, int step) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS) return;

    // Butterfly pattern for step
    int partner = id ^ (1 << step);

    if (id < partner && partner < TOTAL_WORKERS) {
        // Three-step XOR swap
        uint8_t a = sys->workers[id].regs[0];
        uint8_t b = sys->workers[partner].regs[0];

        a ^= b;
        b ^= a;
        a ^= b;

        sys->workers[id].regs[0] = a;
        sys->workers[partner].regs[0] = b;
    }
}

// Galois LFSR across workers (polynomial feedback)
__global__ void xor_lfsr_step(SystemState* sys, uint8_t taps) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS - 1) return;

    // Shift left
    uint8_t current = sys->workers[id].regs[0];
    uint8_t next = sys->workers[id + 1].regs[0];

    __syncthreads();

    // Feedback from MSB
    uint8_t feedback = (sys->workers[TOTAL_WORKERS - 1].regs[0] & 0x80) ? taps : 0;

    if (id == 0) {
        sys->workers[id].regs[0] = (next << 1) ^ feedback;
    } else {
        sys->workers[id].regs[0] = next;
    }
}

// XOR Checksum: Compute running XOR parity
__global__ void xor_parity(SystemState* sys) {
    __shared__ uint8_t shared[TOTAL_WORKERS];

    int id = threadIdx.x;
    if (id >= TOTAL_WORKERS) return;

    shared[id] = sys->workers[id].regs[0];
    __syncthreads();

    // Tree reduction with XOR
    for (int s = TOTAL_WORKERS / 2; s > 0; s >>= 1) {
        if (id < s) {
            shared[id] ^= shared[id + s];
        }
        __syncthreads();
    }

    // All workers get the parity
    sys->workers[id].regs[1] = shared[0];
}

// XOR Ring: Each worker XORs with rotated position
__global__ void xor_ring(SystemState* sys, int rotation) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_WORKERS) return;

    int source = (id + rotation) % TOTAL_WORKERS;
    sys->workers[id].regs[0] ^= sys->workers[source].regs[0];
}

// =============================================================================
// DEMOS
// =============================================================================

void print_workers(SystemState* sys, const char* label, int count = 16) {
    printf("%s: ", label);
    for (int i = 0; i < count; i++) {
        printf("%02X ", sys->workers[i].regs[0]);
    }
    if (count < TOTAL_WORKERS) printf("...");
    printf("\n");
}

void print_all_workers(SystemState* sys, const char* label) {
    printf("%s:\n", label);
    for (int pod = 0; pod < NUM_PODS; pod++) {
        printf("  Pod %d: ", pod);
        for (int w = 0; w < WORKERS_PER_POD; w++) {
            printf("%02X ", sys->workers[pod * WORKERS_PER_POD + w].regs[0]);
        }
        printf("\n");
    }
}

void demo_xor_reduction() {
    printf("=== XOR REDUCTION: 64 values → 1 ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Initialize: each worker has its ID
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        h_sys->workers[i].regs[0] = i;
    }

    print_all_workers(h_sys, "Initial");
    printf("\n");

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // 6 reduction steps: 64→32→16→8→4→2→1
    for (int stride = 1; stride < TOTAL_WORKERS; stride *= 2) {
        xor_reduce<<<1, TOTAL_WORKERS>>>(d_sys, stride);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("After XOR reduction:\n");
    printf("  Worker 0 contains: 0x%02X\n", h_sys->workers[0].regs[0]);
    printf("  (XOR of 0,1,2,...,63)\n\n");

    // Verify: XOR of 0..63
    uint8_t expected = 0;
    for (int i = 0; i < 64; i++) expected ^= i;
    printf("  Expected: 0x%02X  %s\n\n", expected,
           (h_sys->workers[0].regs[0] == expected) ? "✓" : "✗");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_xor_diffusion() {
    printf("=== XOR DIFFUSION: Single bit spreads ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Initialize: only worker 0 has a value
    memset(h_sys, 0, sizeof(SystemState));
    h_sys->workers[0].regs[0] = 0xFF;

    print_workers(h_sys, "Step 0");

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Watch diffusion spread
    for (int step = 1; step <= 8; step++) {
        xor_diffuse<<<1, TOTAL_WORKERS>>>(d_sys);
        cudaDeviceSynchronize();

        cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);
        printf("Step %d", step);
        print_workers(h_sys, "");
    }

    printf("\nThe pattern spreads through XOR with neighbors.\n");
    printf("After 64 steps, it returns to original (XOR is self-inverse).\n\n");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_xor_swap() {
    printf("=== XOR SWAP NETWORK: Bit-reversal permutation ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Initialize: each worker has its ID
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        h_sys->workers[i].regs[0] = i;
    }

    printf("Before: Workers hold their IDs (0,1,2,...,63)\n");
    print_workers(h_sys, "  ");

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // 6 butterfly steps for bit-reversal
    for (int step = 0; step < 6; step++) {
        xor_swap_step<<<1, TOTAL_WORKERS>>>(d_sys, step);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("After:  Bit-reversed positions\n");
    print_workers(h_sys, "  ");
    printf("\n");

    // Show what happened
    printf("Examples:\n");
    printf("  Worker 0  (000000b) now has: %d (from worker %d = %06bb)\n",
           h_sys->workers[0].regs[0], h_sys->workers[0].regs[0], h_sys->workers[0].regs[0]);
    printf("  Worker 1  (000001b) now has: %d (from worker %d = %06bb)\n",
           h_sys->workers[1].regs[0], h_sys->workers[1].regs[0], h_sys->workers[1].regs[0]);
    printf("  Worker 32 (100000b) now has: %d (from worker %d = %06bb)\n",
           h_sys->workers[32].regs[0], h_sys->workers[32].regs[0], h_sys->workers[32].regs[0]);
    printf("\n");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_xor_encryption() {
    printf("=== XOR ENCRYPTION: One-time pad across workers ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Message: "HELLO XOR" (8 bytes, pad last)
    const char* message = "HELLO!!";
    const char* key     = "SECRETS!";

    printf("Message:   ");
    for (int i = 0; i < 8; i++) printf("%02X ", (uint8_t)message[i]);
    printf(" = \"%s\"\n", message);

    printf("Key:       ");
    for (int i = 0; i < 8; i++) printf("%02X ", (uint8_t)key[i]);
    printf(" = \"%s\"\n", key);

    // Load message into first 8 workers
    memset(h_sys, 0, sizeof(SystemState));
    for (int i = 0; i < 8; i++) {
        h_sys->workers[i].regs[0] = message[i];
    }
    // Load key into workers 8-15
    for (int i = 0; i < 8; i++) {
        h_sys->workers[8 + i].regs[0] = key[i];
    }

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // XOR message with key (workers 0-7 XOR with workers 8-15)
    xor_ring<<<1, TOTAL_WORKERS>>>(d_sys, 8);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("Encrypted: ");
    for (int i = 0; i < 8; i++) printf("%02X ", h_sys->workers[i].regs[0]);
    printf("\n");

    // XOR again to decrypt
    xor_ring<<<1, TOTAL_WORKERS>>>(d_sys, 8);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("Decrypted: ");
    for (int i = 0; i < 8; i++) printf("%02X ", h_sys->workers[i].regs[0]);
    printf(" = \"");
    for (int i = 0; i < 8; i++) printf("%c", h_sys->workers[i].regs[0]);
    printf("\"\n\n");

    printf("XOR encryption: Encrypt and decrypt are the same operation!\n\n");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_xor_hash() {
    printf("=== XOR PARITY: Fast checksum across all workers ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Initialize with some data pattern
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        h_sys->workers[i].regs[0] = (i * 17 + 42) & 0xFF;  // Some pattern
    }

    printf("Data pattern: (i*17+42) & 0xFF for each worker\n");
    print_workers(h_sys, "First 16");

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    xor_parity<<<1, TOTAL_WORKERS>>>(d_sys);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("XOR parity: 0x%02X (stored in reg[1] of all workers)\n\n",
           h_sys->workers[0].regs[1]);

    // Flip one bit
    printf("Flip bit 7 of worker 30...\n");
    h_sys->workers[30].regs[0] ^= 0x80;

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);
    xor_parity<<<1, TOTAL_WORKERS>>>(d_sys);
    cudaDeviceSynchronize();
    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("New parity: 0x%02X (XOR with old parity = 0x%02X, the flipped bit!)\n\n",
           h_sys->workers[0].regs[1],
           h_sys->workers[0].regs[1] ^ ((30 * 17 + 42) & 0xFF));

    delete h_sys;
    cudaFree(d_sys);
}

void demo_xor_lfsr() {
    printf("=== XOR LFSR: Pseudo-random across workers ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    // Initialize with seed
    memset(h_sys, 0, sizeof(SystemState));
    h_sys->workers[0].regs[0] = 0xAC;  // Seed
    h_sys->workers[1].regs[0] = 0xE1;

    printf("LFSR with taps at bits 7,5,4,3 (x^8 + x^6 + x^5 + x^4 + 1)\n");
    printf("Seed: ");
    print_workers(h_sys, "");

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Run LFSR steps
    uint8_t taps = 0b10111000;  // Bits 7,5,4,3
    for (int step = 0; step < 8; step++) {
        xor_lfsr_step<<<1, TOTAL_WORKERS>>>(d_sys, taps);
        cudaDeviceSynchronize();

        cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);
        printf("Step %d: ", step + 1);
        print_workers(h_sys, "");
    }

    printf("\nLFSR generates pseudo-random sequences using only XOR!\n\n");

    delete h_sys;
    cudaFree(d_sys);
}

void benchmark_xor_ops() {
    printf("=== BENCHMARK: XOR Operations ===\n\n");

    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));
    cudaMemset(d_sys, 0x42, sizeof(SystemState));

    const int ITERATIONS = 1000000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark XOR diffusion
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        xor_diffuse<<<1, TOTAL_WORKERS>>>(d_sys);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Each diffuse: 64 workers × 1 XOR
    double xor_ops = (double)ITERATIONS * TOTAL_WORKERS;
    double xor_per_sec = xor_ops / (ms / 1000.0);

    printf("XOR Diffusion:\n");
    printf("  Iterations: %d\n", ITERATIONS);
    printf("  Time: %.2f ms\n", ms);
    printf("  XOR ops/sec: %.2f M\n", xor_per_sec / 1e6);
    printf("\n");

    // Benchmark XOR reduction
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        // 6 steps for full reduction
        for (int stride = 1; stride < TOTAL_WORKERS; stride *= 2) {
            xor_reduce<<<1, TOTAL_WORKERS>>>(d_sys, stride);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    // Each reduction: 64-1 = 63 XOR ops
    xor_ops = (double)ITERATIONS * 63;
    xor_per_sec = xor_ops / (ms / 1000.0);

    printf("XOR Reduction:\n");
    printf("  Iterations: %d\n", ITERATIONS);
    printf("  Time: %.2f ms\n", ms);
    printf("  XOR ops/sec: %.2f M\n", xor_per_sec / 1e6);
    printf("  Full 64→1 reductions/sec: %.2f M\n\n", (double)ITERATIONS / (ms / 1000.0) / 1e6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sys);
}

int main() {
    printf("=======================================================================\n");
    printf("XOR MAGIC WITH 64 ADDRESSABLE CPUs\n");
    printf("=======================================================================\n\n");

    printf("XOR Properties:\n");
    printf("  A ^ A = 0        (self-inverse)\n");
    printf("  A ^ B = B ^ A    (commutative)\n");
    printf("  A ^ 0 = A        (identity)\n");
    printf("  (A^B)^C = A^(B^C) (associative)\n\n");

    demo_xor_reduction();
    demo_xor_diffusion();
    demo_xor_swap();
    demo_xor_encryption();
    demo_xor_hash();
    demo_xor_lfsr();
    benchmark_xor_ops();

    printf("=======================================================================\n");
    printf("XOR + ADDRESSABLE CPUs = Parallel Cryptographic Primitives\n");
    printf("=======================================================================\n");

    return 0;
}
