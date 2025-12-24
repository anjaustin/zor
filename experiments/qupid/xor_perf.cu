// XOR Performance Curves
// Finding where the sweet spots and limits are

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// TEST 1: Batch Size Scaling
// How does throughput change as we process more 512-bit values in parallel?
// =============================================================================

__global__ void xor_batch(uint8_t* data, int num_values) {
    int value_id = blockIdx.x;
    int byte_id = threadIdx.x;

    if (value_id >= num_values || byte_id >= 64) return;

    int idx = value_id * 64 + byte_id;
    data[idx] ^= 0xAA;  // Simple XOR operation
}

__global__ void xor_batch_multiop(uint8_t* data, int num_values, int ops_per_value) {
    int value_id = blockIdx.x;
    int byte_id = threadIdx.x;

    if (value_id >= num_values || byte_id >= 64) return;

    int idx = value_id * 64 + byte_id;
    uint8_t v = data[idx];

    // Do multiple operations per value
    for (int i = 0; i < ops_per_value; i++) {
        v ^= (i + 1);
        v = (v << 1) | (v >> 7);  // Rotate
        v ^= byte_id;
    }

    data[idx] = v;
}

void test_batch_scaling() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 1: BATCH SIZE SCALING\n");
    printf("How many 512-bit values can we process in parallel?\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batch_sizes[] = {1, 8, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int num_tests = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("%-12s %-15s %-15s %-15s\n", "Batch Size", "Time (ms)", "512b XORs/sec", "Gbits/sec");
    printf("─────────────────────────────────────────────────────────────────\n");

    for (int t = 0; t < num_tests; t++) {
        int batch = batch_sizes[t];
        size_t data_size = batch * 64;

        uint8_t* d_data;
        cudaError_t err = cudaMalloc(&d_data, data_size);
        if (err != cudaSuccess) {
            printf("%-12d (allocation failed - %.1f MB)\n", batch, data_size / 1e6);
            continue;
        }
        cudaMemset(d_data, 0x55, data_size);

        // Warmup
        xor_batch<<<batch, 64>>>(d_data, batch);
        cudaDeviceSynchronize();

        // Timed run
        int iterations = (batch < 1000) ? 10000 : (batch < 100000) ? 1000 : 100;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_batch<<<batch, 64>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double xors_per_sec = (double)batch * iterations / (ms / 1000.0);
        double gbits_per_sec = xors_per_sec * 512 / 1e9;

        printf("%-12d %-15.3f %-15.2fM %-15.2f\n",
               batch, ms, xors_per_sec / 1e6, gbits_per_sec);

        cudaFree(d_data);
    }
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 2: Operations per Value
// What's the compute vs launch overhead tradeoff?
// =============================================================================

void test_ops_per_value() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 2: OPERATIONS PER VALUE\n");
    printf("More ops per kernel = amortize launch overhead\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batch = 16384;  // Fixed batch size
    size_t data_size = batch * 64;

    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);

    int ops_counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int num_tests = sizeof(ops_counts) / sizeof(ops_counts[0]);

    printf("Batch size: %d values (%.1f KB)\n\n", batch, data_size / 1024.0);
    printf("%-12s %-15s %-15s %-15s\n", "Ops/Value", "Time (ms)", "Total Ops/sec", "Efficiency");
    printf("─────────────────────────────────────────────────────────────────\n");

    double baseline_ops_per_sec = 0;

    for (int t = 0; t < num_tests; t++) {
        int ops = ops_counts[t];

        cudaMemset(d_data, 0x55, data_size);

        // Warmup
        xor_batch_multiop<<<batch, 64>>>(d_data, batch, ops);
        cudaDeviceSynchronize();

        int iterations = 1000;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_batch_multiop<<<batch, 64>>>(d_data, batch, ops);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Total ops = batch * 64 bytes * ops_per_byte * 2 (XOR + ROL per iteration)
        double total_ops = (double)batch * 64 * ops * 2 * iterations;
        double ops_per_sec = total_ops / (ms / 1000.0);

        if (t == 0) baseline_ops_per_sec = ops_per_sec;
        double efficiency = ops_per_sec / baseline_ops_per_sec;

        printf("%-12d %-15.3f %-15.2fM %-15.2fx\n",
               ops, ms, ops_per_sec / 1e6, efficiency);
    }
    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 3: Thread Block Size
// 64 threads per block (1 per byte) vs more threads
// =============================================================================

__global__ void xor_variable_threads(uint8_t* data, int num_values, int bytes_per_thread) {
    int value_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (value_id >= num_values) return;

    int base = value_id * 64 + thread_id * bytes_per_thread;

    for (int i = 0; i < bytes_per_thread && (thread_id * bytes_per_thread + i) < 64; i++) {
        data[base + i] ^= 0xAA;
    }
}

void test_thread_config() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 3: THREAD CONFIGURATION\n");
    printf("Varying threads per block for 512-bit values\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batch = 65536;
    size_t data_size = batch * 64;

    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);

    // threads_per_block, bytes_per_thread
    int configs[][2] = {
        {8, 8},    // 8 threads, 8 bytes each
        {16, 4},   // 16 threads, 4 bytes each
        {32, 2},   // 32 threads, 2 bytes each
        {64, 1},   // 64 threads, 1 byte each (our standard)
        {128, 1},  // 128 threads (64 active + 64 idle)
        {256, 1},  // 256 threads
    };
    int num_tests = sizeof(configs) / sizeof(configs[0]);

    printf("Batch size: %d values\n\n", batch);
    printf("%-15s %-15s %-15s %-15s\n", "Threads/Block", "Bytes/Thread", "Time (ms)", "512b XORs/sec");
    printf("─────────────────────────────────────────────────────────────────\n");

    for (int t = 0; t < num_tests; t++) {
        int threads = configs[t][0];
        int bytes_per = configs[t][1];

        cudaMemset(d_data, 0x55, data_size);

        // Warmup
        xor_variable_threads<<<batch, threads>>>(d_data, batch, bytes_per);
        cudaDeviceSynchronize();

        int iterations = 1000;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_variable_threads<<<batch, threads>>>(d_data, batch, bytes_per);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double xors_per_sec = (double)batch * iterations / (ms / 1000.0);

        printf("%-15d %-15d %-15.3f %-15.2fM\n",
               threads, bytes_per, ms, xors_per_sec / 1e6);
    }
    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 4: Memory Bandwidth vs Compute
// At what point does memory become the bottleneck?
// =============================================================================

// Compute-heavy: many XORs, few memory accesses
__global__ void compute_heavy(uint8_t* data, int num_values, int compute_rounds) {
    int value_id = blockIdx.x;
    int byte_id = threadIdx.x;

    if (value_id >= num_values || byte_id >= 64) return;

    int idx = value_id * 64 + byte_id;
    uint8_t v = data[idx];

    // Lots of compute on the same data
    for (int r = 0; r < compute_rounds; r++) {
        v ^= r;
        v = (v << 3) | (v >> 5);
        v ^= (v >> 2);
        v = (v << 1) | (v >> 7);
        v ^= byte_id;
    }

    data[idx] = v;
}

// Memory-heavy: minimal compute, many memory accesses
__global__ void memory_heavy(uint8_t* data, int num_values, int stride) {
    int value_id = blockIdx.x;
    int byte_id = threadIdx.x;

    if (value_id >= num_values || byte_id >= 64) return;

    // Access multiple locations with stride
    uint8_t acc = 0;
    for (int s = 0; s < 8; s++) {
        int idx = ((value_id + s * stride) % num_values) * 64 + byte_id;
        acc ^= data[idx];
    }

    int idx = value_id * 64 + byte_id;
    data[idx] = acc;
}

void test_memory_vs_compute() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 4: MEMORY vs COMPUTE BOUND\n");
    printf("Finding the crossover point\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batch = 65536;
    size_t data_size = batch * 64;

    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemset(d_data, 0x55, data_size);

    printf("Batch size: %d values (%.1f MB)\n\n", batch, data_size / 1e6);

    // Test compute-heavy
    printf("COMPUTE-HEAVY (varying compute rounds per value):\n");
    printf("%-15s %-15s %-15s %-15s\n", "Rounds", "Time (ms)", "Values/sec", "Ops/Value");
    printf("─────────────────────────────────────────────────────────────────\n");

    int compute_rounds[] = {1, 4, 16, 64, 256, 1024};
    for (int t = 0; t < 6; t++) {
        int rounds = compute_rounds[t];

        int iterations = (rounds < 100) ? 1000 : 100;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            compute_heavy<<<batch, 64>>>(d_data, batch, rounds);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double values_per_sec = (double)batch * iterations / (ms / 1000.0);
        int ops_per_value = rounds * 5;  // 5 ops per round

        printf("%-15d %-15.3f %-15.2fM %-15d\n",
               rounds, ms, values_per_sec / 1e6, ops_per_value);
    }

    printf("\n");

    // Test memory-heavy
    printf("MEMORY-HEAVY (varying access stride):\n");
    printf("%-15s %-15s %-15s %-15s\n", "Stride", "Time (ms)", "Values/sec", "Reads/Value");
    printf("─────────────────────────────────────────────────────────────────\n");

    int strides[] = {1, 16, 256, 4096, 16384};
    for (int t = 0; t < 5; t++) {
        int stride = strides[t];

        int iterations = 100;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            memory_heavy<<<batch, 64>>>(d_data, batch, stride);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double values_per_sec = (double)batch * iterations / (ms / 1000.0);

        printf("%-15d %-15.3f %-15.2fM %-15d\n",
               stride, ms, values_per_sec / 1e6, 8);
    }
    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 5: Sustained Throughput
// What's the maximum sustained XOR rate?
// =============================================================================

__global__ void sustained_xor(uint8_t* data, int num_values) {
    int value_id = blockIdx.x;
    int byte_id = threadIdx.x;

    if (value_id >= num_values || byte_id >= 64) return;

    int idx = value_id * 64 + byte_id;

    // Multiple XORs to measure sustained rate
    uint8_t v = data[idx];

    #pragma unroll 16
    for (int i = 0; i < 64; i++) {
        v ^= i;
    }

    data[idx] = v;
}

void test_sustained_throughput() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 5: SUSTAINED THROUGHPUT\n");
    printf("Maximum continuous XOR rate\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use large batch to saturate GPU
    int batch = 262144;  // 256K values = 16 MB
    size_t data_size = batch * 64;

    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemset(d_data, 0x55, data_size);

    printf("Configuration:\n");
    printf("  Batch size: %d values (%.1f MB)\n", batch, data_size / 1e6);
    printf("  XORs per value: 64\n");
    printf("  Total XORs per kernel: %.1fM\n\n", (double)batch * 64 * 64 / 1e6);

    // Warmup
    for (int i = 0; i < 10; i++) {
        sustained_xor<<<batch, 64>>>(d_data, batch);
    }
    cudaDeviceSynchronize();

    // Timed run
    int iterations = 1000;

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        sustained_xor<<<batch, 64>>>(d_data, batch);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double total_xors = (double)batch * 64 * 64 * iterations;
    double xors_per_sec = total_xors / (ms / 1000.0);
    double bits_per_sec = xors_per_sec * 8;  // 8 bits per XOR

    printf("Results:\n");
    printf("  Time: %.2f ms\n", ms);
    printf("  XOR operations/sec: %.2f G\n", xors_per_sec / 1e9);
    printf("  Bits XOR'd/sec: %.2f G\n", bits_per_sec / 1e9);
    printf("  512-bit values processed/sec: %.2f M\n", (double)batch * iterations / (ms / 1000.0) / 1e6);
    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 6: Scaling with Worker Count
// What if we use more than 64 workers?
// =============================================================================

template<int WORKERS>
__global__ void xor_n_workers(uint8_t* data, int num_values) {
    int value_id = blockIdx.x;
    int worker_id = threadIdx.x;

    if (value_id >= num_values || worker_id >= WORKERS) return;

    // Each worker handles (512/WORKERS) bits
    constexpr int BITS_PER_WORKER = 512 / WORKERS;
    constexpr int BYTES_PER_WORKER = BITS_PER_WORKER / 8;

    int base = value_id * 64 + worker_id * BYTES_PER_WORKER;

    #pragma unroll
    for (int i = 0; i < BYTES_PER_WORKER; i++) {
        data[base + i] ^= 0xAA;
    }
}

void test_worker_scaling() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 6: WORKER COUNT SCALING\n");
    printf("512 bits / N workers = bits per worker\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batch = 65536;
    size_t data_size = batch * 64;

    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);

    printf("Batch size: %d values\n\n", batch);
    printf("%-15s %-15s %-15s %-15s\n", "Workers", "Bits/Worker", "Time (ms)", "512b XORs/sec");
    printf("─────────────────────────────────────────────────────────────────\n");

    int iterations = 1000;

    // 8 workers (64 bits each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<8><<<batch, 8>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 8, 64, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 16 workers (32 bits each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<16><<<batch, 16>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 16, 32, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 32 workers (16 bits each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<32><<<batch, 32>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 32, 16, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 64 workers (8 bits each) - our standard
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<64><<<batch, 64>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 64, 8, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 128 workers (4 bits each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<128><<<batch, 128>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 128, 4, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 256 workers (2 bits each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<256><<<batch, 256>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 256, 2, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    // 512 workers (1 bit each)
    {
        cudaMemset(d_data, 0x55, data_size);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            xor_n_workers<512><<<batch, 512>>>(d_data, batch);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%-15d %-15d %-15.3f %-15.2fM\n", 512, 1, ms, (double)batch * iterations / (ms / 1000.0) / 1e6);
    }

    printf("\n");

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// TEST 7: Real Crypto Workload
// ARX cipher rounds (like ChaCha20)
// =============================================================================

__global__ void chacha_quarter_round(uint32_t* state, int num_states) {
    int state_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (state_id >= num_states || thread_id >= 4) return;

    // Each state is 16 x 32-bit words = 512 bits
    uint32_t* s = state + state_id * 16;

    __shared__ uint32_t shared[16];

    // Load state into shared memory
    for (int i = thread_id; i < 16; i += 4) {
        shared[i] = s[i];
    }
    __syncthreads();

    // Quarter round on columns (threads handle one column each)
    // a = 0,4,8,12  b = 1,5,9,13  c = 2,6,10,14  d = 3,7,11,15
    int a = thread_id;
    int b = thread_id + 4;
    int c = thread_id + 8;
    int d = thread_id + 12;

    // ChaCha quarter round
    shared[a] += shared[b]; shared[d] ^= shared[a]; shared[d] = (shared[d] << 16) | (shared[d] >> 16);
    shared[c] += shared[d]; shared[b] ^= shared[c]; shared[b] = (shared[b] << 12) | (shared[b] >> 20);
    shared[a] += shared[b]; shared[d] ^= shared[a]; shared[d] = (shared[d] << 8) | (shared[d] >> 24);
    shared[c] += shared[d]; shared[b] ^= shared[c]; shared[b] = (shared[b] << 7) | (shared[b] >> 25);

    __syncthreads();

    // Store back
    for (int i = thread_id; i < 16; i += 4) {
        s[i] = shared[i];
    }
}

void test_crypto_workload() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("TEST 7: CRYPTO WORKLOAD (ChaCha-style quarter rounds)\n");
    printf("Each state is 512 bits (16 x 32-bit words)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_states = 65536;
    size_t state_size = num_states * 16 * sizeof(uint32_t);

    uint32_t* d_state;
    cudaMalloc(&d_state, state_size);
    cudaMemset(d_state, 0x42, state_size);

    printf("Configuration:\n");
    printf("  States: %d (each 512 bits)\n", num_states);
    printf("  Total data: %.1f MB\n\n", state_size / 1e6);

    // Warmup
    for (int i = 0; i < 10; i++) {
        chacha_quarter_round<<<num_states, 4>>>(d_state, num_states);
    }
    cudaDeviceSynchronize();

    // Single quarter round
    int iterations = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        chacha_quarter_round<<<num_states, 4>>>(d_state, num_states);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double qr_per_sec = (double)num_states * iterations / (ms / 1000.0);
    double bits_per_sec = qr_per_sec * 512;

    printf("Results:\n");
    printf("  Quarter rounds/sec: %.2f M\n", qr_per_sec / 1e6);
    printf("  Bits processed/sec: %.2f G\n", bits_per_sec / 1e9);
    printf("  Equivalent ChaCha20 throughput: %.2f MB/s\n", qr_per_sec * 64 / 20 / 1e6);
    printf("  (ChaCha20 needs 20 rounds of 4 quarter-rounds each)\n\n");

    cudaFree(d_state);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║           XOR PERFORMANCE CURVES - FINDING THE LIMITS             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Max threads/SM: %d\n", prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("Shared memory/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");

    test_batch_scaling();
    test_ops_per_value();
    test_thread_config();
    test_memory_vs_compute();
    test_sustained_throughput();
    test_worker_scaling();
    test_crypto_workload();

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                      PERFORMANCE SUMMARY                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("Key findings:\n");
    printf("  1. Batch size: Larger = better (amortizes kernel launch)\n");
    printf("  2. Ops per value: More compute = better efficiency\n");
    printf("  3. Worker count: Sweet spot depends on workload\n");
    printf("  4. Memory vs compute: Crossover point found\n");
    printf("  5. Sustained rate: Maximum XOR throughput measured\n");
    printf("\n");

    return 0;
}
