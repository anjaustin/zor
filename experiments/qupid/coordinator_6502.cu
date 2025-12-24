/**
 * 6502 Coordination Engine
 *
 * Not a 6502 emulator. A 6502-STYLE coordination architecture.
 *
 * The 6502's power came from:
 *   - Zero page as routing table
 *   - Memory-mapped fabric configuration
 *   - Specialized chips (VIC, SID, CIA) as fabrics
 *   - Simple coordinator, powerful peripherals
 *
 * This implements that pattern on GPU.
 *
 * Build: nvcc -std=c++17 -O3 -o coordinator_6502 coordinator_6502.cu
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =============================================================================
// MEMORY MAP (inspired by C64)
// =============================================================================
//
// $0000-$00FF: Routing Table (256 bytes, like zero page)
// $0100-$01FF: Coordinator Stack
// $0200-$7FFF: Data Space (32KB)
// $8000-$8FFF: Fabric 0 - Graphics (4KB)
// $9000-$9FFF: Fabric 1 - Compute (4KB)
// $A000-$AFFF: Fabric 2 - Memory (4KB)
// $B000-$BFFF: Fabric 3 - Custom (4KB)
//
// =============================================================================

#define MEM_SIZE 65536
#define ROUTING_BASE 0x0000
#define STACK_BASE   0x0100
#define DATA_BASE    0x0200
#define FABRIC0_BASE 0x8000
#define FABRIC1_BASE 0x9000
#define FABRIC2_BASE 0xA000
#define FABRIC3_BASE 0xB000

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Route entry: 2 bytes, like a zero page pointer
// But instead of address, it encodes: source → destination
struct RouteEntry {
    uint8_t src;  // High nibble: fabric, Low nibble: port
    uint8_t dst;  // High nibble: fabric, Low nibble: port
};

// Fabric configuration: 32 bytes of registers
struct FabricConfig {
    uint8_t mode;           // Operating mode
    uint8_t control;        // Control flags (enable, interrupt, etc.)
    uint8_t param[14];      // Mode-specific parameters
    uint8_t input[8];       // Input ports
    uint8_t output[8];      // Output ports
};

// Complete memory space
struct MemorySpace {
    RouteEntry routes[128];     // $0000-$00FF: Routing table
    uint8_t stack[256];         // $0100-$01FF: Stack
    uint8_t data[32512];        // $0200-$7FFF: Data
    FabricConfig fabric[4];     // $8000-$BFFF: Fabric configs (128 bytes, rest is fabric workspace)
    uint8_t fabric_mem[4][4096-32]; // Fabric workspace
};

// =============================================================================
// FABRIC IMPLEMENTATIONS
// =============================================================================

// Fabric 0: Graphics-like operations
// Mode 0: Tile lookup (input[0-1] = x,y coords, output[0] = tile value)
// Mode 1: Color blend (input[0-2] = r,g,b, input[3] = alpha, output[0-2] = blended)
// Mode 2: Sprite check (input[0-3] = x1,y1,x2,y2, output[0] = collision)
__device__ void fabric_graphics(FabricConfig* config, uint8_t* workspace) {
    switch (config->mode) {
        case 0: {  // Tile lookup
            uint8_t x = config->input[0];
            uint8_t y = config->input[1];
            uint8_t tile_width = config->param[0] ? config->param[0] : 8;
            uint16_t offset = (y / tile_width) * 32 + (x / tile_width);
            config->output[0] = workspace[offset % 4064];
            break;
        }
        case 1: {  // Color blend
            uint16_t alpha = config->input[3];
            config->output[0] = (config->input[0] * alpha + workspace[0] * (255 - alpha)) / 255;
            config->output[1] = (config->input[1] * alpha + workspace[1] * (255 - alpha)) / 255;
            config->output[2] = (config->input[2] * alpha + workspace[2] * (255 - alpha)) / 255;
            break;
        }
        case 2: {  // Sprite collision
            int16_t x1 = config->input[0], y1 = config->input[1];
            int16_t x2 = config->input[2], y2 = config->input[3];
            int16_t w = config->param[0] ? config->param[0] : 8;
            int16_t h = config->param[1] ? config->param[1] : 8;
            bool collide = !(x1 + w < x2 || x2 + w < x1 || y1 + h < y2 || y2 + h < y1);
            config->output[0] = collide ? 1 : 0;
            break;
        }
    }
}

// Fabric 1: Compute operations
// Mode 0: Arithmetic (input[0] op input[1] → output[0])
// Mode 1: Compare (input[0] cmp input[1] → output[0] = flags)
// Mode 2: Bitwise (input[0] op input[1] → output[0])
// Mode 3: Table lookup (input[0] = index, output[0] = table[index])
__device__ void fabric_compute(FabricConfig* config, uint8_t* workspace) {
    uint8_t a = config->input[0];
    uint8_t b = config->input[1];
    uint8_t op = config->param[0];

    switch (config->mode) {
        case 0: {  // Arithmetic
            switch (op) {
                case 0: config->output[0] = a + b; break;
                case 1: config->output[0] = a - b; break;
                case 2: config->output[0] = a * b; break;
                case 3: config->output[0] = b ? a / b : 0; break;
                case 4: config->output[0] = b ? a % b : 0; break;
            }
            break;
        }
        case 1: {  // Compare
            uint8_t flags = 0;
            if (a == b) flags |= 0x01;  // Zero
            if (a < b)  flags |= 0x02;  // Less
            if (a > b)  flags |= 0x04;  // Greater
            config->output[0] = flags;
            break;
        }
        case 2: {  // Bitwise
            switch (op) {
                case 0: config->output[0] = a & b; break;
                case 1: config->output[0] = a | b; break;
                case 2: config->output[0] = a ^ b; break;
                case 3: config->output[0] = ~a; break;
                case 4: config->output[0] = a << (b & 7); break;
                case 5: config->output[0] = a >> (b & 7); break;
            }
            break;
        }
        case 3: {  // Table lookup
            config->output[0] = workspace[a % 4064];
            break;
        }
    }
}

// Fabric 2: Memory operations
// Mode 0: Copy (param[0-1] = src, param[2-3] = dst, param[4] = len)
// Mode 1: Fill (param[0-1] = dst, param[2] = value, param[3] = len)
// Mode 2: Search (param[0-1] = start, param[2] = value, output[0-1] = found addr)
__device__ void fabric_memory(FabricConfig* config, uint8_t* workspace, uint8_t* data) {
    switch (config->mode) {
        case 0: {  // Copy (within workspace for now)
            uint16_t src = config->param[0];
            uint16_t dst = config->param[2];
            uint8_t len = config->param[4];
            for (int i = 0; i < len && i < 64; i++) {
                workspace[(dst + i) % 4064] = workspace[(src + i) % 4064];
            }
            break;
        }
        case 1: {  // Fill
            uint16_t dst = config->param[0];
            uint8_t value = config->param[2];
            uint8_t len = config->param[3];
            for (int i = 0; i < len && i < 64; i++) {
                workspace[(dst + i) % 4064] = value;
            }
            break;
        }
        case 2: {  // Search
            uint16_t start = config->param[0];
            uint8_t value = config->param[2];
            for (int i = 0; i < 256; i++) {
                if (workspace[(start + i) % 4064] == value) {
                    config->output[0] = (start + i) & 0xFF;
                    config->output[1] = ((start + i) >> 8) & 0xFF;
                    return;
                }
            }
            config->output[0] = 0xFF;
            config->output[1] = 0xFF;
            break;
        }
    }
}

// Fabric 3: Custom/programmable
// Mode 0: State machine (workspace[0] = state, transitions in workspace)
// Mode 1: Counter (count up/down based on input)
// Mode 2: Timer (decrement, signal on zero)
__device__ void fabric_custom(FabricConfig* config, uint8_t* workspace) {
    switch (config->mode) {
        case 0: {  // State machine
            uint8_t state = workspace[0];
            uint8_t input = config->input[0];
            // Transition table at workspace[1..]: (state, input) → next_state
            // Simple: next = workspace[1 + state * 16 + input]
            uint8_t next = workspace[1 + (state & 0xF) * 16 + (input & 0xF)];
            workspace[0] = next;
            config->output[0] = next;
            break;
        }
        case 1: {  // Counter
            int16_t count = workspace[0] | (workspace[1] << 8);
            if (config->input[0] & 0x01) count++;
            if (config->input[0] & 0x02) count--;
            if (config->input[0] & 0x04) count = 0;
            workspace[0] = count & 0xFF;
            workspace[1] = (count >> 8) & 0xFF;
            config->output[0] = workspace[0];
            config->output[1] = workspace[1];
            break;
        }
        case 2: {  // Timer
            if (workspace[0] > 0) {
                workspace[0]--;
                config->output[0] = 0;
            } else {
                config->output[0] = 1;  // Timer expired signal
                workspace[0] = config->param[0];  // Reload
            }
            break;
        }
    }
}

// =============================================================================
// COORDINATOR KERNEL
// =============================================================================

__global__ void route_data(MemorySpace* mem) {
    // Each thread handles one route
    int route_id = threadIdx.x;
    if (route_id >= 128) return;

    RouteEntry route = mem->routes[route_id];

    // Inactive routes have src = 0xFF
    if (route.src == 0xFF) return;

    // Decode source
    int src_fabric = (route.src >> 4) & 0x0F;
    int src_port = route.src & 0x0F;

    // Decode destination
    int dst_fabric = (route.dst >> 4) & 0x0F;
    int dst_port = route.dst & 0x0F;

    // Route data from source output to destination input
    if (src_fabric < 4 && dst_fabric < 4 && src_port < 8 && dst_port < 8) {
        uint8_t value = mem->fabric[src_fabric].output[src_port];
        mem->fabric[dst_fabric].input[dst_port] = value;
    }
}

__global__ void execute_fabrics(MemorySpace* mem) {
    // Each block is a fabric
    int fabric_id = blockIdx.x;
    if (fabric_id >= 4) return;

    FabricConfig* config = &mem->fabric[fabric_id];
    uint8_t* workspace = mem->fabric_mem[fabric_id];

    // Check if fabric is enabled
    if (!(config->control & 0x01)) return;

    // Execute based on fabric type
    switch (fabric_id) {
        case 0: fabric_graphics(config, workspace); break;
        case 1: fabric_compute(config, workspace); break;
        case 2: fabric_memory(config, workspace, mem->data); break;
        case 3: fabric_custom(config, workspace); break;
    }
}

// =============================================================================
// HOST-SIDE HELPERS
// =============================================================================

void set_route(MemorySpace* mem, int route_id,
               int src_fabric, int src_port,
               int dst_fabric, int dst_port) {
    mem->routes[route_id].src = ((src_fabric & 0xF) << 4) | (src_port & 0xF);
    mem->routes[route_id].dst = ((dst_fabric & 0xF) << 4) | (dst_port & 0xF);
}

void clear_routes(MemorySpace* mem) {
    for (int i = 0; i < 128; i++) {
        mem->routes[i].src = 0xFF;
        mem->routes[i].dst = 0xFF;
    }
}

void enable_fabric(MemorySpace* mem, int fabric_id, uint8_t mode) {
    mem->fabric[fabric_id].mode = mode;
    mem->fabric[fabric_id].control = 0x01;  // Enable bit
}

// =============================================================================
// DEMO: A simple "game" using the coordination engine
// =============================================================================

void demo_game() {
    printf("=== DEMO: Coordination Engine Game Loop ===\n\n");

    MemorySpace* h_mem = new MemorySpace();
    MemorySpace* d_mem;
    cudaMalloc(&d_mem, sizeof(MemorySpace));

    // Clear everything
    memset(h_mem, 0, sizeof(MemorySpace));
    clear_routes(h_mem);

    // Setup: A simple "bounce" demo
    // - Fabric 3 (Custom): Counter for position
    // - Fabric 1 (Compute): Bounds check
    // - Fabric 0 (Graphics): Collision detect (unused here, just for show)

    // Enable fabrics
    enable_fabric(h_mem, 1, 1);  // Compute: Compare mode
    enable_fabric(h_mem, 3, 1);  // Custom: Counter mode

    // Route: Counter output → Compute input for bounds check
    set_route(h_mem, 0, 3, 0, 1, 0);  // Counter.out[0] → Compute.in[0]

    // Set bounds in compute fabric
    h_mem->fabric[1].param[0] = 0;  // Will be used for compare
    h_mem->fabric[1].input[1] = 200;  // Compare against 200

    // Initialize counter in custom fabric workspace
    h_mem->fabric_mem[3][0] = 100;  // Start position

    // Copy to device
    cudaMemcpy(d_mem, h_mem, sizeof(MemorySpace), cudaMemcpyHostToDevice);

    printf("Simulating 10 frames of game loop:\n\n");

    for (int frame = 0; frame < 10; frame++) {
        // Set counter direction (alternate up/down)
        h_mem->fabric[3].input[0] = (frame % 2 == 0) ? 0x01 : 0x02;  // Up or down
        cudaMemcpy(&d_mem->fabric[3].input[0], &h_mem->fabric[3].input[0], 1, cudaMemcpyHostToDevice);

        // Run one coordination cycle
        route_data<<<1, 128>>>(d_mem);
        execute_fabrics<<<4, 1>>>(d_mem);
        cudaDeviceSynchronize();

        // Read results
        cudaMemcpy(h_mem, d_mem, sizeof(MemorySpace), cudaMemcpyDeviceToHost);

        uint8_t position = h_mem->fabric[3].output[0];
        uint8_t compare_flags = h_mem->fabric[1].output[0];

        printf("Frame %2d: Position = %3d, Flags = 0x%02X", frame, position, compare_flags);
        if (compare_flags & 0x01) printf(" [EQUAL]");
        if (compare_flags & 0x02) printf(" [LESS]");
        if (compare_flags & 0x04) printf(" [GREATER]");
        printf("\n");
    }

    printf("\n");

    delete h_mem;
    cudaFree(d_mem);
}

// =============================================================================
// DEMO: Data transformation pipeline
// =============================================================================

void demo_pipeline() {
    printf("=== DEMO: Data Pipeline ===\n\n");

    MemorySpace* h_mem = new MemorySpace();
    MemorySpace* d_mem;
    cudaMalloc(&d_mem, sizeof(MemorySpace));

    memset(h_mem, 0, sizeof(MemorySpace));
    clear_routes(h_mem);

    // Setup a pipeline:
    // Input → Compute(XOR) → Compute(ADD) → Output
    //
    // Fabric 1: XOR with mask
    // Fabric 1 also does ADD (we'll run two cycles with mode change)
    //
    // Actually, let's use two compute operations in sequence

    enable_fabric(h_mem, 1, 2);  // Compute: Bitwise mode (XOR)
    h_mem->fabric[1].param[0] = 2;  // Operation: XOR

    // Input: byte 0x55
    h_mem->fabric[1].input[0] = 0x55;
    h_mem->fabric[1].input[1] = 0xAA;  // XOR mask

    cudaMemcpy(d_mem, h_mem, sizeof(MemorySpace), cudaMemcpyHostToDevice);

    // Stage 1: XOR
    execute_fabrics<<<4, 1>>>(d_mem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_mem, d_mem, sizeof(MemorySpace), cudaMemcpyDeviceToHost);
    printf("Stage 1 (XOR 0x55 ^ 0xAA): 0x%02X\n", h_mem->fabric[1].output[0]);

    // Stage 2: Add 0x10
    h_mem->fabric[1].mode = 0;  // Arithmetic mode
    h_mem->fabric[1].param[0] = 0;  // ADD operation
    h_mem->fabric[1].input[0] = h_mem->fabric[1].output[0];  // Result from stage 1
    h_mem->fabric[1].input[1] = 0x10;  // Add 0x10

    cudaMemcpy(d_mem, h_mem, sizeof(MemorySpace), cudaMemcpyHostToDevice);

    execute_fabrics<<<4, 1>>>(d_mem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_mem, d_mem, sizeof(MemorySpace), cudaMemcpyDeviceToHost);
    printf("Stage 2 (+ 0x10): 0x%02X\n", h_mem->fabric[1].output[0]);

    printf("\nExpected: 0x55 ^ 0xAA = 0xFF, 0xFF + 0x10 = 0x0F (overflow)\n");

    delete h_mem;
    cudaFree(d_mem);
}

// =============================================================================
// BENCHMARK
// =============================================================================

void benchmark() {
    printf("=== BENCHMARK: Coordination Cycles ===\n\n");

    MemorySpace* d_mem;
    cudaMalloc(&d_mem, sizeof(MemorySpace));

    MemorySpace* h_mem = new MemorySpace();
    memset(h_mem, 0, sizeof(MemorySpace));

    // Setup: All fabrics active, full routing table
    for (int i = 0; i < 4; i++) {
        h_mem->fabric[i].control = 0x01;
        h_mem->fabric[i].mode = 0;
    }
    for (int i = 0; i < 64; i++) {
        set_route(h_mem, i, i % 4, i % 8, (i + 1) % 4, (i + 1) % 8);
    }

    cudaMemcpy(d_mem, h_mem, sizeof(MemorySpace), cudaMemcpyHostToDevice);

    // Warm up
    for (int i = 0; i < 100; i++) {
        route_data<<<1, 128>>>(d_mem);
        execute_fabrics<<<4, 1>>>(d_mem);
    }
    cudaDeviceSynchronize();

    // Benchmark
    int iterations = 100000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        route_data<<<1, 128>>>(d_mem);
        execute_fabrics<<<4, 1>>>(d_mem);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double cycles_per_sec = iterations / (ms / 1000.0);
    double routes_per_sec = cycles_per_sec * 64;  // 64 active routes

    printf("Coordination cycles: %d\n", iterations);
    printf("Time: %.2f ms\n", ms);
    printf("Cycles/sec: %.2f M\n", cycles_per_sec / 1e6);
    printf("Routes/sec: %.2f M\n", routes_per_sec / 1e6);
    printf("\n");

    printf("This is NOT about raw ops/sec.\n");
    printf("It's about coordination: routing data between specialized fabrics.\n");
    printf("\n");

    delete h_mem;
    cudaFree(d_mem);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("6502 COORDINATION ENGINE\n");
    printf("=======================================================================\n\n");

    printf("The 6502 wasn't powerful because of its ALU.\n");
    printf("It was powerful because it coordinated specialized chips.\n");
    printf("\n");
    printf("This engine implements that pattern:\n");
    printf("  - Routing table (like zero page)\n");
    printf("  - Specialized fabrics (like VIC, SID, CIA)\n");
    printf("  - Memory-mapped configuration\n");
    printf("  - Simple coordinator, powerful peripherals\n\n");

    demo_game();
    printf("\n");

    demo_pipeline();
    printf("\n");

    benchmark();

    printf("=======================================================================\n");
    printf("THE INSIGHT\n");
    printf("=======================================================================\n\n");

    printf("The 6502 pattern is:\n");
    printf("  1. Small coordinator maintains routing tables\n");
    printf("  2. Specialized fabrics do the real work\n");
    printf("  3. Data flows between fabrics via routes\n");
    printf("  4. Configuration is just memory writes\n\n");

    printf("This is Hollywood Squares at the hardware level.\n");
    printf("The coordinator assigns. The fabrics compute.\n");
    printf("The routing IS the program.\n\n");

    return 0;
}
