/*
 * GILLIES Core Implementation
 *
 * Context management and graph execution.
 * Uses CUDA with unified memory on Jetson Thor.
 *
 * (c) 2025 TriX Project
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/gillies.h"

/* ============================================================================
 * CUDA ERROR CHECKING
 * ============================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ============================================================================
 * EXECUTION KERNEL
 *
 * Each thread executes one invocation.
 * Reads from input ports, applies shape, writes to output port.
 * ============================================================================ */

__global__ void gillies_execute_kernel(
    float* ports,
    const gillies_invocation_t* invocations,
    const uint32_t* execution_order,
    uint32_t num_invocations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_invocations) return;

    // Get invocation in execution order
    uint32_t inv_idx = execution_order[idx];
    gillies_invocation_t inv = invocations[inv_idx];

    // Read inputs
    float a = ports[inv.inputs[0]];
    float b = (inv.num_inputs > 1) ? ports[inv.inputs[1]] : 0.0f;

    // Execute shape
    float result = gillies_dispatch_f32(inv.shape_id, a, b);

    // Write output
    ports[inv.outputs[0]] = result;
}

/* ============================================================================
 * CONTEXT MANAGEMENT
 * ============================================================================ */

gillies_context_t* gillies_create(void) {
    gillies_context_t* ctx;

    // Allocate in unified memory (accessible by both CPU and GPU)
    CUDA_CHECK(cudaMallocManaged(&ctx, sizeof(gillies_context_t)));

    // Initialize
    memset(ctx, 0, sizeof(gillies_context_t));
    ctx->tracing_enabled = false;

    return ctx;
}

void gillies_destroy(gillies_context_t* ctx) {
    if (ctx) {
        CUDA_CHECK(cudaFree(ctx));
    }
}

void gillies_reset(gillies_context_t* ctx) {
    ctx->num_invocations = 0;
    ctx->execution_count = 0;
    ctx->num_events = 0;
    memset(ctx->executed, 0, sizeof(ctx->executed));
    memset(ctx->execution_order, 0, sizeof(ctx->execution_order));
}

void gillies_clear_ports(gillies_context_t* ctx) {
    memset(&ctx->ports, 0, sizeof(gillies_ports_t));
}

/* ============================================================================
 * PORT ACCESS
 * ============================================================================ */

void gillies_set_port(gillies_context_t* ctx, uint16_t port, float value) {
    if (port < GILLIES_MAX_PORTS) {
        ctx->ports.data[port] = value;
        ctx->ports.valid[port] = 1;
    }
}

float gillies_get_port(gillies_context_t* ctx, uint16_t port) {
    if (port < GILLIES_MAX_PORTS) {
        return ctx->ports.data[port];
    }
    return 0.0f;
}

bool gillies_port_valid(gillies_context_t* ctx, uint16_t port) {
    if (port < GILLIES_MAX_PORTS) {
        return ctx->ports.valid[port] != 0;
    }
    return false;
}

/* ============================================================================
 * INVOCATION BUILDING
 * ============================================================================ */

int gillies_invoke(
    gillies_context_t* ctx,
    gillies_shape_t shape,
    uint16_t in0,
    uint16_t in1,
    uint16_t out0
) {
    if (ctx->num_invocations >= GILLIES_MAX_INVOCATIONS) {
        fprintf(stderr, "GILLIES: Maximum invocations reached\n");
        return -1;
    }

    uint32_t idx = ctx->num_invocations++;
    gillies_invocation_t* inv = &ctx->invocations[idx];

    inv->shape_id = (uint8_t)shape;
    inv->inputs[0] = in0;
    inv->inputs[1] = in1;
    inv->outputs[0] = out0;
    inv->flags = 0;

    // Determine number of inputs based on shape
    switch (shape) {
        case GILLIES_SHAPE_NOT:
        case GILLIES_SHAPE_IDENTITY:
            inv->num_inputs = 1;
            break;
        default:
            inv->num_inputs = 2;
            break;
    }
    inv->num_outputs = 1;

    return (int)idx;
}

/* ============================================================================
 * DEPENDENCY RESOLUTION
 *
 * Simple topological sort: invocations that read from ports written by
 * earlier invocations must execute after them.
 *
 * For prototype: assume invocations are added in valid order.
 * TODO: Full topological sort for arbitrary graphs.
 * ============================================================================ */

static void gillies_resolve_dependencies(gillies_context_t* ctx) {
    // For now: execute in order added
    // This works if user adds invocations in dependency order
    for (uint32_t i = 0; i < ctx->num_invocations; i++) {
        ctx->execution_order[i] = i;
    }
    ctx->execution_count = ctx->num_invocations;
}

/* ============================================================================
 * GPU EXECUTION
 * ============================================================================ */

void gillies_execute(gillies_context_t* ctx) {
    if (ctx->num_invocations == 0) return;

    // Resolve dependencies
    gillies_resolve_dependencies(ctx);

    // For now: execute sequentially to respect dependencies
    // Each invocation may depend on previous outputs
    // TODO: Parallel execution of independent invocations

    for (uint32_t i = 0; i < ctx->execution_count; i++) {
        // Launch kernel for single invocation
        // (Inefficient but correct - optimize later)
        gillies_execute_kernel<<<1, 1>>>(
            ctx->ports.data,
            ctx->invocations,
            &ctx->execution_order[i],
            1
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Record event if tracing
        if (ctx->tracing_enabled && ctx->num_events < GILLIES_MAX_EVENTS) {
            uint32_t inv_idx = ctx->execution_order[i];
            gillies_invocation_t* inv = &ctx->invocations[inv_idx];

            gillies_event_t* evt = &ctx->events[ctx->num_events++];
            evt->invocation_id = inv_idx;
            evt->shape_id = inv->shape_id;
            evt->input_a = ctx->ports.data[inv->inputs[0]];
            evt->input_b = (inv->num_inputs > 1) ? ctx->ports.data[inv->inputs[1]] : 0.0f;
            evt->output = ctx->ports.data[inv->outputs[0]];

            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            evt->timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        }

        // Update statistics
        ctx->total_invocations++;
        uint32_t inv_idx = ctx->execution_order[i];
        ctx->shape_counts[ctx->invocations[inv_idx].shape_id]++;

        // Mark as executed
        ctx->executed[inv_idx] = 1;

        // Mark output port as valid
        ctx->ports.valid[ctx->invocations[inv_idx].outputs[0]] = 1;
    }
}

/* ============================================================================
 * CPU EXECUTION (Fallback / Comparison)
 * ============================================================================ */

void gillies_execute_cpu(gillies_context_t* ctx) {
    if (ctx->num_invocations == 0) return;

    // Resolve dependencies
    gillies_resolve_dependencies(ctx);

    // Execute on CPU
    for (uint32_t i = 0; i < ctx->execution_count; i++) {
        uint32_t inv_idx = ctx->execution_order[i];
        gillies_invocation_t* inv = &ctx->invocations[inv_idx];

        // Read inputs
        float a = ctx->ports.data[inv->inputs[0]];
        float b = (inv->num_inputs > 1) ? ctx->ports.data[inv->inputs[1]] : 0.0f;

        // Execute shape
        float result = gillies_dispatch_f32(inv->shape_id, a, b);

        // Write output
        ctx->ports.data[inv->outputs[0]] = result;
        ctx->ports.valid[inv->outputs[0]] = 1;

        // Update statistics
        ctx->total_invocations++;
        ctx->shape_counts[inv->shape_id]++;
        ctx->executed[inv_idx] = 1;
    }
}

/* ============================================================================
 * OBSERVABILITY
 * ============================================================================ */

void gillies_set_tracing(gillies_context_t* ctx, bool enabled) {
    ctx->tracing_enabled = enabled;
}

uint32_t gillies_event_count(gillies_context_t* ctx) {
    return ctx->num_events;
}

const gillies_event_t* gillies_get_event(gillies_context_t* ctx, uint32_t index) {
    if (index < ctx->num_events) {
        return &ctx->events[index];
    }
    return NULL;
}

/* ============================================================================
 * DEBUGGING / STATISTICS
 * ============================================================================ */

void gillies_print_stats(gillies_context_t* ctx) {
    printf("GILLIES Statistics:\n");
    printf("  Total invocations: %lu\n", (unsigned long)ctx->total_invocations);
    printf("  Shape usage:\n");
    for (int i = 0; i < GILLIES_NUM_SHAPES; i++) {
        if (ctx->shape_counts[i] > 0) {
            printf("    %s: %lu\n", GILLIES_SHAPE_NAMES[i],
                   (unsigned long)ctx->shape_counts[i]);
        }
    }
}

void gillies_print_graph(gillies_context_t* ctx) {
    printf("GILLIES Invocation Graph:\n");
    for (uint32_t i = 0; i < ctx->num_invocations; i++) {
        gillies_invocation_t* inv = &ctx->invocations[i];
        printf("  [%u] %s(port[%u]",
               i, GILLIES_SHAPE_NAMES[inv->shape_id], inv->inputs[0]);
        if (inv->num_inputs > 1) {
            printf(", port[%u]", inv->inputs[1]);
        }
        printf(") -> port[%u]\n", inv->outputs[0]);
    }
}
