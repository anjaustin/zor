/*
 * GILLIES - Geometric Instruction Language Layer In Every System
 *
 * The Shape Layer: where all computation meets, and all paradigms become one.
 *
 * A GILLIES program is a graph of shape invocations.
 * Each invocation specifies:
 *   - Which shape (frozen polynomial)
 *   - Where to read inputs (port indices)
 *   - Where to write outputs (port indices)
 *
 * The executor resolves dependencies and dispatches to available substrates
 * (CPU, GPU, FPGA, neural accelerators, future unknown).
 *
 * "The routing IS the program. The shapes ARE the instruction set."
 *
 * (c) 2025 TriX Project
 */

#ifndef GILLIES_H
#define GILLIES_H

#include <stdint.h>
#include <stdbool.h>
#include "shapes_device.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define GILLIES_MAX_PORTS 256
#define GILLIES_MAX_INVOCATIONS 1024
#define GILLIES_MAX_EVENTS 4096

/* ============================================================================
 * INVOCATION - The universal primitive
 *
 * "The shape invocation is to GILLIES what the instruction is to CPUs.
 *  But unlike instructions, invocations are substrate-agnostic."
 * ============================================================================ */

typedef struct {
    uint8_t shape_id;       // Which shape (from gillies_shape_t)
    uint8_t num_inputs;     // Number of input ports used (1-2)
    uint8_t num_outputs;    // Number of output ports used (1)
    uint8_t flags;          // Reserved for precision/substrate hints

    uint16_t inputs[2];     // Port indices for inputs
    uint16_t outputs[1];    // Port indices for outputs
} gillies_invocation_t;

/* ============================================================================
 * PORT SPACE - Unified memory for all data
 *
 * Ports are named locations. Invocations read from and write to ports.
 * On Thor's unified memory, CPU and GPU share the same port space.
 * ============================================================================ */

typedef struct {
    float data[GILLIES_MAX_PORTS];      // The actual values
    uint8_t valid[GILLIES_MAX_PORTS];   // Which ports have been written
} gillies_ports_t;

/* ============================================================================
 * EVENT - Observability (Glassbox)
 *
 * Every invocation can be traced. No black boxes.
 * ============================================================================ */

typedef struct {
    uint32_t invocation_id;     // Which invocation
    uint8_t shape_id;           // Which shape was executed
    float input_a;              // Input value A
    float input_b;              // Input value B
    float output;               // Output value
    uint64_t timestamp_ns;      // When it executed (nanoseconds)
} gillies_event_t;

/* ============================================================================
 * CONTEXT - Holds the complete execution state
 * ============================================================================ */

typedef struct {
    // Port space (unified memory)
    gillies_ports_t ports;

    // Invocation graph
    gillies_invocation_t invocations[GILLIES_MAX_INVOCATIONS];
    uint32_t num_invocations;

    // Execution state
    uint8_t executed[GILLIES_MAX_INVOCATIONS];
    uint32_t execution_order[GILLIES_MAX_INVOCATIONS];
    uint32_t execution_count;

    // Observability
    gillies_event_t events[GILLIES_MAX_EVENTS];
    uint32_t num_events;
    bool tracing_enabled;

    // Statistics
    uint64_t total_invocations;
    uint64_t shape_counts[GILLIES_NUM_SHAPES];
} gillies_context_t;

/* ============================================================================
 * API - The interface to GILLIES
 * ============================================================================ */

/**
 * Create a new GILLIES context.
 * Allocates unified memory accessible by both CPU and GPU.
 */
gillies_context_t* gillies_create(void);

/**
 * Destroy a GILLIES context and free all resources.
 */
void gillies_destroy(gillies_context_t* ctx);

/**
 * Reset the context for a new computation.
 * Clears invocations and execution state, keeps port data.
 */
void gillies_reset(gillies_context_t* ctx);

/**
 * Clear all ports (set to zero, mark invalid).
 */
void gillies_clear_ports(gillies_context_t* ctx);

/**
 * Set a port value.
 */
void gillies_set_port(gillies_context_t* ctx, uint16_t port, float value);

/**
 * Get a port value.
 */
float gillies_get_port(gillies_context_t* ctx, uint16_t port);

/**
 * Check if a port has been written.
 */
bool gillies_port_valid(gillies_context_t* ctx, uint16_t port);

/**
 * Add a shape invocation to the graph.
 *
 * @param ctx      The context
 * @param shape    Which shape to execute
 * @param in0      First input port
 * @param in1      Second input port (ignored for unary shapes)
 * @param out0     Output port
 * @return         Invocation index, or -1 on error
 */
int gillies_invoke(
    gillies_context_t* ctx,
    gillies_shape_t shape,
    uint16_t in0,
    uint16_t in1,
    uint16_t out0
);

/**
 * Execute all invocations in the graph.
 *
 * Resolves dependencies (topological sort) and dispatches to GPU.
 * On Thor, this uses CUDA with unified memory.
 */
void gillies_execute(gillies_context_t* ctx);

/**
 * Execute on CPU only (for comparison/fallback).
 */
void gillies_execute_cpu(gillies_context_t* ctx);

/**
 * Enable/disable execution tracing.
 */
void gillies_set_tracing(gillies_context_t* ctx, bool enabled);

/**
 * Get number of recorded events.
 */
uint32_t gillies_event_count(gillies_context_t* ctx);

/**
 * Get an event by index.
 */
const gillies_event_t* gillies_get_event(gillies_context_t* ctx, uint32_t index);

/**
 * Print execution statistics.
 */
void gillies_print_stats(gillies_context_t* ctx);

/**
 * Print the invocation graph (for debugging).
 */
void gillies_print_graph(gillies_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GILLIES_H */
