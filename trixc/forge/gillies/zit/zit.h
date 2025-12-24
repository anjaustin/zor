/*
 * ZIT - Zero-Instruction Topology
 *
 * Minimal interface to homeo-adaptive topological learning.
 *
 * The fabric learns its own connectivity through resistance.
 * No gradients. No loss function. No supervision.
 * Just: resistant nodes try new neighbors.
 *
 * Usage:
 *     zit_fabric_t* f = zit_create(8);  // 8^3 = 512 nodes
 *     zit_run(f, 0);                    // run to convergence
 *     zit_print(f);                     // see results
 *     zit_destroy(f);
 *
 * For the theory: see papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md
 */

#ifndef ZIT_H
#define ZIT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque fabric handle */
typedef struct zit_fabric zit_fabric_t;

/* === Lifecycle === */

/*
 * Create a fabric of dim^3 nodes in a 3D torus topology.
 * Recommended: dim=8 (512 nodes) for demos.
 * Returns NULL on allocation failure.
 */
zit_fabric_t* zit_create(int dim);

/*
 * Destroy fabric and free all resources.
 */
void zit_destroy(zit_fabric_t* f);

/* === Seeding === */

/*
 * Seed the random state for reproducibility.
 * Default: time-based seeding.
 * The "Second Star Constant" (1122911624) reproduces paper results.
 */
void zit_seed(zit_fabric_t* f, uint32_t seed);

/* === Execution === */

/*
 * Run one cycle (6 phases of comparison + plasticity).
 * Returns number of resonant nodes after this cycle.
 */
int zit_step(zit_fabric_t* f);

/*
 * Run until converged (100% resonance) or max_cycles reached.
 * Pass 0 for max_cycles to run indefinitely until convergence.
 * Returns number of cycles taken.
 */
int zit_run(zit_fabric_t* f, int max_cycles);

/* === Observation === */

/* Number of nodes that are resonant (unchanged this cycle) */
int zit_resonant(zit_fabric_t* f);

/* Total number of nodes in the fabric */
int zit_total(zit_fabric_t* f);

/* Total rewiring attempts so far */
int zit_rewires(zit_fabric_t* f);

/* True if 100% resonant */
bool zit_converged(zit_fabric_t* f);

/* Current cycle count */
int zit_cycle(zit_fabric_t* f);

/* === Output === */

/*
 * Print summary to stdout.
 */
void zit_print(zit_fabric_t* f);

/*
 * Print progress line (for use in loops).
 */
void zit_print_progress(zit_fabric_t* f);

/*
 * Export topology to JSON file.
 * Each node's neighbor indices are written.
 * Returns 0 on success, -1 on failure.
 */
int zit_export(zit_fabric_t* f, const char* path);

/* === Visualization Support === */

/*
 * Get fabric dimension (dim^3 = total nodes).
 */
int zit_dim(zit_fabric_t* f);

/*
 * Get node state (value 0-255) for visualization.
 */
uint8_t zit_node_state(zit_fabric_t* f, int node_id);

/*
 * Get node resistance level (0-255) for visualization.
 */
uint8_t zit_node_resistance(zit_fabric_t* f, int node_id);

/*
 * Check if node is currently resonant.
 */
bool zit_node_resonant(zit_fabric_t* f, int node_id);

/*
 * Check if node is currently rewiring.
 */
bool zit_node_rewiring(zit_fabric_t* f, int node_id);

/*
 * Get node's neighbor at given direction (0-5).
 * Directions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
 */
int zit_node_neighbor(zit_fabric_t* f, int node_id, int direction);

/*
 * Get 3D coordinates for a node (for visualization).
 * Returns x, y, z in the range [0, dim-1].
 */
void zit_node_coords(zit_fabric_t* f, int node_id, int* x, int* y, int* z);

/*
 * Step callback type for real-time visualization.
 * Called after each cycle with fabric pointer and user data.
 */
typedef void (*zit_step_callback_t)(zit_fabric_t* f, void* user_data);

/*
 * Set callback to be called after each step.
 * Pass NULL to disable.
 */
void zit_set_step_callback(zit_fabric_t* f, zit_step_callback_t cb, void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* ZIT_H */
