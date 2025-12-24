/*
 * ZIT - Zero-Instruction Topology
 *
 * Implementation of homeo-adaptive topological learning.
 *
 * The core insight: resistant nodes rewire. Topology learns.
 */

#include "zit.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* === Internal Node Structure === */

typedef struct {
    uint8_t state;              /* Current value (0-255) */
    uint8_t resistance;         /* Resistance counter */
    uint8_t resonant;           /* Was resonant this cycle? */
    uint8_t rewire_dir;         /* Current rewire direction (0-5) */
    uint32_t neighbors[6];      /* Neighbor indices */
    uint16_t lfsr;              /* Random state (16-bit LFSR) */
    uint8_t rewiring;           /* Currently evaluating new neighbor? */
    uint8_t eval_cycles;        /* Cycles spent in evaluation */
    uint32_t old_neighbor;      /* Backup for revert */
    uint8_t pre_resistance;     /* Resistance before rewire attempt */
    uint8_t has_participated;   /* Participated in a swap this cycle */
} zit_node_t;

struct zit_fabric {
    zit_node_t* nodes;
    uint8_t* state_snapshot;    /* Frozen state for phase reads */
    int dim;
    int total;
    int cycle;
    int rewire_count;
    int resonant_count;
    uint32_t seed;
    zit_step_callback_t step_callback;
    void* callback_user_data;
};

/* === Constants === */

#define RESISTANCE_THRESHOLD 8
#define EVAL_PERIOD 8

/* === Helper: 3D indexing with toroidal wrap === */

static inline int idx(int dim, int x, int y, int z) {
    x = (x + dim) % dim;
    y = (y + dim) % dim;
    z = (z + dim) % dim;
    return x + y * dim + z * dim * dim;
}

/* === Helper: 16-bit LFSR (matches CUDA version) === */

static inline uint16_t lfsr_step(uint16_t s) {
    return (s >> 1) | (((s ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1) << 15);
}

/* === Lifecycle === */

zit_fabric_t* zit_create(int dim) {
    zit_fabric_t* f = calloc(1, sizeof(zit_fabric_t));
    if (!f) return NULL;

    f->dim = dim;
    f->total = dim * dim * dim;
    f->nodes = calloc(f->total, sizeof(zit_node_t));
    f->state_snapshot = calloc(f->total, sizeof(uint8_t));
    if (!f->nodes || !f->state_snapshot) {
        free(f->nodes);
        free(f->state_snapshot);
        free(f);
        return NULL;
    }

    /* Initialize callback */
    f->step_callback = NULL;
    f->callback_user_data = NULL;

    /* Initialize with time-based seed */
    zit_seed(f, (uint32_t)time(NULL));

    return f;
}

void zit_seed(zit_fabric_t* f, uint32_t seed) {
    f->seed = seed;
    f->cycle = 0;
    f->rewire_count = 0;
    f->resonant_count = 0;

    int dim = f->dim;
    for (int i = 0; i < f->total; i++) {
        zit_node_t* n = &f->nodes[i];

        /* Initial state: deterministic from index */
        n->state = (uint8_t)((i * 3) & 0xFF);

        /* LFSR init (matches CUDA: 0xBEEF ^ (idx * 0x1337) ^ (idx << 8)) */
        n->lfsr = (uint16_t)(0xBEEF ^ (i * 0x1337) ^ (i << 8));

        /* Initial topology: 6-connected 3D torus */
        int z = i / (dim * dim);
        int y = (i / dim) % dim;
        int x = i % dim;

        n->neighbors[0] = idx(dim, x+1, y, z);  /* +X */
        n->neighbors[1] = idx(dim, x-1, y, z);  /* -X */
        n->neighbors[2] = idx(dim, x, y+1, z);  /* +Y */
        n->neighbors[3] = idx(dim, x, y-1, z);  /* -Y */
        n->neighbors[4] = idx(dim, x, y, z+1);  /* +Z */
        n->neighbors[5] = idx(dim, x, y, z-1);  /* -Z */

        /* Clear state */
        n->resistance = 0;
        n->resonant = 1;
        n->rewiring = 0;
        n->rewire_dir = 0;
        n->eval_cycles = 0;
        n->old_neighbor = 0;
        n->pre_resistance = 0;
        n->has_participated = 0;

        /* Initialize snapshot */
        f->state_snapshot[i] = n->state;
    }
}

void zit_destroy(zit_fabric_t* f) {
    if (f) {
        free(f->nodes);
        free(f->state_snapshot);
        free(f);
    }
}

/* === Sync states to snapshot === */

static void sync_states(zit_fabric_t* f) {
    for (int i = 0; i < f->total; i++) {
        f->state_snapshot[i] = f->nodes[i].state;
    }
}

/* === The Frozen Shape: Comparator ===
 *
 * This is the fixed operation. It never changes.
 * Each node reads its neighbor from the frozen snapshot,
 * and updates only its own state.
 *
 * Phase 0 (+X): if my_val > nb_val: I take nb_val
 * Phase 1 (-X): if nb_val > my_val: I take nb_val
 * Phase 2 (+Y): if my_val > nb_val: I take nb_val
 * Phase 3 (-Y): if nb_val > my_val: I take nb_val
 * Phase 4 (+Z): if my_val > nb_val: I take nb_val
 * Phase 5 (-Z): if nb_val > my_val: I take nb_val
 */

static void do_phase(zit_fabric_t* f, int phase) {
    int positive = (phase % 2 == 0);

    for (int i = 0; i < f->total; i++) {
        zit_node_t* n = &f->nodes[i];

        /* Read neighbor value from frozen snapshot */
        int nb_idx = n->neighbors[phase];
        uint8_t nb_val = f->state_snapshot[nb_idx];
        uint8_t my_val = n->state;

        /* Comparator logic */
        int should_take;
        if (positive) {
            should_take = (my_val > nb_val);  /* I'm too big, take smaller */
        } else {
            should_take = (nb_val > my_val);  /* Neighbor bigger, take it */
        }

        if (should_take) {
            n->state = nb_val;
            n->resonant = 0;
        }

        n->has_participated = 1;
    }
}

/* === Plasticity: Resistance and Rewiring ===
 *
 * This is the learning algorithm:
 *
 * if (!rewiring):
 *     if (!resonant): resistance++
 *     else: resistance decay
 *     if (resistance >= threshold): start rewire
 * else:
 *     wait EVAL_PERIOD cycles
 *     if (more resistance): revert
 *     else: keep new neighbor
 */

static void do_plasticity(zit_fabric_t* f) {
    for (int i = 0; i < f->total; i++) {
        zit_node_t* n = &f->nodes[i];

        uint8_t res = n->resonant;
        uint8_t resist = n->resistance;
        uint16_t rnd = n->lfsr;
        uint8_t rewiring = n->rewiring;
        uint8_t rewire_dir = n->rewire_dir;
        uint8_t has_part = n->has_participated;

        if (!rewiring) {
            /* Update resistance */
            if (!res && has_part) {
                if (resist < 255) resist++;
            } else if (res) {
                resist >>= 1;  /* Decay */
            }

            /* Check if should start rewiring */
            if (resist >= RESISTANCE_THRESHOLD) {
                rewiring = 1;
                rnd = lfsr_step(rnd);

                n->old_neighbor = n->neighbors[rewire_dir];
                n->pre_resistance = resist;
                n->eval_cycles = 0;

                /* Pick random new neighbor */
                int new_nb = rnd % f->total;
                if (new_nb == i) new_nb = (new_nb + 1) % f->total;
                n->neighbors[rewire_dir] = new_nb;

                resist = 0;
                f->rewire_count++;
            }
        } else {
            /* In evaluation period */
            n->eval_cycles++;
            if (!res && resist < 255) resist++;

            if (n->eval_cycles >= EVAL_PERIOD) {
                /* Evaluation complete */
                if (resist >= n->pre_resistance) {
                    /* Revert - new neighbor didn't help */
                    n->neighbors[rewire_dir] = n->old_neighbor;
                }
                rewire_dir = (rewire_dir + 1) % 6;
                rewiring = 0;
                resist = 0;
            }
        }

        n->resistance = resist;
        n->rewiring = rewiring;
        n->rewire_dir = rewire_dir;
        n->lfsr = lfsr_step(rnd);
    }
}

/* === Execution === */

int zit_step(zit_fabric_t* f) {
    /* Reset resonance and participation flags */
    for (int i = 0; i < f->total; i++) {
        f->nodes[i].resonant = 1;
        f->nodes[i].has_participated = 0;
    }

    /* 6 sequential phases of comparison (the frozen shape) */
    for (int phase = 0; phase < 6; phase++) {
        do_phase(f, phase);
        /* Sync states after each phase */
        sync_states(f);
    }

    /* Topological plasticity (the learning) */
    do_plasticity(f);

    /* Count resonant nodes */
    f->resonant_count = 0;
    for (int i = 0; i < f->total; i++) {
        if (f->nodes[i].resonant)
            f->resonant_count++;
    }

    f->cycle++;

    /* Call step callback if set */
    if (f->step_callback) {
        f->step_callback(f, f->callback_user_data);
    }

    return f->resonant_count;
}

int zit_run(zit_fabric_t* f, int max_cycles) {
    if (max_cycles <= 0) max_cycles = 10000;

    while (f->cycle < max_cycles && !zit_converged(f)) {
        zit_step(f);
    }
    return f->cycle;
}

/* === Observation === */

int zit_resonant(zit_fabric_t* f) {
    return f->resonant_count;
}

int zit_total(zit_fabric_t* f) {
    return f->total;
}

int zit_rewires(zit_fabric_t* f) {
    return f->rewire_count;
}

int zit_cycle(zit_fabric_t* f) {
    return f->cycle;
}

bool zit_converged(zit_fabric_t* f) {
    return f->resonant_count == f->total;
}

/* === Output === */

void zit_print(zit_fabric_t* f) {
    printf("ZIT Fabric: %dx%dx%d = %d nodes\n",
           f->dim, f->dim, f->dim, f->total);
    printf("Cycle: %d\n", f->cycle);
    printf("Resonant: %d/%d (%.1f%%)\n",
           f->resonant_count, f->total,
           100.0 * f->resonant_count / f->total);
    printf("Rewires: %d\n", f->rewire_count);
    printf("Status: %s\n", zit_converged(f) ? "CONVERGED" : "running");
}

void zit_print_progress(zit_fabric_t* f) {
    printf("Cycle %4d: %d/%d resonant, %d rewires\n",
           f->cycle, f->resonant_count, f->total, f->rewire_count);
}

int zit_export(zit_fabric_t* f, const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) return -1;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"dim\": %d,\n", f->dim);
    fprintf(fp, "  \"total\": %d,\n", f->total);
    fprintf(fp, "  \"cycle\": %d,\n", f->cycle);
    fprintf(fp, "  \"rewires\": %d,\n", f->rewire_count);
    fprintf(fp, "  \"nodes\": [\n");

    for (int i = 0; i < f->total; i++) {
        zit_node_t* n = &f->nodes[i];
        fprintf(fp, "    {\"neighbors\": [%u,%u,%u,%u,%u,%u]}%s\n",
                n->neighbors[0], n->neighbors[1],
                n->neighbors[2], n->neighbors[3],
                n->neighbors[4], n->neighbors[5],
                (i < f->total - 1) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}

/* === Visualization Support === */

int zit_dim(zit_fabric_t* f) {
    return f->dim;
}

uint8_t zit_node_state(zit_fabric_t* f, int node_id) {
    if (node_id < 0 || node_id >= f->total) return 0;
    return f->nodes[node_id].state;
}

uint8_t zit_node_resistance(zit_fabric_t* f, int node_id) {
    if (node_id < 0 || node_id >= f->total) return 0;
    return f->nodes[node_id].resistance;
}

bool zit_node_resonant(zit_fabric_t* f, int node_id) {
    if (node_id < 0 || node_id >= f->total) return false;
    return f->nodes[node_id].resonant != 0;
}

bool zit_node_rewiring(zit_fabric_t* f, int node_id) {
    if (node_id < 0 || node_id >= f->total) return false;
    return f->nodes[node_id].rewiring != 0;
}

int zit_node_neighbor(zit_fabric_t* f, int node_id, int direction) {
    if (node_id < 0 || node_id >= f->total) return -1;
    if (direction < 0 || direction > 5) return -1;
    return (int)f->nodes[node_id].neighbors[direction];
}

void zit_node_coords(zit_fabric_t* f, int node_id, int* x, int* y, int* z) {
    if (node_id < 0 || node_id >= f->total) {
        if (x) *x = 0;
        if (y) *y = 0;
        if (z) *z = 0;
        return;
    }
    int dim = f->dim;
    if (z) *z = node_id / (dim * dim);
    if (y) *y = (node_id / dim) % dim;
    if (x) *x = node_id % dim;
}

void zit_set_step_callback(zit_fabric_t* f, zit_step_callback_t cb, void* user_data) {
    f->step_callback = cb;
    f->callback_user_data = user_data;
}
