# ZIT Examples

Practical examples demonstrating ZIT usage patterns.

---

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Progress Monitoring](#progress-monitoring)
3. [Experiments and Analysis](#experiments-and-analysis)
4. [Visualization Examples](#visualization-examples)
5. [Advanced Patterns](#advanced-patterns)

---

## Basic Examples

### Example 1: Minimal Demo

The simplest possible ZIT program.

```c
/* minimal.c */
#include "zit.h"

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_run(f, 0);
    zit_print(f);
    zit_destroy(f);
    return 0;
}
```

**Build and run:**
```bash
gcc -O3 -o minimal minimal.c zit.c -lm
./minimal
```

**Expected output:**
```
ZIT Fabric: 8x8x8 = 512 nodes
Cycle: 114
Resonant: 512/512 (100.0%)
Rewires: 1340
Status: CONVERGED
```

---

### Example 2: Reproducible Results

Using the Second Star Constant for reproducibility.

```c
/* reproducible.c */
#include "zit.h"
#include <stdio.h>

#define SECOND_STAR 1122911624

int main() {
    zit_fabric_t* f = zit_create(8);

    /* Same seed = same results every time */
    zit_seed(f, SECOND_STAR);

    int cycles = zit_run(f, 0);
    int rewires = zit_rewires(f);

    printf("Cycles: %d (should be 114)\n", cycles);
    printf("Rewires: %d (should be 1340)\n", rewires);

    zit_destroy(f);
    return 0;
}
```

---

### Example 3: Different Dimensions

Comparing fabric sizes.

```c
/* dimensions.c */
#include "zit.h"
#include <stdio.h>

void test_dimension(int dim) {
    zit_fabric_t* f = zit_create(dim);
    zit_seed(f, 1122911624);

    int cycles = zit_run(f, 1000);
    int nodes = zit_total(f);
    int rewires = zit_rewires(f);
    int converged = zit_converged(f);

    printf("%dx%dx%d = %5d nodes: %3d cycles, %5d rewires %s\n",
           dim, dim, dim, nodes, cycles, rewires,
           converged ? "[CONVERGED]" : "[timeout]");

    zit_destroy(f);
}

int main() {
    printf("ZIT Dimension Comparison\n");
    printf("========================\n\n");

    test_dimension(4);   /* 64 nodes */
    test_dimension(6);   /* 216 nodes */
    test_dimension(8);   /* 512 nodes */
    test_dimension(10);  /* 1000 nodes */
    test_dimension(12);  /* 1728 nodes */
    test_dimension(16);  /* 4096 nodes */

    return 0;
}
```

**Expected output:**
```
ZIT Dimension Comparison
========================

4x4x4 =    64 nodes: 158 cycles,   169 rewires [CONVERGED]
6x6x6 =   216 nodes: 100 cycles,   515 rewires [CONVERGED]
8x8x8 =   512 nodes: 114 cycles,  1340 rewires [CONVERGED]
10x10x10 =  1000 nodes: 158 cycles,  2955 rewires [CONVERGED]
12x12x12 =  1728 nodes: 175 cycles,  5414 rewires [CONVERGED]
16x16x16 =  4096 nodes: 202 cycles, 13231 rewires [CONVERGED]
```

---

## Progress Monitoring

### Example 4: Step-by-Step Output

Watch the fabric converge.

```c
/* progress.c */
#include "zit.h"
#include <stdio.h>

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);

    printf("Cycle  Resonant  Rewires\n");
    printf("-----  --------  -------\n");

    while (!zit_converged(f)) {
        zit_step(f);

        /* Print every 25 cycles */
        if (zit_cycle(f) % 25 == 0) {
            printf("%5d   %3d/%3d    %5d\n",
                   zit_cycle(f),
                   zit_resonant(f),
                   zit_total(f),
                   zit_rewires(f));
        }
    }

    /* Final state */
    printf("%5d   %3d/%3d    %5d  *** CONVERGED ***\n",
           zit_cycle(f),
           zit_resonant(f),
           zit_total(f),
           zit_rewires(f));

    zit_destroy(f);
    return 0;
}
```

---

### Example 5: Callback-Based Monitoring

Using the callback mechanism.

```c
/* callback.c */
#include "zit.h"
#include <stdio.h>

void step_monitor(zit_fabric_t* f, void* user_data) {
    int* last_report = (int*)user_data;
    int cycle = zit_cycle(f);

    /* Report every 10 cycles */
    if (cycle - *last_report >= 10) {
        float ratio = 100.0f * zit_resonant(f) / zit_total(f);
        printf("Cycle %3d: %.1f%% resonant\n", cycle, ratio);
        *last_report = cycle;
    }
}

int main() {
    int last_report = 0;

    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);
    zit_set_step_callback(f, step_monitor, &last_report);

    zit_run(f, 0);

    printf("\n*** Converged at cycle %d ***\n", zit_cycle(f));

    zit_destroy(f);
    return 0;
}
```

---

### Example 6: Progress Bar

ASCII progress visualization.

```c
/* progress_bar.c */
#include "zit.h"
#include <stdio.h>
#include <string.h>

#define BAR_WIDTH 40

void print_bar(float ratio) {
    int filled = (int)(ratio * BAR_WIDTH);
    char bar[BAR_WIDTH + 1];

    memset(bar, '#', filled);
    memset(bar + filled, '-', BAR_WIDTH - filled);
    bar[BAR_WIDTH] = '\0';

    printf("\r[%s] %.1f%%", bar, ratio * 100);
    fflush(stdout);
}

void progress_callback(zit_fabric_t* f, void* user_data) {
    float ratio = (float)zit_resonant(f) / zit_total(f);
    print_bar(ratio);
}

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);
    zit_set_step_callback(f, progress_callback, NULL);

    printf("Running ZIT fabric...\n\n");
    zit_run(f, 0);
    printf("\n\nConverged in %d cycles!\n", zit_cycle(f));

    zit_destroy(f);
    return 0;
}
```

---

## Experiments and Analysis

### Example 7: Monte Carlo Seeds

Testing convergence across random seeds.

```c
/* monte_carlo.c */
#include "zit.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TRIALS 100

int main() {
    int min_cycles = 999999;
    int max_cycles = 0;
    long total_cycles = 0;
    long total_rewires = 0;

    srand(time(NULL));

    printf("Running %d trials...\n", NUM_TRIALS);

    for (int i = 0; i < NUM_TRIALS; i++) {
        zit_fabric_t* f = zit_create(8);
        uint32_t seed = rand();
        zit_seed(f, seed);

        int cycles = zit_run(f, 500);

        if (zit_converged(f)) {
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
            total_cycles += cycles;
            total_rewires += zit_rewires(f);
        }

        zit_destroy(f);

        if ((i + 1) % 10 == 0) {
            printf("  %d/%d complete\n", i + 1, NUM_TRIALS);
        }
    }

    printf("\nResults:\n");
    printf("  Min cycles: %d\n", min_cycles);
    printf("  Max cycles: %d\n", max_cycles);
    printf("  Avg cycles: %.1f\n", (float)total_cycles / NUM_TRIALS);
    printf("  Avg rewires: %.1f\n", (float)total_rewires / NUM_TRIALS);

    return 0;
}
```

---

### Example 8: Topology Analysis

Analyzing the learned topology.

```c
/* topology_analysis.c */
#include "zit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int manhattan_distance(int dim, int n1, int n2) {
    int x1, y1, z1, x2, y2, z2;

    z1 = n1 / (dim * dim); y1 = (n1 / dim) % dim; x1 = n1 % dim;
    z2 = n2 / (dim * dim); y2 = (n2 / dim) % dim; x2 = n2 % dim;

    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);
    int dz = abs(z1 - z2);

    /* Account for torus wrap */
    if (dx > dim/2) dx = dim - dx;
    if (dy > dim/2) dy = dim - dy;
    if (dz > dim/2) dz = dim - dz;

    return dx + dy + dz;
}

void analyze_topology(zit_fabric_t* f) {
    int dim = zit_dim(f);
    int total = zit_total(f);

    int local_edges = 0;
    int shortcut_edges = 0;
    int total_distance = 0;

    for (int i = 0; i < total; i++) {
        for (int d = 0; d < 6; d++) {
            int neighbor = zit_node_neighbor(f, i, d);
            int dist = manhattan_distance(dim, i, neighbor);

            total_distance += dist;

            if (dist == 1) {
                local_edges++;
            } else {
                shortcut_edges++;
            }
        }
    }

    printf("Topology Analysis\n");
    printf("=================\n");
    printf("Total edges: %d\n", total * 6);
    printf("Local edges (distance=1): %d (%.1f%%)\n",
           local_edges, 100.0f * local_edges / (total * 6));
    printf("Shortcut edges (distance>1): %d (%.1f%%)\n",
           shortcut_edges, 100.0f * shortcut_edges / (total * 6));
    printf("Average edge distance: %.2f\n",
           (float)total_distance / (total * 6));
}

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);

    printf("Before learning:\n\n");
    analyze_topology(f);

    zit_run(f, 0);

    printf("\n\nAfter learning (%d cycles):\n\n", zit_cycle(f));
    analyze_topology(f);

    zit_destroy(f);
    return 0;
}
```

---

### Example 9: Export and Visualize

Export topology for external visualization.

```c
/* export_topology.c */
#include "zit.h"
#include <stdio.h>

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);
    zit_run(f, 0);

    /* Export to JSON */
    if (zit_export(f, "fabric_topology.json") == 0) {
        printf("Exported to fabric_topology.json\n");
    } else {
        printf("Export failed!\n");
    }

    zit_destroy(f);
    return 0;
}
```

**Python visualization script:**

```python
# visualize_topology.py
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with open('fabric_topology.json') as f:
    data = json.load(f)

dim = data['dim']
nodes = data['nodes']
total = data['total']

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
xs, ys, zs = [], [], []
for i in range(total):
    z = i // (dim * dim)
    y = (i // dim) % dim
    x = i % dim
    xs.append(x)
    ys.append(y)
    zs.append(z)

ax.scatter(xs, ys, zs, c='blue', s=20)

# Plot edges (sample for clarity)
for i in range(0, total, 10):
    z = i // (dim * dim)
    y = (i // dim) % dim
    x = i % dim

    for nb in nodes[i]['neighbors'][:3]:  # Just first 3 directions
        nz = nb // (dim * dim)
        ny = (nb // dim) % dim
        nx = nb % dim

        ax.plot([x, nx], [y, ny], [z, nz], 'gray', alpha=0.3, linewidth=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'ZIT Topology ({dim}x{dim}x{dim})')

plt.savefig('topology.png', dpi=150)
plt.show()
```

---

## Visualization Examples

### Example 10: Node State Dump

Dump all node states for debugging.

```c
/* dump_states.c */
#include "zit.h"
#include <stdio.h>

void dump_slice(zit_fabric_t* f, int z) {
    int dim = zit_dim(f);

    printf("Z=%d slice:\n", z);
    printf("    ");
    for (int x = 0; x < dim; x++) printf("%3d", x);
    printf("\n");

    for (int y = 0; y < dim; y++) {
        printf("%2d: ", y);
        for (int x = 0; x < dim; x++) {
            int id = x + y * dim + z * dim * dim;
            uint8_t state = zit_node_state(f, id);
            printf("%3d", state);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    zit_fabric_t* f = zit_create(4);
    zit_seed(f, 1122911624);

    printf("=== Initial State ===\n\n");
    for (int z = 0; z < 4; z++) {
        dump_slice(f, z);
    }

    zit_run(f, 0);

    printf("=== After Convergence ===\n\n");
    for (int z = 0; z < 4; z++) {
        dump_slice(f, z);
    }

    zit_destroy(f);
    return 0;
}
```

---

### Example 11: Resistance Heatmap

Track resistance distribution.

```c
/* resistance_heatmap.c */
#include "zit.h"
#include <stdio.h>
#include <string.h>

void print_resistance_distribution(zit_fabric_t* f) {
    int buckets[9] = {0};  /* 0, 1, 2, 3, 4, 5, 6, 7, 8+ */
    int total = zit_total(f);

    for (int i = 0; i < total; i++) {
        uint8_t r = zit_node_resistance(f, i);
        if (r > 8) r = 8;
        buckets[r]++;
    }

    printf("Resistance distribution:\n");
    for (int i = 0; i <= 8; i++) {
        int bar_len = buckets[i] * 50 / total;
        char bar[51];
        memset(bar, '#', bar_len);
        bar[bar_len] = '\0';

        printf("  %d%s: %4d %s\n",
               i, i == 8 ? "+" : " ",
               buckets[i], bar);
    }
}

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);

    printf("Cycle 0:\n");
    print_resistance_distribution(f);

    for (int c = 1; c <= 50; c++) {
        zit_step(f);
        if (c % 10 == 0) {
            printf("\nCycle %d:\n", c);
            print_resistance_distribution(f);
        }
    }

    zit_destroy(f);
    return 0;
}
```

---

## Advanced Patterns

### Example 12: Parallel Experiments (OpenMP)

Run multiple fabrics in parallel.

```c
/* parallel_experiments.c */
#include "zit.h"
#include <stdio.h>
#include <omp.h>

#define NUM_EXPERIMENTS 100

int main() {
    int results[NUM_EXPERIMENTS];

    #pragma omp parallel for
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        zit_fabric_t* f = zit_create(8);
        zit_seed(f, 1000 + i);
        results[i] = zit_run(f, 500);
        zit_destroy(f);
    }

    /* Analyze results */
    int sum = 0;
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        sum += results[i];
    }
    printf("Average cycles: %.1f\n", (float)sum / NUM_EXPERIMENTS);

    return 0;
}
```

**Build:**
```bash
gcc -O3 -fopenmp -o parallel parallel_experiments.c zit.c -lm
```

---

### Example 13: Convergence Race

Race different seeds to convergence.

```c
/* convergence_race.c */
#include "zit.h"
#include <stdio.h>
#include <stdbool.h>

#define NUM_RACERS 4

int main() {
    zit_fabric_t* racers[NUM_RACERS];
    uint32_t seeds[NUM_RACERS] = {1111, 2222, 3333, 4444};
    bool finished[NUM_RACERS] = {false};
    int winner = -1;

    /* Create racers */
    for (int i = 0; i < NUM_RACERS; i++) {
        racers[i] = zit_create(8);
        zit_seed(racers[i], seeds[i]);
    }

    printf("Racing %d fabrics to convergence...\n\n", NUM_RACERS);

    /* Race! */
    int cycle = 0;
    while (winner == -1 && cycle < 500) {
        cycle++;

        for (int i = 0; i < NUM_RACERS; i++) {
            if (!finished[i]) {
                zit_step(racers[i]);

                if (zit_converged(racers[i])) {
                    finished[i] = true;
                    if (winner == -1) {
                        winner = i;
                    }
                }
            }
        }

        /* Status every 25 cycles */
        if (cycle % 25 == 0) {
            printf("Cycle %3d: ", cycle);
            for (int i = 0; i < NUM_RACERS; i++) {
                printf("[%d: %3d/%3d] ",
                       i, zit_resonant(racers[i]), zit_total(racers[i]));
            }
            printf("\n");
        }
    }

    printf("\n*** WINNER: Racer %d (seed %u) at cycle %d ***\n",
           winner, seeds[winner], cycle);

    /* Cleanup */
    for (int i = 0; i < NUM_RACERS; i++) {
        zit_destroy(racers[i]);
    }

    return 0;
}
```

---

### Example 14: State Injection

Inject custom initial states.

```c
/* state_injection.c */
#include "zit.h"
#include <stdio.h>

/* Note: This requires modifying zit.h to expose state setter,
   or accessing internals directly. For demo purposes only. */

extern void zit_set_node_state(zit_fabric_t* f, int id, uint8_t state);

void inject_gradient(zit_fabric_t* f) {
    int total = zit_total(f);
    int dim = zit_dim(f);

    for (int i = 0; i < total; i++) {
        int x, y, z;
        zit_node_coords(f, i, &x, &y, &z);

        /* State based on distance from origin */
        int dist = x + y + z;
        uint8_t state = (uint8_t)(dist * 255 / (3 * (dim - 1)));

        zit_set_node_state(f, i, state);
    }
}

int main() {
    zit_fabric_t* f = zit_create(8);

    /* Inject custom initial state */
    inject_gradient(f);

    printf("Injected gradient pattern\n");
    printf("Running...\n");

    int cycles = zit_run(f, 500);

    printf("Converged: %s\n", zit_converged(f) ? "yes" : "no");
    printf("Cycles: %d\n", cycles);

    zit_destroy(f);
    return 0;
}
```

---

### Example 15: Live Statistics Logger

Log statistics to CSV for later analysis.

```c
/* stats_logger.c */
#include "zit.h"
#include <stdio.h>

FILE* log_file;

void log_callback(zit_fabric_t* f, void* user_data) {
    fprintf(log_file, "%d,%d,%d,%d\n",
            zit_cycle(f),
            zit_resonant(f),
            zit_total(f),
            zit_rewires(f));
}

int main() {
    log_file = fopen("zit_stats.csv", "w");
    if (!log_file) {
        printf("Failed to open log file\n");
        return 1;
    }

    /* CSV header */
    fprintf(log_file, "cycle,resonant,total,rewires\n");

    zit_fabric_t* f = zit_create(8);
    zit_seed(f, 1122911624);
    zit_set_step_callback(f, log_callback, NULL);

    zit_run(f, 0);

    fclose(log_file);
    printf("Statistics logged to zit_stats.csv\n");

    zit_destroy(f);
    return 0;
}
```

**Analyze in Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('zit_stats.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(df['cycle'], df['resonant'] / df['total'])
ax1.set_ylabel('Resonance Ratio')
ax1.set_title('ZIT Convergence')

ax2.plot(df['cycle'], df['rewires'])
ax2.set_xlabel('Cycle')
ax2.set_ylabel('Cumulative Rewires')

plt.tight_layout()
plt.savefig('zit_analysis.png')
```

---

## Building All Examples

**Makefile:**
```makefile
CC = gcc
CFLAGS = -O3 -Wall -std=c99

EXAMPLES = minimal reproducible dimensions progress callback \
           progress_bar monte_carlo topology_analysis export_topology \
           dump_states resistance_heatmap convergence_race stats_logger

all: $(EXAMPLES)

%: %.c zit.c zit.h
	$(CC) $(CFLAGS) -o $@ $< zit.c -lm

parallel_experiments: parallel_experiments.c zit.c zit.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< zit.c -lm

clean:
	rm -f $(EXAMPLES) parallel_experiments *.json *.csv

.PHONY: all clean
```

---

*The topology IS the learned model.*
