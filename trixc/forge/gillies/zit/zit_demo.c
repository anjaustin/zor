/*
 * ZIT Demo - Watch topology learn
 *
 * Compile: make
 * Run:     ./zit_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include "zit.h"

int main(int argc, char** argv) {
    int dim = 8;  /* 8x8x8 = 512 nodes */

    if (argc > 1) {
        dim = atoi(argv[1]);
        if (dim < 2 || dim > 64) {
            fprintf(stderr, "Dimension must be 2-64\n");
            return 1;
        }
    }

    printf("ZIT: Homeo-Adaptive Topological Learning\n");
    printf("=========================================\n\n");

    zit_fabric_t* f = zit_create(dim);
    if (!f) {
        fprintf(stderr, "Failed to create fabric\n");
        return 1;
    }

    zit_seed(f, 1122911624);  /* Second Star Constant */

    printf("Fabric: %dx%dx%d = %d nodes\n\n", dim, dim, dim, zit_total(f));

    printf("Cycle  Resonant  Rewires\n");
    printf("-----  --------  -------\n");

    while (!zit_converged(f)) {
        zit_step(f);

        if (zit_cycle(f) % 25 == 0 || zit_converged(f)) {
            printf("%5d  %4d/%4d  %7d\n",
                   zit_cycle(f),
                   zit_resonant(f),
                   zit_total(f),
                   zit_rewires(f));
        }
    }

    printf("\n*** CONVERGED at cycle %d ***\n\n", zit_cycle(f));
    printf("The topology learned.\n");
    printf("Resistance dissolved.\n");

    zit_destroy(f);
    return 0;
}
