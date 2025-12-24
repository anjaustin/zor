/*
 * Test program for ZIT visualization API
 */

#include "zit.h"
#include <stdio.h>

int main() {
    printf("Testing ZIT Visualization API\n");
    printf("==============================\n\n");

    zit_fabric_t* f = zit_create(4);  /* 4x4x4 = 64 nodes for quick test */
    zit_seed(f, 1122911624);  /* Second Star Constant */

    /* Test dimension accessor */
    printf("Dimension: %d\n", zit_dim(f));
    printf("Total nodes: %d\n", zit_total(f));

    /* Test coordinate conversion */
    int x, y, z;
    zit_node_coords(f, 0, &x, &y, &z);
    printf("Node 0 coords: (%d, %d, %d)\n", x, y, z);

    zit_node_coords(f, 27, &x, &y, &z);  /* Should be (3, 2, 1) in 4x4x4 */
    printf("Node 27 coords: (%d, %d, %d)\n", x, y, z);

    /* Test node state access */
    printf("\nInitial states for first 8 nodes:\n");
    for (int i = 0; i < 8; i++) {
        printf("  Node %d: state=%3d, resistance=%d, resonant=%s\n",
               i, zit_node_state(f, i), zit_node_resistance(f, i),
               zit_node_resonant(f, i) ? "yes" : "no");
    }

    /* Test neighbor access */
    printf("\nNode 0 neighbors:\n");
    const char* dirs[] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
    for (int d = 0; d < 6; d++) {
        int nb = zit_node_neighbor(f, 0, d);
        zit_node_coords(f, nb, &x, &y, &z);
        printf("  %s: node %d at (%d, %d, %d)\n", dirs[d], nb, x, y, z);
    }

    /* Run a few steps and check state changes */
    printf("\nRunning 10 cycles...\n");
    for (int i = 0; i < 10; i++) {
        zit_step(f);
    }

    printf("\nAfter 10 cycles:\n");
    printf("  Cycle: %d\n", zit_cycle(f));
    printf("  Resonant: %d/%d\n", zit_resonant(f), zit_total(f));
    printf("  Rewires: %d\n", zit_rewires(f));

    /* Check for rewiring nodes */
    int rewiring_count = 0;
    for (int i = 0; i < zit_total(f); i++) {
        if (zit_node_rewiring(f, i)) rewiring_count++;
    }
    printf("  Currently rewiring: %d nodes\n", rewiring_count);

    zit_destroy(f);

    printf("\nVisualization API test complete.\n");
    return 0;
}
