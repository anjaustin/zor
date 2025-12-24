// =============================================================================
// ZIT_FABRIC_TB.v - Testbench: Topology is Program
// =============================================================================
//
// This testbench demonstrates the core principle:
//   "Same kernel + different wiring = different behavior"
//
// Test 1: 4x4 grid of Zits performing bubble sort
// Test 2: Show that the sorting converges (all nodes reach resonance)
// Test 3: Measure cycles to convergence
//
// The frozen shape (comparator) never changes.
// Only the wiring determines the behavior.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_fabric_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter GRID_WIDTH = 4;
    parameter GRID_HEIGHT = 4;
    parameter STATE_WIDTH = 8;
    parameter CLK_PERIOD = 10;  // 100 MHz

    // =========================================================================
    // SIGNALS
    // =========================================================================

    reg clk;
    reg rst_n;
    reg enable;
    reg single_step;

    wire [1:0] phase;
    wire phase_strobe;
    wire cycle_complete;
    wire all_resonant;
    wire [GRID_WIDTH*GRID_HEIGHT-1:0] resonance_map;

    // Seed interface
    reg [STATE_WIDTH-1:0] seed_data;
    reg [3:0] seed_addr;
    reg seed_write;

    // =========================================================================
    // DUT INSTANTIATION
    // =========================================================================

    zit_fabric_controller controller (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .single_step(single_step),
        .phase(phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete)
    );

    zit_fabric #(
        .GRID_WIDTH(GRID_WIDTH),
        .GRID_HEIGHT(GRID_HEIGHT),
        .STATE_WIDTH(STATE_WIDTH)
    ) fabric (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .phase_strobe(phase_strobe),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .all_resonant(all_resonant),
        .resonance_map(resonance_map)
    );

    // =========================================================================
    // CLOCK GENERATION
    // =========================================================================

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // =========================================================================
    // TEST SEQUENCE
    // =========================================================================

    integer cycle_count;
    integer i;

    // Test values for 16 nodes (scrambled for sorting)
    reg [STATE_WIDTH-1:0] init_values [0:15];

    initial begin
        // Initialize test values (unsorted)
        init_values[0]  = 8'd42;
        init_values[1]  = 8'd17;
        init_values[2]  = 8'd93;
        init_values[3]  = 8'd8;
        init_values[4]  = 8'd55;
        init_values[5]  = 8'd71;
        init_values[6]  = 8'd23;
        init_values[7]  = 8'd64;
        init_values[8]  = 8'd12;
        init_values[9]  = 8'd88;
        init_values[10] = 8'd36;
        init_values[11] = 8'd99;
        init_values[12] = 8'd5;
        init_values[13] = 8'd77;
        init_values[14] = 8'd44;
        init_values[15] = 8'd29;

        $display("");
        $display("================================================================");
        $display("       ZIT FABRIC TESTBENCH - Topology is Program");
        $display("================================================================");
        $display("");
        $display("   \"Same kernel + different wiring = different behavior\"");
        $display("");

        // Initialize
        rst_n = 0;
        enable = 0;
        single_step = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;
        cycle_count = 0;

        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // =====================================================================
        // SEED: Load initial values into nodes
        // =====================================================================

        $display("PHASE: Seeding initial values");
        $display("----------------------------------------------------------------");

        for (i = 0; i < 16; i = i + 1) begin
            seed_addr = i[3:0];
            seed_data = init_values[i];
            seed_write = 1;
            #(CLK_PERIOD);
            seed_write = 0;
            #(CLK_PERIOD);
            $display("  Node %2d: %3d", i, init_values[i]);
        end
        $display("");

        // =====================================================================
        // TEST 1: Initial State (Unsorted)
        // =====================================================================

        $display("TEST 1: Initial State");
        $display("----------------------------------------------------------------");
        $display("  Grid (4x4):");
        $display("    Row 0: [%3d] [%3d] [%3d] [%3d]", init_values[0], init_values[1], init_values[2], init_values[3]);
        $display("    Row 1: [%3d] [%3d] [%3d] [%3d]", init_values[4], init_values[5], init_values[6], init_values[7]);
        $display("    Row 2: [%3d] [%3d] [%3d] [%3d]", init_values[8], init_values[9], init_values[10], init_values[11]);
        $display("    Row 3: [%3d] [%3d] [%3d] [%3d]", init_values[12], init_values[13], init_values[14], init_values[15]);
        $display("");

        // =====================================================================
        // TEST 2: Run Fabric
        // =====================================================================

        $display("TEST 2: Running Fabric (Bubble Sort Kernel)");
        $display("----------------------------------------------------------------");

        enable = 1;
        cycle_count = 0;

        // Run until all resonant or max cycles
        while (!all_resonant && cycle_count < 100) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            if (cycle_count <= 10 || cycle_count % 10 == 0) begin
                $display("  Cycle %3d: resonance = %b", cycle_count, resonance_map);
            end
        end

        enable = 0;
        #(CLK_PERIOD * 10);

        $display("");
        if (all_resonant) begin
            $display("  [PASS] CONVERGED in %0d cycles", cycle_count);
        end else begin
            $display("  [FAIL] DID NOT CONVERGE after %0d cycles", cycle_count);
        end
        $display("");

        // =====================================================================
        // SUMMARY
        // =====================================================================

        $display("================================================================");
        $display("SUMMARY");
        $display("================================================================");
        $display("  Grid size:          %0d x %0d = %0d nodes", GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH * GRID_HEIGHT);
        $display("  State width:        %0d bits", STATE_WIDTH);
        $display("  Cycles to converge: %0d", cycle_count);
        $display("  All resonant:       %s", all_resonant ? "YES" : "NO");
        $display("");
        $display("  THE KEY INSIGHT:");
        $display("  -----------------");
        $display("  The frozen shape (comparator) NEVER CHANGED.");
        $display("  The wiring (2D toroidal grid) defined the behavior.");
        $display("  Change the wiring, change the algorithm.");
        $display("");
        $display("  TOPOLOGY IS PROGRAM.");
        $display("");

        $finish;
    end

    // =========================================================================
    // WAVEFORM DUMP
    // =========================================================================

    initial begin
        $dumpfile("zit_fabric_tb.vcd");
        $dumpvars(0, zit_fabric_tb);
    end

endmodule


// =============================================================================
// 1D LINE TESTBENCH - Simpler Bubble Sort Demo
// =============================================================================
//
// A simpler demonstration: 8 nodes in a line (1D), performing bubble sort.
// This is a behavioral model showing the concept before the full fabric.
//
// =============================================================================

module bubble_sort_1d_tb;

    parameter N = 8;
    parameter WIDTH = 8;
    parameter CLK_PERIOD = 10;

    reg clk;
    reg [WIDTH-1:0] values [0:N-1];
    reg [WIDTH-1:0] temp;

    integer i, cycle, swaps, total_swaps;

    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test
    initial begin
        $display("");
        $display("================================================================");
        $display("       1D BUBBLE SORT - Same Kernel, Line Topology");
        $display("================================================================");
        $display("");

        // Initialize with unsorted values
        values[0] = 8'd42;
        values[1] = 8'd17;
        values[2] = 8'd93;
        values[3] = 8'd8;
        values[4] = 8'd55;
        values[5] = 8'd71;
        values[6] = 8'd23;
        values[7] = 8'd64;

        $display("Initial: ");
        print_array();
        $display("");

        total_swaps = 0;

        // Odd-even transposition sort (parallel bubble sort)
        for (cycle = 0; cycle < N; cycle = cycle + 1) begin
            swaps = 0;

            // Even phase: compare (0,1), (2,3), (4,5), (6,7)
            for (i = 0; i < N-1; i = i + 2) begin
                if (values[i] > values[i+1]) begin
                    // Swap
                    temp = values[i];
                    values[i] = values[i+1];
                    values[i+1] = temp;
                    swaps = swaps + 1;
                end
            end

            // Odd phase: compare (1,2), (3,4), (5,6)
            for (i = 1; i < N-1; i = i + 2) begin
                if (values[i] > values[i+1]) begin
                    // Swap
                    temp = values[i];
                    values[i] = values[i+1];
                    values[i+1] = temp;
                    swaps = swaps + 1;
                end
            end

            total_swaps = total_swaps + swaps;

            $display("Cycle %2d (%2d swaps): ", cycle + 1, swaps);
            print_array();

            if (swaps == 0) begin
                $display("");
                $display("================================================================");
                $display("[PASS] SORTED in %0d cycles (%0d total swaps)", cycle + 1, total_swaps);
                $display("================================================================");
                $display("");
                $display("  The kernel:  'if neighbor > me, swap'");
                $display("  The wiring:  1D line with even/odd phases");
                $display("  The result:  Sorted array");
                $display("");
                $display("  Same kernel + different wiring = different behavior");
                $display("");
                $finish;
            end
        end

        $display("");
        $display("[INFO] Completed %0d cycles", N);
        $finish;
    end

    task print_array;
        integer j;
        begin
            $write("  [");
            for (j = 0; j < N; j = j + 1) begin
                $write("%3d", values[j]);
                if (j < N-1) $write(", ");
            end
            $display("]");
        end
    endtask

endmodule
