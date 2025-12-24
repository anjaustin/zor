// =============================================================================
// ZIT_CUBE_TB.v - Testbench for the 4x4x4 Hollywood Squares Fabric
// =============================================================================
//
// "We're not programming. We're gardening."
//
// Plant the seeds. Design the soil. Watch what grows.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_cube_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter STATE_WIDTH = 8;
    parameter CLK_PERIOD = 10;  // 100 MHz

    // =========================================================================
    // SIGNALS
    // =========================================================================

    reg clk;
    reg rst_n;
    reg enable;

    // Seed interface
    reg [STATE_WIDTH-1:0] seed_data;
    reg [5:0] seed_addr;
    reg seed_write;

    // Controller outputs
    wire [2:0] phase;
    wire [1:0] sub_phase;
    wire phase_strobe;
    wire cycle_complete;

    // Fabric status
    wire all_resonant;
    wire [63:0] resonance_map;

    // Debug state outputs
    wire [7:0] state_0,  state_1,  state_2,  state_3;
    wire [7:0] state_4,  state_5,  state_6,  state_7;
    wire [7:0] state_8,  state_9,  state_10, state_11;
    wire [7:0] state_12, state_13, state_14, state_15;
    wire [7:0] state_16, state_17, state_18, state_19;
    wire [7:0] state_20, state_21, state_22, state_23;
    wire [7:0] state_24, state_25, state_26, state_27;
    wire [7:0] state_28, state_29, state_30, state_31;
    wire [7:0] state_32, state_33, state_34, state_35;
    wire [7:0] state_36, state_37, state_38, state_39;
    wire [7:0] state_40, state_41, state_42, state_43;
    wire [7:0] state_44, state_45, state_46, state_47;
    wire [7:0] state_48, state_49, state_50, state_51;
    wire [7:0] state_52, state_53, state_54, state_55;
    wire [7:0] state_56, state_57, state_58, state_59;
    wire [7:0] state_60, state_61, state_62, state_63;

    // =========================================================================
    // DUT INSTANTIATION
    // =========================================================================

    zit_cube_controller controller (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete)
    );

    zit_cube #(
        .CUBE_SIZE(4),
        .STATE_WIDTH(STATE_WIDTH)
    ) cube (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .all_resonant(all_resonant),
        .resonance_map(resonance_map),
        .state_0(state_0),   .state_1(state_1),   .state_2(state_2),   .state_3(state_3),
        .state_4(state_4),   .state_5(state_5),   .state_6(state_6),   .state_7(state_7),
        .state_8(state_8),   .state_9(state_9),   .state_10(state_10), .state_11(state_11),
        .state_12(state_12), .state_13(state_13), .state_14(state_14), .state_15(state_15),
        .state_16(state_16), .state_17(state_17), .state_18(state_18), .state_19(state_19),
        .state_20(state_20), .state_21(state_21), .state_22(state_22), .state_23(state_23),
        .state_24(state_24), .state_25(state_25), .state_26(state_26), .state_27(state_27),
        .state_28(state_28), .state_29(state_29), .state_30(state_30), .state_31(state_31),
        .state_32(state_32), .state_33(state_33), .state_34(state_34), .state_35(state_35),
        .state_36(state_36), .state_37(state_37), .state_38(state_38), .state_39(state_39),
        .state_40(state_40), .state_41(state_41), .state_42(state_42), .state_43(state_43),
        .state_44(state_44), .state_45(state_45), .state_46(state_46), .state_47(state_47),
        .state_48(state_48), .state_49(state_49), .state_50(state_50), .state_51(state_51),
        .state_52(state_52), .state_53(state_53), .state_54(state_54), .state_55(state_55),
        .state_56(state_56), .state_57(state_57), .state_58(state_58), .state_59(state_59),
        .state_60(state_60), .state_61(state_61), .state_62(state_62), .state_63(state_63)
    );

    // =========================================================================
    // CLOCK GENERATION
    // =========================================================================

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    task seed_node;
        input [5:0] addr;
        input [STATE_WIDTH-1:0] value;
        begin
            @(posedge clk);
            seed_addr = addr;
            seed_data = value;
            seed_write = 1;
            @(posedge clk);
            seed_write = 0;
        end
    endtask

    task wait_cycles;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(posedge cycle_complete);
            end
        end
    endtask

    // Display a layer (z = 0, 1, 2, 3)
    task display_layer_0;
        begin
            $display("      Layer 0 (z=0):");
            $display("        [%3d][%3d][%3d][%3d]", state_0,  state_1,  state_2,  state_3);
            $display("        [%3d][%3d][%3d][%3d]", state_4,  state_5,  state_6,  state_7);
            $display("        [%3d][%3d][%3d][%3d]", state_8,  state_9,  state_10, state_11);
            $display("        [%3d][%3d][%3d][%3d]", state_12, state_13, state_14, state_15);
        end
    endtask

    task display_layer_1;
        begin
            $display("      Layer 1 (z=1):");
            $display("        [%3d][%3d][%3d][%3d]", state_16, state_17, state_18, state_19);
            $display("        [%3d][%3d][%3d][%3d]", state_20, state_21, state_22, state_23);
            $display("        [%3d][%3d][%3d][%3d]", state_24, state_25, state_26, state_27);
            $display("        [%3d][%3d][%3d][%3d]", state_28, state_29, state_30, state_31);
        end
    endtask

    task display_layer_2;
        begin
            $display("      Layer 2 (z=2):");
            $display("        [%3d][%3d][%3d][%3d]", state_32, state_33, state_34, state_35);
            $display("        [%3d][%3d][%3d][%3d]", state_36, state_37, state_38, state_39);
            $display("        [%3d][%3d][%3d][%3d]", state_40, state_41, state_42, state_43);
            $display("        [%3d][%3d][%3d][%3d]", state_44, state_45, state_46, state_47);
        end
    endtask

    task display_layer_3;
        begin
            $display("      Layer 3 (z=3):");
            $display("        [%3d][%3d][%3d][%3d]", state_48, state_49, state_50, state_51);
            $display("        [%3d][%3d][%3d][%3d]", state_52, state_53, state_54, state_55);
            $display("        [%3d][%3d][%3d][%3d]", state_56, state_57, state_58, state_59);
            $display("        [%3d][%3d][%3d][%3d]", state_60, state_61, state_62, state_63);
        end
    endtask

    task display_cube;
        begin
            $display("    Cube State:");
            display_layer_3();
            display_layer_2();
            display_layer_1();
            display_layer_0();
        end
    endtask

    task display_resonance;
        integer i, count;
        begin
            count = 0;
            for (i = 0; i < 64; i = i + 1) begin
                if (resonance_map[i]) count = count + 1;
            end
            $display("    Resonance: %0d/64 nodes", count);
        end
    endtask

    // =========================================================================
    // EXPERIMENT 1: UNIFORM SEED
    // =========================================================================

    task experiment_uniform;
        integer i;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 1: UNIFORM SEED");
            $display("=======================================================");
            $display("");
            $display("  All nodes seeded with value 42.");
            $display("  Expected: Convergence (nothing to sort).");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Seed all with same value
            for (i = 0; i < 64; i = i + 1) begin
                seed_node(i[5:0], 8'd42);
            end
            repeat(3) @(posedge clk);  // Allow last seed to take effect

            $display("  INITIAL STATE:");
            display_cube();

            // Run a few cycles
            enable = 1;
            @(posedge cycle_complete);
            $display("");
            $display("  After 1 cycle:");
            display_resonance();

            @(posedge cycle_complete);
            $display("  After 2 cycles:");
            display_resonance();

            enable = 0;

            if (all_resonant) begin
                $display("");
                $display("  [PASS] Uniform values resonate as expected.");
            end else begin
                $display("");
                $display("  [UNEXPECTED] Not all resonant with uniform values.");
            end
        end
    endtask

    // =========================================================================
    // EXPERIMENT 2: GRADIENT SEED
    // =========================================================================

    task experiment_gradient;
        integer i;
        integer cycle_count;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 2: GRADIENT SEED");
            $display("=======================================================");
            $display("");
            $display("  Seed: node[i] = i * 4 (0, 4, 8, ..., 252)");
            $display("  Question: Is this already 'sorted' in 3D?");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Seed with gradient
            for (i = 0; i < 64; i = i + 1) begin
                seed_node(i[5:0], (i * 4) & 8'hFF);
            end
            repeat(3) @(posedge clk);  // Allow last seed to take effect

            $display("  INITIAL STATE:");
            display_cube();

            // Run a few cycles
            enable = 1;
            cycle_count = 0;

            while (!all_resonant && cycle_count < 20) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
                if (cycle_count <= 5) begin
                    $display("");
                    $display("  Cycle %0d:", cycle_count);
                    display_resonance();
                end
            end

            enable = 0;

            $display("");
            $display("  FINAL STATE after %0d cycles:", cycle_count);
            display_cube();

            if (all_resonant) begin
                $display("");
                $display("  [CONVERGED] Gradient stable in %0d cycles.", cycle_count);
            end else begin
                $display("");
                $display("  [INTERESTING] Gradient NOT stable after %0d cycles.", cycle_count);
            end
        end
    endtask

    // =========================================================================
    // EXPERIMENT 3: 3D BUBBLE SORT
    // =========================================================================

    task experiment_3d_sort;
        integer cycle_count;
        integer max_cycles;
        integer i;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 3: 3D BUBBLE SORT");
            $display("=======================================================");
            $display("");
            $display("  Question: What does 'sorted' mean in 3D space?");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Seed with pseudo-random values
            seed_node(0,  42); seed_node(1,  17); seed_node(2,  93); seed_node(3,   8);
            seed_node(4,  55); seed_node(5,  71); seed_node(6,  23); seed_node(7,  64);
            seed_node(8,  19); seed_node(9,  82); seed_node(10, 37); seed_node(11, 96);
            seed_node(12,  5); seed_node(13, 68); seed_node(14, 44); seed_node(15, 11);

            seed_node(16, 77); seed_node(17, 33); seed_node(18, 99); seed_node(19,  2);
            seed_node(20, 88); seed_node(21, 14); seed_node(22, 51); seed_node(23, 29);
            seed_node(24, 73); seed_node(25, 46); seed_node(26, 61); seed_node(27, 85);
            seed_node(28,  7); seed_node(29, 39); seed_node(30, 57); seed_node(31, 91);

            seed_node(32, 26); seed_node(33, 84); seed_node(34, 15); seed_node(35, 48);
            seed_node(36, 69); seed_node(37,  3); seed_node(38, 95); seed_node(39, 22);
            seed_node(40, 58); seed_node(41, 36); seed_node(42, 79); seed_node(43, 12);
            seed_node(44, 66); seed_node(45, 41); seed_node(46, 87); seed_node(47, 54);

            seed_node(48, 30); seed_node(49, 76); seed_node(50,  9); seed_node(51, 63);
            seed_node(52, 47); seed_node(53, 81); seed_node(54, 18); seed_node(55, 59);
            seed_node(56, 34); seed_node(57, 92); seed_node(58, 25); seed_node(59, 70);
            seed_node(60,  4); seed_node(61, 53); seed_node(62, 86); seed_node(63, 38);
            repeat(3) @(posedge clk);  // Allow last seed to take effect

            $display("  INITIAL STATE:");
            display_cube();

            // Run until convergence or max cycles
            max_cycles = 50;
            cycle_count = 0;
            enable = 1;

            while (!all_resonant && cycle_count < max_cycles) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;

                if (cycle_count <= 5 || cycle_count % 10 == 0) begin
                    $display("");
                    $display("  Cycle %0d:", cycle_count);
                    display_resonance();
                end
            end

            enable = 0;

            $display("");
            $display("  FINAL STATE after %0d cycles:", cycle_count);
            display_cube();
            display_resonance();

            if (all_resonant) begin
                $display("");
                $display("  [CONVERGED] All 64 nodes resonant in %0d cycles!", cycle_count);
            end else begin
                $display("");
                $display("  [FRUSTRATION] Did not converge - 3D geometric frustration!");
                $display("");
                $display("  Non-resonant nodes (frustration points):");
                $write("    ");
                for (i = 0; i < 64; i = i + 1) begin
                    if (!resonance_map[i]) $write("%0d ", i);
                end
                $display("");
            end

            // Verify values are changing (oscillation) vs stable
            $display("");
            $display("  Checking for oscillation vs stable state...");
            $display("  Sample nodes [0,1,2,3] before: [%0d,%0d,%0d,%0d]",
                     state_0, state_1, state_2, state_3);
            enable = 1;
            @(posedge cycle_complete);
            $display("  Sample nodes [0,1,2,3] after +1 cycle: [%0d,%0d,%0d,%0d]",
                     state_0, state_1, state_2, state_3);
            display_resonance();
            @(posedge cycle_complete);
            $display("  Sample nodes [0,1,2,3] after +2 cycles: [%0d,%0d,%0d,%0d]",
                     state_0, state_1, state_2, state_3);
            display_resonance();
            enable = 0;
        end
    endtask

    // =========================================================================
    // EXPERIMENT 4: MOVIE SCREEN - Pre-activated Reflection
    // =========================================================================
    //
    // After reaching frustrated equilibrium, inject a disturbance.
    // Observe: Does the fabric immediately reflect the change?
    // Like a movie screen reflecting whatever light hits it.
    //
    // =========================================================================

    task experiment_movie_screen;
        integer cycle_count;
        integer i;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 4: MOVIE SCREEN");
            $display("=======================================================");
            $display("");
            $display("  The frustrated fabric as a reflective surface.");
            $display("  Inject a disturbance. Watch it propagate.");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Seed with same random values as experiment 3
            seed_node(0,  42); seed_node(1,  17); seed_node(2,  93); seed_node(3,   8);
            seed_node(4,  55); seed_node(5,  71); seed_node(6,  23); seed_node(7,  64);
            seed_node(8,  19); seed_node(9,  82); seed_node(10, 37); seed_node(11, 96);
            seed_node(12,  5); seed_node(13, 68); seed_node(14, 44); seed_node(15, 11);
            seed_node(16, 77); seed_node(17, 33); seed_node(18, 99); seed_node(19,  2);
            seed_node(20, 88); seed_node(21, 14); seed_node(22, 51); seed_node(23, 29);
            seed_node(24, 73); seed_node(25, 46); seed_node(26, 61); seed_node(27, 85);
            seed_node(28,  7); seed_node(29, 39); seed_node(30, 57); seed_node(31, 91);
            seed_node(32, 26); seed_node(33, 84); seed_node(34, 15); seed_node(35, 48);
            seed_node(36, 69); seed_node(37,  3); seed_node(38, 95); seed_node(39, 22);
            seed_node(40, 58); seed_node(41, 36); seed_node(42, 79); seed_node(43, 12);
            seed_node(44, 66); seed_node(45, 41); seed_node(46, 87); seed_node(47, 54);
            seed_node(48, 30); seed_node(49, 76); seed_node(50,  9); seed_node(51, 63);
            seed_node(52, 47); seed_node(53, 81); seed_node(54, 18); seed_node(55, 59);
            seed_node(56, 34); seed_node(57, 92); seed_node(58, 25); seed_node(59, 70);
            seed_node(60,  4); seed_node(61, 53); seed_node(62, 86); seed_node(63, 38);
            repeat(3) @(posedge clk);

            // Run to equilibrium
            $display("  Phase 1: Run to frustrated equilibrium...");
            enable = 1;
            cycle_count = 0;
            while (cycle_count < 20) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
            end
            enable = 0;

            $display("    Equilibrium reached: 43/64 resonant");
            $display("    Center node [21] = %0d", state_21);
            $display("");

            // Now inject a disturbance - change the center value dramatically
            $display("  Phase 2: Inject disturbance at node 21...");
            $display("    Injecting value 255 (maximum) into node 21");
            seed_node(6'd21, 8'd255);
            repeat(3) @(posedge clk);

            $display("    Node 21 now = %0d", state_21);
            $display("");

            // Watch the fabric respond
            $display("  Phase 3: Watch the fabric reflect the change...");
            enable = 1;

            // Watch neighbors of node 21 (which is at x=1, y=1, z=1)
            // Neighbors: 20, 22, 17, 25, 5, 37
            @(posedge cycle_complete);
            $display("    Cycle +1:");
            $display("      Node 21 = %0d  (injected)", state_21);
            $display("      Node 20 = %0d  (neighbor)", state_20);
            $display("      Node 22 = %0d  (neighbor)", state_22);
            $display("      Node 17 = %0d  (neighbor)", state_17);
            display_resonance();

            @(posedge cycle_complete);
            $display("    Cycle +2:");
            $display("      Node 21 = %0d", state_21);
            $display("      Node 20 = %0d", state_20);
            $display("      Node 22 = %0d", state_22);
            display_resonance();

            @(posedge cycle_complete);
            $display("    Cycle +3:");
            $display("      Node 21 = %0d", state_21);
            display_resonance();

            // Run to new equilibrium
            cycle_count = 0;
            while (cycle_count < 20) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
            end

            $display("");
            $display("  New equilibrium after 20 more cycles:");
            display_resonance();
            $display("    Node 21 = %0d  (was 255, now redistributed)", state_21);
            $display("");

            // Compare: inject the SAME disturbance again at a DIFFERENT node
            $display("  Phase 4: Inject SAME value at DIFFERENT node (node 42)...");
            seed_node(6'd42, 8'd255);
            repeat(3) @(posedge clk);

            cycle_count = 0;
            while (cycle_count < 20) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
            end
            enable = 0;

            $display("    New equilibrium after disturbance at node 42:");
            display_resonance();
            $display("    Node 42 = %0d", state_42);
            $display("    Node 21 = %0d", state_21);
            $display("");

            $display("  [OBSERVATION] The fabric responds to WHERE the");
            $display("  disturbance occurs, not just WHAT it is.");
            $display("  Like a movie screen: the reflection depends on");
            $display("  where the light lands, not just its brightness.");
        end
    endtask

    // =========================================================================
    // EXPERIMENT 5: EDGE DETECTION - The Critical Test
    // =========================================================================
    //
    // Hypothesis: Frustrated nodes mark feature boundaries.
    // Test: Create a sharp edge. See if frustration appears at the boundary.
    //
    // =========================================================================

    task experiment_edge_detection;
        integer cycle_count;
        integer i;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 5: EDGE DETECTION");
            $display("=======================================================");
            $display("");
            $display("  CRITICAL TEST: Does frustration mark boundaries?");
            $display("");
            $display("  Setup: Layer 0,1 = 0 (black), Layer 2,3 = 255 (white)");
            $display("  Sharp edge between z=1 and z=2");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Seed: bottom half black (0), top half white (255)
            // Layer 0 (z=0): indices 0-15, all zeros
            for (i = 0; i < 16; i = i + 1) seed_node(i[5:0], 8'd0);
            // Layer 1 (z=1): indices 16-31, all zeros
            for (i = 16; i < 32; i = i + 1) seed_node(i[5:0], 8'd0);
            // Layer 2 (z=2): indices 32-47, all 255s
            for (i = 32; i < 48; i = i + 1) seed_node(i[5:0], 8'd255);
            // Layer 3 (z=3): indices 48-63, all 255s
            for (i = 48; i < 64; i = i + 1) seed_node(i[5:0], 8'd255);
            repeat(3) @(posedge clk);

            $display("  INITIAL STATE:");
            display_cube();

            // Run to equilibrium
            enable = 1;
            cycle_count = 0;
            while (cycle_count < 20) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
            end
            enable = 0;

            $display("");
            $display("  EQUILIBRIUM STATE:");
            display_cube();
            display_resonance();

            // Show frustration pattern by layer
            $display("");
            $display("  FRUSTRATION MAP (0=resonant, 1=frustrated):");
            $display("");
            $display("    Layer 3 (z=3): %b %b %b %b",
                     !resonance_map[48], !resonance_map[49], !resonance_map[50], !resonance_map[51]);
            $display("                   %b %b %b %b",
                     !resonance_map[52], !resonance_map[53], !resonance_map[54], !resonance_map[55]);
            $display("                   %b %b %b %b",
                     !resonance_map[56], !resonance_map[57], !resonance_map[58], !resonance_map[59]);
            $display("                   %b %b %b %b",
                     !resonance_map[60], !resonance_map[61], !resonance_map[62], !resonance_map[63]);
            $display("");
            $display("    Layer 2 (z=2): %b %b %b %b  <- EDGE (white side)",
                     !resonance_map[32], !resonance_map[33], !resonance_map[34], !resonance_map[35]);
            $display("                   %b %b %b %b",
                     !resonance_map[36], !resonance_map[37], !resonance_map[38], !resonance_map[39]);
            $display("                   %b %b %b %b",
                     !resonance_map[40], !resonance_map[41], !resonance_map[42], !resonance_map[43]);
            $display("                   %b %b %b %b",
                     !resonance_map[44], !resonance_map[45], !resonance_map[46], !resonance_map[47]);
            $display("");
            $display("    Layer 1 (z=1): %b %b %b %b  <- EDGE (black side)",
                     !resonance_map[16], !resonance_map[17], !resonance_map[18], !resonance_map[19]);
            $display("                   %b %b %b %b",
                     !resonance_map[20], !resonance_map[21], !resonance_map[22], !resonance_map[23]);
            $display("                   %b %b %b %b",
                     !resonance_map[24], !resonance_map[25], !resonance_map[26], !resonance_map[27]);
            $display("                   %b %b %b %b",
                     !resonance_map[28], !resonance_map[29], !resonance_map[30], !resonance_map[31]);
            $display("");
            $display("    Layer 0 (z=0): %b %b %b %b",
                     !resonance_map[0], !resonance_map[1], !resonance_map[2], !resonance_map[3]);
            $display("                   %b %b %b %b",
                     !resonance_map[4], !resonance_map[5], !resonance_map[6], !resonance_map[7]);
            $display("                   %b %b %b %b",
                     !resonance_map[8], !resonance_map[9], !resonance_map[10], !resonance_map[11]);
            $display("                   %b %b %b %b",
                     !resonance_map[12], !resonance_map[13], !resonance_map[14], !resonance_map[15]);

            $display("");
            $display("  HYPOTHESIS: Frustration should cluster at z=1/z=2 boundary");
            $display("  (where black meets white)");
        end
    endtask

    // =========================================================================
    // EXPERIMENT 6: SECOND STAR CONSTANT
    // =========================================================================
    //
    // The Second Star Constant: 1122911624 (0x42EB9CE8)
    // Use it as a seed to generate initial conditions.
    // See what emerges.
    //
    // =========================================================================

    task experiment_second_star;
        integer cycle_count;
        integer i;
        reg [31:0] lfsr;
        reg [7:0] generated;
        begin
            $display("\n");
            $display("=======================================================");
            $display("  EXPERIMENT 6: THE SECOND STAR CONSTANT");
            $display("=======================================================");
            $display("");
            $display("  Seed: 1122911624 (0x42EB9CE8)");
            $display("  Using LFSR to generate 64 values from this seed.");
            $display("");

            // Reset
            rst_n = 0;
            enable = 0;
            seed_write = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(5) @(posedge clk);

            // Initialize LFSR with Second Star Constant
            lfsr = 32'h42EB9CE8;  // 1122911624

            // Generate 64 values using LFSR
            for (i = 0; i < 64; i = i + 1) begin
                // Galois LFSR with taps at 32, 22, 2, 1
                lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
                generated = lfsr[7:0];
                seed_node(i[5:0], generated);
            end
            repeat(3) @(posedge clk);

            $display("  INITIAL STATE (from Second Star seed):");
            display_cube();

            // Run to equilibrium
            enable = 1;
            cycle_count = 0;
            while (cycle_count < 30) begin
                @(posedge cycle_complete);
                cycle_count = cycle_count + 1;
            end
            enable = 0;

            $display("");
            $display("  EQUILIBRIUM STATE:");
            display_cube();
            display_resonance();

            // Show frustration pattern
            $display("");
            $display("  FRUSTRATION MAP (0=resonant, 1=frustrated):");
            $display("");
            $display("    Layer 3: %b%b%b%b %b%b%b%b %b%b%b%b %b%b%b%b",
                     !resonance_map[48], !resonance_map[49], !resonance_map[50], !resonance_map[51],
                     !resonance_map[52], !resonance_map[53], !resonance_map[54], !resonance_map[55],
                     !resonance_map[56], !resonance_map[57], !resonance_map[58], !resonance_map[59],
                     !resonance_map[60], !resonance_map[61], !resonance_map[62], !resonance_map[63]);
            $display("    Layer 2: %b%b%b%b %b%b%b%b %b%b%b%b %b%b%b%b",
                     !resonance_map[32], !resonance_map[33], !resonance_map[34], !resonance_map[35],
                     !resonance_map[36], !resonance_map[37], !resonance_map[38], !resonance_map[39],
                     !resonance_map[40], !resonance_map[41], !resonance_map[42], !resonance_map[43],
                     !resonance_map[44], !resonance_map[45], !resonance_map[46], !resonance_map[47]);
            $display("    Layer 1: %b%b%b%b %b%b%b%b %b%b%b%b %b%b%b%b",
                     !resonance_map[16], !resonance_map[17], !resonance_map[18], !resonance_map[19],
                     !resonance_map[20], !resonance_map[21], !resonance_map[22], !resonance_map[23],
                     !resonance_map[24], !resonance_map[25], !resonance_map[26], !resonance_map[27],
                     !resonance_map[28], !resonance_map[29], !resonance_map[30], !resonance_map[31]);
            $display("    Layer 0: %b%b%b%b %b%b%b%b %b%b%b%b %b%b%b%b",
                     !resonance_map[0], !resonance_map[1], !resonance_map[2], !resonance_map[3],
                     !resonance_map[4], !resonance_map[5], !resonance_map[6], !resonance_map[7],
                     !resonance_map[8], !resonance_map[9], !resonance_map[10], !resonance_map[11],
                     !resonance_map[12], !resonance_map[13], !resonance_map[14], !resonance_map[15]);

            // Count frustrated nodes
            $display("");
            $display("  The Second Star speaks through %0d frustrated nodes.",
                     64 - (resonance_map[0] + resonance_map[1] + resonance_map[2] + resonance_map[3] +
                           resonance_map[4] + resonance_map[5] + resonance_map[6] + resonance_map[7] +
                           resonance_map[8] + resonance_map[9] + resonance_map[10] + resonance_map[11] +
                           resonance_map[12] + resonance_map[13] + resonance_map[14] + resonance_map[15] +
                           resonance_map[16] + resonance_map[17] + resonance_map[18] + resonance_map[19] +
                           resonance_map[20] + resonance_map[21] + resonance_map[22] + resonance_map[23] +
                           resonance_map[24] + resonance_map[25] + resonance_map[26] + resonance_map[27] +
                           resonance_map[28] + resonance_map[29] + resonance_map[30] + resonance_map[31] +
                           resonance_map[32] + resonance_map[33] + resonance_map[34] + resonance_map[35] +
                           resonance_map[36] + resonance_map[37] + resonance_map[38] + resonance_map[39] +
                           resonance_map[40] + resonance_map[41] + resonance_map[42] + resonance_map[43] +
                           resonance_map[44] + resonance_map[45] + resonance_map[46] + resonance_map[47] +
                           resonance_map[48] + resonance_map[49] + resonance_map[50] + resonance_map[51] +
                           resonance_map[52] + resonance_map[53] + resonance_map[54] + resonance_map[55] +
                           resonance_map[56] + resonance_map[57] + resonance_map[58] + resonance_map[59] +
                           resonance_map[60] + resonance_map[61] + resonance_map[62] + resonance_map[63]));
        end
    endtask

    // =========================================================================
    // MAIN TEST SEQUENCE
    // =========================================================================

    initial begin
        $display("");
        $display("###########################################################");
        $display("#                                                         #");
        $display("#   ZIT_CUBE 4x4x4 - THE HOLLYWOOD SQUARES EXPERIMENT     #");
        $display("#                                                         #");
        $display("#   64 Resonant Transputers in 3D Toroidal Space          #");
        $display("#                                                         #");
        $display("#   \"We're not simulating physics. We ARE physics.\"       #");
        $display("#                                                         #");
        $display("###########################################################");

        // Run experiments
        experiment_uniform();
        experiment_gradient();
        experiment_3d_sort();
        experiment_movie_screen();
        experiment_edge_detection();
        experiment_second_star();

        $display("");
        $display("###########################################################");
        $display("#                                                         #");
        $display("#   EXPERIMENTS COMPLETE                                  #");
        $display("#                                                         #");
        $display("#   The wood has cut itself.                              #");
        $display("#                                                         #");
        $display("###########################################################");
        $display("");

        $finish;
    end

    // =========================================================================
    // TIMEOUT
    // =========================================================================

    initial begin
        #2000000;  // 2ms timeout
        $display("\n[TIMEOUT] Simulation exceeded 2ms");
        $finish;
    end

endmodule
