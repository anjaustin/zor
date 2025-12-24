// =============================================================================
// ZIT_PLASTIC_TB.v - Testbench for Topological Plasticity
// =============================================================================
//
// "Can topology learn from frustration alone?"
//
// Experiments:
//   1. Does frustration decrease over time?
//   2. Does the fabric self-organize?
//   3. Does topology change?
//
// =============================================================================

`timescale 1ns / 1ps

module zit_plastic_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter CLK_PERIOD = 10;
    parameter NUM_CYCLES = 500;
    parameter SAMPLE_INTERVAL = 10;

    // =========================================================================
    // SIGNALS
    // =========================================================================

    reg         clk;
    reg         rst_n;
    reg         enable;

    wire [2:0]  phase;
    wire [1:0]  sub_phase;
    wire        phase_strobe;
    wire        cycle_complete;

    reg  [7:0]  seed_data;
    reg  [5:0]  seed_addr;
    reg         seed_write;

    wire        all_resonant;
    wire [63:0] resonance_map;
    wire [9:0]  global_frustration;
    wire [9:0]  resonant_count;
    wire [63:0] rewiring_map;

    // State outputs
    wire [7:0] states [63:0];
    wire [7:0] state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7;
    wire [7:0] state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15;
    wire [7:0] state_16, state_17, state_18, state_19, state_20, state_21, state_22, state_23;
    wire [7:0] state_24, state_25, state_26, state_27, state_28, state_29, state_30, state_31;
    wire [7:0] state_32, state_33, state_34, state_35, state_36, state_37, state_38, state_39;
    wire [7:0] state_40, state_41, state_42, state_43, state_44, state_45, state_46, state_47;
    wire [7:0] state_48, state_49, state_50, state_51, state_52, state_53, state_54, state_55;
    wire [7:0] state_56, state_57, state_58, state_59, state_60, state_61, state_62, state_63;

    // =========================================================================
    // CLOCK GENERATION
    // =========================================================================

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // DUT INSTANTIATION
    // =========================================================================

    zit_plastic_controller ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete)
    );

    zit_plastic_fabric #(
        .CUBE_SIZE(4),
        .STATE_WIDTH(8),
        .FRUSTRATION_BITS(8),
        .REWIRE_THRESHOLD(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .all_resonant(all_resonant),
        .resonance_map(resonance_map),
        .global_frustration(global_frustration),
        .resonant_count(resonant_count),
        .rewiring_map(rewiring_map),
        .state_0(state_0), .state_1(state_1), .state_2(state_2), .state_3(state_3),
        .state_4(state_4), .state_5(state_5), .state_6(state_6), .state_7(state_7),
        .state_8(state_8), .state_9(state_9), .state_10(state_10), .state_11(state_11),
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
    // HELPER TASKS
    // =========================================================================

    task seed_node;
        input [5:0] addr;
        input [7:0] value;
        begin
            @(posedge clk);
            seed_addr <= addr;
            seed_data <= value;
            seed_write <= 1;
            @(posedge clk);
            seed_write <= 0;
            repeat(3) @(posedge clk);
        end
    endtask

    task run_cycles;
        input integer count;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                @(posedge cycle_complete);
            end
        end
    endtask

    // PRNG for seeding
    reg [31:0] prng;
    function [31:0] next_rand;
        input [31:0] current;
        begin
            next_rand = current * 1103515245 + 12345;
        end
    endfunction

    // =========================================================================
    // METRICS TRACKING
    // =========================================================================

    reg [9:0] frustration_history [0:99];
    reg [9:0] resonance_history [0:99];
    reg [9:0] rewire_history [0:99];
    integer history_idx;

    function integer count_ones;
        input [63:0] bitmap;
        integer c, b;
        begin
            c = 0;
            for (b = 0; b < 64; b = b + 1) begin
                if (bitmap[b]) c = c + 1;
            end
            count_ones = c;
        end
    endfunction

    // =========================================================================
    // MAIN TEST
    // =========================================================================

    integer cycle_count;
    integer i;
    integer initial_resonance;
    integer final_resonance;
    integer initial_frustration;
    integer final_frustration;
    integer rewire_events;

    initial begin
        $display("");
        $display("============================================================");
        $display("     ZIT PLASTIC FABRIC - TOPOLOGICAL LEARNING TEST");
        $display("============================================================");
        $display("");
        $display("Hypothesis: Topology can learn from frustration alone.");
        $display("");

        // =====================================================================
        // INITIALIZATION
        // =====================================================================

        rst_n = 0;
        enable = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;
        prng = 32'hDEADBEEF;
        history_idx = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // =====================================================================
        // EXPERIMENT 1: Random Seed - Does Frustration Decrease?
        // =====================================================================

        $display("");
        $display("------------------------------------------------------------");
        $display("EXPERIMENT 1: Random Seed - Self-Organization Test");
        $display("------------------------------------------------------------");
        $display("");

        // Seed with random values
        for (i = 0; i < 64; i = i + 1) begin
            prng = next_rand(prng);
            seed_node(i[5:0], prng[7:0]);
        end

        $display("Seeded 64 nodes with random values.");
        $display("");

        // Enable and let it run
        enable = 1;

        // Wait for stabilization
        run_cycles(20);

        // Record initial metrics
        initial_resonance = resonant_count;
        initial_frustration = global_frustration;

        $display("Initial state:");
        $display("  Resonant nodes: %0d / 64", initial_resonance);
        $display("  Global frustration: %0d", initial_frustration);
        $display("");

        // Run for many cycles, sampling periodically
        $display("Running %0d cycles with plasticity enabled...", NUM_CYCLES);
        $display("");
        $display("Cycle | Resonant | Frustration | Rewiring");
        $display("------+----------+-------------+---------");

        cycle_count = 0;
        rewire_events = 0;

        for (i = 0; i < NUM_CYCLES; i = i + 1) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Count rewiring events
            rewire_events = rewire_events + count_ones(rewiring_map);

            // Sample at intervals
            if (cycle_count % SAMPLE_INTERVAL == 0) begin
                $display("%5d | %8d | %11d | %7d",
                    cycle_count, resonant_count, global_frustration, count_ones(rewiring_map));

                // Record history
                if (history_idx < 100) begin
                    frustration_history[history_idx] = global_frustration;
                    resonance_history[history_idx] = resonant_count;
                    rewire_history[history_idx] = count_ones(rewiring_map);
                    history_idx = history_idx + 1;
                end
            end
        end

        // Record final metrics
        final_resonance = resonant_count;
        final_frustration = global_frustration;

        $display("");
        $display("Final state:");
        $display("  Resonant nodes: %0d / 64", final_resonance);
        $display("  Global frustration: %0d", final_frustration);
        $display("  Total rewire attempts: %0d", rewire_events);
        $display("");

        // =====================================================================
        // ANALYSIS
        // =====================================================================

        $display("------------------------------------------------------------");
        $display("ANALYSIS");
        $display("------------------------------------------------------------");
        $display("");

        if (final_frustration < initial_frustration) begin
            $display("[PASS] Frustration DECREASED: %0d -> %0d",
                initial_frustration, final_frustration);
            $display("       The fabric is learning!");
        end else if (final_frustration == initial_frustration) begin
            $display("[NEUTRAL] Frustration unchanged: %0d", final_frustration);
        end else begin
            $display("[UNEXPECTED] Frustration increased: %0d -> %0d",
                initial_frustration, final_frustration);
        end

        $display("");

        if (final_resonance > initial_resonance) begin
            $display("[PASS] Resonance INCREASED: %0d -> %0d",
                initial_resonance, final_resonance);
        end else if (final_resonance == initial_resonance) begin
            $display("[NEUTRAL] Resonance unchanged: %0d", final_resonance);
        end else begin
            $display("[UNEXPECTED] Resonance decreased: %0d -> %0d",
                initial_resonance, final_resonance);
        end

        $display("");

        if (rewire_events > 0) begin
            $display("[PASS] Topology changed: %0d rewire attempts observed", rewire_events);
        end else begin
            $display("[UNEXPECTED] No rewiring occurred");
        end

        $display("");

        // =====================================================================
        // EXPERIMENT 2: Gradient Input - Domain Specialization
        // =====================================================================

        $display("");
        $display("------------------------------------------------------------");
        $display("EXPERIMENT 2: Gradient Input - Domain Specialization");
        $display("------------------------------------------------------------");
        $display("");

        // Reset
        rst_n = 0;
        enable = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Seed with gradient (sorted values)
        for (i = 0; i < 64; i = i + 1) begin
            seed_node(i[5:0], i[7:0] * 4);  // 0, 4, 8, 12, ... 252
        end

        $display("Seeded 64 nodes with gradient (sorted values).");
        $display("");

        enable = 1;

        // Let it stabilize
        run_cycles(50);

        $display("After 50 cycles with gradient input:");
        $display("  Resonant nodes: %0d / 64", resonant_count);
        $display("  Global frustration: %0d", global_frustration);
        $display("");

        // Now perturb with random values and see if it adapts
        $display("Perturbing with random values...");

        for (i = 0; i < 64; i = i + 1) begin
            prng = next_rand(prng);
            seed_node(i[5:0], prng[7:0]);
        end

        run_cycles(100);

        $display("After 100 cycles post-perturbation:");
        $display("  Resonant nodes: %0d / 64", resonant_count);
        $display("  Global frustration: %0d", global_frustration);
        $display("");

        // =====================================================================
        // SUMMARY
        // =====================================================================

        $display("");
        $display("============================================================");
        $display("                      SUMMARY");
        $display("============================================================");
        $display("");
        $display("The plastic fabric was tested for self-organization.");
        $display("");
        $display("Key observations:");
        $display("  1. Frustration tracking: Active");
        $display("  2. Rewiring attempts: %0d", rewire_events);
        $display("  3. Final resonance: %0d / 64", final_resonance);
        $display("");
        $display("The hypothesis 'Topology can learn from frustration alone'");
        if (final_frustration < initial_frustration && rewire_events > 0) begin
            $display("is SUPPORTED by this experiment.");
        end else begin
            $display("requires further investigation.");
        end
        $display("");
        $display("============================================================");
        $display("");

        enable = 0;
        #100;
        $finish;
    end

endmodule
