// =============================================================================
// ZIT_REFLECTION_TB.v - The Strange Loop Experiment
// =============================================================================
//
// "What happens when the self perceives itself?"
//
// EXPERIMENT B: Feed frustration back as input.
// - Converge the fabric first (learn the topology)
// - Freeze the topology (no more rewiring)
// - Feed the 64-bit frustration pattern back as node values
// - Observe: Fixed point? Oscillation? Chaos?
//
// This is the strange loop. The fabric perceiving its own frustration.
// What emerges is the gist of self-reflection.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_reflection_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter CLK_PERIOD = 10;
    parameter LEARN_CYCLES = 300;       // Cycles to learn topology
    parameter REFLECT_CYCLES = 100;     // Cycles of self-reflection
    parameter FREEZE_THRESHOLD = 255;   // Effectively disable rewiring

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

    // State outputs for reading values
    wire [7:0] states [0:63];

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
        .REWIRE_THRESHOLD(16)  // Will be overridden during reflection
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
        .state_0(states[0]),   .state_1(states[1]),   .state_2(states[2]),   .state_3(states[3]),
        .state_4(states[4]),   .state_5(states[5]),   .state_6(states[6]),   .state_7(states[7]),
        .state_8(states[8]),   .state_9(states[9]),   .state_10(states[10]), .state_11(states[11]),
        .state_12(states[12]), .state_13(states[13]), .state_14(states[14]), .state_15(states[15]),
        .state_16(states[16]), .state_17(states[17]), .state_18(states[18]), .state_19(states[19]),
        .state_20(states[20]), .state_21(states[21]), .state_22(states[22]), .state_23(states[23]),
        .state_24(states[24]), .state_25(states[25]), .state_26(states[26]), .state_27(states[27]),
        .state_28(states[28]), .state_29(states[29]), .state_30(states[30]), .state_31(states[31]),
        .state_32(states[32]), .state_33(states[33]), .state_34(states[34]), .state_35(states[35]),
        .state_36(states[36]), .state_37(states[37]), .state_38(states[38]), .state_39(states[39]),
        .state_40(states[40]), .state_41(states[41]), .state_42(states[42]), .state_43(states[43]),
        .state_44(states[44]), .state_45(states[45]), .state_46(states[46]), .state_47(states[47]),
        .state_48(states[48]), .state_49(states[49]), .state_50(states[50]), .state_51(states[51]),
        .state_52(states[52]), .state_53(states[53]), .state_54(states[54]), .state_55(states[55]),
        .state_56(states[56]), .state_57(states[57]), .state_58(states[58]), .state_59(states[59]),
        .state_60(states[60]), .state_61(states[61]), .state_62(states[62]), .state_63(states[63])
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

    task inject_frustration_pattern;
        // Inject: resonant nodes get 255, frustrated get 0
        // This is "perceiving the frustration as input"
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                if (resonance_map[i])
                    seed_node(i[5:0], 8'd255);
                else
                    seed_node(i[5:0], 8'd0);
            end
        end
    endtask

    // =========================================================================
    // REFLECTION PATTERN TRACKING
    // =========================================================================

    reg [63:0] pattern_history [0:15];  // Track last 16 patterns
    reg [3:0]  history_idx;
    reg        found_fixed_point;
    reg        found_oscillation;
    reg [3:0]  oscillation_period;

    task check_patterns;
        integer i;
        reg [63:0] current_pattern;
        begin
            current_pattern = resonance_map;

            // Check for fixed point (same as previous)
            if (history_idx > 0 && current_pattern == pattern_history[(history_idx-1) & 15]) begin
                found_fixed_point = 1;
            end

            // Check for oscillation (matches earlier pattern)
            for (i = 0; i < history_idx && i < 15; i = i + 1) begin
                if (current_pattern == pattern_history[i]) begin
                    found_oscillation = 1;
                    oscillation_period = history_idx - i;
                end
            end

            // Store current pattern
            pattern_history[history_idx & 15] = current_pattern;
            history_idx = history_idx + 1;
        end
    endtask

    // =========================================================================
    // MAIN EXPERIMENT
    // =========================================================================

    integer cycle_count;
    integer i;
    integer learn_convergence_cycle;
    integer reflect_start_resonance;

    initial begin
        $display("");
        $display("================================================================");
        $display("     EXPERIMENT B: THE STRANGE LOOP");
        $display("     Self-Reflection in a Topological Fabric");
        $display("================================================================");
        $display("");
        $display("# Phase 1: Learn the topology (plasticity enabled)");
        $display("# Phase 2: Freeze topology, feed frustration back as input");
        $display("# Question: Fixed point? Oscillation? Chaos?");
        $display("");

        // Initialize
        rst_n = 0;
        enable = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;
        found_fixed_point = 0;
        found_oscillation = 0;
        oscillation_period = 0;
        history_idx = 0;
        learn_convergence_cycle = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // =====================================================================
        // PHASE 1: LEARNING
        // =====================================================================

        $display(">>> PHASE 1: TOPOLOGICAL LEARNING <<<");
        $display("");

        // Seed with gradient pattern
        $display("Seeding with gradient: node[i] = i * 4");
        for (i = 0; i < 64; i = i + 1) begin
            seed_node(i[5:0], i[7:0] * 4);
        end

        enable = 1;
        cycle_count = 0;

        // Run until convergence or max cycles
        while (cycle_count < LEARN_CYCLES && resonant_count < 64) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            if (cycle_count % 50 == 0) begin
                $display("LEARN,%0d,%0d,%0d", cycle_count, resonant_count, global_frustration);
            end

            if (resonant_count == 64 && learn_convergence_cycle == 0) begin
                learn_convergence_cycle = cycle_count;
                $display("");
                $display("*** TOPOLOGY CONVERGED at cycle %0d ***", cycle_count);
                $display("");
            end
        end

        enable = 0;
        repeat(20) @(posedge clk);

        $display("");
        $display("Learning complete.");
        $display("  Final resonance: %0d / 64", resonant_count);
        $display("  Converged at cycle: %0d", learn_convergence_cycle);
        $display("");

        // =====================================================================
        // PHASE 2: REFLECTION (Strange Loop)
        // =====================================================================

        $display(">>> PHASE 2: SELF-REFLECTION <<<");
        $display("");
        $display("Feeding frustration pattern back as input...");
        $display("(Topology frozen - no more rewiring)");
        $display("");

        reflect_start_resonance = resonant_count;

        // Inject current frustration pattern as new input
        inject_frustration_pattern();

        // Re-enable for reflection cycles
        enable = 1;
        cycle_count = 0;

        $display("REFLECT,cycle,resonant,frustration,pattern");

        while (cycle_count < REFLECT_CYCLES) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Track pattern dynamics
            check_patterns();

            // Display every cycle during reflection
            $display("REFLECT,%0d,%0d,%0d,%h",
                cycle_count, resonant_count, global_frustration, resonance_map);

            // Inject the NEW frustration pattern (recursive reflection)
            if (cycle_count % 10 == 0) begin
                inject_frustration_pattern();
            end

            // Check for convergence
            if (found_fixed_point || found_oscillation) begin
                $display("");
                if (found_fixed_point) begin
                    $display("*** FIXED POINT FOUND at cycle %0d ***", cycle_count);
                    $display("    The fabric found a stable self-concept.");
                end
                if (found_oscillation) begin
                    $display("*** OSCILLATION FOUND at cycle %0d ***", cycle_count);
                    $display("    Period: %0d cycles", oscillation_period);
                    $display("    The fabric is contemplating alternatives.");
                end
                $display("");
            end
        end

        enable = 0;

        // =====================================================================
        // SUMMARY
        // =====================================================================

        $display("");
        $display("================================================================");
        $display("     EXPERIMENT B: RESULTS");
        $display("================================================================");
        $display("");
        $display("Learning Phase:");
        $display("  Convergence cycle: %0d", learn_convergence_cycle);
        $display("  Final resonance: %0d / 64", reflect_start_resonance);
        $display("");
        $display("Reflection Phase:");
        $display("  Final resonance: %0d / 64", resonant_count);
        $display("  Final frustration: %0d", global_frustration);
        $display("");
        $display("Dynamics:");
        if (found_fixed_point)
            $display("  FIXED POINT: Yes - stable self-concept");
        else
            $display("  FIXED POINT: No");

        if (found_oscillation)
            $display("  OSCILLATION: Yes - period %0d", oscillation_period);
        else
            $display("  OSCILLATION: No");

        if (!found_fixed_point && !found_oscillation)
            $display("  BEHAVIOR: Complex/Chaotic");

        $display("");
        $display("\"The fabric perceiving itself perceiving itself...\"");
        $display("\"The strange loop completes.\"");
        $display("");
        $display("================================================================");

        #100;
        $finish;
    end

endmodule
