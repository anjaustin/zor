// =============================================================================
// ZIT_LINE_TB.v - 1D Line Testbench: Topology is Program
// =============================================================================
//
// This testbench demonstrates the core principle with a 1D topology:
//   "Same kernel + different wiring = different behavior"
//
// 8 nodes in a line performing odd-even transposition sort.
// The frozen shape (comparator) never changes.
// The wiring (1D line with odd/even phases) defines the algorithm.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_line_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter NUM_NODES = 8;
    parameter STATE_WIDTH = 8;
    parameter CLK_PERIOD = 10;

    // =========================================================================
    // SIGNALS
    // =========================================================================

    reg clk;
    reg rst_n;
    reg enable;
    reg single_step;

    wire [1:0] phase;
    wire phase_strobe;
    wire odd_phase;
    wire cycle_complete;
    wire all_resonant;
    wire [NUM_NODES-1:0] resonance_map;
    wire [STATE_WIDTH*NUM_NODES-1:0] all_states;

    // Seed interface
    reg [STATE_WIDTH-1:0] seed_data;
    reg [2:0] seed_addr;
    reg seed_write;

    // =========================================================================
    // DUT INSTANTIATION
    // =========================================================================

    zit_line_controller controller (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .single_step(single_step),
        .phase(phase),
        .phase_strobe(phase_strobe),
        .odd_phase(odd_phase),
        .cycle_complete(cycle_complete)
    );

    zit_line #(
        .NUM_NODES(NUM_NODES),
        .STATE_WIDTH(STATE_WIDTH)
    ) line (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .phase_strobe(phase_strobe),
        .odd_phase(odd_phase),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .all_resonant(all_resonant),
        .resonance_map(resonance_map),
        .all_states(all_states)
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
    reg [STATE_WIDTH-1:0] init_values [0:NUM_NODES-1];

    // Convergence tracking
    reg prev_all_resonant;
    reg prev_odd_phase;
    reg converged;

    initial begin
        // Initialize test values (unsorted)
        init_values[0] = 8'd42;
        init_values[1] = 8'd17;
        init_values[2] = 8'd93;
        init_values[3] = 8'd8;
        init_values[4] = 8'd55;
        init_values[5] = 8'd71;
        init_values[6] = 8'd23;
        init_values[7] = 8'd64;

        $display("");
        $display("================================================================");
        $display("      ZIT LINE TESTBENCH - Topology is Program (1D)");
        $display("================================================================");
        $display("");
        $display("   \"Same kernel + different wiring = different behavior\"");
        $display("");
        $display("   Kernel: if (neighbor > me) swap");
        $display("   Wiring: 1D line with odd-even phases");
        $display("   Result: Sorted array");
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

        for (i = 0; i < NUM_NODES; i = i + 1) begin
            seed_addr = i[2:0];
            seed_data = init_values[i];
            seed_write = 1;
            #(CLK_PERIOD);
            seed_write = 0;
            #(CLK_PERIOD);
        end

        $display("  Initial: [%3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d]",
            init_values[0], init_values[1], init_values[2], init_values[3],
            init_values[4], init_values[5], init_values[6], init_values[7]);
        $display("");

        // =====================================================================
        // TEST: Run Line Fabric
        // =====================================================================

        $display("TEST: Running 1D Line Fabric (Bubble Sort)");
        $display("----------------------------------------------------------------");

        enable = 1;
        cycle_count = 0;

        // Track consecutive all-resonant cycles (need both even and odd phases)
        prev_all_resonant = 0;
        converged = 0;

        // Run until both phases are resonant in sequence or max cycles
        while (!converged && cycle_count < 20) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Extract current values from all_states
            $display("  Cycle %2d [ran %s]: [%3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d]  res=%b",
                cycle_count,
                odd_phase ? "even" : "odd ",  // Inverted because toggle happened
                all_states[7:0],
                all_states[15:8],
                all_states[23:16],
                all_states[31:24],
                all_states[39:32],
                all_states[47:40],
                all_states[55:48],
                all_states[63:56],
                resonance_map);

            // Converged when two consecutive cycles (even + odd) are both all-resonant
            if (all_resonant && prev_all_resonant && (odd_phase != prev_odd_phase)) begin
                converged = 1;
            end
            prev_all_resonant = all_resonant;
            prev_odd_phase = odd_phase;
        end

        enable = 0;
        #(CLK_PERIOD * 10);

        // =====================================================================
        // RESULTS
        // =====================================================================

        $display("");
        $display("================================================================");
        if (converged) begin
            $display("[PASS] SORTED in %0d cycles", cycle_count);
            $display("  Final: [%3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d]",
                all_states[7:0], all_states[15:8], all_states[23:16], all_states[31:24],
                all_states[39:32], all_states[47:40], all_states[55:48], all_states[63:56]);
        end else begin
            $display("[FAIL] DID NOT CONVERGE after %0d cycles", cycle_count);
        end
        $display("================================================================");
        $display("");
        $display("  THE KEY INSIGHT:");
        $display("  -----------------");
        $display("  The frozen shape (comparator) NEVER CHANGED.");
        $display("  The wiring (1D line + odd/even phases) IS the algorithm.");
        $display("");
        $display("  Same kernel + 1D line = Bubble Sort");
        $display("  Same kernel + 2D grid = ???");
        $display("  Same kernel + 3D cube = ???");
        $display("");
        $display("  TOPOLOGY IS PROGRAM.");
        $display("");

        $finish;
    end

    // =========================================================================
    // WAVEFORM DUMP
    // =========================================================================

    initial begin
        $dumpfile("zit_line_tb.vcd");
        $dumpvars(0, zit_line_tb);
    end

endmodule
