// =============================================================================
// ZIT DETECTOR - Core Recognition Circuit for the Neural Geometric Processor
// =============================================================================
//
// The Zit detector determines whether an input signature resonates with the
// system's accumulated resonance state.
//
// Equation: Zit = popcount(S XOR vx) < theta
//
// Components:
//   1. XOR array (512 parallel XOR gates)
//   2. Popcount tree (9-level binary adder tree)
//   3. Comparator (10-bit less-than comparison)
//
// "The Zit fires when structure recognizes structure."
// =============================================================================

`timescale 1ns / 1ps

module zit_detector #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10
)(
    input  wire [WIDTH-1:0]       S,        // Resonance state
    input  wire [WIDTH-1:0]       vx,       // Input signature
    input  wire [THRESH_BITS-1:0] theta,    // Activation threshold
    output wire                   zit,      // Activation signal (1 = resonates)
    output wire [THRESH_BITS-1:0] hamming   // Hamming distance (for shape selection)
);

    // =========================================================================
    // Stage 1: XOR Array - Compute bitwise difference
    // =========================================================================
    wire [WIDTH-1:0] diff;
    assign diff = S ^ vx;

    // =========================================================================
    // Stage 2: Popcount Tree - Count differing bits
    // =========================================================================
    popcount_512 u_popcount (
        .in(diff),
        .count(hamming)
    );

    // =========================================================================
    // Stage 3: Comparator - Check against threshold
    // =========================================================================
    assign zit = (hamming < theta);

endmodule


// =============================================================================
// POPCOUNT_512 - Population count for 512-bit vector
// =============================================================================
//
// Implements a parallel binary adder tree to count set bits.
// Total latency: 9 gate levels (log2(512))
//
// Level 0: 512 single bits
// Level 1: 256 2-bit sums
// Level 2: 128 3-bit sums
// Level 3:  64 4-bit sums
// Level 4:  32 5-bit sums
// Level 5:  16 6-bit sums
// Level 6:   8 7-bit sums
// Level 7:   4 8-bit sums
// Level 8:   2 9-bit sums
// Level 9:   1 10-bit sum (final)
// =============================================================================

module popcount_512 (
    input  wire [511:0] in,
    output wire [9:0]   count
);

    // Level 1: 256 pairs -> 256 2-bit sums
    wire [1:0] l1 [255:0];
    genvar i;
    generate
        for (i = 0; i < 256; i = i + 1) begin : level1
            assign l1[i] = {1'b0, in[2*i]} + {1'b0, in[2*i + 1]};
        end
    endgenerate

    // Level 2: 128 pairs -> 128 3-bit sums
    wire [2:0] l2 [127:0];
    generate
        for (i = 0; i < 128; i = i + 1) begin : level2
            assign l2[i] = {1'b0, l1[2*i]} + {1'b0, l1[2*i + 1]};
        end
    endgenerate

    // Level 3: 64 pairs -> 64 4-bit sums
    wire [3:0] l3 [63:0];
    generate
        for (i = 0; i < 64; i = i + 1) begin : level3
            assign l3[i] = {1'b0, l2[2*i]} + {1'b0, l2[2*i + 1]};
        end
    endgenerate

    // Level 4: 32 pairs -> 32 5-bit sums
    wire [4:0] l4 [31:0];
    generate
        for (i = 0; i < 32; i = i + 1) begin : level4
            assign l4[i] = {1'b0, l3[2*i]} + {1'b0, l3[2*i + 1]};
        end
    endgenerate

    // Level 5: 16 pairs -> 16 6-bit sums
    wire [5:0] l5 [15:0];
    generate
        for (i = 0; i < 16; i = i + 1) begin : level5
            assign l5[i] = {1'b0, l4[2*i]} + {1'b0, l4[2*i + 1]};
        end
    endgenerate

    // Level 6: 8 pairs -> 8 7-bit sums
    wire [6:0] l6 [7:0];
    generate
        for (i = 0; i < 8; i = i + 1) begin : level6
            assign l6[i] = {1'b0, l5[2*i]} + {1'b0, l5[2*i + 1]};
        end
    endgenerate

    // Level 7: 4 pairs -> 4 8-bit sums
    wire [7:0] l7 [3:0];
    generate
        for (i = 0; i < 4; i = i + 1) begin : level7
            assign l7[i] = {1'b0, l6[2*i]} + {1'b0, l6[2*i + 1]};
        end
    endgenerate

    // Level 8: 2 pairs -> 2 9-bit sums
    wire [8:0] l8 [1:0];
    generate
        for (i = 0; i < 2; i = i + 1) begin : level8
            assign l8[i] = {1'b0, l7[2*i]} + {1'b0, l7[2*i + 1]};
        end
    endgenerate

    // Level 9: Final 10-bit sum
    assign count = {1'b0, l8[0]} + {1'b0, l8[1]};

endmodule


// =============================================================================
// RESONANCE_REGISTER - XOR Accumulation State
// =============================================================================
//
// The resonance register accumulates inputs via XOR.
// S' = S XOR input
//
// This is the "memory" of the system - but it's not storage,
// it's entanglement. Every input becomes part of the resonance.
// =============================================================================

module resonance_register #(
    parameter WIDTH = 512
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                enable,      // Write enable
    input  wire [WIDTH-1:0]    data_in,     // Input to entangle
    output reg  [WIDTH-1:0]    S            // Resonance state
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            S <= {WIDTH{1'b0}};
        end else if (enable) begin
            S <= S ^ data_in;  // XOR accumulation
        end
    end

endmodule


// =============================================================================
// ZIT_UNIT - Complete Zit Detection Unit with Resonance
// =============================================================================
//
// Combines:
//   - Resonance register (state)
//   - Zit detector (recognition)
//   - Resonance update logic
//
// This is the heart of the NGP.
// =============================================================================

module zit_unit #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10,
    parameter DEFAULT_THETA = 64
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,         // Process input
    input  wire [WIDTH-1:0]        vx,             // Input signature
    input  wire [THRESH_BITS-1:0]  theta,          // Threshold (or use default)
    input  wire                    use_default,    // Use DEFAULT_THETA

    output wire                    zit,            // Zit fires
    output wire [THRESH_BITS-1:0]  hamming,        // Distance for routing
    output wire [WIDTH-1:0]        S_out           // Current state (for debug)
);

    // Internal signals
    wire [WIDTH-1:0] S;
    wire [THRESH_BITS-1:0] theta_mux;

    // Resonance register
    resonance_register #(.WIDTH(WIDTH)) u_resonance (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_in(vx),
        .S(S)
    );

    // Threshold selection
    assign theta_mux = use_default ? DEFAULT_THETA[THRESH_BITS-1:0] : theta;

    // Zit detector (combinational)
    zit_detector #(
        .WIDTH(WIDTH),
        .THRESH_BITS(THRESH_BITS)
    ) u_zit (
        .S(S),
        .vx(vx),
        .theta(theta_mux),
        .zit(zit),
        .hamming(hamming)
    );

    // Debug output
    assign S_out = S;

endmodule


// =============================================================================
// ZIT_TESTBENCH - Verification testbench
// =============================================================================

`ifdef SIMULATION

module zit_testbench;

    // Parameters
    parameter WIDTH = 512;
    parameter THRESH_BITS = 10;
    parameter CLK_PERIOD = 10;

    // Signals
    reg                       clk;
    reg                       rst_n;
    reg                       enable;
    reg  [WIDTH-1:0]          vx;
    reg  [THRESH_BITS-1:0]    theta;
    wire                      zit;
    wire [THRESH_BITS-1:0]    hamming;
    wire [WIDTH-1:0]          S_out;

    // DUT
    zit_unit #(
        .WIDTH(WIDTH),
        .THRESH_BITS(THRESH_BITS),
        .DEFAULT_THETA(64)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .vx(vx),
        .theta(theta),
        .use_default(1'b1),
        .zit(zit),
        .hamming(hamming),
        .S_out(S_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test sequence
    initial begin
        $display("=== ZIT DETECTOR TESTBENCH ===");

        // Reset
        rst_n = 0;
        enable = 0;
        vx = {WIDTH{1'b0}};
        theta = 64;
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);

        // Test 1: Input with all zeros (should fire - hamming = 0)
        $display("Test 1: Zero input, zero state");
        vx = {WIDTH{1'b0}};
        enable = 1;
        #(CLK_PERIOD);
        $display("  hamming = %d, zit = %b (expected: 0, 1)", hamming, zit);
        enable = 0;
        #(CLK_PERIOD);

        // Test 2: Input same pattern (should fire)
        $display("Test 2: Same pattern (perfect resonance)");
        vx = 512'hDEADBEEF_CAFEBABE_12345678_FEEDFACE_DEADBEEF_CAFEBABE_12345678_FEEDFACE_DEADBEEF_CAFEBABE_12345678_FEEDFACE_DEADBEEF_CAFEBABE_12345678_FEEDFACE;
        enable = 1;
        #(CLK_PERIOD);
        enable = 0;
        #(CLK_PERIOD);
        // Query with same pattern
        $display("  After entangle, query same: hamming = %d, zit = %b", hamming, zit);

        // Test 3: Input opposite pattern (should not fire)
        $display("Test 3: Opposite pattern (dissonance)");
        vx = ~vx;
        #(CLK_PERIOD);
        $display("  Opposite query: hamming = %d, zit = %b (expected: high, 0)", hamming, zit);

        // Test 4: Partial match
        $display("Test 4: Partial match (50%% bits)");
        vx = {256'hFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF, 256'h0};
        #(CLK_PERIOD);
        $display("  Half match: hamming = %d, zit = %b", hamming, zit);

        $display("=== TESTBENCH COMPLETE ===");
        $finish;
    end

endmodule

`endif
