// =============================================================================
// NGP CORE - Neural Geometric Processor v2
// =============================================================================
//
// The NGP is a shape-native processor that resonates with input rather than
// executing instructions. It recognizes patterns through XOR interference.
//
// Architecture:
//   - 1 Resonance Register (512 bits)
//   - 1 Zit Detector (recognition circuit)
//   - 1 Shape Fabric (frozen shape execution)
//   - Shape Selection via Hamming distance bands
//
// Key Metrics:
//   - ~53K gates total
//   - 512-bit resonance width
//   - 32-64 Tbits/sec theoretical throughput
//   - Single-cycle recognition
//
// "Geometry is computation. The shapes resonate with input."
// =============================================================================

`timescale 1ns / 1ps


// =============================================================================
// NGP_CORE - Top-level processor
// =============================================================================

module ngp_core #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10,
    parameter NUM_SHAPE_BANDS = 4
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input interface
    input  wire                    valid_in,
    input  wire [WIDTH-1:0]        signature_in,   // Input signature

    // Configuration
    input  wire [THRESH_BITS-1:0]  theta,          // Zit threshold
    input  wire [THRESH_BITS-1:0]  band_thresholds [NUM_SHAPE_BANDS-1:0], // Shape selection bands

    // Output interface
    output wire                    valid_out,
    output wire                    zit,            // Resonance detected
    output wire [THRESH_BITS-1:0]  hamming,        // Hamming distance
    output wire [1:0]              shape_band,     // Selected shape band (0-3)
    output wire [WIDTH-1:0]        result,         // Shape computation result

    // Debug outputs
    output wire [WIDTH-1:0]        S_debug         // Current resonance state
);

    // =========================================================================
    // Resonance State Register
    // =========================================================================

    reg [WIDTH-1:0] S;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            S <= {WIDTH{1'b0}};
        end else if (valid_in) begin
            S <= S ^ signature_in;  // XOR accumulation
        end
    end

    assign S_debug = S;

    // =========================================================================
    // Zit Detector
    // =========================================================================

    wire [WIDTH-1:0] diff;
    assign diff = S ^ signature_in;

    // Popcount tree
    popcount_512 u_popcount (
        .in(diff),
        .count(hamming)
    );

    // Zit activation
    assign zit = (hamming < theta);

    // =========================================================================
    // Shape Band Selection
    // =========================================================================
    //
    // The hamming distance determines which "band" of shapes to activate.
    // Lower distance = higher confidence = more refined shapes.
    //
    // Band 0: hamming < band_thresholds[0] (highest confidence)
    // Band 1: band_thresholds[0] <= hamming < band_thresholds[1]
    // Band 2: band_thresholds[1] <= hamming < band_thresholds[2]
    // Band 3: hamming >= band_thresholds[2] (fallback)
    // =========================================================================

    reg [1:0] shape_band_reg;

    always @(*) begin
        if (hamming < band_thresholds[0])
            shape_band_reg = 2'd0;
        else if (hamming < band_thresholds[1])
            shape_band_reg = 2'd1;
        else if (hamming < band_thresholds[2])
            shape_band_reg = 2'd2;
        else
            shape_band_reg = 2'd3;
    end

    assign shape_band = shape_band_reg;

    // =========================================================================
    // Shape Fabric - Frozen Computation
    // =========================================================================
    //
    // The shape fabric executes all frozen shapes in parallel.
    // The shape_band selects which result to output.
    //
    // For now, we implement basic operations. Full fabric is extensible.
    // =========================================================================

    // Logic operations on diff (the interference pattern)
    wire [WIDTH-1:0] logic_result;

    logic_fabric #(.WIDTH(WIDTH)) u_logic (
        .a(S),
        .b(signature_in),
        .shape_sel(3'd0),  // XOR by default
        .y(logic_result)
    );

    // Output selection based on band
    // Band 0-1: Return the interference pattern (diff)
    // Band 2-3: Return logic result
    reg [WIDTH-1:0] result_mux;

    always @(*) begin
        case (shape_band_reg)
            2'd0: result_mux = diff;        // Direct interference
            2'd1: result_mux = diff;        // Direct interference
            2'd2: result_mux = logic_result; // Processed
            2'd3: result_mux = {WIDTH{1'b0}}; // No match - zero output
            default: result_mux = {WIDTH{1'b0}};
        endcase
    end

    assign result = result_mux;

    // =========================================================================
    // Output Valid
    // =========================================================================

    // Pipeline the valid signal
    reg valid_out_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_out_reg <= 1'b0;
        else
            valid_out_reg <= valid_in;
    end

    assign valid_out = valid_out_reg;

endmodule


// =============================================================================
// NGP_SIMPLE - Minimal NGP for embedded applications
// =============================================================================
//
// A simplified NGP with just:
//   - Resonance register
//   - Zit detector
//   - No shape fabric (just outputs zit/hamming)
//
// Ideal for:
//   - CubeSat missions
//   - Resource-constrained devices
//   - Pure recognition (no computation)
// =============================================================================

module ngp_simple #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10,
    parameter DEFAULT_THETA = 64
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input
    input  wire                    valid_in,
    input  wire [WIDTH-1:0]        signature_in,

    // Output
    output wire                    zit,
    output wire [THRESH_BITS-1:0]  hamming
);

    // Resonance state
    reg [WIDTH-1:0] S;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            S <= {WIDTH{1'b0}};
        else if (valid_in)
            S <= S ^ signature_in;
    end

    // Zit detection (combinational)
    wire [WIDTH-1:0] diff;
    assign diff = S ^ signature_in;

    popcount_512 u_popcount (
        .in(diff),
        .count(hamming)
    );

    assign zit = (hamming < DEFAULT_THETA[THRESH_BITS-1:0]);

endmodule


// =============================================================================
// NGP_ARRAY - Array of NGP cores for parallel processing
// =============================================================================
//
// Multiple NGP cores, each with independent resonance state.
// Useful for:
//   - Multi-class recognition
//   - Ensemble methods
//   - Parallel pattern matching
// =============================================================================

module ngp_array #(
    parameter WIDTH = 512,
    parameter NUM_CORES = 4,
    parameter THRESH_BITS = 10
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input (broadcast to all cores)
    input  wire                    valid_in,
    input  wire [WIDTH-1:0]        signature_in,

    // Per-core thresholds
    input  wire [THRESH_BITS-1:0]  theta [NUM_CORES-1:0],

    // Outputs (from all cores)
    output wire [NUM_CORES-1:0]    zit_array,
    output wire [THRESH_BITS-1:0]  hamming_array [NUM_CORES-1:0],

    // Best match
    output wire [$clog2(NUM_CORES)-1:0] best_core,
    output wire                         any_match
);

    // Instantiate cores
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i = i + 1) begin : cores
            ngp_simple #(
                .WIDTH(WIDTH),
                .THRESH_BITS(THRESH_BITS),
                .DEFAULT_THETA(64)
            ) u_core (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .signature_in(signature_in),
                .zit(zit_array[i]),
                .hamming(hamming_array[i])
            );
        end
    endgenerate

    // Find best match (lowest hamming)
    reg [$clog2(NUM_CORES)-1:0] best_idx;
    reg [THRESH_BITS-1:0] best_hamming;

    integer j;
    always @(*) begin
        best_idx = 0;
        best_hamming = hamming_array[0];

        for (j = 1; j < NUM_CORES; j = j + 1) begin
            if (hamming_array[j] < best_hamming) begin
                best_idx = j[$clog2(NUM_CORES)-1:0];
                best_hamming = hamming_array[j];
            end
        end
    end

    assign best_core = best_idx;
    assign any_match = |zit_array;

endmodule


// =============================================================================
// NGP_WITH_ROUTING - NGP with shape routing table
// =============================================================================
//
// Full NGP with configurable routing:
//   - Zit detection
//   - Hamming-based shape selection
//   - External routing table interface
// =============================================================================

module ngp_with_routing #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10,
    parameter ROUTE_BITS = 8
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input
    input  wire                    valid_in,
    input  wire [WIDTH-1:0]        signature_in,

    // Configuration
    input  wire [THRESH_BITS-1:0]  theta,

    // Routing interface
    output wire [THRESH_BITS-1:0]  hamming_out,     // To routing table
    input  wire [ROUTE_BITS-1:0]   route_in,        // From routing table

    // Output
    output wire                    zit,
    output wire [ROUTE_BITS-1:0]   shape_id         // Selected shape
);

    // Resonance state
    reg [WIDTH-1:0] S;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            S <= {WIDTH{1'b0}};
        else if (valid_in)
            S <= S ^ signature_in;
    end

    // Zit detection
    wire [WIDTH-1:0] diff;
    assign diff = S ^ signature_in;

    popcount_512 u_popcount (
        .in(diff),
        .count(hamming_out)
    );

    assign zit = (hamming_out < theta);

    // Shape ID from routing table
    assign shape_id = zit ? route_in : {ROUTE_BITS{1'b0}};

endmodule
