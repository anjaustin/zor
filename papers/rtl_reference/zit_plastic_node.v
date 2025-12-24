// =============================================================================
// ZIT_PLASTIC_NODE.v - A Node That Can Learn Its Topology
// =============================================================================
//
// "The topology IS the self. And the self can GROW."
//
// This extends zit_cube_node with topological plasticity:
// - Tracks frustration (how often this node fails to resonate)
// - Can rewire to different neighbors when frustrated
// - Keeps connections that reduce frustration
//
// The hypothesis: Topology can learn from frustration alone.
// No gradients. No global loss. Just local dissatisfaction.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_plastic_node #(
    parameter STATE_WIDTH = 8,
    parameter NODE_ID = 0,
    parameter NUM_NODES = 64,
    parameter FRUSTRATION_BITS = 8,
    parameter REWIRE_THRESHOLD = 16,
    parameter DECAY_SHIFT = 1,           // Frustration decay rate
    parameter LFSR_SEED = 16'hACE1       // Unique per node
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Phase control
    input  wire [2:0]                   phase,
    input  wire [1:0]                   sub_phase,
    input  wire                         phase_strobe,
    input  wire                         cycle_complete,

    // Configuration
    input  wire [STATE_WIDTH-1:0]       init_value,
    input  wire                         init_load,

    // Initial topology (from generate block)
    input  wire [5:0]                   init_neighbor_px,
    input  wire [5:0]                   init_neighbor_mx,
    input  wire [5:0]                   init_neighbor_py,
    input  wire [5:0]                   init_neighbor_my,
    input  wire [5:0]                   init_neighbor_pz,
    input  wire [5:0]                   init_neighbor_mz,

    // All node values (for dynamic neighbor selection)
    input  wire [64*STATE_WIDTH-1:0]    all_node_values,
    input  wire [63:0]                  all_node_valid,

    // Outputs
    output wire                         tx_valid,
    output wire [STATE_WIDTH-1:0]       tx_data,
    output wire                         resonance,
    output wire [STATE_WIDTH-1:0]       state_out,

    // Plasticity outputs (for observation)
    output wire [FRUSTRATION_BITS-1:0]  frustration_out,
    output wire                         rewiring,
    output wire [5:0]                   neighbor_px_out,
    output wire [5:0]                   neighbor_mx_out,
    output wire [5:0]                   neighbor_py_out,
    output wire [5:0]                   neighbor_my_out,
    output wire [5:0]                   neighbor_pz_out,
    output wire [5:0]                   neighbor_mz_out
);

    // =========================================================================
    // INTERNAL STATE
    // =========================================================================

    reg [STATE_WIDTH-1:0] S;

    // Dynamic neighbor list
    reg [5:0] neighbor_idx [5:0];  // 6 neighbors, each is a 6-bit index

    // Neighbor data latches
    reg [STATE_WIDTH-1:0] neighbor_data [5:0];
    reg [5:0] neighbor_received;

    reg [STATE_WIDTH-1:0] tx_data_reg;
    reg tx_valid_reg;
    reg resonance_reg;
    reg react_done;
    reg has_participated;

    // Plasticity state
    reg [FRUSTRATION_BITS-1:0] frustration_count;
    reg rewiring_active;
    reg [5:0] candidate_neighbor;
    reg [5:0] old_neighbor;
    reg [2:0] rewire_direction;   // Which of 6 neighbors to rewire
    reg [3:0] eval_cycles;        // Cycles to evaluate new connection
    reg [FRUSTRATION_BITS-1:0] pre_rewire_frustration;

    // LFSR for random neighbor selection
    reg [15:0] lfsr;

    // Sub-phase encoding
    localparam SUB_LISTEN = 2'd0;
    localparam SUB_REACT  = 2'd1;
    localparam SUB_SHOVE  = 2'd2;

    // =========================================================================
    // LFSR for Randomness
    // =========================================================================

    wire [15:0] lfsr_next = {lfsr[14:0], lfsr[15] ^ lfsr[14] ^ lfsr[12] ^ lfsr[3]};

    // =========================================================================
    // NEIGHBOR VALUE EXTRACTION
    // =========================================================================

    // Extract value for a specific node index from the flattened array
    function [STATE_WIDTH-1:0] get_node_value;
        input [5:0] idx;
        integer bit_offset;
        begin
            bit_offset = idx * STATE_WIDTH;
            get_node_value = all_node_values[bit_offset +: STATE_WIDTH];
        end
    endfunction

    // =========================================================================
    // FROZEN SHAPE: 3D COMPARATOR (same as original)
    // =========================================================================

    wire [STATE_WIDTH-1:0] active_neighbor_data = neighbor_data[phase[2:0] < 6 ? phase[2:0] : 0];
    wire active_received = neighbor_received[phase[2:0] < 6 ? phase[2:0] : 0];

    // Positive directions: even phase numbers (0, 2, 4)
    wire positive_dir = (phase[0] == 1'b0);

    // The comparison - the frozen shape
    wire should_swap = active_received &&
                       (positive_dir ? (S > active_neighbor_data) : (active_neighbor_data > S));

    // =========================================================================
    // PLASTICITY LOGIC
    // =========================================================================

    // Pick a random node to try as new neighbor (excluding self)
    wire [5:0] random_node_idx = (lfsr[5:0] == NODE_ID[5:0]) ? lfsr[5:0] + 1 : lfsr[5:0];

    // Find most frustrated direction (direction with most swap failures)
    // For simplicity, cycle through directions
    wire [2:0] next_rewire_dir = rewire_direction + 1;

    // =========================================================================
    // STATE MACHINE
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            S <= 0;
            neighbor_idx[0] <= 0;
            neighbor_idx[1] <= 0;
            neighbor_idx[2] <= 0;
            neighbor_idx[3] <= 0;
            neighbor_idx[4] <= 0;
            neighbor_idx[5] <= 0;
            neighbor_data[0] <= 0;
            neighbor_data[1] <= 0;
            neighbor_data[2] <= 0;
            neighbor_data[3] <= 0;
            neighbor_data[4] <= 0;
            neighbor_data[5] <= 0;
            neighbor_received <= 0;
            tx_data_reg <= 0;
            tx_valid_reg <= 0;
            resonance_reg <= 0;
            react_done <= 0;
            has_participated <= 0;
            frustration_count <= 0;
            rewiring_active <= 0;
            candidate_neighbor <= 0;
            old_neighbor <= 0;
            rewire_direction <= 0;
            eval_cycles <= 0;
            pre_rewire_frustration <= 0;
            lfsr <= LFSR_SEED ^ NODE_ID;
        end else begin
            // Initialize value
            if (init_load) begin
                S <= init_value;
                // Initialize topology from generate block
                neighbor_idx[0] <= init_neighbor_px;
                neighbor_idx[1] <= init_neighbor_mx;
                neighbor_idx[2] <= init_neighbor_py;
                neighbor_idx[3] <= init_neighbor_my;
                neighbor_idx[4] <= init_neighbor_pz;
                neighbor_idx[5] <= init_neighbor_mz;
            end

            // Advance LFSR every clock
            lfsr <= lfsr_next;

            case (sub_phase)

                SUB_LISTEN: begin
                    tx_valid_reg <= 0;
                    react_done <= 0;

                    // Latch neighbor values based on dynamic neighbor list
                    if (all_node_valid[neighbor_idx[0]]) begin
                        neighbor_data[0] <= get_node_value(neighbor_idx[0]);
                        neighbor_received[0] <= 1;
                    end
                    if (all_node_valid[neighbor_idx[1]]) begin
                        neighbor_data[1] <= get_node_value(neighbor_idx[1]);
                        neighbor_received[1] <= 1;
                    end
                    if (all_node_valid[neighbor_idx[2]]) begin
                        neighbor_data[2] <= get_node_value(neighbor_idx[2]);
                        neighbor_received[2] <= 1;
                    end
                    if (all_node_valid[neighbor_idx[3]]) begin
                        neighbor_data[3] <= get_node_value(neighbor_idx[3]);
                        neighbor_received[3] <= 1;
                    end
                    if (all_node_valid[neighbor_idx[4]]) begin
                        neighbor_data[4] <= get_node_value(neighbor_idx[4]);
                        neighbor_received[4] <= 1;
                    end
                    if (all_node_valid[neighbor_idx[5]]) begin
                        neighbor_data[5] <= get_node_value(neighbor_idx[5]);
                        neighbor_received[5] <= 1;
                    end
                end

                SUB_REACT: begin
                    if (!react_done) begin
                        react_done <= 1;

                        // Compute resonance
                        resonance_reg <= has_participated && active_received && ~should_swap;

                        if (should_swap) begin
                            S <= active_neighbor_data;
                        end
                    end
                end

                SUB_SHOVE: begin
                    tx_valid_reg <= 1;
                    tx_data_reg <= S;
                    has_participated <= 1;
                    neighbor_received <= 0;
                end

            endcase

            // =====================================================================
            // PLASTICITY LOGIC - runs on cycle completion
            // =====================================================================
            if (cycle_complete) begin
                if (!rewiring_active) begin
                    // Track frustration
                    if (!resonance_reg && has_participated) begin
                        // Frustrated - increment counter
                        if (frustration_count < {FRUSTRATION_BITS{1'b1}}) begin
                            frustration_count <= frustration_count + 1;
                        end
                    end else begin
                        // Resonant - decay frustration
                        frustration_count <= frustration_count >> DECAY_SHIFT;
                    end

                    // Check if we should try rewiring
                    if (frustration_count >= REWIRE_THRESHOLD) begin
                        rewiring_active <= 1;
                        old_neighbor <= neighbor_idx[rewire_direction];
                        candidate_neighbor <= random_node_idx;
                        pre_rewire_frustration <= frustration_count;
                        eval_cycles <= 0;

                        // Try the new neighbor
                        neighbor_idx[rewire_direction] <= random_node_idx;

                        // Reset frustration for fair evaluation
                        frustration_count <= 0;
                    end
                end else begin
                    // We're evaluating a new connection
                    eval_cycles <= eval_cycles + 1;

                    // Track frustration during evaluation
                    if (!resonance_reg && has_participated) begin
                        if (frustration_count < {FRUSTRATION_BITS{1'b1}}) begin
                            frustration_count <= frustration_count + 1;
                        end
                    end

                    // After 8 cycles, decide whether to keep new connection
                    if (eval_cycles >= 8) begin
                        if (frustration_count >= pre_rewire_frustration) begin
                            // New connection is worse or same - revert
                            neighbor_idx[rewire_direction] <= old_neighbor;
                        end
                        // else: keep new connection (it's better)

                        // Move to next direction for future rewiring attempts
                        rewire_direction <= (rewire_direction == 5) ? 0 : rewire_direction + 1;
                        rewiring_active <= 0;
                        frustration_count <= 0;
                    end
                end
            end
        end
    end

    // =========================================================================
    // OUTPUTS
    // =========================================================================

    assign tx_valid = tx_valid_reg;
    assign tx_data = tx_data_reg;
    assign resonance = resonance_reg;
    assign state_out = S;
    assign frustration_out = frustration_count;
    assign rewiring = rewiring_active;
    assign neighbor_px_out = neighbor_idx[0];
    assign neighbor_mx_out = neighbor_idx[1];
    assign neighbor_py_out = neighbor_idx[2];
    assign neighbor_my_out = neighbor_idx[3];
    assign neighbor_pz_out = neighbor_idx[4];
    assign neighbor_mz_out = neighbor_idx[5];

endmodule
