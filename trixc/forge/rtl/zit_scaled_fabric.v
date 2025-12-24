// =============================================================================
// ZIT_SCALED_FABRIC.v - Parameterized N^3 Self-Organizing Fabric
// =============================================================================
//
// "Does emergence scale? Let's find out."
//
// EXPERIMENT C: The 8x8x8 challenge.
// - 512 nodes instead of 64
// - Same plasticity mechanism
// - Same physics, bigger space
//
// Question: Does the topology learn at scale?
//
// =============================================================================

`timescale 1ns / 1ps

module zit_scaled_fabric #(
    parameter CUBE_SIZE = 8,                        // 8x8x8 = 512 nodes
    parameter NUM_NODES = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE,
    parameter STATE_WIDTH = 8,
    parameter FRUSTRATION_BITS = 8,
    parameter REWIRE_THRESHOLD = 16,
    parameter ADDR_WIDTH = $clog2(NUM_NODES)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Phase control
    input  wire [2:0]                   phase,
    input  wire [1:0]                   sub_phase,
    input  wire                         phase_strobe,
    input  wire                         cycle_complete,

    // Seed interface
    input  wire [STATE_WIDTH-1:0]       seed_data,
    input  wire [ADDR_WIDTH-1:0]        seed_addr,
    input  wire                         seed_write,

    // Status
    output wire                         all_resonant,
    output wire [NUM_NODES-1:0]         resonance_map,

    // Plasticity metrics
    output wire [15:0]                  global_frustration,
    output wire [15:0]                  resonant_count,
    output wire [NUM_NODES-1:0]         rewiring_map
);

    // =========================================================================
    // NODE STATE ARRAYS
    // =========================================================================

    wire tx_valid [0:NUM_NODES-1];
    wire [STATE_WIDTH-1:0] tx_data [0:NUM_NODES-1];
    wire node_resonance [0:NUM_NODES-1];
    wire [STATE_WIDTH-1:0] node_state [0:NUM_NODES-1];
    wire [FRUSTRATION_BITS-1:0] node_frustration [0:NUM_NODES-1];
    wire node_rewiring [0:NUM_NODES-1];

    // For scaled fabric, we use a different approach for neighbor values
    // Instead of full broadcast, each node samples neighbors on demand

    // =========================================================================
    // COORDINATE CALCULATION FUNCTIONS
    // =========================================================================

    function [ADDR_WIDTH-1:0] calc_neighbor;
        input [ADDR_WIDTH-1:0] node_id;
        input [2:0] direction;  // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
        reg [7:0] x, y, z;
        reg [7:0] new_x, new_y, new_z;
        begin
            x = node_id % CUBE_SIZE;
            y = (node_id / CUBE_SIZE) % CUBE_SIZE;
            z = node_id / (CUBE_SIZE * CUBE_SIZE);

            case (direction)
                3'd0: begin new_x = (x + 1) % CUBE_SIZE; new_y = y; new_z = z; end
                3'd1: begin new_x = (x + CUBE_SIZE - 1) % CUBE_SIZE; new_y = y; new_z = z; end
                3'd2: begin new_x = x; new_y = (y + 1) % CUBE_SIZE; new_z = z; end
                3'd3: begin new_x = x; new_y = (y + CUBE_SIZE - 1) % CUBE_SIZE; new_z = z; end
                3'd4: begin new_x = x; new_y = y; new_z = (z + 1) % CUBE_SIZE; end
                3'd5: begin new_x = x; new_y = y; new_z = (z + CUBE_SIZE - 1) % CUBE_SIZE; end
                default: begin new_x = x; new_y = y; new_z = z; end
            endcase

            calc_neighbor = new_x + new_y * CUBE_SIZE + new_z * CUBE_SIZE * CUBE_SIZE;
        end
    endfunction

    // =========================================================================
    // FLATTEN NODE VALUES FOR BROADCAST (like 4x4x4)
    // =========================================================================

    wire [NUM_NODES*STATE_WIDTH-1:0] all_node_values_flat;
    wire [NUM_NODES-1:0] all_node_valid;

    genvar flatten_idx;
    generate
        for (flatten_idx = 0; flatten_idx < NUM_NODES; flatten_idx = flatten_idx + 1) begin : gen_flatten
            assign all_node_values_flat[flatten_idx*STATE_WIDTH +: STATE_WIDTH] = tx_data[flatten_idx];
            assign all_node_valid[flatten_idx] = tx_valid[flatten_idx];
        end
    endgenerate

    // =========================================================================
    // GENERATE PLASTIC NODES
    // =========================================================================

    genvar idx;
    generate
        for (idx = 0; idx < NUM_NODES; idx = idx + 1) begin : gen_nodes

            // Calculate x, y, z from linear index
            localparam [7:0] X = idx % CUBE_SIZE;
            localparam [7:0] Y = (idx / CUBE_SIZE) % CUBE_SIZE;
            localparam [7:0] Z = idx / (CUBE_SIZE * CUBE_SIZE);

            // Initial torus neighbors
            localparam [ADDR_WIDTH-1:0] INIT_PX = ((X + 1) % CUBE_SIZE) + (Y * CUBE_SIZE) + (Z * CUBE_SIZE * CUBE_SIZE);
            localparam [ADDR_WIDTH-1:0] INIT_MX = ((X + CUBE_SIZE - 1) % CUBE_SIZE) + (Y * CUBE_SIZE) + (Z * CUBE_SIZE * CUBE_SIZE);
            localparam [ADDR_WIDTH-1:0] INIT_PY = X + (((Y + 1) % CUBE_SIZE) * CUBE_SIZE) + (Z * CUBE_SIZE * CUBE_SIZE);
            localparam [ADDR_WIDTH-1:0] INIT_MY = X + (((Y + CUBE_SIZE - 1) % CUBE_SIZE) * CUBE_SIZE) + (Z * CUBE_SIZE * CUBE_SIZE);
            localparam [ADDR_WIDTH-1:0] INIT_PZ = X + (Y * CUBE_SIZE) + (((Z + 1) % CUBE_SIZE) * CUBE_SIZE * CUBE_SIZE);
            localparam [ADDR_WIDTH-1:0] INIT_MZ = X + (Y * CUBE_SIZE) + (((Z + CUBE_SIZE - 1) % CUBE_SIZE) * CUBE_SIZE * CUBE_SIZE);

            wire seed_select = seed_write && (seed_addr == idx);

            // Unique LFSR seed per node
            localparam [15:0] NODE_LFSR_SEED = 16'hBEEF ^ (idx * 16'h1337);

            zit_scaled_node #(
                .STATE_WIDTH(STATE_WIDTH),
                .NODE_ID(idx),
                .NUM_NODES(NUM_NODES),
                .ADDR_WIDTH(ADDR_WIDTH),
                .FRUSTRATION_BITS(FRUSTRATION_BITS),
                .REWIRE_THRESHOLD(REWIRE_THRESHOLD),
                .LFSR_SEED(NODE_LFSR_SEED)
            ) node_inst (
                .clk(clk),
                .rst_n(rst_n),

                .phase(phase),
                .sub_phase(sub_phase),
                .phase_strobe(phase_strobe),
                .cycle_complete(cycle_complete),

                .init_value(seed_data),
                .init_load(seed_select),

                .init_neighbor_px(INIT_PX),
                .init_neighbor_mx(INIT_MX),
                .init_neighbor_py(INIT_PY),
                .init_neighbor_my(INIT_MY),
                .init_neighbor_pz(INIT_PZ),
                .init_neighbor_mz(INIT_MZ),

                // All node values for dynamic neighbor access
                .all_node_values(all_node_values_flat),
                .all_node_valid(all_node_valid),

                .tx_valid(tx_valid[idx]),
                .tx_data(tx_data[idx]),
                .resonance(node_resonance[idx]),
                .state_out(node_state[idx]),
                .frustration_out(node_frustration[idx]),
                .rewiring(node_rewiring[idx])
            );

            assign resonance_map[idx] = node_resonance[idx];
            assign rewiring_map[idx] = node_rewiring[idx];
        end
    endgenerate

    // =========================================================================
    // GLOBAL METRICS
    // =========================================================================

    reg [15:0] frustration_sum;
    reg [15:0] resonance_sum;
    integer i;

    always @(*) begin
        frustration_sum = 0;
        resonance_sum = 0;
        for (i = 0; i < NUM_NODES; i = i + 1) begin
            frustration_sum = frustration_sum + node_frustration[i];
            resonance_sum = resonance_sum + node_resonance[i];
        end
    end

    assign all_resonant = &resonance_map;
    assign global_frustration = frustration_sum;
    assign resonant_count = resonance_sum;

endmodule


// =============================================================================
// ZIT_SCALED_NODE - Simplified plastic node for large fabrics
// =============================================================================

module zit_scaled_node #(
    parameter STATE_WIDTH = 8,
    parameter NODE_ID = 0,
    parameter NUM_NODES = 512,
    parameter ADDR_WIDTH = 9,
    parameter FRUSTRATION_BITS = 8,
    parameter REWIRE_THRESHOLD = 16,
    parameter DECAY_SHIFT = 1,
    parameter [15:0] LFSR_SEED = 16'hACE1
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire [2:0]                   phase,
    input  wire [1:0]                   sub_phase,
    input  wire                         phase_strobe,
    input  wire                         cycle_complete,

    input  wire [STATE_WIDTH-1:0]       init_value,
    input  wire                         init_load,

    // Initial neighbors
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_px,
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_mx,
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_py,
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_my,
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_pz,
    input  wire [ADDR_WIDTH-1:0]        init_neighbor_mz,

    // All node values for dynamic neighbor access
    input  wire [NUM_NODES*STATE_WIDTH-1:0] all_node_values,
    input  wire [NUM_NODES-1:0]             all_node_valid,

    output reg                          tx_valid,
    output reg  [STATE_WIDTH-1:0]       tx_data,
    output reg                          resonance,
    output wire [STATE_WIDTH-1:0]       state_out,
    output wire [FRUSTRATION_BITS-1:0]  frustration_out,
    output reg                          rewiring
);

    // =========================================================================
    // STATE REGISTERS
    // =========================================================================

    reg [STATE_WIDTH-1:0] state_reg;
    reg [STATE_WIDTH-1:0] accumulated;
    reg [2:0] rx_count;
    reg has_participated;

    // Frustration tracking
    reg [FRUSTRATION_BITS-1:0] frustration_count;

    // Neighbor indices (can be rewired)
    reg [ADDR_WIDTH-1:0] neighbor_idx [0:5];

    // LFSR for random rewiring
    reg [15:0] lfsr;

    // =========================================================================
    // DYNAMIC NEIGHBOR VALUE EXTRACTION
    // =========================================================================

    // Get values from dynamically selected neighbors
    wire [STATE_WIDTH-1:0] nb_val [0:5];
    assign nb_val[0] = all_node_values[neighbor_idx[0]*STATE_WIDTH +: STATE_WIDTH];
    assign nb_val[1] = all_node_values[neighbor_idx[1]*STATE_WIDTH +: STATE_WIDTH];
    assign nb_val[2] = all_node_values[neighbor_idx[2]*STATE_WIDTH +: STATE_WIDTH];
    assign nb_val[3] = all_node_values[neighbor_idx[3]*STATE_WIDTH +: STATE_WIDTH];
    assign nb_val[4] = all_node_values[neighbor_idx[4]*STATE_WIDTH +: STATE_WIDTH];
    assign nb_val[5] = all_node_values[neighbor_idx[5]*STATE_WIDTH +: STATE_WIDTH];

    // =========================================================================
    // FROZEN SHAPE: Median-like comparator
    // =========================================================================

    wire [STATE_WIDTH-1:0] median_result;
    wire [2:0] count_below;
    wire [2:0] count_above;

    assign count_below = (nb_val[0] < state_reg) + (nb_val[1] < state_reg) +
                         (nb_val[2] < state_reg) + (nb_val[3] < state_reg) +
                         (nb_val[4] < state_reg) + (nb_val[5] < state_reg);

    assign count_above = (nb_val[0] > state_reg) + (nb_val[1] > state_reg) +
                         (nb_val[2] > state_reg) + (nb_val[3] > state_reg) +
                         (nb_val[4] > state_reg) + (nb_val[5] > state_reg);

    assign median_result = (count_below > count_above) ? state_reg - 1 :
                           (count_above > count_below) ? state_reg + 1 :
                           state_reg;

    // =========================================================================
    // RESONANCE CHECK
    // =========================================================================

    wire resonance_check = (count_below == count_above) ||
                           (count_below <= 1 && count_above <= 1);

    // =========================================================================
    // MAIN STATE MACHINE
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= NODE_ID[STATE_WIDTH-1:0];
            tx_valid <= 0;
            tx_data <= 0;
            accumulated <= 0;
            rx_count <= 0;
            has_participated <= 0;
            resonance <= 0;
            frustration_count <= 0;
            rewiring <= 0;
            lfsr <= LFSR_SEED;

            // Initialize with torus neighbors
            neighbor_idx[0] <= init_neighbor_px;
            neighbor_idx[1] <= init_neighbor_mx;
            neighbor_idx[2] <= init_neighbor_py;
            neighbor_idx[3] <= init_neighbor_my;
            neighbor_idx[4] <= init_neighbor_pz;
            neighbor_idx[5] <= init_neighbor_mz;
        end else begin
            // Load initial value
            if (init_load) begin
                state_reg <= init_value;
            end

            // LFSR tick
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[14] ^ lfsr[12] ^ lfsr[3]};

            rewiring <= 0;

            if (phase_strobe) begin
                case (sub_phase)
                    2'd0: begin  // LISTEN
                        accumulated <= 0;
                        rx_count <= 0;
                        tx_valid <= 0;
                    end

                    2'd1: begin  // REACT
                        // Apply frozen shape
                        state_reg <= median_result;
                        resonance <= resonance_check;
                        has_participated <= 1;

                        // Update frustration
                        if (!resonance_check && has_participated) begin
                            if (frustration_count < {FRUSTRATION_BITS{1'b1}})
                                frustration_count <= frustration_count + 1;
                        end else begin
                            frustration_count <= frustration_count >> DECAY_SHIFT;
                        end

                        // Check for rewiring trigger
                        if (frustration_count >= REWIRE_THRESHOLD) begin
                            // Random rewire one direction
                            rewiring <= 1;
                            neighbor_idx[lfsr[2:0] % 6] <= lfsr[ADDR_WIDTH-1+3:3] % NUM_NODES;
                            frustration_count <= 0;
                        end
                    end

                    2'd2: begin  // SHOVE
                        tx_valid <= 1;
                        tx_data <= state_reg;
                    end

                    default: ;
                endcase
            end

            if (cycle_complete) begin
                tx_valid <= 0;
            end
        end
    end

    assign state_out = state_reg;
    assign frustration_out = frustration_count;

endmodule
