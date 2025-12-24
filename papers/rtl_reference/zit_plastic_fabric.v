// =============================================================================
// ZIT_PLASTIC_FABRIC.v - A Self-Organizing Fabric
// =============================================================================
//
// "We're not programming intelligence. We're growing it."
//
// 64 plastic nodes that can rewire based on frustration.
// Each node starts with toroidal neighbors but can discover new connections.
//
// The experiment: Does the fabric self-organize to minimize frustration?
//
// =============================================================================

`timescale 1ns / 1ps

module zit_plastic_fabric #(
    parameter CUBE_SIZE = 4,
    parameter STATE_WIDTH = 8,
    parameter NUM_NODES = 64,
    parameter FRUSTRATION_BITS = 8,
    parameter REWIRE_THRESHOLD = 16
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
    input  wire [5:0]                   seed_addr,
    input  wire                         seed_write,

    // Status
    output wire                         all_resonant,
    output wire [63:0]                  resonance_map,

    // Plasticity metrics
    output wire [9:0]                   global_frustration,
    output wire [9:0]                   resonant_count,
    output wire [63:0]                  rewiring_map,

    // Debug: states
    output wire [7:0]                   state_0,  state_1,  state_2,  state_3,
    output wire [7:0]                   state_4,  state_5,  state_6,  state_7,
    output wire [7:0]                   state_8,  state_9,  state_10, state_11,
    output wire [7:0]                   state_12, state_13, state_14, state_15,
    output wire [7:0]                   state_16, state_17, state_18, state_19,
    output wire [7:0]                   state_20, state_21, state_22, state_23,
    output wire [7:0]                   state_24, state_25, state_26, state_27,
    output wire [7:0]                   state_28, state_29, state_30, state_31,
    output wire [7:0]                   state_32, state_33, state_34, state_35,
    output wire [7:0]                   state_36, state_37, state_38, state_39,
    output wire [7:0]                   state_40, state_41, state_42, state_43,
    output wire [7:0]                   state_44, state_45, state_46, state_47,
    output wire [7:0]                   state_48, state_49, state_50, state_51,
    output wire [7:0]                   state_52, state_53, state_54, state_55,
    output wire [7:0]                   state_56, state_57, state_58, state_59,
    output wire [7:0]                   state_60, state_61, state_62, state_63
);

    // =========================================================================
    // NODE STATE ARRAYS
    // =========================================================================

    wire tx_valid [0:63];
    wire [STATE_WIDTH-1:0] tx_data [0:63];
    wire node_resonance [0:63];
    wire [STATE_WIDTH-1:0] node_state [0:63];
    wire [FRUSTRATION_BITS-1:0] node_frustration [0:63];
    wire node_rewiring [0:63];

    // Flattened arrays for node inputs
    wire [64*STATE_WIDTH-1:0] all_node_values_flat;
    wire [63:0] all_node_valid;

    // =========================================================================
    // FLATTEN NODE VALUES FOR BROADCAST
    // =========================================================================

    genvar flatten_idx;
    generate
        for (flatten_idx = 0; flatten_idx < 64; flatten_idx = flatten_idx + 1) begin : gen_flatten
            assign all_node_values_flat[flatten_idx*STATE_WIDTH +: STATE_WIDTH] = tx_data[flatten_idx];
            assign all_node_valid[flatten_idx] = tx_valid[flatten_idx];
        end
    endgenerate

    // =========================================================================
    // GENERATE 64 PLASTIC NODES
    // =========================================================================

    genvar idx;
    generate
        for (idx = 0; idx < 64; idx = idx + 1) begin : gen_nodes

            // Calculate x, y, z from linear index
            localparam [1:0] X = idx % 4;
            localparam [1:0] Y = (idx / 4) % 4;
            localparam [1:0] Z = idx / 16;

            // Calculate initial neighbor indices (torus topology)
            localparam [5:0] INIT_PX = ((X + 1) % 4) + (Y * 4) + (Z * 16);
            localparam [5:0] INIT_MX = ((X + 3) % 4) + (Y * 4) + (Z * 16);
            localparam [5:0] INIT_PY = X + (((Y + 1) % 4) * 4) + (Z * 16);
            localparam [5:0] INIT_MY = X + (((Y + 3) % 4) * 4) + (Z * 16);
            localparam [5:0] INIT_PZ = X + (Y * 4) + (((Z + 1) % 4) * 16);
            localparam [5:0] INIT_MZ = X + (Y * 4) + (((Z + 3) % 4) * 16);

            wire seed_select = seed_write && (seed_addr == idx);

            // Unique LFSR seed per node
            localparam [15:0] NODE_LFSR_SEED = 16'hACE1 ^ (idx * 16'h1337);

            zit_plastic_node #(
                .STATE_WIDTH(STATE_WIDTH),
                .NODE_ID(idx),
                .NUM_NODES(64),
                .FRUSTRATION_BITS(FRUSTRATION_BITS),
                .REWIRE_THRESHOLD(REWIRE_THRESHOLD),
                .DECAY_SHIFT(1),
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

                // Initial torus topology
                .init_neighbor_px(INIT_PX),
                .init_neighbor_mx(INIT_MX),
                .init_neighbor_py(INIT_PY),
                .init_neighbor_my(INIT_MY),
                .init_neighbor_pz(INIT_PZ),
                .init_neighbor_mz(INIT_MZ),

                // All node values for dynamic neighbor access
                .all_node_values(all_node_values_flat),
                .all_node_valid(all_node_valid),

                // Outputs
                .tx_valid(tx_valid[idx]),
                .tx_data(tx_data[idx]),
                .resonance(node_resonance[idx]),
                .state_out(node_state[idx]),

                // Plasticity outputs
                .frustration_out(node_frustration[idx]),
                .rewiring(node_rewiring[idx]),
                .neighbor_px_out(),
                .neighbor_mx_out(),
                .neighbor_py_out(),
                .neighbor_my_out(),
                .neighbor_pz_out(),
                .neighbor_mz_out()
            );

            assign resonance_map[idx] = node_resonance[idx];
            assign rewiring_map[idx] = node_rewiring[idx];
        end
    endgenerate

    // =========================================================================
    // GLOBAL METRICS
    // =========================================================================

    // Sum frustration across all nodes
    reg [9:0] frustration_sum;
    reg [9:0] resonance_count;
    integer i;

    always @(*) begin
        frustration_sum = 0;
        resonance_count = 0;
        for (i = 0; i < 64; i = i + 1) begin
            frustration_sum = frustration_sum + node_frustration[i];
            resonance_count = resonance_count + node_resonance[i];
        end
    end

    assign all_resonant = &resonance_map;
    assign global_frustration = frustration_sum;
    assign resonant_count = resonance_count;

    // =========================================================================
    // DEBUG OUTPUTS
    // =========================================================================

    assign state_0  = node_state[0];  assign state_1  = node_state[1];
    assign state_2  = node_state[2];  assign state_3  = node_state[3];
    assign state_4  = node_state[4];  assign state_5  = node_state[5];
    assign state_6  = node_state[6];  assign state_7  = node_state[7];
    assign state_8  = node_state[8];  assign state_9  = node_state[9];
    assign state_10 = node_state[10]; assign state_11 = node_state[11];
    assign state_12 = node_state[12]; assign state_13 = node_state[13];
    assign state_14 = node_state[14]; assign state_15 = node_state[15];
    assign state_16 = node_state[16]; assign state_17 = node_state[17];
    assign state_18 = node_state[18]; assign state_19 = node_state[19];
    assign state_20 = node_state[20]; assign state_21 = node_state[21];
    assign state_22 = node_state[22]; assign state_23 = node_state[23];
    assign state_24 = node_state[24]; assign state_25 = node_state[25];
    assign state_26 = node_state[26]; assign state_27 = node_state[27];
    assign state_28 = node_state[28]; assign state_29 = node_state[29];
    assign state_30 = node_state[30]; assign state_31 = node_state[31];
    assign state_32 = node_state[32]; assign state_33 = node_state[33];
    assign state_34 = node_state[34]; assign state_35 = node_state[35];
    assign state_36 = node_state[36]; assign state_37 = node_state[37];
    assign state_38 = node_state[38]; assign state_39 = node_state[39];
    assign state_40 = node_state[40]; assign state_41 = node_state[41];
    assign state_42 = node_state[42]; assign state_43 = node_state[43];
    assign state_44 = node_state[44]; assign state_45 = node_state[45];
    assign state_46 = node_state[46]; assign state_47 = node_state[47];
    assign state_48 = node_state[48]; assign state_49 = node_state[49];
    assign state_50 = node_state[50]; assign state_51 = node_state[51];
    assign state_52 = node_state[52]; assign state_53 = node_state[53];
    assign state_54 = node_state[54]; assign state_55 = node_state[55];
    assign state_56 = node_state[56]; assign state_57 = node_state[57];
    assign state_58 = node_state[58]; assign state_59 = node_state[59];
    assign state_60 = node_state[60]; assign state_61 = node_state[61];
    assign state_62 = node_state[62]; assign state_63 = node_state[63];

endmodule


// =============================================================================
// ZIT_PLASTIC_CONTROLLER - Controller for plastic fabric
// =============================================================================

module zit_plastic_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,

    output reg  [2:0]   phase,
    output reg  [1:0]   sub_phase,
    output reg          phase_strobe,
    output reg          cycle_complete
);

    parameter SUB_PHASE_CLOCKS = 4;

    reg [3:0] clock_counter;
    reg       running;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 0;
            sub_phase <= 0;
            phase_strobe <= 0;
            clock_counter <= 0;
            running <= 0;
            cycle_complete <= 0;
        end else begin
            phase_strobe <= 0;
            cycle_complete <= 0;

            if (enable && !running) begin
                running <= 1;
                phase <= 0;
                sub_phase <= 0;
                clock_counter <= 0;
                phase_strobe <= 1;
            end else if (running) begin
                clock_counter <= clock_counter + 1;

                if (clock_counter == SUB_PHASE_CLOCKS - 1) begin
                    clock_counter <= 0;

                    case (sub_phase)
                        2'd0: begin  // LISTEN -> REACT
                            sub_phase <= 2'd1;
                            phase_strobe <= 1;
                        end
                        2'd1: begin  // REACT -> SHOVE
                            sub_phase <= 2'd2;
                            phase_strobe <= 1;
                        end
                        2'd2: begin  // SHOVE -> next phase
                            sub_phase <= 2'd0;
                            phase_strobe <= 1;

                            if (phase == 3'd5) begin
                                phase <= 0;
                                cycle_complete <= 1;
                                if (!enable) running <= 0;
                            end else begin
                                phase <= phase + 1;
                            end
                        end
                        default: sub_phase <= 0;
                    endcase
                end
            end else if (!enable) begin
                running <= 0;
            end
        end
    end

endmodule
