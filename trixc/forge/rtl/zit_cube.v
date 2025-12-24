// =============================================================================
// ZIT_CUBE.v - The 4x4x4 Hollywood Squares Fabric
// =============================================================================
//
// "We're not simulating physics. We ARE physics."
//
// 64 Resonant Transputers arranged in 3D toroidal space.
// This is an experiment, not a product. We're creating conditions
// for emergence and observing what happens.
//
// The ontological frame:
//   - Symbolic machines manipulate representations (maps)
//   - Resonant machines ARE physical analogs (territory)
//   - A resonant machine cannot lie - it can only resonate or be silent
//
// =============================================================================

`timescale 1ns / 1ps

// =============================================================================
// ZIT_CUBE_NODE - A node extended for 6 neighbors
// =============================================================================

module zit_cube_node #(
    parameter STATE_WIDTH = 8,
    parameter NODE_ID = 0
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Phase control (6 directions)
    input  wire [2:0]               phase,          // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    input  wire [1:0]               sub_phase,      // 0=LISTEN, 1=REACT, 2=SHOVE
    input  wire                     phase_strobe,

    // Configuration
    input  wire [STATE_WIDTH-1:0]   init_value,
    input  wire                     init_load,

    // 6 Neighbor interfaces
    input  wire                     neighbor_valid_px,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_px,
    input  wire                     neighbor_valid_mx,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_mx,
    input  wire                     neighbor_valid_py,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_py,
    input  wire                     neighbor_valid_my,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_my,
    input  wire                     neighbor_valid_pz,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_pz,
    input  wire                     neighbor_valid_mz,
    input  wire [STATE_WIDTH-1:0]   neighbor_data_mz,

    // Outputs (broadcast to all 6)
    output wire                     tx_valid,
    output wire [STATE_WIDTH-1:0]   tx_data,

    // Status
    output wire                     resonance,
    output wire [STATE_WIDTH-1:0]   state_out
);

    // =========================================================================
    // INTERNAL STATE
    // =========================================================================

    reg [STATE_WIDTH-1:0] S;

    // Neighbor latches (6 directions)
    reg [STATE_WIDTH-1:0] latch_px, latch_mx;
    reg [STATE_WIDTH-1:0] latch_py, latch_my;
    reg [STATE_WIDTH-1:0] latch_pz, latch_mz;
    reg [5:0] neighbor_received;

    reg [STATE_WIDTH-1:0] tx_data_reg;
    reg tx_valid_reg;
    reg resonance_reg;
    reg react_done;
    reg has_participated;

    // Sub-phase encoding
    localparam SUB_LISTEN = 2'd0;
    localparam SUB_REACT  = 2'd1;
    localparam SUB_SHOVE  = 2'd2;

    // Direction encoding
    localparam DIR_PX = 3'd0;
    localparam DIR_MX = 3'd1;
    localparam DIR_PY = 3'd2;
    localparam DIR_MY = 3'd3;
    localparam DIR_PZ = 3'd4;
    localparam DIR_MZ = 3'd5;

    // =========================================================================
    // FROZEN SHAPE: 3D COMPARATOR
    // =========================================================================
    //
    // The same kernel as 1D, extended to 6 directions.
    // Positive directions (+X, +Y, +Z): swap if me > neighbor
    // Negative directions (-X, -Y, -Z): swap if neighbor > me
    //
    // This creates a "flow" of values through the 3D space.
    // =========================================================================

    // Select active neighbor based on current phase
    reg [STATE_WIDTH-1:0] active_neighbor;
    reg active_received;

    always @(*) begin
        case (phase)
            DIR_PX: begin active_neighbor = latch_px; active_received = neighbor_received[0]; end
            DIR_MX: begin active_neighbor = latch_mx; active_received = neighbor_received[1]; end
            DIR_PY: begin active_neighbor = latch_py; active_received = neighbor_received[2]; end
            DIR_MY: begin active_neighbor = latch_my; active_received = neighbor_received[3]; end
            DIR_PZ: begin active_neighbor = latch_pz; active_received = neighbor_received[4]; end
            DIR_MZ: begin active_neighbor = latch_mz; active_received = neighbor_received[5]; end
            default: begin active_neighbor = 0; active_received = 0; end
        endcase
    end

    // Positive directions: even phase numbers (0, 2, 4)
    wire positive_dir = (phase[0] == 1'b0);

    // The comparison - the frozen shape
    wire should_swap = active_received &&
                       (positive_dir ? (S > active_neighbor) : (active_neighbor > S));

    // =========================================================================
    // STATE MACHINE
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            S <= 0;
            latch_px <= 0; latch_mx <= 0;
            latch_py <= 0; latch_my <= 0;
            latch_pz <= 0; latch_mz <= 0;
            neighbor_received <= 0;
            tx_data_reg <= 0;
            tx_valid_reg <= 0;
            resonance_reg <= 0;
            react_done <= 0;
            has_participated <= 0;
        end else begin
            // Initialize
            if (init_load) begin
                S <= init_value;
            end

            case (sub_phase)

                SUB_LISTEN: begin
                    tx_valid_reg <= 0;
                    react_done <= 0;

                    // Latch all incoming neighbor values
                    if (neighbor_valid_px) begin latch_px <= neighbor_data_px; neighbor_received[0] <= 1; end
                    if (neighbor_valid_mx) begin latch_mx <= neighbor_data_mx; neighbor_received[1] <= 1; end
                    if (neighbor_valid_py) begin latch_py <= neighbor_data_py; neighbor_received[2] <= 1; end
                    if (neighbor_valid_my) begin latch_my <= neighbor_data_my; neighbor_received[3] <= 1; end
                    if (neighbor_valid_pz) begin latch_pz <= neighbor_data_pz; neighbor_received[4] <= 1; end
                    if (neighbor_valid_mz) begin latch_mz <= neighbor_data_mz; neighbor_received[5] <= 1; end
                end

                SUB_REACT: begin
                    if (!react_done) begin
                        react_done <= 1;

                        // Compute resonance: no swap needed
                        resonance_reg <= has_participated && active_received && ~should_swap;

                        if (should_swap) begin
                            S <= active_neighbor;
                        end
                    end
                end

                SUB_SHOVE: begin
                    // Broadcast current value to all neighbors
                    tx_valid_reg <= 1;
                    tx_data_reg <= S;
                    has_participated <= 1;

                    // Clear for next phase
                    neighbor_received <= 0;
                end

            endcase
        end
    end

    // =========================================================================
    // OUTPUTS
    // =========================================================================

    assign tx_valid = tx_valid_reg;
    assign tx_data = tx_data_reg;
    assign resonance = resonance_reg;
    assign state_out = S;

endmodule


// =============================================================================
// ZIT_CUBE - The 4x4x4 Fabric
// =============================================================================

module zit_cube #(
    parameter CUBE_SIZE = 4,
    parameter STATE_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Phase control
    input  wire [2:0]               phase,
    input  wire [1:0]               sub_phase,
    input  wire                     phase_strobe,

    // Seed interface
    input  wire [STATE_WIDTH-1:0]   seed_data,
    input  wire [5:0]               seed_addr,
    input  wire                     seed_write,

    // Status
    output wire                     all_resonant,
    output wire [63:0]              resonance_map,

    // Debug: all 64 states (flat array for testbench access)
    output wire [7:0]               state_0,  state_1,  state_2,  state_3,
    output wire [7:0]               state_4,  state_5,  state_6,  state_7,
    output wire [7:0]               state_8,  state_9,  state_10, state_11,
    output wire [7:0]               state_12, state_13, state_14, state_15,
    output wire [7:0]               state_16, state_17, state_18, state_19,
    output wire [7:0]               state_20, state_21, state_22, state_23,
    output wire [7:0]               state_24, state_25, state_26, state_27,
    output wire [7:0]               state_28, state_29, state_30, state_31,
    output wire [7:0]               state_32, state_33, state_34, state_35,
    output wire [7:0]               state_36, state_37, state_38, state_39,
    output wire [7:0]               state_40, state_41, state_42, state_43,
    output wire [7:0]               state_44, state_45, state_46, state_47,
    output wire [7:0]               state_48, state_49, state_50, state_51,
    output wire [7:0]               state_52, state_53, state_54, state_55,
    output wire [7:0]               state_56, state_57, state_58, state_59,
    output wire [7:0]               state_60, state_61, state_62, state_63
);

    // =========================================================================
    // NODE ARRAYS
    // =========================================================================

    wire tx_valid [0:63];
    wire [STATE_WIDTH-1:0] tx_data [0:63];
    wire node_resonance [0:63];
    wire [STATE_WIDTH-1:0] node_state [0:63];

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    // Convert (x, y, z) to linear index
    function [5:0] xyz_to_idx;
        input [1:0] x, y, z;
        begin
            xyz_to_idx = x + (y * 4) + (z * 16);
        end
    endfunction

    // Wrap around for toroidal topology
    function [1:0] wrap;
        input [2:0] val;  // Signed-ish: 0-3 normal, 4 = -1 wrapped, -1 = 3 wrapped
        begin
            wrap = val[1:0];  // Just take low 2 bits, wraps naturally
        end
    endfunction

    // =========================================================================
    // GENERATE 64 NODES
    // =========================================================================

    genvar idx;
    generate
        for (idx = 0; idx < 64; idx = idx + 1) begin : gen_nodes

            // Calculate x, y, z from linear index
            localparam [1:0] X = idx % 4;
            localparam [1:0] Y = (idx / 4) % 4;
            localparam [1:0] Z = idx / 16;

            // Calculate neighbor indices with wraparound
            localparam [5:0] IDX_PX = ((X + 1) % 4) + (Y * 4) + (Z * 16);
            localparam [5:0] IDX_MX = ((X + 3) % 4) + (Y * 4) + (Z * 16);  // +3 = -1 mod 4
            localparam [5:0] IDX_PY = X + (((Y + 1) % 4) * 4) + (Z * 16);
            localparam [5:0] IDX_MY = X + (((Y + 3) % 4) * 4) + (Z * 16);
            localparam [5:0] IDX_PZ = X + (Y * 4) + (((Z + 1) % 4) * 16);
            localparam [5:0] IDX_MZ = X + (Y * 4) + (((Z + 3) % 4) * 16);

            wire seed_select = seed_write && (seed_addr == idx);

            zit_cube_node #(
                .STATE_WIDTH(STATE_WIDTH),
                .NODE_ID(idx)
            ) node_inst (
                .clk(clk),
                .rst_n(rst_n),

                .phase(phase),
                .sub_phase(sub_phase),
                .phase_strobe(phase_strobe),

                .init_value(seed_data),
                .init_load(seed_select),

                // +X neighbor
                .neighbor_valid_px(tx_valid[IDX_PX]),
                .neighbor_data_px(tx_data[IDX_PX]),
                // -X neighbor
                .neighbor_valid_mx(tx_valid[IDX_MX]),
                .neighbor_data_mx(tx_data[IDX_MX]),
                // +Y neighbor
                .neighbor_valid_py(tx_valid[IDX_PY]),
                .neighbor_data_py(tx_data[IDX_PY]),
                // -Y neighbor
                .neighbor_valid_my(tx_valid[IDX_MY]),
                .neighbor_data_my(tx_data[IDX_MY]),
                // +Z neighbor
                .neighbor_valid_pz(tx_valid[IDX_PZ]),
                .neighbor_data_pz(tx_data[IDX_PZ]),
                // -Z neighbor
                .neighbor_valid_mz(tx_valid[IDX_MZ]),
                .neighbor_data_mz(tx_data[IDX_MZ]),

                .tx_valid(tx_valid[idx]),
                .tx_data(tx_data[idx]),

                .resonance(node_resonance[idx]),
                .state_out(node_state[idx])
            );

            assign resonance_map[idx] = node_resonance[idx];

        end
    endgenerate

    // =========================================================================
    // GLOBAL STATUS
    // =========================================================================

    assign all_resonant = &resonance_map;

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
// ZIT_CUBE_CONTROLLER - 6-Phase Sequencer
// =============================================================================

module zit_cube_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,

    output reg  [2:0]   phase,          // 0-5 for 6 directions
    output reg  [1:0]   sub_phase,      // 0=LISTEN, 1=REACT, 2=SHOVE
    output reg          phase_strobe,
    output wire         cycle_complete
);

    parameter SUB_PHASE_CLOCKS = 4;

    reg [3:0] clock_counter;
    reg       running;
    reg       cycle_done;

    assign cycle_complete = cycle_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 0;
            sub_phase <= 0;
            phase_strobe <= 0;
            clock_counter <= 0;
            running <= 0;
            cycle_done <= 0;
        end else begin
            phase_strobe <= 0;
            cycle_done <= 0;

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
                        2'd2: begin  // SHOVE -> next phase or cycle complete
                            sub_phase <= 2'd0;
                            phase_strobe <= 1;

                            if (phase == 3'd5) begin
                                // Completed all 6 phases
                                phase <= 0;
                                cycle_done <= 1;

                                if (!enable) begin
                                    running <= 0;
                                end
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
