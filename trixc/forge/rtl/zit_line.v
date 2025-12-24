// =============================================================================
// ZIT_LINE.v - A 1D Line of Hollywood Squares
// =============================================================================
//
// "Topology is Program" - 1D Line Topology
//
// This module implements a linear chain of Zit nodes performing odd-even
// transposition sort (parallel bubble sort).
//
// The WIRING defines the behavior:
//   - 1D Line: Nodes compare with left/right neighbors
//   - Odd-even phases: Alternating comparison directions
//
// =============================================================================

`timescale 1ns / 1ps

module zit_line #(
    parameter NUM_NODES = 8,
    parameter STATE_WIDTH = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Phase control
    input  wire [1:0]                   phase,
    input  wire                         phase_strobe,
    input  wire                         odd_phase,      // 0=even pairs, 1=odd pairs

    // Seed interface
    input  wire [STATE_WIDTH-1:0]       seed_data,
    input  wire [$clog2(NUM_NODES)-1:0] seed_addr,
    input  wire                         seed_write,

    // Status outputs
    output wire                         all_resonant,
    output wire [NUM_NODES-1:0]         resonance_map,
    output wire [STATE_WIDTH*NUM_NODES-1:0] all_states  // For debugging
);

    // Node outputs
    wire [STATE_WIDTH-1:0] node_state [0:NUM_NODES-1];
    wire [3:0] node_valid_out [0:NUM_NODES-1];
    wire [STATE_WIDTH-1:0] node_data_out [0:NUM_NODES-1];
    wire node_resonance [0:NUM_NODES-1];

    // Generate the line of nodes
    genvar i;
    generate
        for (i = 0; i < NUM_NODES; i = i + 1) begin : gen_nodes

            // Determine which direction to compare based on odd/even phase
            // Even phase: even-indexed nodes compare East, odd-indexed compare West
            // Odd phase: odd-indexed nodes compare East, even-indexed compare West
            wire [1:0] my_direction;
            wire my_active;

            // For even phase: pairs are (0,1), (2,3), (4,5), (6,7)
            //   Node 0 compares East (1), Node 1 compares West (3)
            //   Node 2 compares East (1), Node 3 compares West (3)
            // For odd phase: pairs are (1,2), (3,4), (5,6)
            //   Node 1 compares East (1), Node 2 compares West (3)
            //   Node 3 compares East (1), Node 4 compares West (3)

            // Even nodes: compare East on even phase, West on odd phase
            // Odd nodes: compare West on even phase, East on odd phase
            assign my_direction = (i[0] == 1'b0) ?
                                  (odd_phase ? 2'b11 : 2'b01) :  // Even index: odd->West, even->East
                                  (odd_phase ? 2'b01 : 2'b11);   // Odd index: odd->East, even->West

            // Boundary conditions: edge nodes can't compare in certain directions
            // Node 0 can't compare West (is idle on odd phase)
            // Node N-1 can't compare East (is idle on odd phase)
            wire my_idle = (i == 0 && my_direction == 2'b11) ? 1'b1 :           // Node 0 idle when going West
                           (i == NUM_NODES-1 && my_direction == 2'b01) ? 1'b1 :  // Last idle when going East
                           1'b0;

            // Seed select for this node
            wire seed_select = seed_write && (seed_addr == i);

            // Neighbor data wiring
            // East neighbor's data (direction 1)
            wire [STATE_WIDTH-1:0] east_data = (i < NUM_NODES-1) ? node_data_out[i+1] : {STATE_WIDTH{1'b0}};
            wire east_valid = (i < NUM_NODES-1) ? node_valid_out[i+1][3] : 1'b0;  // East neighbor sends West

            // West neighbor's data (direction 3)
            wire [STATE_WIDTH-1:0] west_data = (i > 0) ? node_data_out[i-1] : {STATE_WIDTH{1'b0}};
            wire west_valid = (i > 0) ? node_valid_out[i-1][1] : 1'b0;  // West neighbor sends East

            zit_node #(
                .STATE_WIDTH(STATE_WIDTH),
                .NODE_ID(i)
            ) node_inst (
                .clk(clk),
                .rst_n(rst_n),

                .phase(phase),
                .phase_strobe(phase_strobe),

                // Configuration
                .init_value(seed_data),
                .init_load(seed_select),
                .cfg_direction(my_direction),
                .cfg_idle(my_idle),  // Idle when at boundary

                // Neighbor inputs (only East and West matter for 1D)
                .neighbor_valid_in({west_valid, 1'b0, east_valid, 1'b0}),  // {W, S, E, N}
                .neighbor_data_n({STATE_WIDTH{1'b0}}),
                .neighbor_data_e(east_data),
                .neighbor_data_s({STATE_WIDTH{1'b0}}),
                .neighbor_data_w(west_data),

                .neighbor_valid_out(node_valid_out[i]),
                .neighbor_data_out(node_data_out[i]),

                .resonance(node_resonance[i]),
                .state_out(node_state[i])
            );

            assign resonance_map[i] = node_resonance[i];
            assign all_states[STATE_WIDTH*(i+1)-1 -: STATE_WIDTH] = node_state[i];

        end
    endgenerate

    assign all_resonant = &resonance_map;

endmodule


// =============================================================================
// ZIT_LINE_CONTROLLER - Phase Sequencer with Odd/Even
// =============================================================================

module zit_line_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,
    input  wire         single_step,

    output reg  [1:0]   phase,
    output reg          phase_strobe,
    output reg          odd_phase,      // Toggles each cycle
    output wire         cycle_complete
);

    parameter PHASE_CLOCKS = 4;

    reg [3:0] phase_counter;
    reg       running;
    reg       cycle_done;

    assign cycle_complete = cycle_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'b00;
            phase_strobe <= 1'b0;
            phase_counter <= 0;
            running <= 1'b0;
            cycle_done <= 1'b0;
            odd_phase <= 1'b0;
        end else begin
            phase_strobe <= 1'b0;
            cycle_done <= 1'b0;

            if (enable && !running) begin
                running <= 1'b1;
                phase <= 2'b01;  // LISTEN
                phase_counter <= 0;
                phase_strobe <= 1'b1;
            end else if (running) begin
                phase_counter <= phase_counter + 1;

                if (phase_counter == PHASE_CLOCKS - 1) begin
                    phase_counter <= 0;

                    case (phase)
                        2'b01: begin  // LISTEN -> REACT
                            phase <= 2'b10;
                            phase_strobe <= 1'b1;
                        end
                        2'b10: begin  // REACT -> SHOVE
                            phase <= 2'b11;
                            phase_strobe <= 1'b1;
                        end
                        2'b11: begin  // SHOVE -> LISTEN (or IDLE)
                            cycle_done <= 1'b1;
                            odd_phase <= ~odd_phase;  // Toggle for next cycle

                            if (single_step || !enable) begin
                                phase <= 2'b00;
                                running <= 1'b0;
                            end else begin
                                phase <= 2'b01;
                                phase_strobe <= 1'b1;
                            end
                        end
                        default: phase <= 2'b01;
                    endcase
                end
            end else if (!enable) begin
                running <= 1'b0;
            end
        end
    end

endmodule
