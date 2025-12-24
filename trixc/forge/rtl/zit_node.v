// =============================================================================
// ZIT_NODE.v - The Hollywood Square
// =============================================================================
//
// "Topology is Program. Neighbor is State."
//
// This is not just a chip. This is a Resonant Transputer - an active mesh node
// in the Hollywood Squares fabric. The OS isn't software running ON the chip.
// The OS IS the shape of the connections between chips.
//
// Architecture:
//   - State Register: The "value" of this square
//   - Frozen Shape: The "kernel" (comparator, Zit detector, etc.)
//   - Neighbor Ports: N/S/E/W connections to adjacent squares
//   - Host Port: SPI for configuration and seeding
//
// Protocol: Deterministic Shoving
//   Phase 1 (LISTEN):  Receive values from neighbors
//   Phase 2 (REACT):   Apply frozen shape to decide action
//   Phase 3 (SHOVE):   Send updated values to neighbors
//
// The same kernel + different wiring = different behavior.
//
// =============================================================================

`timescale 1ns / 1ps

// =============================================================================
// ZIT_NODE - The Hollywood Square (Parameterized)
// =============================================================================

module zit_node #(
    parameter STATE_WIDTH = 8,          // Bit width of state (8, 16, 64, 512)
    parameter NODE_ID = 0               // Unique ID for debugging
)(
    // =========================================================================
    // GLOBAL
    // =========================================================================
    input  wire                         clk,
    input  wire                         rst_n,

    // =========================================================================
    // PHASE CONTROL (From Fabric Controller)
    // =========================================================================
    input  wire [1:0]                   phase,          // 0=IDLE, 1=LISTEN, 2=REACT, 3=SHOVE
    input  wire                         phase_strobe,   // Rising edge = advance phase

    // =========================================================================
    // DIRECT CONFIGURATION (Alternative to SPI for simulation)
    // =========================================================================
    input  wire [STATE_WIDTH-1:0]       init_value,     // Initial state value
    input  wire                         init_load,      // Load initial value
    input  wire [1:0]                   cfg_direction,  // Active neighbor direction (0-3)
    input  wire                         cfg_idle,       // If 1, this node is idle (always resonant)

    // =========================================================================
    // NEIGHBOR INTERFACE (The "Square" Ports)
    // =========================================================================
    // Simple valid/data protocol - no packets, just values
    // Direction: 0=North, 1=East, 2=South, 3=West

    // Receive from neighbors (flattened for Verilog-2001 compatibility)
    input  wire [3:0]                   neighbor_valid_in,
    input  wire [STATE_WIDTH-1:0]       neighbor_data_n,
    input  wire [STATE_WIDTH-1:0]       neighbor_data_e,
    input  wire [STATE_WIDTH-1:0]       neighbor_data_s,
    input  wire [STATE_WIDTH-1:0]       neighbor_data_w,

    // Send to neighbors
    output wire [3:0]                   neighbor_valid_out,
    output wire [STATE_WIDTH-1:0]       neighbor_data_out,  // Same data to all directions

    // =========================================================================
    // STATUS
    // =========================================================================
    output wire                         resonance,      // Shape kernel output (no swap needed)
    output wire [STATE_WIDTH-1:0]       state_out       // Current state (debug)
);

    // =========================================================================
    // INTERNAL REGISTERS
    // =========================================================================

    // The State - the "value" of this square
    reg [STATE_WIDTH-1:0] S;

    // Latched neighbor values (captured during LISTEN phase)
    reg [STATE_WIDTH-1:0] neighbor_latched_n, neighbor_latched_e;
    reg [STATE_WIDTH-1:0] neighbor_latched_s, neighbor_latched_w;
    reg [3:0] neighbor_received;

    // Output registers
    reg [3:0] tx_valid;
    reg [STATE_WIDTH-1:0] tx_data;

    // Kernel output
    reg resonance_reg;
    reg has_participated;  // Has this node participated in at least one cycle?
    reg react_done;        // Have we processed REACT for this phase?

    // Configuration
    reg [1:0] active_direction;
    reg is_idle;  // This node is idle for current cycle

    // =========================================================================
    // PHASE STATE MACHINE
    // =========================================================================

    localparam PHASE_IDLE   = 2'b00;
    localparam PHASE_LISTEN = 2'b01;
    localparam PHASE_REACT  = 2'b10;
    localparam PHASE_SHOVE  = 2'b11;

    // =========================================================================
    // FROZEN SHAPE: BUBBLE SORT COMPARATOR
    // =========================================================================
    //
    // This is the first kernel - a simple comparator for sorting.
    //
    // Rule: If neighbor (from specific direction) has value > mine,
    //       we swap. The direction depends on the phase (odd/even).
    //
    // For bubble sort on a 1D line:
    //   - Even phase: compare with East neighbor, swap if needed
    //   - Odd phase: compare with West neighbor, swap if needed
    //
    // The WIRING determines which direction is "active" each phase.
    // This kernel doesn't know if it's in 1D, 2D, or 6D.
    // It just compares with whoever sends a valid signal.
    // =========================================================================

    // Kernel computation (combinational)
    // Use cfg_direction directly (not registered) for immediate response to phase changes
    wire should_swap;
    reg [STATE_WIDTH-1:0] active_neighbor_value;

    // Mux the active neighbor based on CURRENT direction (not registered)
    always @(*) begin
        case (cfg_direction)
            2'b00: active_neighbor_value = neighbor_latched_n;
            2'b01: active_neighbor_value = neighbor_latched_e;
            2'b10: active_neighbor_value = neighbor_latched_s;
            2'b11: active_neighbor_value = neighbor_latched_w;
        endcase
    end

    // Active neighbor received? (use cfg_direction for current phase)
    wire active_neighbor_received = neighbor_received[cfg_direction];

    // The comparison - the frozen shape
    // Direction-aware comparison for bubble sort:
    //   - Comparing East (dir=1): swap if I'm larger (move right), so me > neighbor
    //   - Comparing West (dir=3): swap if neighbor is larger (coming from right), so neighbor > me
    // This ensures both nodes in a pair swap symmetrically
    wire compare_with_east = (cfg_direction == 2'b01);
    assign should_swap = active_neighbor_received && !cfg_idle &&
                         (compare_with_east ? (S > active_neighbor_value) : (active_neighbor_value > S));

    // Resonance output - use the latched resonance_reg (computed during REACT)
    assign resonance = resonance_reg;

    // =========================================================================
    // MAIN STATE MACHINE
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset
            S <= {STATE_WIDTH{1'b0}};
            resonance_reg <= 1'b0;
            neighbor_received <= 4'b0000;
            tx_valid <= 4'b0000;
            tx_data <= {STATE_WIDTH{1'b0}};

            neighbor_latched_n <= {STATE_WIDTH{1'b0}};
            neighbor_latched_e <= {STATE_WIDTH{1'b0}};
            neighbor_latched_s <= {STATE_WIDTH{1'b0}};
            neighbor_latched_w <= {STATE_WIDTH{1'b0}};

            has_participated <= 1'b0;
            react_done <= 1'b0;
            active_direction <= 2'b01;  // Default: East
            is_idle <= 1'b0;

        end else begin
            // Handle direct initialization
            if (init_load) begin
                S <= init_value;
            end

            // Handle configuration (updated during IDLE phase or phase transition)
            // Configuration can change each cycle based on odd/even phase
            if (phase == PHASE_IDLE || phase == PHASE_LISTEN) begin
                active_direction <= cfg_direction;
                is_idle <= cfg_idle;
            end

            case (phase)

                PHASE_IDLE: begin
                    // Clear outputs, wait for next cycle
                    tx_valid <= 4'b0000;
                end

                PHASE_LISTEN: begin
                    // Clear react_done flag for new cycle
                    react_done <= 1'b0;

                    // Capture incoming neighbor values
                    if (neighbor_valid_in[0]) begin
                        neighbor_latched_n <= neighbor_data_n;
                        neighbor_received[0] <= 1'b1;
                    end
                    if (neighbor_valid_in[1]) begin
                        neighbor_latched_e <= neighbor_data_e;
                        neighbor_received[1] <= 1'b1;
                    end
                    if (neighbor_valid_in[2]) begin
                        neighbor_latched_s <= neighbor_data_s;
                        neighbor_received[2] <= 1'b1;
                    end
                    if (neighbor_valid_in[3]) begin
                        neighbor_latched_w <= neighbor_data_w;
                        neighbor_received[3] <= 1'b1;
                    end
                end

                PHASE_REACT: begin
                    // Apply the frozen shape (comparator) - ONCE per phase
                    // Only process on first clock of REACT (when react_done is 0)
                    if (!react_done) begin
                        react_done <= 1'b1;

                        // Use cfg_idle directly for immediate response to phase changes
                        // Idle nodes are always resonant after participation
                        // Active nodes: resonant if received data AND no swap needed
                        if (cfg_idle) begin
                            resonance_reg <= has_participated;
                            tx_data <= S;  // Still send our value even if idle
                        end else begin
                            resonance_reg <= has_participated && active_neighbor_received && ~should_swap;

                            if (should_swap) begin
                                // Swap: take neighbor's value, prepare to send mine
                                tx_data <= S;
                                S <= active_neighbor_value;
                            end else begin
                                // No swap: keep my value, send it anyway
                                tx_data <= S;
                            end
                        end
                    end
                end

                PHASE_SHOVE: begin
                    // Broadcast our CURRENT value (post-swap) to ALL neighbors
                    // This ensures neighbors see our updated state for their next cycle
                    tx_valid <= 4'b1111;
                    tx_data <= S;  // Send current value, not old tx_data from REACT

                    // Mark that we've participated in at least one cycle
                    has_participated <= 1'b1;

                    // Clear received flags for next cycle
                    neighbor_received <= 4'b0000;
                end

            endcase
        end
    end

    // =========================================================================
    // OUTPUT ASSIGNMENTS
    // =========================================================================

    assign state_out = S;
    assign neighbor_valid_out = tx_valid;
    assign neighbor_data_out = tx_data;

endmodule


// =============================================================================
// ZIT_FABRIC - A Grid of Hollywood Squares
// =============================================================================
//
// Wires together an NxN grid of zit_nodes.
// The WIRING is the OS. The TOPOLOGY is the program.
//
// For an 8x8 grid of 8-bit Zits:
//   - 64 nodes
//   - 128 inter-node connections (each node has 4 neighbors)
//   - Perfect for bubble sort, cellular automata, constraint propagation
//
// =============================================================================

module zit_fabric #(
    parameter GRID_WIDTH = 4,           // Nodes per row
    parameter GRID_HEIGHT = 4,          // Nodes per column
    parameter STATE_WIDTH = 8           // Bits per node
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Phase control (broadcast to all nodes)
    input  wire [1:0]                   phase,
    input  wire                         phase_strobe,

    // Global configuration
    input  wire [STATE_WIDTH-1:0]       seed_data,
    input  wire [3:0]                   seed_addr,
    input  wire                         seed_write,

    // Status
    output wire                         all_resonant,   // All nodes stable
    output wire [GRID_WIDTH*GRID_HEIGHT-1:0] resonance_map
);

    // =========================================================================
    // INTER-NODE WIRING
    // =========================================================================
    //
    // Direction convention:
    //   0 = North (y-1)
    //   1 = East  (x+1)
    //   2 = South (y+1)
    //   3 = West  (x-1)
    //
    // This is where the "OS" lives - in the shape of these connections.
    // =========================================================================

    // Node outputs (linearized for Verilog-2001)
    wire [3:0] node_valid_out [0:GRID_WIDTH*GRID_HEIGHT-1];
    wire [STATE_WIDTH-1:0] node_data_out [0:GRID_WIDTH*GRID_HEIGHT-1];

    // Node status
    wire node_resonance [0:GRID_WIDTH*GRID_HEIGHT-1];
    wire [STATE_WIDTH-1:0] node_state [0:GRID_WIDTH*GRID_HEIGHT-1];

    // Seed loading
    wire [GRID_WIDTH*GRID_HEIGHT-1:0] seed_select;

    // =========================================================================
    // INSTANTIATE THE GRID
    // =========================================================================

    genvar idx;
    generate
        for (idx = 0; idx < GRID_WIDTH * GRID_HEIGHT; idx = idx + 1) begin : gen_nodes

            // Calculate x, y from linear index
            localparam X = idx % GRID_WIDTH;
            localparam Y = idx / GRID_WIDTH;

            // Calculate neighbor indices with wrap-around
            localparam NORTH_IDX = ((Y == 0) ? (GRID_HEIGHT - 1) : (Y - 1)) * GRID_WIDTH + X;
            localparam EAST_IDX  = Y * GRID_WIDTH + ((X == GRID_WIDTH - 1) ? 0 : (X + 1));
            localparam SOUTH_IDX = ((Y == GRID_HEIGHT - 1) ? 0 : (Y + 1)) * GRID_WIDTH + X;
            localparam WEST_IDX  = Y * GRID_WIDTH + ((X == 0) ? (GRID_WIDTH - 1) : (X - 1));

            // Seed select for this node
            assign seed_select[idx] = seed_write && (seed_addr == idx);

            zit_node #(
                .STATE_WIDTH(STATE_WIDTH),
                .NODE_ID(idx)
            ) node_inst (
                .clk(clk),
                .rst_n(rst_n),

                .phase(phase),
                .phase_strobe(phase_strobe),

                // Direct configuration
                .init_value(seed_data),
                .init_load(seed_select[idx]),
                .cfg_direction(2'b01),  // Default: compare with East
                .cfg_idle(1'b0),

                // Neighbor inputs from adjacent nodes
                // North input comes from South output of northern neighbor
                .neighbor_valid_in({
                    node_valid_out[WEST_IDX][1],   // West neighbor's East output -> my West input
                    node_valid_out[SOUTH_IDX][0],  // South neighbor's North output -> my South input
                    node_valid_out[EAST_IDX][3],   // East neighbor's West output -> my East input
                    node_valid_out[NORTH_IDX][2]   // North neighbor's South output -> my North input
                }),
                .neighbor_data_n(node_data_out[NORTH_IDX]),
                .neighbor_data_e(node_data_out[EAST_IDX]),
                .neighbor_data_s(node_data_out[SOUTH_IDX]),
                .neighbor_data_w(node_data_out[WEST_IDX]),

                .neighbor_valid_out(node_valid_out[idx]),
                .neighbor_data_out(node_data_out[idx]),

                .resonance(node_resonance[idx]),
                .state_out(node_state[idx])
            );

            // Resonance map output
            assign resonance_map[idx] = node_resonance[idx];

        end
    endgenerate

    // =========================================================================
    // GLOBAL STATUS
    // =========================================================================

    assign all_resonant = &resonance_map;

endmodule


// =============================================================================
// ZIT_FABRIC_CONTROLLER - Phase Sequencer
// =============================================================================
//
// Generates the LISTEN → REACT → SHOVE phase sequence.
// This is the "clock" of the Hollywood Squares OS.
//
// One complete cycle = one "tick" of the cellular automaton.
// =============================================================================

module zit_fabric_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,         // Start/stop the fabric
    input  wire         single_step,    // Execute one cycle then stop

    output reg  [1:0]   phase,
    output reg          phase_strobe,
    output wire         cycle_complete
);

    // Phase timing (clocks per phase)
    parameter PHASE_CLOCKS = 4;

    reg [3:0] phase_counter;
    reg       running;
    reg       cycle_done;

    assign cycle_complete = cycle_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 2'b00;  // IDLE
            phase_strobe <= 1'b0;
            phase_counter <= 0;
            running <= 1'b0;
            cycle_done <= 1'b0;
        end else begin
            phase_strobe <= 1'b0;
            cycle_done <= 1'b0;

            if (enable && !running) begin
                running <= 1'b1;
                phase <= 2'b01;  // Start with LISTEN
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
                            if (single_step) begin
                                phase <= 2'b00;
                                running <= 1'b0;
                            end else if (!enable) begin
                                phase <= 2'b00;
                                running <= 1'b0;
                            end else begin
                                phase <= 2'b01;
                                phase_strobe <= 1'b1;
                            end
                        end
                        default: begin
                            phase <= 2'b01;
                        end
                    endcase
                end
            end else if (!enable) begin
                running <= 1'b0;
            end
        end
    end

endmodule
