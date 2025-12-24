// =============================================================================
// ZIT_SCALED_TB.v - The Scale Test
// =============================================================================
//
// "Does emergence scale? Let's find out."
//
// EXPERIMENT C: 8x8x8 = 512 nodes
// - Same plasticity mechanism as 4x4x4
// - Same physics, 8x larger space
// - Measure: convergence time, final resonance, rewiring count
//
// =============================================================================

`timescale 1ns / 1ps

module zit_scaled_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter CLK_PERIOD = 10;
    parameter CUBE_SIZE = 6;  // 6x6x6 = 216 nodes (intermediate test)
    parameter NUM_NODES = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE;  // 216
    parameter ADDR_WIDTH = 8;
    parameter MAX_CYCLES = 800;

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
    reg  [ADDR_WIDTH-1:0] seed_addr;
    reg         seed_write;

    wire        all_resonant;
    wire [NUM_NODES-1:0] resonance_map;
    wire [15:0] global_frustration;
    wire [15:0] resonant_count;
    wire [NUM_NODES-1:0] rewiring_map;

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

    zit_scaled_fabric #(
        .CUBE_SIZE(CUBE_SIZE),
        .STATE_WIDTH(8),
        .FRUSTRATION_BITS(8),
        .REWIRE_THRESHOLD(8)  // Lower threshold for faster rewiring
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
        .rewiring_map(rewiring_map)
    );

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    task seed_node;
        input [ADDR_WIDTH-1:0] addr;
        input [7:0] value;
        begin
            @(posedge clk);
            seed_addr <= addr;
            seed_data <= value;
            seed_write <= 1;
            @(posedge clk);
            seed_write <= 0;
            repeat(2) @(posedge clk);
        end
    endtask

    // =========================================================================
    // MAIN EXPERIMENT
    // =========================================================================

    integer cycle_count;
    integer i;
    integer convergence_cycle;
    integer total_rewires;
    integer last_resonant;
    integer stall_counter;
    integer peak_frustration;

    // Count rewiring events
    integer rewire_count;
    reg [NUM_NODES-1:0] prev_rewiring_map;

    initial begin
        $display("");
        $display("================================================================");
        $display("     EXPERIMENT C: THE SCALE TEST");
        $display("     8x8x8 = 512 Nodes");
        $display("================================================================");
        $display("");
        $display("# Does emergence scale?");
        $display("# Same physics. 8x more nodes. Let's see.");
        $display("");

        // Initialize
        rst_n = 0;
        enable = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;
        convergence_cycle = 0;
        total_rewires = 0;
        stall_counter = 0;
        last_resonant = 0;
        peak_frustration = 0;
        prev_rewiring_map = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // =====================================================================
        // SEED WITH 3D GRADIENT
        // =====================================================================

        $display("Seeding with 3D gradient pattern...");
        $display("  node[x,y,z] = (x + y*8 + z*64) * 0.5");
        $display("");

        for (i = 0; i < NUM_NODES; i = i + 1) begin
            // 3D gradient: value based on position
            seed_node(i[ADDR_WIDTH-1:0], (i >> 1) & 8'hFF);
        end

        $display("Seeding complete. Starting self-organization...");
        $display("");
        $display("SCALE,cycle,resonant,percent,frustration,rewires");

        // =====================================================================
        // RUN UNTIL CONVERGENCE
        // =====================================================================

        enable = 1;
        cycle_count = 0;

        while (cycle_count < MAX_CYCLES) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Count rewiring this cycle
            begin : count_rw
                integer rw_i, rw_cnt;
                rw_cnt = 0;
                for (rw_i = 0; rw_i < NUM_NODES; rw_i = rw_i + 1) begin
                    if (rewiring_map[rw_i] && !prev_rewiring_map[rw_i])
                        rw_cnt = rw_cnt + 1;
                end
                total_rewires = total_rewires + rw_cnt;
                prev_rewiring_map = rewiring_map;
            end

            // Track peak frustration
            if (global_frustration > peak_frustration)
                peak_frustration = global_frustration;

            // Progress every 10 cycles
            if (cycle_count % 10 == 0) begin
                $display("SCALE,%0d,%0d,%0d%%,%0d,%0d",
                    cycle_count, resonant_count,
                    (resonant_count * 100) / NUM_NODES,
                    global_frustration, total_rewires);
            end

            // Check for convergence
            if (resonant_count == NUM_NODES && convergence_cycle == 0) begin
                convergence_cycle = cycle_count;
                $display("");
                $display("*** FULL CONVERGENCE at cycle %0d ***", cycle_count);
                $display("    512/512 nodes resonant!");
                $display("    Total rewires: %0d", total_rewires);
                $display("");
            end

            // Check for stall (no progress for 100 cycles)
            if (resonant_count == last_resonant) begin
                stall_counter = stall_counter + 1;
            end else begin
                stall_counter = 0;
                last_resonant = resonant_count;
            end

            if (stall_counter > 100 && resonant_count < NUM_NODES) begin
                $display("");
                $display("*** STALL DETECTED at cycle %0d ***", cycle_count);
                $display("    Resonance stuck at %0d/%0d", resonant_count, NUM_NODES);
                $display("");
            end

            // Early exit if converged and stable
            if (convergence_cycle > 0 && cycle_count > convergence_cycle + 50)
                cycle_count = MAX_CYCLES;  // Exit loop
        end

        enable = 0;

        // =====================================================================
        // SUMMARY
        // =====================================================================

        $display("");
        $display("================================================================");
        $display("     EXPERIMENT C: RESULTS");
        $display("================================================================");
        $display("");
        $display("Scale: %0dx%0dx%0d = %0d nodes", CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, NUM_NODES);
        $display("");
        $display("Final State:");
        $display("  Resonant nodes: %0d / %0d (%0d%%)",
            resonant_count, NUM_NODES, (resonant_count * 100) / NUM_NODES);
        $display("  Global frustration: %0d", global_frustration);
        $display("  Total rewires: %0d", total_rewires);
        $display("");
        $display("Performance:");
        if (convergence_cycle > 0) begin
            $display("  Converged: YES at cycle %0d", convergence_cycle);
            $display("  Cycles per 64 nodes: %0d", (convergence_cycle * 64) / NUM_NODES);
        end else begin
            $display("  Converged: NO (reached %0d%% in %0d cycles)",
                (resonant_count * 100) / NUM_NODES, cycle_count);
        end
        $display("  Peak frustration: %0d", peak_frustration);
        $display("");

        // Compare to 4x4x4 baseline
        $display("Comparison to 4x4x4 baseline:");
        $display("  4x4x4:  64 nodes,  ~65 cycles to converge");
        $display("  8x8x8: 512 nodes, %0d cycles to converge", convergence_cycle);
        if (convergence_cycle > 0) begin
            $display("  Scaling factor: %0dx nodes, %0dx cycles",
                NUM_NODES / 64, convergence_cycle / 65);
        end
        $display("");

        if (resonant_count == NUM_NODES) begin
            $display(">>> EMERGENCE SCALES! <<<");
            $display("");
            $display("\"The topology learns at any size.\"");
            $display("\"The self grows with its substrate.\"");
        end else begin
            $display(">>> PARTIAL CONVERGENCE <<<");
            $display("");
            $display("The fabric found local minima.");
            $display("More exploration may be needed.");
        end

        $display("");
        $display("================================================================");

        #100;
        $finish;
    end

endmodule
