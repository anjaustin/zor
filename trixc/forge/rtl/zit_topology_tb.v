// =============================================================================
// ZIT_TOPOLOGY_TB.v - Testbench for Topology Visualization
// =============================================================================
//
// "To see is to understand."
//
// Dumps topology to file for visualization.
// Uses hierarchical access to read internal node state.
//
// =============================================================================

`timescale 1ns / 1ps

module zit_topology_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter CLK_PERIOD = 10;
    parameter NUM_CYCLES = 200;
    parameter DUMP_INTERVAL = 20;

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
    reg  [5:0]  seed_addr;
    reg         seed_write;

    wire        all_resonant;
    wire [63:0] resonance_map;
    wire [9:0]  global_frustration;
    wire [9:0]  resonant_count;
    wire [63:0] rewiring_map;

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

    zit_plastic_fabric #(
        .CUBE_SIZE(4),
        .STATE_WIDTH(8),
        .FRUSTRATION_BITS(8),
        .REWIRE_THRESHOLD(16)
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
        .rewiring_map(rewiring_map),
        .state_0(), .state_1(), .state_2(), .state_3(),
        .state_4(), .state_5(), .state_6(), .state_7(),
        .state_8(), .state_9(), .state_10(), .state_11(),
        .state_12(), .state_13(), .state_14(), .state_15(),
        .state_16(), .state_17(), .state_18(), .state_19(),
        .state_20(), .state_21(), .state_22(), .state_23(),
        .state_24(), .state_25(), .state_26(), .state_27(),
        .state_28(), .state_29(), .state_30(), .state_31(),
        .state_32(), .state_33(), .state_34(), .state_35(),
        .state_36(), .state_37(), .state_38(), .state_39(),
        .state_40(), .state_41(), .state_42(), .state_43(),
        .state_44(), .state_45(), .state_46(), .state_47(),
        .state_48(), .state_49(), .state_50(), .state_51(),
        .state_52(), .state_53(), .state_54(), .state_55(),
        .state_56(), .state_57(), .state_58(), .state_59(),
        .state_60(), .state_61(), .state_62(), .state_63()
    );

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    task seed_node;
        input [5:0] addr;
        input [7:0] value;
        begin
            @(posedge clk);
            seed_addr <= addr;
            seed_data <= value;
            seed_write <= 1;
            @(posedge clk);
            seed_write <= 0;
            repeat(3) @(posedge clk);
        end
    endtask

    // Dump sample nodes using hierarchical access (constant indices)
    task dump_sample_topology;
        input integer cycle_num;
        begin
            // Node 0: corner
            $display("TOPO,%0d,0,%0d,%0d,%0d,%0d,%0d,%0d",
                cycle_num,
                dut.gen_nodes[0].node_inst.neighbor_idx[0],
                dut.gen_nodes[0].node_inst.neighbor_idx[1],
                dut.gen_nodes[0].node_inst.neighbor_idx[2],
                dut.gen_nodes[0].node_inst.neighbor_idx[3],
                dut.gen_nodes[0].node_inst.neighbor_idx[4],
                dut.gen_nodes[0].node_inst.neighbor_idx[5]);
            // Node 1
            $display("TOPO,%0d,1,%0d,%0d,%0d,%0d,%0d,%0d",
                cycle_num,
                dut.gen_nodes[1].node_inst.neighbor_idx[0],
                dut.gen_nodes[1].node_inst.neighbor_idx[1],
                dut.gen_nodes[1].node_inst.neighbor_idx[2],
                dut.gen_nodes[1].node_inst.neighbor_idx[3],
                dut.gen_nodes[1].node_inst.neighbor_idx[4],
                dut.gen_nodes[1].node_inst.neighbor_idx[5]);
            // Node 21: middle-ish
            $display("TOPO,%0d,21,%0d,%0d,%0d,%0d,%0d,%0d",
                cycle_num,
                dut.gen_nodes[21].node_inst.neighbor_idx[0],
                dut.gen_nodes[21].node_inst.neighbor_idx[1],
                dut.gen_nodes[21].node_inst.neighbor_idx[2],
                dut.gen_nodes[21].node_inst.neighbor_idx[3],
                dut.gen_nodes[21].node_inst.neighbor_idx[4],
                dut.gen_nodes[21].node_inst.neighbor_idx[5]);
            // Node 32: middle layer
            $display("TOPO,%0d,32,%0d,%0d,%0d,%0d,%0d,%0d",
                cycle_num,
                dut.gen_nodes[32].node_inst.neighbor_idx[0],
                dut.gen_nodes[32].node_inst.neighbor_idx[1],
                dut.gen_nodes[32].node_inst.neighbor_idx[2],
                dut.gen_nodes[32].node_inst.neighbor_idx[3],
                dut.gen_nodes[32].node_inst.neighbor_idx[4],
                dut.gen_nodes[32].node_inst.neighbor_idx[5]);
            // Node 63: far corner
            $display("TOPO,%0d,63,%0d,%0d,%0d,%0d,%0d,%0d",
                cycle_num,
                dut.gen_nodes[63].node_inst.neighbor_idx[0],
                dut.gen_nodes[63].node_inst.neighbor_idx[1],
                dut.gen_nodes[63].node_inst.neighbor_idx[2],
                dut.gen_nodes[63].node_inst.neighbor_idx[3],
                dut.gen_nodes[63].node_inst.neighbor_idx[4],
                dut.gen_nodes[63].node_inst.neighbor_idx[5]);
        end
    endtask

    // =========================================================================
    // MAIN TEST
    // =========================================================================

    integer cycle_count;
    integer i;
    integer converged_cycle;

    initial begin
        $display("# TOPOLOGY VISUALIZATION EXPERIMENT");
        $display("# Format: TOPO,cycle,node,n0,n1,n2,n3,n4,n5");
        $display("# Format: METRICS,cycle,resonant,frustration,rewiring");
        $display("#");
        $display("# Torus initial neighbors for reference:");
        $display("# Node 0:  +X=1, -X=3, +Y=4, -Y=12, +Z=16, -Z=48");
        $display("# Node 63: +X=60, -X=62, +Y=51, -Y=59, +Z=47, -Z=15");
        $display("#");

        rst_n = 0;
        enable = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Seed with gradient
        $display("# Seeding with gradient: node[i] = i * 4");
        for (i = 0; i < 64; i = i + 1) begin
            seed_node(i[5:0], i[7:0] * 4);
        end

        $display("#");
        $display("# Initial topology (should be torus):");
        dump_sample_topology(0);

        enable = 1;
        cycle_count = 0;
        converged_cycle = 0;

        $display("#");
        $display("# Running with plasticity...");
        $display("#");

        for (i = 0; i < NUM_CYCLES; i = i + 1) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Metrics every cycle (count rewiring manually)
            begin : count_rewiring
                integer rw_count, rw_i;
                rw_count = 0;
                for (rw_i = 0; rw_i < 64; rw_i = rw_i + 1)
                    if (rewiring_map[rw_i]) rw_count = rw_count + 1;
                $display("METRICS,%0d,%0d,%0d,%0d",
                    cycle_count, resonant_count, global_frustration, rw_count);
            end

            // Topology at intervals
            if (cycle_count % DUMP_INTERVAL == 0) begin
                dump_sample_topology(cycle_count);
            end

            // Mark convergence
            if (resonant_count == 64 && converged_cycle == 0) begin
                converged_cycle = cycle_count;
                $display("# === CONVERGED at cycle %0d ===", cycle_count);
                dump_sample_topology(cycle_count);
            end
        end

        $display("#");
        $display("# Final topology:");
        dump_sample_topology(cycle_count);

        $display("#");
        $display("# ============================================");
        $display("# SUMMARY");
        $display("# ============================================");
        $display("# Final resonance: %0d / 64", resonant_count);
        $display("# Final frustration: %0d", global_frustration);
        $display("# Converged at cycle: %0d", converged_cycle);
        $display("#");

        enable = 0;
        #100;
        $finish;
    end

endmodule
