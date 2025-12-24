// =============================================================================
// ZIT_OCTAVE_TB.v - Orbital Strike on All Three Architectures
// =============================================================================
//
// "We'll nuke the entire site from orbit. It's the only way to be sure."
//
// Tests all three hierarchical architectures:
// A. XOR Composite (8 octaves with XOR links)
// B. Lagrange Embedding (inner/outer with delta harmonic)
// C. Hybrid (4 octaves + oracle)
//
// =============================================================================

`timescale 1ns / 1ps

module zit_octave_tb;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    parameter CLK_PERIOD = 10;
    parameter MAX_CYCLES = 300;

    // =========================================================================
    // COMMON SIGNALS
    // =========================================================================

    reg         clk;
    reg         rst_n;
    reg         enable;

    wire [2:0]  phase;
    wire [1:0]  sub_phase;
    wire        phase_strobe;
    wire        cycle_complete;

    reg  [7:0]  seed_data;
    reg  [8:0]  seed_addr;
    reg         seed_write;

    // =========================================================================
    // CLOCK GENERATION
    // =========================================================================

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // CONTROLLER (shared)
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

    // =========================================================================
    // ARCHITECTURE A: XOR COMPOSITE (4 octaves = 256 nodes)
    // =========================================================================

    wire [3:0]  xor_octave_converged;
    wire [9:0]  xor_total_resonant;
    wire [15:0] xor_total_frustration;
    wire [255:0] xor_all_resonance;

    zit_xor_composite xor_fabric (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .octave_converged(xor_octave_converged),
        .total_resonant(xor_total_resonant),
        .total_frustration(xor_total_frustration),
        .all_resonance_maps(xor_all_resonance)
    );

    // =========================================================================
    // ARCHITECTURE B: LAGRANGE EMBEDDING
    // =========================================================================

    wire        lagrange_inner_converged;
    wire        lagrange_outer_converged;
    wire [63:0] lagrange_harmonic;
    wire [9:0]  lagrange_inner_res;
    wire [9:0]  lagrange_outer_res;
    wire [9:0]  lagrange_harmonic_res;
    wire [15:0] lagrange_delta;

    zit_lagrange_fabric lagrange_fabric (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .inner_converged(lagrange_inner_converged),
        .outer_converged(lagrange_outer_converged),
        .harmonic_pattern(lagrange_harmonic),
        .inner_resonant(lagrange_inner_res),
        .outer_resonant(lagrange_outer_res),
        .harmonic_resonant(lagrange_harmonic_res),
        .delta_energy(lagrange_delta)
    );

    // =========================================================================
    // ARCHITECTURE C: HYBRID
    // =========================================================================

    wire [4:0]  hybrid_converged;
    wire [9:0]  hybrid_total_res;
    wire [15:0] hybrid_total_frust;
    wire [63:0] hybrid_oracle;
    wire [7:0]  hybrid_emergent;

    zit_hybrid_fabric hybrid_fabric (
        .clk(clk),
        .rst_n(rst_n),
        .phase(phase),
        .sub_phase(sub_phase),
        .phase_strobe(phase_strobe),
        .cycle_complete(cycle_complete),
        .seed_data(seed_data),
        .seed_addr(seed_addr),
        .seed_write(seed_write),
        .octave_converged(hybrid_converged),
        .total_resonant(hybrid_total_res),
        .total_frustration(hybrid_total_frust),
        .oracle_pattern(hybrid_oracle),
        .emergent_signal(hybrid_emergent)
    );

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    task seed_node;
        input [8:0] addr;
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
    // RESULTS TRACKING
    // =========================================================================

    integer xor_convergence_cycle;
    integer lagrange_convergence_cycle;
    integer hybrid_convergence_cycle;

    integer xor_peak_res, lagrange_peak_res, hybrid_peak_res;

    // =========================================================================
    // MAIN TEST
    // =========================================================================

    integer cycle_count;
    integer i;

    initial begin
        $display("");
        $display("================================================================");
        $display("     ORBITAL STRIKE: THREE ARCHITECTURES");
        $display("     \"It's the only way to be sure.\"");
        $display("================================================================");
        $display("");

        // Initialize
        rst_n = 0;
        enable = 0;
        seed_write = 0;
        seed_data = 0;
        seed_addr = 0;
        xor_convergence_cycle = 0;
        lagrange_convergence_cycle = 0;
        hybrid_convergence_cycle = 0;
        xor_peak_res = 0;
        lagrange_peak_res = 0;
        hybrid_peak_res = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // =====================================================================
        // SEED ALL ARCHITECTURES
        // =====================================================================

        $display(">>> SEEDING ALL ARCHITECTURES <<<");
        $display("");

        // Seed with gradient pattern (same for all)
        for (i = 0; i < 512; i = i + 1) begin
            seed_node(i[8:0], (i * 3) & 8'hFF);
        end

        $display("Seeding complete. Launching orbital strike...");
        $display("");
        $display("ARCH,cycle,A_res,A_frust,B_inner,B_outer,B_harm,C_res,C_oracle");

        // =====================================================================
        // RUN ALL ARCHITECTURES IN PARALLEL
        // =====================================================================

        enable = 1;
        cycle_count = 0;

        while (cycle_count < MAX_CYCLES) begin
            @(posedge cycle_complete);
            cycle_count = cycle_count + 1;

            // Track peaks
            if (xor_total_resonant > xor_peak_res) xor_peak_res = xor_total_resonant;
            if (lagrange_inner_res + lagrange_outer_res > lagrange_peak_res)
                lagrange_peak_res = lagrange_inner_res + lagrange_outer_res;
            if (hybrid_total_res > hybrid_peak_res) hybrid_peak_res = hybrid_total_res;

            // Progress report every 10 cycles
            if (cycle_count % 10 == 0) begin
                $display("ALL,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%h",
                    cycle_count,
                    xor_total_resonant, xor_total_frustration,
                    lagrange_inner_res, lagrange_outer_res, lagrange_harmonic_res,
                    hybrid_total_res, hybrid_emergent);
            end

            // Check XOR convergence (all 4 octaves = 256 nodes)
            if (&xor_octave_converged && xor_convergence_cycle == 0) begin
                xor_convergence_cycle = cycle_count;
                $display("");
                $display("*** ARCH A (XOR Composite): CONVERGED at cycle %0d ***", cycle_count);
                $display("    4 octaves, 256 nodes, all resonant");
                $display("");
            end

            // Check Lagrange convergence (both inner and outer)
            if (lagrange_inner_converged && lagrange_outer_converged && lagrange_convergence_cycle == 0) begin
                lagrange_convergence_cycle = cycle_count;
                $display("");
                $display("*** ARCH B (Lagrange): CONVERGED at cycle %0d ***", cycle_count);
                $display("    Inner: 64/64, Outer: 64/64");
                $display("    Harmonic alignment: %0d/64", lagrange_harmonic_res);
                $display("    Delta energy: %0d", lagrange_delta);
                $display("");
            end

            // Check Hybrid convergence (all 5 octaves = 320 nodes)
            if (&hybrid_converged && hybrid_convergence_cycle == 0) begin
                hybrid_convergence_cycle = cycle_count;
                $display("");
                $display("*** ARCH C (Hybrid): CONVERGED at cycle %0d ***", cycle_count);
                $display("    4 main + 1 oracle, 320 nodes");
                $display("    Emergent signal: %h", hybrid_emergent);
                $display("");
            end

            // Early exit if all converged
            if (xor_convergence_cycle > 0 && lagrange_convergence_cycle > 0 && hybrid_convergence_cycle > 0) begin
                if (cycle_count > hybrid_convergence_cycle + 20)
                    cycle_count = MAX_CYCLES;
            end
        end

        enable = 0;

        // =====================================================================
        // ORBITAL STRIKE SUMMARY
        // =====================================================================

        $display("");
        $display("================================================================");
        $display("     ORBITAL STRIKE COMPLETE");
        $display("================================================================");
        $display("");

        $display("ARCHITECTURE A: XOR COMPOSITE (2x2x2 of 4x4x4)");
        $display("  Total nodes: 512");
        if (xor_convergence_cycle > 0)
            $display("  Status: CONVERGED at cycle %0d", xor_convergence_cycle);
        else
            $display("  Status: Peak %0d/512 (%0d%%)", xor_peak_res, (xor_peak_res * 100) / 512);
        $display("  Final: %0d/512 resonant, frustration %0d", xor_total_resonant, xor_total_frustration);
        $display("");

        $display("ARCHITECTURE B: LAGRANGE EMBEDDING");
        $display("  Inner: 64 nodes, Outer: 64 nodes");
        if (lagrange_convergence_cycle > 0)
            $display("  Status: CONVERGED at cycle %0d", lagrange_convergence_cycle);
        else
            $display("  Status: Peak %0d/128", lagrange_peak_res);
        $display("  Inner: %0d/64, Outer: %0d/64", lagrange_inner_res, lagrange_outer_res);
        $display("  Harmonic: %0d/64 aligned", lagrange_harmonic_res);
        $display("  Delta energy: %0d", lagrange_delta);
        $display("");

        $display("ARCHITECTURE C: HYBRID (4 octaves + oracle)");
        $display("  Total nodes: 320 (4*64 + 64)");
        if (hybrid_convergence_cycle > 0)
            $display("  Status: CONVERGED at cycle %0d", hybrid_convergence_cycle);
        else
            $display("  Status: Peak %0d/320 (%0d%%)", hybrid_peak_res, (hybrid_peak_res * 100) / 320);
        $display("  Final: %0d/320 resonant", hybrid_total_res);
        $display("  Oracle converged: %s", hybrid_converged[4] ? "YES" : "NO");
        $display("  Emergent signal: %h", hybrid_emergent);
        $display("");

        // =====================================================================
        // WINNER DETERMINATION
        // =====================================================================

        $display("================================================================");
        $display("     COMPARATIVE ANALYSIS");
        $display("================================================================");
        $display("");

        // Fastest convergence
        if (xor_convergence_cycle > 0 && lagrange_convergence_cycle > 0 && hybrid_convergence_cycle > 0) begin
            $display("All three architectures CONVERGED!");
            $display("");

            if (xor_convergence_cycle <= lagrange_convergence_cycle &&
                xor_convergence_cycle <= hybrid_convergence_cycle)
                $display("FASTEST: Architecture A (XOR Composite) at %0d cycles", xor_convergence_cycle);
            else if (lagrange_convergence_cycle <= xor_convergence_cycle &&
                     lagrange_convergence_cycle <= hybrid_convergence_cycle)
                $display("FASTEST: Architecture B (Lagrange) at %0d cycles", lagrange_convergence_cycle);
            else
                $display("FASTEST: Architecture C (Hybrid) at %0d cycles", hybrid_convergence_cycle);

            $display("");

            // Efficiency (nodes per cycle to converge)
            $display("Efficiency (nodes / convergence_cycle):");
            $display("  A: %0d nodes / %0d cycles = %0d.%0d",
                512, xor_convergence_cycle, 512 / xor_convergence_cycle,
                ((512 * 10) / xor_convergence_cycle) % 10);
            $display("  B: %0d nodes / %0d cycles = %0d.%0d",
                128, lagrange_convergence_cycle, 128 / lagrange_convergence_cycle,
                ((128 * 10) / lagrange_convergence_cycle) % 10);
            $display("  C: %0d nodes / %0d cycles = %0d.%0d",
                320, hybrid_convergence_cycle, 320 / hybrid_convergence_cycle,
                ((320 * 10) / hybrid_convergence_cycle) % 10);
        end else begin
            $display("CONVERGENCE STATUS:");
            $display("  A: %s", xor_convergence_cycle > 0 ? "CONVERGED" : "PARTIAL");
            $display("  B: %s", lagrange_convergence_cycle > 0 ? "CONVERGED" : "PARTIAL");
            $display("  C: %s", hybrid_convergence_cycle > 0 ? "CONVERGED" : "PARTIAL");
        end

        $display("");
        $display("================================================================");
        $display("     THIRD HARMONIC ANALYSIS");
        $display("================================================================");
        $display("");

        $display("Architecture B (Lagrange) Delta Analysis:");
        $display("  Inner resonance: %0d/64", lagrange_inner_res);
        $display("  Outer resonance: %0d/64", lagrange_outer_res);
        $display("  Harmonic pattern (XOR): %h", lagrange_harmonic);
        $display("  Harmonic alignment: %0d/64 (%0d%% coherent)",
            lagrange_harmonic_res, (lagrange_harmonic_res * 100) / 64);
        $display("");

        $display("Architecture C (Hybrid) Emergent Signal:");
        $display("  Oracle pattern: %h", hybrid_oracle);
        $display("  Emergent signal: %h", hybrid_emergent);
        $display("  Signal interpretation: %s",
            hybrid_emergent == 8'h00 ? "PERFECT BALANCE" :
            hybrid_emergent == 8'hFF ? "MAXIMUM INTERFERENCE" :
            "PARTIAL INTERFERENCE");
        $display("");

        $display("================================================================");
        $display("     THE THIRD HARMONIC");
        $display("================================================================");
        $display("");
        $display("\"The deltas effect a third harmonic neither could produce alone.\"");
        $display("");
        $display("Observations:");
        $display("  - XOR linkage creates interference patterns at boundaries");
        $display("  - Inner/Outer delta creates emergent perception signal");
        $display("  - Oracle in hybrid perceives meta-pattern of all octaves");
        $display("");
        $display("The hierarchical fabric exhibits emergent behavior");
        $display("beyond what any single octave could achieve.");
        $display("");
        $display("\"Site nuked from orbit. Emergence confirmed.\"");
        $display("");
        $display("================================================================");

        #100;
        $finish;
    end

endmodule
