# Reflection: Shape Processor

## What's Actually New Here?

**Honest assessment**: We're proposing a dataflow processor optimized for polynomial evaluation. This isn't revolutionary - dataflow architectures exist. CGRAs exist. What's our edge?

### Genuine Novelty
1. **Algebraic abstraction**: Think in polynomials, not gates
2. **Built-in certification**: Proof of correctness at silicon level
3. **Coefficient constraint**: Only {-2,-1,0,1,2} enables extreme simplification
4. **Shape-first design**: Architecture derived from the math, not vice versa

### Not Novel (Honest)
1. Multiply-accumulate units - commodity
2. Dataflow execution - well-studied
3. Reconfigurable arrays - FPGA/CGRA territory
4. Carry-chain optimization - 50 years of literature

## Critical Questions

### 1. Is This Better Than a CPU?
**For shapes specifically**: Yes. No fetch-decode overhead.
**For general compute**: No. CPUs are flexible.

Break-even analysis:
- Shape eval on CPU: ~10 cycles per polynomial term (memory + arithmetic)
- Shape eval on Shape Processor: ~1 cycle per term (hardwired)
- 10x improvement for pure shape workloads

But: Modern CPUs run at 5 GHz. Our chip might run at 500 MHz (FPGA) or 1 GHz (ASIC). Frequency disadvantage erodes gains.

**Verdict**: Only wins for massively parallel shape evaluation. Single shapes on single inputs? CPU might be faster due to clock speed.

### 2. Is This Better Than an FPGA?
FPGAs can implement any logic directly. Why not just synthesize shapes to FPGA?

**FPGA advantages**:
- Mature toolchain
- High clock speeds
- Proven reliability

**Shape Processor advantages**:
- Instant reconfiguration (load new shape in microseconds, not minutes)
- No synthesis step
- Polynomial-level optimization

**Verdict**: Shape Processor wins for rapid reconfiguration scenarios. FPGA wins for fixed, high-performance workloads.

### 3. Is This Better Than a GPU?
GPUs excel at SIMD parallelism. Shapes are parallel. Why not GPU?

**GPU advantages**:
- Massive parallelism (thousands of cores)
- Mature ecosystem (CUDA, etc.)
- High memory bandwidth

**Shape Processor disadvantages**:
- We'd need to build comparable parallelism
- We don't have the toolchain
- We're not cheaper

**Verdict**: For large shapes with lots of parallelism, GPU might win. Our advantage is in the small-coefficient constraint and binary specialization that GPUs can't exploit.

### 4. Who Buys This?
**Potential markets**:

| Segment | Use Case | Willingness to Pay |
|---------|----------|-------------------|
| EDA vendors | RTL simulation acceleration | High ($$$) |
| Embedded | Chip emulation on microcontrollers | Medium |
| Research | Formal verification hardware | Medium |
| Retrocomputing | Run frozen vintage CPUs | Low (hobbyist) |
| Crypto | Constant-time polynomial eval | Medium |

**Most promising**: EDA vendors who need faster simulation.

## Missing Pieces

### 1. Memory Architecture
Shapes can be huge. 2M-bit adder = millions of terms (if expanded). How do we store/stream this?
- On-chip SRAM? Limited capacity.
- Off-chip DRAM? Bandwidth bottleneck.
- Compressed representation? Decompression latency.

**Not solved yet.**

### 2. Multi-Shape Pipelining
Real workloads need multiple shapes. How do we:
- Switch between shapes?
- Pipeline multiple shapes?
- Handle dependencies?

**Not solved yet.**

### 3. I/O Interface
How do inputs/outputs connect to real systems?
- PCIe for datacenter?
- AXI for embedded?
- Simple parallel bus?

**Not specified yet.**

### 4. Programming Model
How do users express shapes?
- Raw coefficient dumps?
- Verilog → Freeze → Load?
- High-level DSL?

**Verilog → Freeze → Load is the obvious path.**

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance doesn't beat CPU | Medium | Fatal | Benchmark early on FPGA |
| Too complex to implement | Medium | High | Start with minimal PE |
| No market demand | Medium | Fatal | Talk to EDA vendors |
| FPGA prototype too slow | Low | Medium | Optimize critical path |
| Memory bandwidth bottleneck | High | High | Use compact shape format |

## Revised Strategy

### Phase 1: Prove It (FPGA)
1. Build minimal PE in Verilog
2. Build 8-PE array
3. Run frozen 4-bit adder
4. Benchmark vs. software

Success criteria: 5x speedup over C on same FPGA's ARM core.

### Phase 2: Scale It
1. Expand to 64-PE array
2. Implement carry-lookahead
3. Run frozen ALU
4. Benchmark larger shapes

Success criteria: 100x speedup for complex shapes.

### Phase 3: Prove the Meta
1. Freeze a 6502 CPU
2. Run it on Shape Processor
3. Execute real 6502 programs
4. A chip running a chip

Success criteria: Working 6502 at reasonable speed.

### Phase 4: Productize
1. ASIC exploration
2. Partner with EDA vendor
3. Or: sell FPGA IP cores

## Final Verdict

**Should we build this?**

YES, but with constraints:
1. FPGA-first (prove before committing to ASIC)
2. Benchmark-driven (no theoretical handwaving)
3. Minimal viable architecture (don't over-engineer)
4. Clear success criteria at each phase

The Shape Processor is worth exploring because:
1. It's architecturally interesting (polynomial-native compute)
2. It has a clear application (accelerate Forge outputs)
3. It can be prototyped cheaply (FPGA)
4. Failure is informative (learn about polynomial compute limits)

The risk is manageable. The potential is real. Proceed with Phase 1.
