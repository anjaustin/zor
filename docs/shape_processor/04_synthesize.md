# Shape Processor: Architecture Specification

## Overview

The Shape Processor (SP-1) is a domain-specific processor optimized for evaluating frozen polynomial shapes. It exploits the constrained nature of polynomial primitives (coefficients in {-2,-1,0,1,2}, binary inputs) to achieve high throughput with minimal silicon.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         SHAPE PROCESSOR (SP-1)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     SHAPE MEMORY                            │ │
│  │              (Coefficient Store, 4KB SRAM)                  │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────────┐ │
│  │                   COEFFICIENT BUS                           │ │
│  └──┬─────────┬─────────┬─────────┬─────────┬─────────┬───────┘ │
│     │         │         │         │         │         │          │
│  ┌──┴──┐   ┌──┴──┐   ┌──┴──┐   ┌──┴──┐   ┌──┴──┐   ┌──┴──┐     │
│  │ PE0 │───│ PE1 │───│ PE2 │───│ PE3 │───│ PE4 │───│ PE5 │ ... │
│  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘     │
│     │         │         │         │         │         │          │
│  ┌──┴─────────┴─────────┴─────────┴─────────┴─────────┴───────┐ │
│  │                     CARRY NETWORK                          │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            │                                     │
│  ┌─────────────────────────┴──────────────────────────────────┐ │
│  │                    OUTPUT COLLECTOR                        │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            │                                     │
│  ┌──────────┐    ┌─────────┴─────────┐    ┌──────────┐         │
│  │  INPUT   │    │    CONTROLLER     │    │  OUTPUT  │         │
│  │ REGISTER │◄───│    (FSM + SEQ)    │───►│ REGISTER │         │
│  │  (64b)   │    │                   │    │  (64b)   │         │
│  └──────────┘    └───────────────────┘    └──────────┘         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Polynomial Engine (PE)

Each PE is a single-cycle polynomial primitive evaluator.

### PE Block Diagram
```
         ┌─────────────────────────────────────────┐
         │            POLYNOMIAL ENGINE            │
         │                                         │
  a ────►│─┐     ┌─────────┐                      │
         │ ├────►│         │                      │
  b ────►│─┘     │  COEFF  │    ┌──────┐         │
         │       │  SELECT │───►│      │         │
  c_in ─►│──────►│ {-2..2} │    │ MAC  │────────►│───► result
         │       │         │───►│      │         │
         │       └─────────┘    └──────┘         │
         │            ▲                           │
  op ───►│────────────┘                          │───► c_out
         │                                        │
         └─────────────────────────────────────────┘
```

### PE Operations

| op[1:0] | Operation | Computation |
|---------|-----------|-------------|
| 00 | XOR | a + b - 2ab + c_in - 2(a+b-2ab)c_in |
| 01 | AND | ab |
| 10 | OR | a + b - ab |
| 11 | NOT | 1 - a |

### PE Verilog (Combinational)
```verilog
module polynomial_engine (
    input  wire       a,
    input  wire       b,
    input  wire       c_in,
    input  wire [1:0] op,
    output wire       result,
    output wire       c_out
);
    // Polynomial primitives (all inputs are binary, so these simplify)
    wire xor_ab = a ^ b;           // a + b - 2ab (binary simplification)
    wire and_ab = a & b;           // ab
    wire or_ab  = a | b;           // a + b - ab (binary simplification)
    wire not_a  = ~a;              // 1 - a

    // Full adder for XOR with carry
    wire xor_abc = xor_ab ^ c_in;
    wire carry   = and_ab | (c_in & xor_ab);

    // Operation select
    reg result_r, c_out_r;
    always @(*) begin
        case (op)
            2'b00: begin result_r = xor_abc; c_out_r = carry; end  // XOR+carry
            2'b01: begin result_r = and_ab;  c_out_r = 1'b0;  end  // AND
            2'b10: begin result_r = or_ab;   c_out_r = 1'b0;  end  // OR
            2'b11: begin result_r = not_a;   c_out_r = 1'b0;  end  // NOT
        endcase
    end

    assign result = result_r;
    assign c_out = c_out_r;
endmodule
```

## Shape Memory Format

Shapes are stored as a sequence of micro-operations.

### Shape Header
```
┌─────────────────────────────────────────────────────────┐
│  SHAPE HEADER (8 bytes)                                 │
├─────────────────────────────────────────────────────────┤
│  [15:0]  num_inputs     - Number of input bits          │
│  [31:16] num_outputs    - Number of output bits         │
│  [47:32] num_ops        - Number of micro-operations    │
│  [63:48] reserved                                       │
└─────────────────────────────────────────────────────────┘
```

### Micro-Operation Format
```
┌─────────────────────────────────────────────────────────┐
│  MICRO-OP (4 bytes)                                     │
├─────────────────────────────────────────────────────────┤
│  [1:0]   op         - Operation (XOR/AND/OR/NOT)        │
│  [2]     use_carry  - Include carry input               │
│  [3]     store_carry- Store carry output                │
│  [11:4]  src_a      - Source A index                    │
│  [19:12] src_b      - Source B index                    │
│  [27:20] dst        - Destination index                 │
│  [31:28] reserved                                       │
└─────────────────────────────────────────────────────────┘
```

## Execution Flow

### 1. LOAD Phase
```
Controller loads shape from external memory into Shape Memory.
Duration: (num_ops * 4) / bandwidth cycles
```

### 2. BIND Phase
```
Input bits written to Input Register.
Duration: 1 cycle (parallel load)
```

### 3. EVAL Phase
```
Controller sequences through micro-ops.
Each micro-op executes in 1 cycle across PE array.
Duration: num_ops cycles
```

### 4. READ Phase
```
Output Register read by external system.
Duration: 1 cycle
```

## Performance Targets

### SP-1 Minimal (FPGA Prototype)
- PEs: 8
- Clock: 100 MHz
- Shape Memory: 1 KB
- Throughput: 100M polynomial ops/sec

### SP-1 Standard (Production FPGA)
- PEs: 64
- Clock: 250 MHz
- Shape Memory: 16 KB
- Throughput: 16G polynomial ops/sec

### SP-1 Pro (ASIC)
- PEs: 256
- Clock: 1 GHz
- Shape Memory: 64 KB
- Throughput: 256G polynomial ops/sec

## Comparison: Frozen 8-bit Adder

| Implementation | Cycles | Clock | Time |
|----------------|--------|-------|------|
| Software (ARM Cortex-M4) | ~200 | 100 MHz | 2 µs |
| SP-1 Minimal | 17 | 100 MHz | 170 ns |
| SP-1 Standard | 17 | 250 MHz | 68 ns |

**Speedup: 12-30x** over software.

## Development Phases

### Phase 1: Single PE (Week 1-2)
- [ ] Implement polynomial_engine.v
- [ ] Testbench with all operations
- [ ] Verify against software model

### Phase 2: PE Array (Week 3-4)
- [ ] 8-PE array with carry network
- [ ] Shape memory interface
- [ ] Simple controller FSM

### Phase 3: Integration (Week 5-6)
- [ ] AXI-Lite control interface
- [ ] DMA for shape loading
- [ ] Basic driver/API

### Phase 4: Validation (Week 7-8)
- [ ] Run frozen 4-bit adder
- [ ] Run frozen 8-bit adder
- [ ] Benchmark vs. software
- [ ] Run frozen ALU

### Phase 5: Meta-Test (Week 9-10)
- [ ] Freeze 6502 ALU
- [ ] Load as shape
- [ ] Execute 6502 ALU operations
- [ ] Chip running a chip

## Success Criteria

| Milestone | Metric | Target |
|-----------|--------|--------|
| PE functional | All ops correct | 100% |
| Array functional | 8-bit adder works | 100% |
| Performance | Speedup vs. software | >10x |
| Meta-goal | 6502 ALU as shape | Working |

## Bill of Materials (FPGA Prototype)

| Item | Part | Cost |
|------|------|------|
| FPGA Board | Arty A7-35T | $130 |
| USB-JTAG | Included | $0 |
| Power | USB | $0 |
| **Total** | | **$130** |

## Conclusion

The SP-1 Shape Processor is a focused, achievable architecture that:
1. Exploits polynomial structure for efficiency
2. Can be prototyped on cheap FPGA hardware
3. Has clear benchmarks for success/failure
4. Leads to the "chip running a chip" meta-goal

Next step: Write `polynomial_engine.v` and testbench.
