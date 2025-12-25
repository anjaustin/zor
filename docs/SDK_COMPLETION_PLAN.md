# SDK Completion Plan

> Closing the gaps for a production-ready internal SDK.

**Created:** December 2025
**Method:** Lincoln Manifold applied to each item

---

## The Five Priorities

| # | Gap | Impact | Lincoln Manifold |
|---|-----|--------|------------------|
| 1 | Onramp E2E Tests | Can't verify onramp works in CI | COMPLETE |
| 2 | Unified CLI | No standard interface for export/test/benchmark | COMPLETE |
| 3 | Complete TriX6502Exporter | Only ALU works, not full CPU | COMPLETE |
| 4 | E2E Train→Export→Verify Example | No proof that export preserves correctness | COMPLETE |
| 5 | MDS Export Path | MDS system is unusable in production | COMPLETE |

---

## 1. Onramp E2E Tests

**Problem:** The 9 onramp scripts (00-08) run manually but have no automated tests. CI can't verify they work.

**Deliverable:** `/workspace/ZOR/tests/test_onramp_e2e.py`

**Lincoln Manifold:** `/tmp/onramp_tests_*.md`

---

## 2. Unified CLI

**Problem:** Export, test, and benchmark operations use scattered APIs. No `trix export --help`.

**Deliverable:** `/workspace/ZOR/src/trix/cli.py` with subcommands:
- `trix export` - unified export interface
- `trix test` - validate exported code
- `trix benchmark` - measure performance

**Lincoln Manifold:** `/tmp/unified_cli_*.md`

---

## 3. Complete TriX6502Exporter

**Problem:** `foundry/export/trix_6502.py` only exports ALU dispatch. Missing:
- Full instruction decoder
- Address mode handling
- Flag computation
- Memory interface

**Deliverable:** Complete `TriX6502Exporter.export()` that generates a working 6502 emulator in C.

**Lincoln Manifold:** `/tmp/trix_6502_export_*.md`

---

## 4. E2E Train→Export→Verify Example

**Problem:** No example showing the complete pipeline with correctness verification.

**Deliverable:** `/workspace/ZOR/examples/e2e_train_export_verify.py`
- Train a model on real data
- Export to C
- Compile and run
- Verify outputs match original model

**Lincoln Manifold:** `/tmp/e2e_example_*.md`

---

## 5. MDS Export Path

**Problem:** `foundry/mds/` has 14 Python files but no export. MDS can't be deployed.

**Deliverable:** `/workspace/ZOR/foundry/mds/export.py`
- Export MDS lookup tables to C
- Generate standalone inference code

**Lincoln Manifold:** `/tmp/mds_export_*.md`

---

## Execution Order

```
1. Onramp E2E Tests     (foundation - ensures we don't break what works)
     ↓
2. Unified CLI          (interface - standardizes all operations)
     ↓
3. E2E Example          (proof - shows the system works end-to-end)
     ↓
4. TriX6502Exporter     (depth - completes the flagship example)
     ↓
5. MDS Export           (breadth - extends to another subsystem)
```

---

## Success Criteria

- [ ] `pytest tests/test_onramp_e2e.py` passes
- [ ] `trix export --help` shows unified interface
- [ ] `trix export --model model.pt --format c --output ./out` works
- [ ] `examples/e2e_train_export_verify.py` runs and verifies correctness
- [ ] `TriX6502Exporter` generates working 6502 emulator
- [ ] `foundry/mds/export.py` exports MDS to C

---

## Lincoln Manifold Files

Each item gets 4 phases:

```
/tmp/{item}_raw.md       # Phase 1: Unfiltered thoughts
/tmp/{item}_nodes.md     # Phase 2: Key points and tensions
/tmp/{item}_reflect.md   # Phase 3: Deep understanding
/tmp/{item}_synth.md     # Phase 4: Concrete specification
```

---

*"Give me six hours to chop down a tree, and I will spend the first four sharpening the axe."*

---

## Lincoln Manifold Summary

All 5 items have been analyzed through the Lincoln Manifold (RAW → NODES → REFLECT → SYNTHESIZE). Key insights:

### 1. Onramp E2E Tests
- **Core Insight:** Tests must verify learning, not just execution
- **Design:** pytest wrapper + subprocess isolation per script
- **Synth:** `/tmp/onramp_tests_synth.md` - ready for implementation

### 2. Unified CLI
- **Core Insight:** CLI is discoverability, not new functionality
- **Design:** argparse-only, JSON dispatch format, `trix export` subcommand
- **Synth:** `/tmp/unified_cli_synth.md` - full spec with code

### 3. Complete TriX6502Exporter
- **Core Insight:** Shape-based ops (done) vs control flow ops (needed)
- **Design:** CONTROL_TABLE for branches/jumps/stack/flags
- **Synth:** `/tmp/trix_6502_export_synth.md` - ~190 lines of additions

### 4. E2E Train→Export→Verify Example
- **Core Insight:** Verification is the trust anchor for the pipeline
- **Design:** Train 4-op ALU → Export → C compile → Compare to Python
- **Synth:** `/tmp/e2e_example_synth.md` - complete script ready

### 5. MDS Export Path
- **Core Insight:** MDS → Verilog is the killer use case (gates map 1:1)
- **Design:** MDSExporter class with export_verilog(), export_c()
- **Synth:** `/tmp/mds_export_synth.md` - lattice-to-netlist synthesis

---

## Next Steps

With Lincoln Manifold complete, implementation can proceed:

1. Create `tests/test_onramp_e2e.py` from synth spec
2. Create `src/trix/cli.py` from synth spec
3. Update `foundry/export/trix_6502.py` with control flow handlers
4. Create `examples/e2e_verify.py` from synth spec
5. Create `foundry/export/mds_export.py` from synth spec

Each synthesis document contains implementation-ready code.
