# =============================================================================
# NGP RTL Makefile
# =============================================================================
#
# Build targets for the Neural Geometric Processor Verilog RTL.
#
# Simulators supported:
#   - Icarus Verilog (iverilog)
#   - Verilator
#   - VCS (if available)
#
# Synthesis targets:
#   - Yosys (for gate count estimation)
#   - Vivado (for Xilinx FPGAs)
#
# =============================================================================

# Source files
SOURCES = \
    zit_detector.v \
    shapes_logic.v \
    ngp_core.v

# Top modules for different targets
TOP_ZIT = zit_detector
TOP_NGP = ngp_core
TOP_SIMPLE = ngp_simple
TOP_TB = zit_testbench

# Output directories
BUILD_DIR = build
SIM_DIR = $(BUILD_DIR)/sim
SYN_DIR = $(BUILD_DIR)/syn

# Tool detection
IVERILOG := $(shell command -v iverilog 2> /dev/null)
VERILATOR := $(shell command -v verilator 2> /dev/null)
YOSYS := $(shell command -v yosys 2> /dev/null)

# Default target
.PHONY: all
all: info

# =============================================================================
# Simulation Targets
# =============================================================================

.PHONY: sim sim-iverilog sim-verilator

sim: sim-iverilog

sim-iverilog: $(SIM_DIR)/zit_tb
	@echo "Running Zit detector testbench..."
	$(SIM_DIR)/zit_tb

$(SIM_DIR)/zit_tb: $(SOURCES)
	@mkdir -p $(SIM_DIR)
ifdef IVERILOG
	iverilog -DSIMULATION -o $@ $^
else
	@echo "Icarus Verilog not found. Install with: apt install iverilog"
	@exit 1
endif

sim-verilator:
ifdef VERILATOR
	@mkdir -p $(SIM_DIR)
	verilator --cc --exe --build -j 0 \
		--top-module $(TOP_ZIT) \
		$(SOURCES)
else
	@echo "Verilator not found. Install with: apt install verilator"
endif

# =============================================================================
# Synthesis Targets (Gate Count Estimation)
# =============================================================================

.PHONY: synth synth-yosys gates

synth: synth-yosys

synth-yosys: $(SYN_DIR)/ngp_synth.json
	@echo "Synthesis complete. See $(SYN_DIR)/ for results."

$(SYN_DIR)/ngp_synth.json: $(SOURCES)
	@mkdir -p $(SYN_DIR)
ifdef YOSYS
	yosys -p "\
		read_verilog $(SOURCES); \
		hierarchy -top $(TOP_SIMPLE); \
		proc; opt; techmap; opt; \
		stat; \
		write_json $@"
else
	@echo "Yosys not found. Install with: apt install yosys"
	@exit 1
endif

gates: $(SOURCES)
	@echo "=== Gate Count Estimation ==="
ifdef YOSYS
	@yosys -q -p "\
		read_verilog $(SOURCES); \
		hierarchy -top $(TOP_SIMPLE); \
		proc; opt; techmap; opt; \
		stat" 2>/dev/null | grep -E "(cells|Chip area)"
else
	@echo "Yosys not found. Install for gate count estimation."
endif

# =============================================================================
# Lint and Checks
# =============================================================================

.PHONY: lint check

lint:
ifdef VERILATOR
	verilator --lint-only --Wall $(SOURCES)
else
	@echo "Verilator not found. Install for linting."
endif

check: lint sim

# =============================================================================
# Documentation
# =============================================================================

.PHONY: docs

docs:
	@echo "=== NGP RTL Documentation ==="
	@echo ""
	@echo "Modules:"
	@echo "  zit_detector    - Core recognition circuit (popcount(S^vx) < theta)"
	@echo "  popcount_512    - 512-bit population count tree"
	@echo "  resonance_register - XOR accumulation state"
	@echo "  zit_unit        - Complete Zit unit with state"
	@echo ""
	@echo "  shape_xor       - XOR shape (opcode 0x00)"
	@echo "  shape_and       - AND shape (opcode 0x01)"
	@echo "  shape_or        - OR shape (opcode 0x02)"
	@echo "  shape_not       - NOT shape (opcode 0x03)"
	@echo "  shape_nand      - NAND shape (opcode 0x04)"
	@echo "  shape_nor       - NOR shape (opcode 0x05)"
	@echo "  shape_xnor      - XNOR shape (opcode 0x06)"
	@echo "  logic_fabric    - All logic shapes in parallel"
	@echo ""
	@echo "  ngp_core        - Full NGP with shape bands"
	@echo "  ngp_simple      - Minimal NGP (recognition only)"
	@echo "  ngp_array       - Array of NGP cores"
	@echo "  ngp_with_routing - NGP with routing table interface"
	@echo ""
	@echo "Estimated Gate Counts:"
	@echo "  popcount_512:   ~900 gates"
	@echo "  zit_detector:   ~1,500 gates"
	@echo "  ngp_simple:     ~2,500 gates"
	@echo "  ngp_core:       ~10,000 gates"
	@echo "  Full NGP v2:    ~53,000 gates"

# =============================================================================
# Clean
# =============================================================================

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR)

# =============================================================================
# Info
# =============================================================================

.PHONY: info

info:
	@echo "=== NGP RTL Build System ==="
	@echo ""
	@echo "Available targets:"
	@echo "  make sim         - Run simulation (Icarus Verilog)"
	@echo "  make synth       - Run synthesis (Yosys)"
	@echo "  make gates       - Estimate gate count"
	@echo "  make lint        - Lint check (Verilator)"
	@echo "  make check       - Run all checks"
	@echo "  make docs        - Show documentation"
	@echo "  make clean       - Clean build artifacts"
	@echo ""
	@echo "Tool availability:"
ifdef IVERILOG
	@echo "  [x] Icarus Verilog (iverilog)"
else
	@echo "  [ ] Icarus Verilog (iverilog) - not found"
endif
ifdef VERILATOR
	@echo "  [x] Verilator"
else
	@echo "  [ ] Verilator - not found"
endif
ifdef YOSYS
	@echo "  [x] Yosys"
else
	@echo "  [ ] Yosys - not found"
endif
