# GILLIES Makefile
#
# Geometric Instruction Language Layer In Every System
#
# Build targets:
#   make demo      - Build the minimal proof demo
#   make run       - Build and run the demo
#   make clean     - Remove build artifacts
#
# Requirements:
#   - CUDA toolkit (nvcc)
#   - Jetson AGX Thor or compatible GPU
#

# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -O3 -std=c++17
NVCC_FLAGS += -I include

# Architecture (Jetson AGX Thor uses Ampere architecture)
# Use native for auto-detection on the target machine
NVCC_FLAGS += -arch=native

# Directories
SRC_DIR = src
DEMO_DIR = demo
BUILD_DIR = build

# Sources
CORE_SRC = $(SRC_DIR)/gillies_core.cu
DEMO_SRC = $(DEMO_DIR)/demo_gillies.cu
TEST_SRC = $(DEMO_DIR)/test_rigorous.cu

# Targets
DEMO_BIN = $(BUILD_DIR)/demo_gillies
TEST_BIN = $(BUILD_DIR)/test_rigorous

# =============================================================================
# BUILD RULES
# =============================================================================

.PHONY: all demo test-rigorous run run-rigorous clean

all: demo test-rigorous

demo: $(DEMO_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DEMO_BIN): $(DEMO_SRC) $(CORE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CORE_SRC) $(DEMO_SRC)
	@echo ""
	@echo "Built: $@"
	@echo ""

test-rigorous: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(CORE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CORE_SRC) $(TEST_SRC)
	@echo ""
	@echo "Built: $@"
	@echo ""

run: demo
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "                     RUNNING GILLIES DEMO"
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo ""
	@./$(DEMO_BIN)

run-rigorous: test-rigorous
	@echo ""
	@./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned."

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

.PHONY: check test

# Quick compile check
check:
	$(NVCC) $(NVCC_FLAGS) -c $(CORE_SRC) -o /dev/null
	@echo "Core compiles OK."

# Alias for run
test: run
