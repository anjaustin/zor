# =============================================================================
# ZIT-1 Experiment Makefile
# =============================================================================
# Usage:
#   make test        - Run standard test suite
#   make test-quick  - Run quick tests only
#   make test-full   - Run full test suite
#   make test-stress - Run stress tests (large scales)
#   make all         - Build all experiments
#   make clean       - Remove binaries
# =============================================================================

NVCC = nvcc
NVCC_FLAGS = -O3

# All experiment binaries
EXPERIMENTS = fabric_4x4x4 fabric_64 fabric_64_v2 fabric_128 fabric_256 \
              fabric_384 fabric_512 sequential_fabric sequential_fabric_8x8x8 \
              sequential_fabric_16x16x16 sequential_fabric_32x32x32 \
              turbo_fabric turbo_fabric_scaled turbo_fabric_warmup test_harness

.PHONY: all clean test test-quick test-full test-stress help

help:
	@echo "ZIT-1 Experiment Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make test        - Run standard test suite"
	@echo "  make test-quick  - Run quick tests only (4x4x4, 8x8x8)"
	@echo "  make test-full   - Run full test suite (up to 262K nodes)"
	@echo "  make test-stress - Run stress tests (up to 16.7M nodes)"
	@echo "  make all         - Build all experiment binaries"
	@echo "  make clean       - Remove all binaries"
	@echo ""
	@echo "Individual experiments:"
	@echo "  make fabric_64   - Build 64x64x64 experiment"
	@echo "  make fabric_128  - Build 128x128x128 experiment"
	@echo "  make fabric_256  - Build 256x256x256 experiment"
	@echo "  make fabric_384  - Build 384x384x384 experiment"

# Test targets
test: test_harness
	./test_harness --full

test-quick: test_harness
	./test_harness --quick

test-full: test_harness
	./test_harness --full -v

test-stress: test_harness
	./test_harness --stress -v

# Build all
all: $(EXPERIMENTS)

# Generic rule for CUDA files
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean
clean:
	rm -f $(EXPERIMENTS)
	rm -f *.o

# Test harness (primary target)
test_harness: test_harness.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
