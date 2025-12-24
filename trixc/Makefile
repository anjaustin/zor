# TRIXC Makefile
#
# The TriX Compiler
#
# "Shapes are opcodes. C is machine code."

CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c11 -I./include
LDFLAGS = -lm

# Directories
INC_DIR = include/trixc
TEST_DIR = test
BUILD_DIR = build

# Targets
.PHONY: all clean test test-rigorous demo size install help

all: build/test_apu build/alu6502 build/test_rigorous

# =============================================================================
# BUILD
# =============================================================================

build/test_apu: $(TEST_DIR)/test_apu.c $(INC_DIR)/apu.h $(INC_DIR)/shapes.h $(INC_DIR)/providence.h $(INC_DIR)/alu6502.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

build/alu6502: $(INC_DIR)/alu6502.h $(INC_DIR)/shapes.h $(INC_DIR)/apu.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTRIX_ALU6502_STANDALONE -x c -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

build/test_rigorous: $(TEST_DIR)/test_rigorous.c $(INC_DIR)/*.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

build/sparse_octave: $(INC_DIR)/sparse_octave.h $(INC_DIR)/providence.h $(INC_DIR)/apu.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTRIX_SPARSE_OCTAVE_STANDALONE -x c -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# =============================================================================
# TEST
# =============================================================================

test: build/test_apu
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║      TRIXC Test Suite                  ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@./$(BUILD_DIR)/test_apu
	@echo ""

test-rigorous: build/test_rigorous
	@echo ""
	@./$(BUILD_DIR)/test_rigorous
	@echo ""

test-all: test test-rigorous
	@echo "All tests complete."

# =============================================================================
# DEMO
# =============================================================================

demo: build/alu6502
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║      6502 ALU Demo                     ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "── Addition ──"
	@./$(BUILD_DIR)/alu6502 ADC 16 32
	@./$(BUILD_DIR)/alu6502 ADC 255 1
	@echo ""
	@echo "── Subtraction ──"
	@./$(BUILD_DIR)/alu6502 SBC 50 30
	@echo ""
	@echo "── Logic ──"
	@./$(BUILD_DIR)/alu6502 AND 255 170
	@./$(BUILD_DIR)/alu6502 ORA 240 15
	@./$(BUILD_DIR)/alu6502 EOR 255 170
	@echo ""
	@echo "── Shift ──"
	@./$(BUILD_DIR)/alu6502 ASL 64 0
	@./$(BUILD_DIR)/alu6502 LSR 2 0
	@echo ""
	@echo "── Increment/Decrement ──"
	@./$(BUILD_DIR)/alu6502 INC 65 0
	@./$(BUILD_DIR)/alu6502 DEC 66 0
	@echo ""

# =============================================================================
# SIZE ANALYSIS
# =============================================================================

size: build/alu6502 build/test_apu
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║      Binary Size Analysis              ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "── File sizes ──"
	@ls -lh $(BUILD_DIR)/alu6502 $(BUILD_DIR)/test_apu
	@echo ""
	@echo "── Stripped ──"
	@cp $(BUILD_DIR)/alu6502 $(BUILD_DIR)/alu6502.stripped
	@strip $(BUILD_DIR)/alu6502.stripped
	@ls -lh $(BUILD_DIR)/alu6502.stripped
	@rm $(BUILD_DIR)/alu6502.stripped
	@echo ""
	@echo "── Code sections ──"
	@size $(BUILD_DIR)/alu6502
	@echo ""
	@echo "The 'text' column is actual code size (~6 KB)."
	@echo "File size includes ELF headers and metadata."
	@echo ""

# =============================================================================
# INSTALL
# =============================================================================

PREFIX ?= /usr/local

install: build/alu6502
	@echo "Installing to $(PREFIX)..."
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/include/trixc
	@cp $(BUILD_DIR)/alu6502 $(PREFIX)/bin/trix-alu6502
	@cp $(INC_DIR)/*.h $(PREFIX)/include/trixc/
	@echo ""
	@echo "Installed:"
	@echo "  $(PREFIX)/bin/trix-alu6502"
	@echo "  $(PREFIX)/include/trixc/*.h"
	@echo ""

# =============================================================================
# CLEAN
# =============================================================================

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned."

# =============================================================================
# HELP
# =============================================================================

help:
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║      TRIXC - The TriX Compiler         ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "Targets:"
	@echo "  all            Build everything (default)"
	@echo "  test           Run the basic test suite (48 tests)"
	@echo "  test-rigorous  Run rigorous tests (65536+ tests)"
	@echo "  test-all       Run all tests"
	@echo "  demo           Run the 6502 ALU demo"
	@echo "  size           Show binary size analysis"
	@echo "  install        Install to PREFIX (default: /usr/local)"
	@echo "  clean          Remove build artifacts"
	@echo "  help           Show this message"
	@echo ""
	@echo "Examples:"
	@echo "  make                           # Build"
	@echo "  make test                      # Basic tests"
	@echo "  make test-rigorous             # Exhaustive tests"
	@echo "  make demo                      # Demo"
	@echo "  make size                      # Size analysis"
	@echo "  make PREFIX=~/.local install   # Install locally"
	@echo ""
	@echo "\"Shapes are opcodes. C is machine code.\""
	@echo ""
