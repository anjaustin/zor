# Skeptics Proof - Makefile
#
# Build and run the compiled XOR resonance proof.
#
#   make           # Build
#   make run       # Build and run
#   make clean     # Clean

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

.PHONY: all run clean python

all: prove_it

prove_it: prove_it.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo ""
	@echo "Built: ./prove_it"
	@echo "Run with: ./prove_it"

run: prove_it
	./prove_it

# Also run Python demos
python:
	@echo "Running Python demos..."
	python3 01_see_it.py
	python3 02_break_it.py
	python3 03_measure_it.py

# Run everything
all-tests: run python
	@echo ""
	@echo "All tests complete."

clean:
	rm -f prove_it results.json

# Show what's here
info:
	@echo ""
	@echo "XOR Resonance - Skeptics Package"
	@echo "================================"
	@echo ""
	@echo "Compiled (C):"
	@echo "  make && ./prove_it     # Build and run C proof"
	@echo ""
	@echo "Interpreted (Python):"
	@echo "  python3 01_see_it.py   # Watch it work"
	@echo "  python3 02_break_it.py # Find the limits"
	@echo "  python3 03_measure_it.py # Get numbers"
	@echo ""
	@echo "The C version uses hardware popcount and"
	@echo "512-bit signatures. No interpreter overhead."
	@echo ""
