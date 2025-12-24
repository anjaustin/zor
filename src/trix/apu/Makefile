# TRIX APU Makefile
#
# Builds the endogenous APU components

CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c11
LDFLAGS = -lm

# Targets
.PHONY: all clean test standalone

all: test_apu alu6502

# Test suite
test_apu: test_apu.c apu.h shapes.h providence.h alu6502.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Standalone ALU executable
alu6502: alu6502.h shapes.h apu.h
	$(CC) $(CFLAGS) -DTRIX_ALU6502_STANDALONE -x c -o $@ $< $(LDFLAGS)

# Run tests
test: test_apu
	./test_apu

# Run standalone ALU examples
demo: alu6502
	@echo "=== 6502 ALU Demo ==="
	./alu6502 ADC 16 32
	./alu6502 ADC 255 1
	./alu6502 SBC 50 30
	./alu6502 AND 255 170
	./alu6502 ORA 240 15
	./alu6502 EOR 255 170
	./alu6502 ASL 64 0
	./alu6502 LSR 2 0
	./alu6502 INC 65 0
	./alu6502 DEC 66 0

# Show binary sizes
size: alu6502 test_apu
	@echo "=== Binary Sizes ==="
	ls -la alu6502 test_apu
	@echo ""
	@echo "Stripped:"
	cp alu6502 alu6502.stripped
	strip alu6502.stripped
	ls -la alu6502.stripped
	rm alu6502.stripped

clean:
	rm -f test_apu alu6502 *.o *.stripped
