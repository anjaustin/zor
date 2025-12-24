# TRIXC Pi - Hello XOR Example
#
# Build: make
# Run:   ./hello_xor
# Clean: make clean

CC = gcc
CFLAGS = -O2 -Wall -I../../include
LDFLAGS = $(shell sdl2-config --libs) -lm

# SDL2 compiler flags
SDL_CFLAGS = $(shell sdl2-config --cflags)

# Source files
SRC = main.c ../../src/trixc_pi.c
TARGET = hello_xor

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Debug build with symbols
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Optimized build
release: CFLAGS = -O3 -Wall -I../../include -DNDEBUG
release: $(TARGET)
