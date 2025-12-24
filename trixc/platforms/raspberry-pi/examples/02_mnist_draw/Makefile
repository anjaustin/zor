# TRIXC Pi - MNIST Draw Example
#
# Build: make
# Run:   ./mnist_draw
# Clean: make clean

CC = gcc
CFLAGS = -O2 -Wall -I../../include
LDFLAGS = $(shell sdl2-config --libs) -lm

SDL_CFLAGS = $(shell sdl2-config --cflags)

SRC = main.c ../../src/trixc_pi.c
TARGET = mnist_draw

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
