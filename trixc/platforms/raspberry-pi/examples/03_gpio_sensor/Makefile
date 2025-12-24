# TRIXC Pi - GPIO Sensor Example
#
# Build: make
# Run:   sudo ./gpio_sensor
# Clean: make clean

CC = gcc
CFLAGS = -O2 -Wall -I../../include -DTRIXC_PI_GPIO
LDFLAGS = $(shell sdl2-config --libs) -lpigpio -lm

SDL_CFLAGS = $(shell sdl2-config --cflags)

SRC = main.c ../../src/trixc_pi.c
TARGET = gpio_sensor

.PHONY: all clean run sim

# Build with GPIO support
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built with GPIO support. Run with: sudo ./$(TARGET)"

clean:
	rm -f $(TARGET) gpio_sensor_sim

# Run with sudo (required for GPIO)
run: $(TARGET)
	@echo "Running with GPIO (requires sudo)..."
	sudo ./$(TARGET)

# Simulation mode (no GPIO hardware needed)
sim:
	$(CC) -O2 -Wall -I../../include $(SDL_CFLAGS) -o gpio_sensor_sim $(SRC) $(shell sdl2-config --libs) -lm
	@echo "Built simulation mode. Run with: ./gpio_sensor_sim"
	@echo "Press 'A' and 'B' keys to simulate buttons."
