# ============================================================================
# TRIXC Pi Platform Makefile
#
# Build the TRIXC Pi runtime and all examples.
#
# Quick start:
#   make setup    # Install dependencies (run once)
#   make          # Build everything
#   make hello    # Build and run hello_xor
#
# ============================================================================

# Compiler settings
CC = gcc
CFLAGS = -O2 -Wall -Wextra -I./include
LDFLAGS = -lm

# SDL2 flags (detected automatically)
SDL_CFLAGS := $(shell sdl2-config --cflags 2>/dev/null)
SDL_LIBS := $(shell sdl2-config --libs 2>/dev/null)

# Check if SDL2 is available
ifeq ($(SDL_LIBS),)
    $(warning SDL2 not found. Run 'make setup' first.)
endif

# Optional features
ifdef GPIO
    CFLAGS += -DTRIXC_PI_GPIO
    LDFLAGS += -lpigpio
endif

ifdef CAMERA
    CFLAGS += -DTRIXC_PI_CAMERA
endif

ifdef AUDIO
    CFLAGS += -DTRIXC_PI_AUDIO
endif

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
EXAMPLES_DIR = examples
MODELS_DIR = models

# Runtime library source
RUNTIME_SRC = $(SRC_DIR)/trixc_pi.c

# ============================================================================
# TARGETS
# ============================================================================

.PHONY: all clean setup examples hello mnist gpio camera help

# Default: build runtime library
all: $(BUILD_DIR)/libtrixc_pi.a examples

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build static library
$(BUILD_DIR)/libtrixc_pi.a: $(RUNTIME_SRC) $(INC_DIR)/trixc_pi.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $(RUNTIME_SRC) -o $(BUILD_DIR)/trixc_pi.o
	ar rcs $@ $(BUILD_DIR)/trixc_pi.o
	@echo "Built: $@"

# ============================================================================
# EXAMPLES
# ============================================================================

examples: hello_xor mnist_draw

# Example 01: Hello XOR
hello_xor: $(BUILD_DIR)/hello_xor
$(BUILD_DIR)/hello_xor: $(EXAMPLES_DIR)/01_hello_xor/main.c $(RUNTIME_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $< $(RUNTIME_SRC) $(SDL_LIBS) $(LDFLAGS)
	@echo "Built: $@"

# Example 02: MNIST Draw
mnist_draw: $(BUILD_DIR)/mnist_draw
$(BUILD_DIR)/mnist_draw: $(EXAMPLES_DIR)/02_mnist_draw/main.c $(RUNTIME_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $< $(RUNTIME_SRC) $(SDL_LIBS) $(LDFLAGS)
	@echo "Built: $@"

# Example 03: GPIO Sensor (requires pigpio)
gpio_sensor: CFLAGS += -DTRIXC_PI_GPIO
gpio_sensor: LDFLAGS += -lpigpio
gpio_sensor: $(BUILD_DIR)/gpio_sensor
$(BUILD_DIR)/gpio_sensor: $(EXAMPLES_DIR)/03_gpio_sensor/main.c $(RUNTIME_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $< $(RUNTIME_SRC) $(SDL_LIBS) $(LDFLAGS)
	@echo "Built: $@ (with GPIO support)"

# Example 04: Camera Classify
camera_classify: CFLAGS += -DTRIXC_PI_CAMERA
camera_classify: $(BUILD_DIR)/camera_classify
$(BUILD_DIR)/camera_classify: $(EXAMPLES_DIR)/04_camera_classify/main.c $(RUNTIME_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $@ $< $(RUNTIME_SRC) $(SDL_LIBS) $(LDFLAGS)
	@echo "Built: $@ (with camera support)"

# ============================================================================
# RUN TARGETS (convenience)
# ============================================================================

hello: $(BUILD_DIR)/hello_xor
	@echo ""
	@echo "Running Hello XOR..."
	@echo "===================="
	./$(BUILD_DIR)/hello_xor

mnist: $(BUILD_DIR)/mnist_draw
	@echo ""
	@echo "Running MNIST Draw..."
	@echo "====================="
	./$(BUILD_DIR)/mnist_draw

gpio: $(BUILD_DIR)/gpio_sensor
	@echo ""
	@echo "Running GPIO Sensor..."
	@echo "======================"
	sudo ./$(BUILD_DIR)/gpio_sensor

camera: $(BUILD_DIR)/camera_classify
	@echo ""
	@echo "Running Camera Classify..."
	@echo "=========================="
	./$(BUILD_DIR)/camera_classify

# ============================================================================
# SETUP
# ============================================================================

setup:
	@echo "=============================================="
	@echo "  TRIXC Pi Setup"
	@echo "  \"It's all in the reflexes.\""
	@echo "=============================================="
	@echo ""
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev
	@echo ""
	@echo "SDL2 installed successfully!"
	@echo ""
	@echo "Optional dependencies:"
	@echo "  GPIO:   sudo apt-get install pigpio python3-pigpio"
	@echo "  Camera: sudo apt-get install libcamera-dev"
	@echo ""
	@echo "=============================================="
	@echo "  Setup complete!"
	@echo ""
	@echo "  Next: make hello"
	@echo "=============================================="

# ============================================================================
# CLEAN
# ============================================================================

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# ============================================================================
# HELP
# ============================================================================

help:
	@echo ""
	@echo "TRIXC Pi Platform"
	@echo "================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install dependencies (SDL2, etc.)"
	@echo ""
	@echo "Build:"
	@echo "  make                Build runtime library and examples"
	@echo "  make hello_xor      Build Hello XOR example"
	@echo "  make mnist_draw     Build MNIST Draw example"
	@echo "  make gpio_sensor    Build GPIO Sensor example (needs pigpio)"
	@echo ""
	@echo "Run:"
	@echo "  make hello          Build and run Hello XOR"
	@echo "  make mnist          Build and run MNIST Draw"
	@echo "  make gpio           Build and run GPIO Sensor (sudo)"
	@echo "  make camera         Build and run Camera Classify"
	@echo ""
	@echo "Other:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make help           Show this help"
	@echo ""
	@echo "Optional features (add to make command):"
	@echo "  GPIO=1              Enable GPIO support"
	@echo "  CAMERA=1            Enable camera support"
	@echo "  AUDIO=1             Enable audio support"
	@echo ""
	@echo "Example:"
	@echo "  make GPIO=1 gpio_sensor"
	@echo ""

# ============================================================================
# DEBUG/RELEASE VARIANTS
# ============================================================================

debug: CFLAGS += -g -O0 -DDEBUG
debug: all

release: CFLAGS = -O3 -Wall -I./include -DNDEBUG
release: all

# ============================================================================
# INSTALL (optional)
# ============================================================================

PREFIX ?= /usr/local

install: $(BUILD_DIR)/libtrixc_pi.a
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/trixc
	install -m 644 $(BUILD_DIR)/libtrixc_pi.a $(PREFIX)/lib/
	install -m 644 $(INC_DIR)/trixc_pi.h $(PREFIX)/include/trixc/
	@echo "Installed to $(PREFIX)"

uninstall:
	rm -f $(PREFIX)/lib/libtrixc_pi.a
	rm -f $(PREFIX)/include/trixc/trixc_pi.h
	rmdir $(PREFIX)/include/trixc 2>/dev/null || true
	@echo "Uninstalled from $(PREFIX)"
