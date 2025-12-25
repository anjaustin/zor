# GILLIES Vulkan Makefile

CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lvulkan -lm

# Targets
TARGETS = gillies_vulkan_shapes gillies_vulkan_bench

all: $(TARGETS)

gillies_vulkan_shapes: gillies_vulkan_shapes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

gillies_vulkan_bench: gillies_vulkan_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Shader compilation (requires spirv-tools)
xor_shader.spv: xor_shader.spvasm
	spirv-as $< -o $@
	spirv-val $@

# Run targets
run: gillies_vulkan_shapes
	./gillies_vulkan_shapes

bench: gillies_vulkan_bench
	./gillies_vulkan_bench

clean:
	rm -f $(TARGETS)

.PHONY: all run bench clean
