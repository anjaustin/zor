# GILLIES Vulkan Makefile

CC = gcc
CFLAGS = -O3 -Wall -fPIC
LDFLAGS = -lvulkan -lm

# Targets
TARGETS = gillies_vulkan_shapes gillies_vulkan_bench rigorous_test libgillies.so

all: $(TARGETS)

gillies_vulkan_shapes: gillies_vulkan_shapes.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

gillies_vulkan_bench: gillies_vulkan_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

rigorous_test: rigorous_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Shared library for Python bindings
libgillies.so: gillies_lib.c
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

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
