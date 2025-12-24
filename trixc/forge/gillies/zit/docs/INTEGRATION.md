# ZIT Integration Guide

How to embed and use ZIT in your projects.

---

## Table of Contents

1. [Quick Integration](#quick-integration)
2. [Build Options](#build-options)
3. [Language Bindings](#language-bindings)
4. [Embedding Patterns](#embedding-patterns)
5. [Real-Time Visualization](#real-time-visualization)
6. [Performance Tuning](#performance-tuning)
7. [Common Integrations](#common-integrations)

---

## Quick Integration

### Minimal Setup (2 files)

Copy these files to your project:
- `zit.h` - The API header
- `zit.c` - The implementation

```c
// your_program.c
#include "zit.h"

int main() {
    zit_fabric_t* f = zit_create(8);
    zit_run(f, 0);
    zit_print(f);
    zit_destroy(f);
    return 0;
}
```

Compile:
```bash
gcc -O3 -o your_program your_program.c zit.c -lm
```

That's it. No dependencies. No build system required.

---

## Build Options

### As a Static Library

```makefile
# Makefile
CC = gcc
CFLAGS = -O3 -Wall -std=c99

libzit.a: zit.o
	ar rcs $@ $^

zit.o: zit.c zit.h
	$(CC) $(CFLAGS) -c zit.c

clean:
	rm -f libzit.a zit.o
```

Link against it:
```bash
gcc -o myapp myapp.c -L. -lzit -lm
```

### With CMake

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Add ZIT as a library
add_library(zit STATIC
    path/to/zit.c
    path/to/zit.h
)
target_include_directories(zit PUBLIC path/to/)

# Link your application
add_executable(myapp main.cpp)
target_link_libraries(myapp zit m)
```

### With Meson

```meson
# meson.build
project('myproject', 'c')

zit_lib = static_library('zit',
    'path/to/zit.c',
    include_directories: include_directories('path/to/')
)

executable('myapp',
    'main.c',
    link_with: zit_lib,
    dependencies: [dependency('m')]
)
```

---

## Language Bindings

### C++ Wrapper

```cpp
// zit_wrapper.hpp
#pragma once

extern "C" {
#include "zit.h"
}

#include <memory>
#include <functional>
#include <stdexcept>

class ZitFabric {
public:
    explicit ZitFabric(int dim = 8)
        : fabric_(zit_create(dim), zit_destroy)
    {
        if (!fabric_) {
            throw std::runtime_error("Failed to create ZIT fabric");
        }
    }

    void seed(uint32_t s) { zit_seed(fabric_.get(), s); }
    int step() { return zit_step(fabric_.get()); }
    int run(int max_cycles = 0) { return zit_run(fabric_.get(), max_cycles); }

    int cycle() const { return zit_cycle(fabric_.get()); }
    int resonant() const { return zit_resonant(fabric_.get()); }
    int total() const { return zit_total(fabric_.get()); }
    int rewires() const { return zit_rewires(fabric_.get()); }
    bool converged() const { return zit_converged(fabric_.get()); }

    // Visualization accessors
    int dim() const { return zit_dim(fabric_.get()); }
    uint8_t nodeState(int id) const { return zit_node_state(fabric_.get(), id); }
    uint8_t nodeResistance(int id) const { return zit_node_resistance(fabric_.get(), id); }
    bool nodeResonant(int id) const { return zit_node_resonant(fabric_.get(), id); }
    bool nodeRewiring(int id) const { return zit_node_rewiring(fabric_.get(), id); }

    void print() const { zit_print(fabric_.get()); }
    int exportJson(const char* path) const { return zit_export(fabric_.get(), path); }

    zit_fabric_t* raw() { return fabric_.get(); }

private:
    std::unique_ptr<zit_fabric_t, decltype(&zit_destroy)> fabric_;
};
```

Usage:
```cpp
#include "zit_wrapper.hpp"
#include <iostream>

int main() {
    ZitFabric fabric(8);
    fabric.seed(1122911624);

    while (!fabric.converged()) {
        fabric.step();
        std::cout << "Cycle " << fabric.cycle()
                  << ": " << fabric.resonant() << "/" << fabric.total()
                  << std::endl;
    }

    fabric.print();
    return 0;
}
```

### Python Bindings (ctypes)

```python
# zit.py
import ctypes
from pathlib import Path

# Load the shared library
_lib_path = Path(__file__).parent / "libzit.so"
_lib = ctypes.CDLL(str(_lib_path))

# Define types
class ZitFabric(ctypes.c_void_p):
    pass

# Function signatures
_lib.zit_create.argtypes = [ctypes.c_int]
_lib.zit_create.restype = ZitFabric

_lib.zit_destroy.argtypes = [ZitFabric]
_lib.zit_destroy.restype = None

_lib.zit_seed.argtypes = [ZitFabric, ctypes.c_uint32]
_lib.zit_seed.restype = None

_lib.zit_step.argtypes = [ZitFabric]
_lib.zit_step.restype = ctypes.c_int

_lib.zit_run.argtypes = [ZitFabric, ctypes.c_int]
_lib.zit_run.restype = ctypes.c_int

_lib.zit_resonant.argtypes = [ZitFabric]
_lib.zit_resonant.restype = ctypes.c_int

_lib.zit_total.argtypes = [ZitFabric]
_lib.zit_total.restype = ctypes.c_int

_lib.zit_cycle.argtypes = [ZitFabric]
_lib.zit_cycle.restype = ctypes.c_int

_lib.zit_converged.argtypes = [ZitFabric]
_lib.zit_converged.restype = ctypes.c_bool

_lib.zit_rewires.argtypes = [ZitFabric]
_lib.zit_rewires.restype = ctypes.c_int


class Fabric:
    """Python wrapper for ZIT fabric."""

    def __init__(self, dim: int = 8):
        self._handle = _lib.zit_create(dim)
        if not self._handle:
            raise MemoryError("Failed to create ZIT fabric")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.zit_destroy(self._handle)

    def seed(self, s: int) -> None:
        _lib.zit_seed(self._handle, s)

    def step(self) -> int:
        return _lib.zit_step(self._handle)

    def run(self, max_cycles: int = 0) -> int:
        return _lib.zit_run(self._handle, max_cycles)

    @property
    def cycle(self) -> int:
        return _lib.zit_cycle(self._handle)

    @property
    def resonant(self) -> int:
        return _lib.zit_resonant(self._handle)

    @property
    def total(self) -> int:
        return _lib.zit_total(self._handle)

    @property
    def converged(self) -> bool:
        return _lib.zit_converged(self._handle)

    @property
    def rewires(self) -> int:
        return _lib.zit_rewires(self._handle)


# Usage
if __name__ == "__main__":
    f = Fabric(8)
    f.seed(1122911624)
    cycles = f.run()
    print(f"Converged in {cycles} cycles with {f.rewires} rewires")
```

Build shared library for Python:
```bash
gcc -O3 -shared -fPIC -o libzit.so zit.c -lm
```

### Rust Bindings (FFI)

```rust
// zit.rs
use std::ffi::c_void;

#[repr(C)]
pub struct ZitFabric {
    _private: [u8; 0],
}

extern "C" {
    fn zit_create(dim: i32) -> *mut ZitFabric;
    fn zit_destroy(f: *mut ZitFabric);
    fn zit_seed(f: *mut ZitFabric, seed: u32);
    fn zit_step(f: *mut ZitFabric) -> i32;
    fn zit_run(f: *mut ZitFabric, max_cycles: i32) -> i32;
    fn zit_resonant(f: *mut ZitFabric) -> i32;
    fn zit_total(f: *mut ZitFabric) -> i32;
    fn zit_cycle(f: *mut ZitFabric) -> i32;
    fn zit_converged(f: *mut ZitFabric) -> bool;
    fn zit_rewires(f: *mut ZitFabric) -> i32;
}

pub struct Fabric {
    handle: *mut ZitFabric,
}

impl Fabric {
    pub fn new(dim: i32) -> Option<Self> {
        let handle = unsafe { zit_create(dim) };
        if handle.is_null() {
            None
        } else {
            Some(Fabric { handle })
        }
    }

    pub fn seed(&mut self, s: u32) {
        unsafe { zit_seed(self.handle, s) }
    }

    pub fn step(&mut self) -> i32 {
        unsafe { zit_step(self.handle) }
    }

    pub fn run(&mut self, max_cycles: i32) -> i32 {
        unsafe { zit_run(self.handle, max_cycles) }
    }

    pub fn resonant(&self) -> i32 {
        unsafe { zit_resonant(self.handle) }
    }

    pub fn total(&self) -> i32 {
        unsafe { zit_total(self.handle) }
    }

    pub fn cycle(&self) -> i32 {
        unsafe { zit_cycle(self.handle) }
    }

    pub fn converged(&self) -> bool {
        unsafe { zit_converged(self.handle) }
    }

    pub fn rewires(&self) -> i32 {
        unsafe { zit_rewires(self.handle) }
    }
}

impl Drop for Fabric {
    fn drop(&mut self) {
        unsafe { zit_destroy(self.handle) }
    }
}

// Make it thread-safe (each instance is independent)
unsafe impl Send for Fabric {}
```

---

## Embedding Patterns

### Pattern 1: Run to Completion

```c
zit_fabric_t* f = zit_create(8);
zit_seed(f, 1122911624);
int cycles = zit_run(f, 0);
// Use results...
zit_destroy(f);
```

### Pattern 2: Interactive Stepping

```c
zit_fabric_t* f = zit_create(8);

while (!zit_converged(f)) {
    zit_step(f);

    // Update UI, check for cancel, etc.
    if (user_cancelled()) break;

    // Yield to other tasks
    sleep_ms(10);
}

zit_destroy(f);
```

### Pattern 3: Callback-Based Progress

```c
typedef struct {
    int update_interval;
    void (*on_progress)(int cycle, int resonant, int total);
} ProgressContext;

void progress_callback(zit_fabric_t* f, void* user_data) {
    ProgressContext* ctx = (ProgressContext*)user_data;

    if (zit_cycle(f) % ctx->update_interval == 0) {
        ctx->on_progress(zit_cycle(f), zit_resonant(f), zit_total(f));
    }
}

void run_with_progress(void (*on_progress)(int, int, int)) {
    ProgressContext ctx = { .update_interval = 10, .on_progress = on_progress };

    zit_fabric_t* f = zit_create(8);
    zit_set_step_callback(f, progress_callback, &ctx);
    zit_run(f, 0);
    zit_destroy(f);
}
```

### Pattern 4: Multiple Fabrics

```c
#define NUM_EXPERIMENTS 10

void run_experiments() {
    zit_fabric_t* fabrics[NUM_EXPERIMENTS];
    int results[NUM_EXPERIMENTS];

    // Create all fabrics with different seeds
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        fabrics[i] = zit_create(8);
        zit_seed(fabrics[i], 1000 + i);
    }

    // Run all to completion
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        results[i] = zit_run(fabrics[i], 0);
    }

    // Analyze results
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        printf("Experiment %d: %d cycles, %d rewires\n",
               i, results[i], zit_rewires(fabrics[i]));
        zit_destroy(fabrics[i]);
    }
}
```

### Pattern 5: Topology Export and Analysis

```c
void analyze_topology(zit_fabric_t* f) {
    int dim = zit_dim(f);
    int total = zit_total(f);

    // Count non-local connections (rewired edges)
    int rewired_edges = 0;

    for (int i = 0; i < total; i++) {
        int x, y, z;
        zit_node_coords(f, i, &x, &y, &z);

        for (int d = 0; d < 6; d++) {
            int neighbor = zit_node_neighbor(f, i, d);
            int nx, ny, nz;
            zit_node_coords(f, neighbor, &nx, &ny, &nz);

            // Check if this is a non-adjacent connection
            int dx = abs(x - nx);
            int dy = abs(y - ny);
            int dz = abs(z - nz);

            // Account for toroidal wrap
            if (dx > dim/2) dx = dim - dx;
            if (dy > dim/2) dy = dim - dy;
            if (dz > dim/2) dz = dim - dz;

            if (dx + dy + dz > 1) {
                rewired_edges++;
            }
        }
    }

    printf("Rewired edges: %d / %d (%.1f%%)\n",
           rewired_edges, total * 6,
           100.0 * rewired_edges / (total * 6));
}
```

---

## Real-Time Visualization

### OpenGL Integration

```c
// In your render loop
void render_fabric(zit_fabric_t* f) {
    int total = zit_total(f);
    int dim = zit_dim(f);

    for (int i = 0; i < total; i++) {
        int x, y, z;
        zit_node_coords(f, i, &x, &y, &z);

        // Color based on state
        float r, g, b;
        if (zit_node_rewiring(f, i)) {
            r = 1.0f; g = 0.0f; b = 0.0f;  // Red for rewiring
        } else if (zit_node_resonant(f, i)) {
            r = 0.0f; g = 0.8f; b = 0.7f;  // Teal for resonant
        } else {
            float resistance = zit_node_resistance(f, i) / 8.0f;
            r = resistance; g = 0.3f; b = 1.0f - resistance;
        }

        draw_sphere(x - dim/2.0f, y - dim/2.0f, z - dim/2.0f, 0.15f, r, g, b);
    }
}
```

### WebGL via WebAssembly

Compile to WASM:
```bash
emcc -O3 -o zit.js zit.c \
    -s EXPORTED_FUNCTIONS='["_zit_create","_zit_destroy","_zit_step",...]' \
    -s EXPORTED_RUNTIME_METHODS='["cwrap"]'
```

JavaScript usage:
```javascript
const zit = {
    create: Module.cwrap('zit_create', 'number', ['number']),
    destroy: Module.cwrap('zit_destroy', null, ['number']),
    step: Module.cwrap('zit_step', 'number', ['number']),
    // ... etc
};

const fabric = zit.create(8);
// Run animation loop with zit.step(fabric)
```

---

## Performance Tuning

### Compiler Optimizations

```bash
# Maximum optimization
gcc -O3 -march=native -flto -o zit_fast zit.c your_app.c -lm

# With profile-guided optimization
gcc -O3 -fprofile-generate -o zit_profile zit.c your_app.c -lm
./zit_profile  # Run typical workload
gcc -O3 -fprofile-use -o zit_fast zit.c your_app.c -lm
```

### Memory Alignment

The internal structures are already well-aligned, but for SIMD:

```c
// In zit.c, replace malloc with aligned allocation
#include <stdlib.h>

void* aligned_malloc(size_t size) {
    void* ptr;
    posix_memalign(&ptr, 64, size);  // 64-byte alignment for AVX-512
    return ptr;
}
```

### Batch Operations

For Monte Carlo studies, batch fabric creation:

```c
// Pre-allocate a pool of fabrics
zit_fabric_t** fabric_pool = malloc(N * sizeof(zit_fabric_t*));
for (int i = 0; i < N; i++) {
    fabric_pool[i] = zit_create(8);
}

// Run experiments by re-seeding, not recreating
for (int experiment = 0; experiment < 1000; experiment++) {
    int idx = experiment % N;
    zit_seed(fabric_pool[idx], experiment);
    zit_run(fabric_pool[idx], 0);
    record_result(experiment, zit_cycle(fabric_pool[idx]));
}
```

---

## Common Integrations

### Game Engine (Unity via C#)

```csharp
// ZitNative.cs
using System;
using System.Runtime.InteropServices;

public static class ZitNative
{
    [DllImport("zit")]
    public static extern IntPtr zit_create(int dim);

    [DllImport("zit")]
    public static extern void zit_destroy(IntPtr f);

    [DllImport("zit")]
    public static extern int zit_step(IntPtr f);

    [DllImport("zit")]
    public static extern bool zit_converged(IntPtr f);

    // ... etc
}

// ZitFabric.cs
public class ZitFabric : IDisposable
{
    private IntPtr handle;

    public ZitFabric(int dim = 8)
    {
        handle = ZitNative.zit_create(dim);
    }

    public void Step() => ZitNative.zit_step(handle);
    public bool Converged => ZitNative.zit_converged(handle);

    public void Dispose()
    {
        if (handle != IntPtr.Zero)
        {
            ZitNative.zit_destroy(handle);
            handle = IntPtr.Zero;
        }
    }
}
```

### Scientific Computing (NumPy)

```python
import numpy as np
from zit import Fabric

def fabric_to_numpy(f: Fabric) -> dict:
    """Extract fabric state as NumPy arrays."""
    total = f.total
    dim = int(round(total ** (1/3)))

    states = np.zeros((dim, dim, dim), dtype=np.uint8)
    # Fill from accessor functions...

    return {
        'states': states,
        'cycle': f.cycle,
        'resonant': f.resonant,
        'rewires': f.rewires
    }
```

### Jupyter Notebook

```python
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from zit import Fabric

def visualize_convergence(dim=8, seed=1122911624):
    f = Fabric(dim)
    f.seed(seed)

    cycles = []
    resonance = []

    fig, ax = plt.subplots()

    while not f.converged:
        f.step()
        cycles.append(f.cycle)
        resonance.append(f.resonant / f.total)

        if f.cycle % 10 == 0:
            clear_output(wait=True)
            ax.clear()
            ax.plot(cycles, resonance)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Resonance Ratio')
            ax.set_ylim(0, 1.1)
            display(fig)

    plt.close()
    return f.cycle, f.rewires
```

---

## Troubleshooting Integration

| Issue | Cause | Solution |
|-------|-------|----------|
| Segfault on create | Out of memory | Reduce dimension |
| Slow performance | Debug build | Use -O3 optimization |
| Wrong results | Uninitialized seed | Always call zit_seed() |
| Memory leak | Missing destroy | Use RAII wrapper |
| Thread issues | Shared fabric | One fabric per thread |

---

*The topology IS the learned model.*
