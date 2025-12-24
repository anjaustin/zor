# ZIT Fabric Visualizer

Real-time 3D visualization of homeo-adaptive topological learning.

## Requirements

- Qt 6.2 or later
- CMake 3.16 or later
- C++17 compiler
- OpenGL 3.3+ capable GPU

### Install Qt6 on Ubuntu/Debian

```bash
sudo apt install qt6-base-dev libqt6opengl6-dev qt6-base-dev-tools cmake
```

For headless/software rendering, also install:

```bash
sudo apt install xvfb libgl1-mesa-dri
```

### Install Qt6 on macOS

```bash
brew install qt@6
```

## Build

```bash
cd viz
mkdir build && cd build
cmake ..
make
```

## Run

### With a Display (Desktop/Monitor)

```bash
./zit_viz
```

### Headless (SSH/Server)

If running without a physical display, use Xvfb:

```bash
# Install Xvfb if needed
sudo apt install xvfb

# Start virtual display and run
Xvfb :99 -screen 0 1280x800x24 &
DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 ./zit_viz
```

### On Jetson/Embedded

For direct framebuffer access (no X server):

```bash
QT_QPA_PLATFORM=eglfs ./zit_viz
```

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| Space | Play/Pause simulation |
| R | Reset to initial state |
| 1 | Color by state (hue gradient) |
| 2 | Color by resistance level |
| 3 | Color by resonance (default) |
| E | Toggle edge visibility |

### Mouse

| Action | Effect |
|--------|--------|
| Left drag | Rotate camera |
| Scroll | Zoom in/out |

## Visual Language

### Node Colors (Resonance Mode)

| Color | Meaning |
|-------|---------|
| Teal | Resonant (stable, no resistance) |
| Blue | Low resistance |
| Purple | Building resistance |
| Orange | High resistance (near threshold) |
| Red (pulsing) | Currently rewiring |

### Edges

- Gray lines show current neighbor connections
- When a node rewires, its connectivity changes

## What You're Seeing

512 nodes arranged in an 8x8x8 3D torus. Each cycle:

1. Every node compares its value with neighbors
2. Values swap according to the comparator rule
3. Nodes that couldn't participate accumulate resistance
4. High-resistance nodes try random new neighbors
5. Better neighbors are kept, worse ones reverted

Watch as the fabric converges from chaotic initial state to 100% resonance.

The topology IS the learned model.

---

*Second Star Constant: 1122911624*
