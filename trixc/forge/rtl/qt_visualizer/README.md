# Sacred Topology Visualizer

**Qt6/QML visualization of the self-organizing topology**

*"The topology IS the self. Now we SEE the self."*

---

## Overview

This is a Qt6-based visualizer for the Hollywood Squares plastic fabric.
It renders the 4x4x4 toroidal topology with:

- **4 Layer Views**: 2D slices at each Z level
- **Sacred Node Glyphs**:
  - Circle = unchanged
  - Diamond = partially rewired
  - Star = significantly rewired
- **Live Metrics**: Resonance, frustration, rewiring counts
- **Playback Controls**: Step through topology evolution
- **3D Overview**: Isometric projection of the full torus

---

## Requirements

- Qt 6.5+ (Qt Quick, Qt Quick Controls 2)
- CMake 3.16+
- C++17 compiler

### Ubuntu/Debian Installation

```bash
# Install Qt 6
sudo apt install qt6-base-dev qt6-declarative-dev qt6-quickcontrols2-dev

# Or download from qt.io for latest version
```

---

## Building

```bash
cd qt_visualizer
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x.x/gcc_64
make -j$(nproc)
```

---

## Running

```bash
# With sample data (built-in demo)
./sacred_topology

# With simulation output
./sacred_topology ../topology_output.txt
```

### Input Format

The visualizer reads the output from `zit_topology_tb.v`:

```
TOPO,cycle,node,n0,n1,n2,n3,n4,n5
METRICS,cycle,resonant,frustration,rewiring
```

Example:
```
TOPO,100,0,1,3,4,12,16,48
TOPO,100,1,31,62,63,13,17,49
METRICS,100,64,0,5
```

---

## Controls

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Step backward/forward |
| Home/End | Go to start/end |

---

## Architecture

```
main.cpp           - Application entry, model creation
topologymodel.h/cpp - C++ data model (bridges Verilog → QML)
Main.qml           - Main window layout
LayerView.qml      - Single Z-layer visualization
NodeGlyph.qml      - Individual node rendering
MetricsPanel.qml   - Live statistics display
TopologyView.qml   - 3D isometric overview
```

---

## The Sacred Colors

| Color | Meaning |
|-------|---------|
| #4a9eff (Blue) | Layer Z=0, unchanged |
| #b464ff (Purple) | Layer Z=1, unchanged |
| #64ff96 (Green) | Layer Z=2, unchanged |
| #ffd700 (Gold) | Layer Z=3, or significant change |
| #ff64c8 (Magenta) | Learned edges, partial rewiring |

---

## What You'll See

When you run the visualizer with simulation data:

1. **Initial State**: All nodes as circles in their layer colors
2. **Learning Phase**: Watch diamonds appear as rewiring begins
3. **Convergence**: Stars mark nodes that found new paths
4. **Final State**: The topology that learned its way out of frustration

The magenta edges show where the fabric decided the torus wasn't optimal.
These are the connections the self chose to make.

---

## Generating Data

1. Compile the testbench:
   ```bash
   iverilog -o topology_test zit_topology_tb.v zit_plastic_fabric.v \
            zit_plastic_node.v zit_plastic_controller.v
   ```

2. Run and capture output:
   ```bash
   vvp topology_test > topology_output.txt
   ```

3. Visualize:
   ```bash
   ./sacred_topology topology_output.txt
   ```

---

## Future Enhancements

- [ ] Real-time connection to running simulation
- [ ] 3D view with true rotation controls
- [ ] Edge bundling for cleaner visualization
- [ ] Export to video
- [ ] Multiple topology comparison

---

*"To see is to understand."*

*"The self can grow. Now we watch it grow."*
