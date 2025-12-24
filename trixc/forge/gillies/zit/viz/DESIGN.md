# ZIT Fabric Visualizer - Design Document

## Vision

A real-time visualization of homeo-adaptive topology that makes the invisible visible. Watch 512 nodes discover their optimal connectivity through resistance and rewiring.

**Design Language:** BMW iDrive-inspired. Dark mode. Precision typography. Subtle gradients. Information density without clutter.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZitVizApp (QApplication)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  MainWindow (QMainWindow)                   │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │           FabricView (QOpenGLWidget)                │   │  │
│  │  │                                                      │   │  │
│  │  │   3D rendering of node lattice                      │   │  │
│  │  │   - Nodes as spheres (color = state)                │   │  │
│  │  │   - Edges as lines (opacity = resistance)           │   │  │
│  │  │   - Rewiring as animated arcs                       │   │  │
│  │  │   - Camera: orbit, zoom, pan                        │   │  │
│  │  │                                                      │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │                                                             │  │
│  │  ┌──────────────────────┐  ┌────────────────────────────┐  │  │
│  │  │   ControlPanel       │  │      MetricsPanel          │  │  │
│  │  │   ──────────────     │  │      ────────────          │  │  │
│  │  │   [▶ Play] [⏸ Pause] │  │   Cycle: 114               │  │  │
│  │  │   Speed: ████░░ 2x   │  │   Resonant: 512/512        │  │  │
│  │  │   ──────────────     │  │   Rewires: 1,340           │  │  │
│  │  │   View: ○3D ●Slice   │  │   ──────────────────       │  │  │
│  │  │   Show: ☑Edges       │  │   [Resonance Chart    ]    │  │  │
│  │  │         ☑Rewires     │  │   [████████████████████]   │  │  │
│  │  └──────────────────────┘  └────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. ZitEngine (C++ wrapper around zit.c)

```cpp
class ZitEngine : public QObject {
    Q_OBJECT

public:
    ZitEngine(int dim = 8);
    ~ZitEngine();

    // Control
    void step();
    void reset(uint32_t seed = 0);
    void setSpeed(float multiplier);

    // State queries (for visualization)
    int dim() const;
    int total() const;
    int cycle() const;
    int resonant() const;
    int rewires() const;
    bool converged() const;

    // Per-node queries
    uint8_t nodeState(int id) const;
    uint8_t nodeResistance(int id) const;
    bool nodeResonant(int id) const;
    bool nodeRewiring(int id) const;
    QVector3D nodePosition(int id) const;
    QList<int> nodeNeighbors(int id) const;

signals:
    void stepped();
    void convergedSignal();
    void rewiringStarted(int nodeId, int oldNeighbor, int newNeighbor);

private:
    zit_fabric_t* m_fabric;
    QTimer* m_timer;
};
```

### 2. FabricRenderer (OpenGL 3D scene)

```cpp
class FabricRenderer : public QOpenGLWidget {
    Q_OBJECT

public:
    FabricRenderer(ZitEngine* engine, QWidget* parent = nullptr);

    // Visual modes
    enum ColorMode { ByState, ByResistance, ByResonance };
    void setColorMode(ColorMode mode);

    // View controls
    void setShowEdges(bool show);
    void setShowRewires(bool show);
    void setSliceZ(int z);  // -1 for full 3D

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    void renderNodes();
    void renderEdges();
    void renderRewireArcs();

    ZitEngine* m_engine;

    // Shaders
    QOpenGLShaderProgram* m_nodeShader;
    QOpenGLShaderProgram* m_edgeShader;

    // Geometry
    QOpenGLBuffer m_sphereVBO;
    QOpenGLBuffer m_lineVBO;

    // Camera
    QVector3D m_cameraPos;
    float m_rotX, m_rotY;
    float m_zoom;

    // State
    ColorMode m_colorMode;
    bool m_showEdges;
    bool m_showRewires;
    int m_sliceZ;
};
```

### 3. MetricsChart (Real-time graph)

```cpp
class MetricsChart : public QWidget {
    Q_OBJECT

public:
    MetricsChart(QWidget* parent = nullptr);

    void addDataPoint(int cycle, int resonant, int total);
    void reset();

protected:
    void paintEvent(QPaintEvent* e) override;

private:
    QVector<QPointF> m_resonanceData;
    int m_maxCycles;
};
```

---

## Visual Design

### Color Palette (BMW iDrive inspired)

```
Background:     #0a0a0f (near black)
Surface:        #14141a
Surface raised: #1e1e28
Accent:         #1e90ff (electric blue)
Accent hover:   #3ba0ff
Text primary:   #ffffff
Text secondary: #808080
Success:        #00d4aa (resonant nodes)
Warning:        #ffa500 (high resistance)
Danger:         #ff4444 (rewiring)
```

### Node Visualization

| State | Color | Meaning |
|-------|-------|---------|
| Resonant | Bright teal (#00d4aa) | Stable, no resistance |
| Low resistance | Cool blue (#4488ff) | Slight strain |
| Mid resistance | Purple (#8844ff) | Building tension |
| High resistance | Orange (#ffa500) | Near threshold |
| Rewiring | Pulsing red (#ff4444) | Evaluating new neighbor |

### Edge Visualization

- **Normal edge:** Thin gray line, 30% opacity
- **Active (used this cycle):** Blue line, 80% opacity
- **Broken (old neighbor):** Fading red dotted line
- **New (being evaluated):** Pulsing yellow arc

### Typography

```
Primary font:    SF Pro Display / Segoe UI (system)
Mono font:       JetBrains Mono / Consolas
Metric numbers:  Tabular lining figures
```

---

## Animation System

### State Transitions

```cpp
struct NodeAnimation {
    int nodeId;
    QColor fromColor;
    QColor toColor;
    float progress;  // 0.0 -> 1.0
    float duration;  // seconds
};
```

### Rewire Arc Animation

```cpp
struct RewireAnimation {
    int nodeId;
    QVector3D from;     // Position of old neighbor
    QVector3D to;       // Position of new neighbor
    float progress;
    bool successful;    // Determines end animation
};
```

### Camera Auto-Focus

When a rewire starts, camera can optionally smoothly pan to show the rewiring node clearly:

```cpp
void FabricRenderer::focusOnNode(int nodeId, float duration = 0.5f);
```

---

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| R | Reset simulation |
| 1/2/3 | Color mode (state/resistance/resonance) |
| E | Toggle edges |
| W | Toggle rewire arcs |
| Z/X | Slice up/down |
| Scroll | Zoom |
| Drag | Rotate |

### Touch/Mouse

- **Left drag:** Rotate camera
- **Right drag:** Pan camera
- **Scroll:** Zoom
- **Double click node:** Focus + show details

---

## Performance Budget

Target: 60 FPS with 512 nodes, 3072 potential edges

| Component | Budget |
|-----------|--------|
| Simulation step | < 1ms |
| Node rendering | < 5ms |
| Edge rendering | < 3ms |
| Animation updates | < 1ms |
| UI updates | < 1ms |
| Frame headroom | ~6ms |

### Optimizations

1. **Instanced rendering** for nodes (single draw call)
2. **Edge batching** (update only changed edges)
3. **LOD system** for large fabrics (>1000 nodes)
4. **Dirty rect** for metrics panel

---

## File Structure

```
viz/
├── DESIGN.md           # This document
├── CMakeLists.txt      # Build configuration
├── main.cpp            # Application entry
├── src/
│   ├── ZitEngine.h/cpp     # C wrapper
│   ├── FabricRenderer.h/cpp # 3D OpenGL view
│   ├── MetricsChart.h/cpp   # Real-time graph
│   ├── ControlPanel.h/cpp   # UI controls
│   └── MainWindow.h/cpp     # Top-level window
├── shaders/
│   ├── node.vert/frag      # Sphere rendering
│   ├── edge.vert/frag      # Line rendering
│   └── arc.vert/frag       # Bezier arcs
├── resources/
│   ├── style.qss           # BMW-inspired stylesheet
│   └── icons/              # UI icons
└── test/
    └── benchmark.cpp       # Performance tests
```

---

## Build Requirements

- Qt 6.2+ (Core, Widgets, OpenGL)
- CMake 3.16+
- C++17 compiler
- OpenGL 3.3+ capable GPU

### Build Commands

```bash
mkdir build && cd build
cmake ..
make
./zit_viz
```

---

## Future Enhancements

1. **VR Mode:** Immerse yourself inside the fabric
2. **Network Mode:** Visualize remote Thor GPU experiments
3. **Time Scrubbing:** Replay recorded simulations
4. **Comparison Mode:** Side-by-side different seeds
5. **Sound Design:** Map rewires to audio (cymatics feedback)

---

## The Goal

When someone runs this visualizer, they should:

1. **Immediately understand** that nodes are connected in 3D space
2. **See resistance build** as colors shift from teal to orange
3. **Watch rewiring happen** with dramatic arc animations
4. **Feel convergence** as the fabric settles into uniform teal
5. **Understand viscerally** that topology learned

*The visualization IS the explanation.*

---

*"Shape = Compute. The fabric learns itself."*
