# TRIX Forge CLI

**Watch neural networks think.**

```
▀█▀ █▀█ █ ▀▄▀   █▀▀ █▀█ █▀█ █▀▀ █▀▀
 █  █▀▄ █ █ █   █▀  █ █ █▀▄ █ █ █▀▀
 ▀  ▀ ▀ ▀ ▀ ▀   ▀   ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀
```

A terminal interface so clear you can watch neural networks think.

---

## Quick Start

```bash
# Run the full demo
python main.py demo

# Build a model
python main.py build model.trix --target arm64-linux

# Build with live visibility
python main.py build model.trix --target arm64-linux --glass

# Launch dashboard
python main.py dashboard model.trix

# List available targets
python main.py targets
```

---

## Three Modes

### Mode 1: SIMPLE (default)

Clean, minimal output. Anyone can use it.

```
   ▀█▀ █▀█ █ ▀▄▀   █▀▀ █▀█ █▀█ █▀▀ █▀▀
    █  █▀▄ █ █ █   █▀  █ █ █▀▄ █ █ █▀▀
    ▀  ▀ ▀ ▀ ▀ ▀   ▀   ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀  v1.0

   Building tiny_mlp.trix → arm64-linux

   [████████████████████████████████████████] 100%

   ✓ Output: tiny_mlp.nge (1.8 KB)
   ✓ Target: arm64-linux (Raspberry Pi 4+)
   ✓ Time:   0.18s
```

### Mode 2: GLASS (`--glass` flag)

Live visibility into the pipeline. See every event, every decision.

```
┌─ TRIX Forge ─────────────────────────────────────────────────────────┐
│  Model: tiny_mlp.trix                    Target: arm64-linux         │
│  Shapes: 4 frozen, 6 tensors             Size est: ~1.8 KB           │
├─ Pipeline ───────────────────────────────────────────────────────────┤
│  Parse ───→ Validate ───→ Optimize ───→ Emit ───→ Package           │
│    ✓           ✓            ✓           ✓          ◐                │
├─ Events ─────────────────────────────────────────────────────────────┤
│  [0001] parse.complete     4 shapes, 6 tensors loaded                │
│  [0002] validate.pass      all shape dimensions compatible           │
│  [0003] optimize.fuse      Linear→ReLU fused (2 ops → 1)            │
│  [0004] emit.complete      generated 847 ARM64 instructions          │
├─ Levers (active) ────────────────────────────────────────────────────┤
│  precision   │ fp16          │ size vs accuracy tradeoff             │
│  optimize    │ aggressive    │ maximum fusion and folding            │
└──────────────────────────────────────────────────────────────────────┘
```

### Mode 3: DASHBOARD (`forge dashboard`)

Full TUI development environment with:
- Model architecture visualization
- Live event stream
- Lever controls (interactive)
- Frame inspector

```
┌─ TRIX Forge Dashboard ──────────────────────────────────────── v1.0 ─┐
├─ Model Architecture ─────────────────┬─ Event Stream ────────────────┤
│  [Input] → [Hidden] → [Output]       │  f0127  │ layer[0].act        │
│   [4]       [8]        [2]           │  f0127  │ layer[1].act        │
├─ Lever Control ──────────────────────┼─ Frame Inspector ─────────────┤
│  precision   [fp32|►fp16◄|fp8]       │  Frame 127                    │
│  optimize    [none|►aggressive◄]      │  h[0]: ████████░░ 0.82        │
│  fuse_ops    [►on◄|off]              │  Output: class_1 (91%) ✓     │
├──────────────────────────────────────┴───────────────────────────────┤
│  [b]uild  [g]lass  [l]evers  [i]nspect  [r]eplay  [h]elp  [q]uit    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Frame Replay: The Killer Feature

Watch neural networks think, frame by frame.

```
┌─ Frame 127 ────────────────────────────────────────────────────────────┐
│  Input                                                                 │
│  x[0]:  0.72  ████████████████████░░░░░░░░░░                          │
│  x[1]:  0.31  ████████░░░░░░░░░░░░░░░░░░░░░░                          │
│                          │                                             │
│                          ▼                                             │
│  Hidden Layer (Linear + ReLU)                                          │
│  h[0]:  0.64  ████████████████░░░░░░░░  activated                     │
│  h[1]:  0.91  ██████████████████████░░  activated (strongest)         │
│  h[2]:  0.00  ░░░░░░░░░░░░░░░░░░░░░░░░  killed by ReLU                │
│                          │                                             │
│                          ▼                                             │
│  Output (Softmax)                                                      │
│  class_0:  0.10  ███░░░░░░░░░░░░░░░░░░░░  9.7%                        │
│  class_1:  0.90  ██████████████████████░░  90.3% ◀── SELECTED         │
│                                                                        │
│  Decision: class_1    Confidence: 90.3%    ✓ CORRECT                  │
├────────────────────────────────────────────────────────────────────────┤
│  [←] prev  [→] next  [space] play/pause  [j] jump  [q] quit           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Commands

| Command | Description |
|---------|-------------|
| `forge build <model> --target <target>` | Build model for target platform |
| `forge build <model> -t <target> --glass` | Build with live visibility |
| `forge dashboard <model>` | Launch full TUI dashboard |
| `forge replay <trace>` | Frame-by-frame replay viewer |
| `forge targets` | List available compilation targets |
| `forge inspect <file>` | Examine model or NGE file |
| `forge demo` | Run full feature demonstration |
| `forge help` | Show help |

## Flags

| Flag | Description |
|------|-------------|
| `--glass`, `-g` | Enable GLASS mode (live events) |
| `--quiet`, `-q` | Machine-readable output (CI/CD) |
| `--no-color` | Disable colors (accessibility) |
| `--trace <file>` | Write trace file for replay |
| `--output`, `-o` | Specify output file path |

---

## Available Targets

| Target | Description |
|--------|-------------|
| `c` | Portable C99 (any platform) |
| `arm64-linux` | Raspberry Pi 4+, Jetson Nano |
| `arm64-macos` | Apple Silicon (M1/M2/M3) |
| `x86-64-linux` | Linux x86-64 servers |
| `x86-64-windows` | Windows x86-64 |
| `wasm` | WebAssembly (browsers) |
| `cuda` | NVIDIA GPUs (compute 7.0+) |
| `metal` | Apple Metal GPUs |

---

## Architecture

```
forge/cli/
├── main.py           # Entry point
├── ui/
│   ├── theme.py      # Visual design system
│   ├── simple.py     # SIMPLE mode renderer
│   ├── glass.py      # GLASS mode renderer
│   ├── dashboard.py  # DASHBOARD TUI
│   └── replay.py     # Frame replay viewer
└── core/
    ├── forge.py      # Backend integration
    ├── events.py     # Event stream handling
    └── trace.py      # Trace file I/O
```

---

## Design Philosophy

### Trust Through Transparency

The interface quality signals system quality. A polished CLI builds confidence in the underlying compiler.

### Glassbox All The Way Down

The CLI embodies the Glassbox principle - it's transparent about its own operation, not just the model's.

### Progressive Power

Simple surface, infinite depth. The same tool serves kids and ML engineers.

```
SIMPLE   →  GLASS   →  DASHBOARD
  ↓           ↓           ↓
Anyone    Developers   Power Users
```

---

## Color Semantics

Colors carry meaning, not just aesthetics:

| Color | Meaning |
|-------|---------|
| Green ✓ | Success, correct |
| Red ✗ | Error, incorrect |
| Amber ◐ | In progress |
| Gray · | Pending |
| Cyan | Accent, selected |
| Heat map | Activation intensity |

All colors are paired with icons for accessibility.

---

## Keyboard Shortcuts (Dashboard/Replay)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `h`, `?` | Help |
| `j`, `↓` | Down |
| `k`, `↑` | Up |
| `←`, `→` | Previous/Next frame |
| `Space` | Play/Pause |
| `a`, `d` | Jump 10 frames |
| `/` | Search |
| `Enter` | Select/Apply |
| `r` | Reset |

---

## Requirements

- Python 3.8+
- No external dependencies for core functionality
- Optional: `rich`, `textual` for enhanced rendering

---

*"No one's laughing us out of anywhere."*

---

```
    "Like I told my last wife, I says:
     'Honey, I never compile faster than I can see.
      Besides that, it's all in the reflexes.'"

                              — Jack Burton, debugging neural networks
```

```
    "When some wild-eyed, eight-foot-tall neural network grabs your
     softmax layer and shakes it like a sack of dead ReLUs, you just
     stare that big sucker right back in the eye and you say:

     'I can SEE you now, frame by frame.'"

                              — Jack Burton, on the Frame Replay feature
```

---

*Watch neural networks think.*
