"""
DASHBOARD Mode - Full TUI experience.

The complete development environment in the terminal:
- Model architecture visualization
- Live event stream
- Lever controls
- Frame inspector
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from .theme import THEME, LOGO, format_size, format_time


@dataclass
class DashboardState:
    """Complete state for the dashboard."""
    model_name: str = "unnamed"
    target: str = "c"

    # Architecture
    layers: List[Dict[str, Any]] = field(default_factory=list)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Levers
    levers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Current frame for inspection
    current_frame: Optional[Dict[str, Any]] = None
    frame_id: int = 0

    # UI state
    active_panel: str = "events"  # events, levers, frame
    scroll_offset: int = 0


class Dashboard:
    """
    Full TUI dashboard for TRIX Forge.

    Layout:
    ┌─────────────────────┬──────────────────────┐
    │   Model Arch        │    Event Stream      │
    │                     │                      │
    ├─────────────────────┼──────────────────────┤
    │   Lever Control     │    Frame Inspector   │
    │                     │                      │
    └─────────────────────┴──────────────────────┘
    """

    def __init__(self, color: bool = True):
        self.color = color
        self.theme = THEME
        self.state = DashboardState()
        self._width = 78
        self._height = 30

    def _c(self, text: str, color: str) -> str:
        """Apply color."""
        if not self.color:
            return text
        colors = {
            THEME.success: "\033[92m",
            THEME.error: "\033[91m",
            THEME.warning: "\033[93m",
            THEME.info: "\033[94m",
            THEME.progress: "\033[93m",
            THEME.muted: "\033[90m",
            THEME.header: "\033[97m",
            THEME.text: "\033[37m",
            THEME.accent: "\033[96m",
            THEME.dim: "\033[90m",
        }
        code = colors.get(color, "\033[37m")
        return f"{code}{text}\033[0m"

    def clear_screen(self) -> None:
        """Clear terminal."""
        print("\033[2J\033[H", end="")

    def _pad(self, text: str, width: int) -> str:
        """Pad text to width, handling ANSI codes."""
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', text)
        padding = width - len(visible)
        if padding > 0:
            return text + " " * padding
        return text[:width]

    def render_architecture_panel(self, width: int, height: int) -> List[str]:
        """Render the model architecture panel."""
        t = self.theme
        lines = []

        # Header
        header = f"{t.box_tl}{t.box_h} {self._c('Model Architecture', THEME.header)} {t.box_h * (width - 23)}{t.box_t_down}"
        lines.append(header)

        # Architecture diagram
        lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        if self.state.layers:
            # Draw layers as boxes
            diagram_line = "  "
            for i, layer in enumerate(self.state.layers[:4]):
                name = layer.get("name", f"L{i}")[:8]
                diagram_line += f"[{self._c(name, THEME.accent)}]"
                if i < len(self.state.layers) - 1:
                    diagram_line += f" {t.arrow_right} "

            lines.append(f"{t.box_v}{self._pad(diagram_line, width - 1)}{t.box_v}")

            # Shapes
            shapes_line = "  "
            for i, layer in enumerate(self.state.layers[:4]):
                shape = layer.get("shape", "?")
                shapes_line += f" {self._c(str(shape), THEME.muted):^8}"
                if i < len(self.state.layers) - 1:
                    shapes_line += "   "

            lines.append(f"{t.box_v}{self._pad(shapes_line, width - 1)}{t.box_v}")
        else:
            lines.append(f"{t.box_v}  {self._c('No model loaded', THEME.muted)}{' ' * (width - 20)}{t.box_v}")
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        # Fill remaining height
        while len(lines) < height - 1:
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        return lines

    def render_events_panel(self, width: int, height: int) -> List[str]:
        """Render the event stream panel."""
        t = self.theme
        lines = []

        # Header
        header = f"{t.box_t_down}{t.box_h} {self._c('Event Stream', THEME.header)} {t.box_h * (width - 17)}{t.box_tr}"
        lines.append(header)

        lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        # Events
        max_events = height - 4
        recent = self.state.events[-max_events:] if self.state.events else []

        for event in recent:
            frame = event.get("frame", 0)
            etype = event.get("type", "")[:15]
            msg = event.get("message", "")[:20]

            line = f"  {self._c(f'f{frame:04d}', THEME.muted)} {t.box_v} {self._c(etype, THEME.accent)}"
            lines.append(f"{t.box_v}{self._pad(line, width - 1)}{t.box_v}")

        # Fill
        while len(lines) < height - 1:
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        # Footer
        footer = f"  [{self._c('f', THEME.accent)}] filter  [{self._c('c', THEME.accent)}] clear"
        lines.append(f"{t.box_v}{self._pad(footer, width - 1)}{t.box_v}")

        return lines

    def render_levers_panel(self, width: int, height: int) -> List[str]:
        """Render the lever control panel."""
        t = self.theme
        lines = []

        # Header
        header = f"{t.box_t_right}{t.box_h} {self._c('Lever Control', THEME.header)} {t.box_h * (width - 19)}{t.box_cross}"
        lines.append(header)

        lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        # Levers
        for name, lever in list(self.state.levers.items())[:5]:
            value = lever.get("value", lever.get("default", "?"))
            options = lever.get("options", [])

            if options:
                # Enum lever - show as selector
                opt_str = ""
                for opt in options[:3]:
                    if opt == value:
                        opt_str += f"[{self._c('►' + str(opt) + '◄', THEME.accent)}]"
                    else:
                        opt_str += f"[{opt}]"
                line = f"  {self._c(name, THEME.header):12} {opt_str}"
            else:
                # Numeric lever - show as slider
                line = f"  {self._c(name, THEME.header):12} {self._c(str(value), THEME.accent)}"

            lines.append(f"{t.box_v}{self._pad(line, width - 1)}{t.box_v}")

        # Fill
        while len(lines) < height - 2:
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        # Footer
        footer = f"  [{self._c('Enter', THEME.accent)}] apply  [{self._c('r', THEME.accent)}] reset"
        lines.append(f"{t.box_v}{self._pad(footer, width - 1)}{t.box_v}")

        return lines

    def render_frame_panel(self, width: int, height: int) -> List[str]:
        """Render the frame inspector panel."""
        t = self.theme
        lines = []

        # Header
        header = f"{t.box_cross}{t.box_h} {self._c('Frame Inspector', THEME.header)} {t.box_h * (width - 20)}{t.box_t_left}"
        lines.append(header)

        lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        frame = self.state.current_frame
        if frame:
            lines.append(f"{t.box_v}  {self._c(f'Frame {self.state.frame_id}', THEME.header)}{' ' * (width - 15)}{t.box_v}")
            lines.append(f"{t.box_v}  {t.box_h * (width - 5)}{' '}{t.box_v}")

            # Input
            inp = frame.get("input", [])
            inp_str = f"Input: [{', '.join(f'{v:.1f}' for v in inp[:4])}]"
            lines.append(f"{t.box_v}  {self._c(inp_str, THEME.text)}{' ' * (width - len(inp_str) - 4)}{t.box_v}")
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

            # Activations
            for layer_name, acts in frame.get("activations", {}).items():
                lines.append(f"{t.box_v}  {self._c(layer_name, THEME.accent)}{' ' * (width - len(layer_name) - 4)}{t.box_v}")
                for i, val in enumerate(acts[:3]):
                    bar_len = int(abs(val) * 15)
                    bar = t.bar_full * bar_len + t.bar_one * (15 - bar_len)
                    ann = ""
                    if val < 0.01:
                        ann = self._c(" (dead)", THEME.dim)
                    lines.append(f"{t.box_v}  h[{i}]: {self._c(bar, THEME.cool)} {val:.2f}{ann}{' ' * 5}{t.box_v}")

            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

            # Output
            out = frame.get("output", {})
            pred = out.get("class", "?")
            conf = out.get("confidence", 0)
            lines.append(f"{t.box_v}  Output: {self._c(f'class_{pred}', THEME.header)} ({conf:.0%}) {self._c(t.icon_success, THEME.success)}{' ' * (width - 35)}{t.box_v}")
        else:
            lines.append(f"{t.box_v}  {self._c('No frame selected', THEME.muted)}{' ' * (width - 22)}{t.box_v}")

        # Fill
        while len(lines) < height - 1:
            lines.append(f"{t.box_v}{' ' * (width - 1)}{t.box_v}")

        return lines

    def render_footer(self) -> str:
        """Render the bottom control bar."""
        t = self.theme
        controls = f"  [{self._c('b', THEME.accent)}]uild  [{self._c('g', THEME.accent)}]lass  [{self._c('l', THEME.accent)}]evers  [{self._c('i', THEME.accent)}]nspect  [{self._c('r', THEME.accent)}]eplay  [{self._c('h', THEME.accent)}]elp  [{self._c('q', THEME.accent)}]uit"
        footer = f"{t.box_bl}{t.box_h * (self._width - 2)}{t.box_br}"
        return f"{t.box_v}{self._pad(controls, self._width - 2)}{t.box_v}\n{footer}"

    def render(self) -> None:
        """Render the complete dashboard."""
        t = self.theme

        # Calculate panel sizes
        left_width = self._width // 2
        right_width = self._width - left_width + 1
        top_height = 10
        bottom_height = 12

        # Get panel contents
        arch_lines = self.render_architecture_panel(left_width, top_height)
        events_lines = self.render_events_panel(right_width, top_height)
        levers_lines = self.render_levers_panel(left_width, bottom_height)
        frame_lines = self.render_frame_panel(right_width, bottom_height)

        self.clear_screen()

        # Title bar
        title = f"{t.box_tl}{t.box_h} {self._c('TRIX Forge Dashboard', THEME.header)} {t.box_h * (self._width - 27)} {self._c('v1.0', THEME.muted)} {t.box_tr}"
        print(title)
        print(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Combine panels horizontally
        for i in range(max(len(arch_lines), len(events_lines))):
            left = arch_lines[i] if i < len(arch_lines) else f"{t.box_v}{' ' * (left_width - 1)}{t.box_v}"
            right = events_lines[i] if i < len(events_lines) else f"{t.box_v}{' ' * (right_width - 1)}{t.box_v}"
            # Merge the middle border
            print(left[:-1] + right)

        for i in range(max(len(levers_lines), len(frame_lines))):
            left = levers_lines[i] if i < len(levers_lines) else f"{t.box_v}{' ' * (left_width - 1)}{t.box_v}"
            right = frame_lines[i] if i < len(frame_lines) else f"{t.box_v}{' ' * (right_width - 1)}{t.box_v}"
            print(left[:-1] + right)

        # Footer
        print(self.render_footer())

    def set_model(self, name: str, layers: List[Dict[str, Any]]) -> None:
        """Set model information."""
        self.state.model_name = name
        self.state.layers = layers

    def add_event(self, frame: int, etype: str, message: str) -> None:
        """Add an event."""
        self.state.events.append({
            "frame": frame,
            "type": etype,
            "message": message
        })

    def set_lever(self, name: str, lever: Dict[str, Any]) -> None:
        """Set lever configuration."""
        self.state.levers[name] = lever

    def set_frame(self, frame_id: int, frame_data: Dict[str, Any]) -> None:
        """Set current frame for inspection."""
        self.state.frame_id = frame_id
        self.state.current_frame = frame_data


def demo_dashboard():
    """Demo the dashboard."""
    dash = Dashboard(color=True)

    # Set up model
    dash.set_model("tiny_mlp.trix", [
        {"name": "Input", "shape": "[4]"},
        {"name": "Hidden", "shape": "[8]"},
        {"name": "Output", "shape": "[2]"},
    ])

    # Add events
    for i in range(8):
        dash.add_event(127 + i, f"layer[{i % 3}].act", f"activation computed")

    # Set levers
    dash.set_lever("precision", {"value": "fp16", "options": ["fp32", "fp16", "fp8"]})
    dash.set_lever("optimize", {"value": "aggressive", "options": ["none", "aggressive"]})
    dash.set_lever("fuse_ops", {"value": "on", "options": ["on", "off"]})

    # Set frame
    dash.set_frame(127, {
        "input": [0.72, 0.31, 0.89, 0.15],
        "activations": {
            "Hidden Layer": [0.82, 0.51, 0.00]
        },
        "output": {"class": 1, "confidence": 0.91}
    })

    dash.render()

    print(f"\n  {dash._c('Dashboard demo complete.', THEME.info)}")
    print(f"  {dash._c('In production, use keyboard for navigation.', THEME.muted)}\n")


if __name__ == "__main__":
    demo_dashboard()
