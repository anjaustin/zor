"""
GLASS Mode Renderer - Live visibility into the pipeline.

The Glassbox principle visualized: see every event, every decision.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .theme import THEME, LOGO, format_size, format_time, PIPELINE_STAGES


@dataclass
class Event:
    """A single Glassbox event."""
    id: int
    type: str
    source: str
    message: str
    frame: int = 0
    timestamp: float = 0.0
    data: Optional[Dict[str, Any]] = None


@dataclass
class PipelineState:
    """State of the compilation pipeline."""
    stages: List[str] = field(default_factory=lambda: PIPELINE_STAGES.copy())
    current: int = 0
    completed: List[bool] = field(default_factory=lambda: [False] * 5)


class GlassRenderer:
    """
    GLASS mode renderer - live event stream and pipeline visibility.

    Shows:
    - Pipeline progress with stage indicators
    - Live event stream from the Glassbox
    - Active lever configuration
    - Model metadata
    """

    def __init__(self, color: bool = True):
        self.color = color
        self.theme = THEME
        self.events: List[Event] = []
        self.pipeline = PipelineState()
        self.levers: Dict[str, Any] = {}
        self.model_info: Dict[str, Any] = {}
        self._width = 70

    def _c(self, text: str, color: str) -> str:
        """Apply color if enabled."""
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

    def _box_top(self, title: str = "") -> str:
        """Draw box top with optional title."""
        t = self.theme
        if title:
            title_part = f" {title} "
            line_len = self._width - len(title_part) - 2
            return f"{t.box_tl}{t.box_h} {self._c(title, THEME.header)} {t.box_h * line_len}{t.box_tr}"
        return f"{t.box_tl}{t.box_h * (self._width - 2)}{t.box_tr}"

    def _box_bottom(self) -> str:
        """Draw box bottom."""
        t = self.theme
        return f"{t.box_bl}{t.box_h * (self._width - 2)}{t.box_br}"

    def _box_line(self, content: str = "") -> str:
        """Draw box line with content."""
        t = self.theme
        # Strip ANSI for length calculation
        import re
        visible_len = len(re.sub(r'\033\[[0-9;]*m', '', content))
        padding = self._width - visible_len - 4
        return f"{t.box_v}  {content}{' ' * max(0, padding)}{t.box_v}"

    def _box_divider(self, title: str = "") -> str:
        """Draw box divider with optional title."""
        t = self.theme
        if title:
            title_part = f" {title} "
            line_len = self._width - len(title_part) - 2
            return f"{t.box_t_right}{t.box_h} {self._c(title, THEME.muted)} {t.box_h * line_len}{t.box_t_left}"
        return f"{t.box_t_right}{t.box_h * (self._width - 2)}{t.box_t_left}"

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def render_header(self, model: str, target: str) -> List[str]:
        """Render the header section."""
        lines = []
        lines.append(self._box_top("TRIX Forge"))
        lines.append(self._box_line())
        lines.append(self._box_line(
            f"Model: {self._c(model, THEME.header):30} Target: {self._c(target, THEME.accent)}"
        ))

        shapes = self.model_info.get("shapes", 0)
        tensors = self.model_info.get("tensors", 0)
        size_est = self.model_info.get("size_est", "~2 KB")
        lines.append(self._box_line(
            f"Shapes: {self._c(str(shapes), THEME.info)} frozen, {self._c(str(tensors), THEME.info)} tensors       Size est: {self._c(size_est, THEME.accent)}"
        ))
        lines.append(self._box_line())

        return lines

    def render_pipeline(self) -> List[str]:
        """Render the pipeline progress section."""
        lines = []
        lines.append(self._box_divider("Pipeline"))
        lines.append(self._box_line())

        # Pipeline flow
        t = self.theme
        flow_parts = []
        for i, stage in enumerate(self.pipeline.stages):
            flow_parts.append(stage)
            if i < len(self.pipeline.stages) - 1:
                flow_parts.append(f" {t.arrow_flow} ")

        lines.append(self._box_line("  ".join(self.pipeline.stages[0:3]) + f"  {t.arrow_right}"))
        lines.append(self._box_line("       " + "  ".join(self.pipeline.stages[3:])))

        # Status indicators
        status_line = "  "
        for i, stage in enumerate(self.pipeline.stages):
            if self.pipeline.completed[i]:
                status_line += self._c(t.icon_success, THEME.success)
            elif i == self.pipeline.current:
                status_line += self._c(t.icon_progress, THEME.progress)
            else:
                status_line += self._c(t.icon_pending, THEME.muted)
            status_line += "         "

        lines.append(self._box_line(status_line[:self._width - 6]))
        lines.append(self._box_line())

        return lines

    def render_events(self, max_events: int = 8) -> List[str]:
        """Render the event stream section."""
        lines = []
        lines.append(self._box_divider("Events"))
        lines.append(self._box_line())

        recent = self.events[-max_events:] if self.events else []
        for event in recent:
            event_id = f"[{event.id:04d}]"
            event_type = f"{event.type:18}"
            message = event.message[:35]
            line = f"{self._c(event_id, THEME.muted)} {self._c(event_type, THEME.accent)} {message}"
            lines.append(self._box_line(line))

        # Pad if fewer events
        for _ in range(max_events - len(recent)):
            lines.append(self._box_line())

        lines.append(self._box_line())
        return lines

    def render_levers(self) -> List[str]:
        """Render the active levers section."""
        lines = []
        lines.append(self._box_divider("Levers (active)"))
        lines.append(self._box_line())

        for name, value in list(self.levers.items())[:4]:
            desc = self._get_lever_description(name)
            line = f"{self._c(name, THEME.header):14} {self.theme.box_v} {self._c(str(value), THEME.accent):14} {self.theme.box_v} {self._c(desc, THEME.dim)}"
            lines.append(self._box_line(line[:self._width - 6]))

        if not self.levers:
            lines.append(self._box_line(self._c("No levers configured", THEME.muted)))

        lines.append(self._box_line())
        return lines

    def _get_lever_description(self, name: str) -> str:
        """Get description for a lever."""
        descriptions = {
            "precision": "size vs accuracy tradeoff",
            "optimize": "fusion and folding level",
            "debug_info": "include debug symbols",
            "target_arch": "compilation target",
            "fuse_ops": "operator fusion",
            "quantize": "weight quantization",
        }
        return descriptions.get(name, "")

    def render_full(self, model: str, target: str) -> None:
        """Render the complete GLASS view."""
        output = []
        output.extend(self.render_header(model, target))
        output.extend(self.render_pipeline())
        output.extend(self.render_events())
        output.extend(self.render_levers())
        output.append(self._box_bottom())

        self.clear_screen()
        for line in output:
            print(line)

    def add_event(self, event: Event) -> None:
        """Add an event to the stream."""
        self.events.append(event)

    def set_stage(self, stage_index: int, completed: bool = False) -> None:
        """Update pipeline stage."""
        if completed and stage_index < len(self.pipeline.completed):
            self.pipeline.completed[stage_index] = True
            self.pipeline.current = stage_index + 1
        else:
            self.pipeline.current = stage_index

    def set_lever(self, name: str, value: Any) -> None:
        """Set a lever value."""
        self.levers[name] = value

    def set_model_info(self, info: Dict[str, Any]) -> None:
        """Set model metadata."""
        self.model_info = info


def demo_glass():
    """Demo the GLASS renderer."""
    renderer = GlassRenderer(color=True)

    model = "tiny_mlp.trix"
    target = "arm64-linux"

    renderer.set_model_info({
        "shapes": 4,
        "tensors": 6,
        "size_est": "~1.8 KB"
    })

    renderer.set_lever("precision", "fp16")
    renderer.set_lever("optimize", "aggressive")
    renderer.set_lever("debug_info", "off")
    renderer.set_lever("target_arch", "arm64-linux")

    # Simulate pipeline
    event_id = 0
    events_data = [
        ("parse.start", "parse", "loading model.trix"),
        ("parse.complete", "parse", "4 shapes, 6 tensors loaded"),
        ("validate.start", "validate", "checking constraints"),
        ("validate.pass", "validate", "all shape dimensions compatible"),
        ("optimize.start", "optimize", "analyzing operations"),
        ("optimize.fuse", "optimize", "Linear→ReLU fused (2 ops → 1)"),
        ("optimize.quantize", "optimize", "fp32 → fp16 (precision lever)"),
        ("emit.start", "emit", "generating target code"),
        ("emit.complete", "emit", "generated 847 ARM64 instructions"),
        ("package.start", "package", "writing NGE header"),
        ("package.weights", "package", "embedding quantized weights"),
        ("package.complete", "package", "output: tiny_mlp.nge (1847 bytes)"),
    ]

    stage_map = {"parse": 0, "validate": 1, "optimize": 2, "emit": 3, "package": 4}

    for event_type, source, message in events_data:
        event_id += 1
        renderer.add_event(Event(
            id=event_id,
            type=event_type,
            source=source,
            message=message,
            timestamp=time.time()
        ))

        # Update stage
        stage_idx = stage_map.get(source, 0)
        if ".complete" in event_type or ".pass" in event_type:
            renderer.set_stage(stage_idx, completed=True)
        else:
            renderer.set_stage(stage_idx)

        renderer.render_full(model, target)
        time.sleep(0.3)

    # Final render
    time.sleep(0.5)
    print()
    print(f"   {renderer._c('✓', THEME.success)} Build complete: {renderer._c('tiny_mlp.nge', THEME.header)} ({format_size(1847)})")
    print()


if __name__ == "__main__":
    demo_glass()
