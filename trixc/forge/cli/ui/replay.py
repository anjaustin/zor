"""
Frame Replay Viewer - Watch neural networks think.

THE KILLER FEATURE: Frame-by-frame visualization of neural decisions.
See exactly what the network sees, how it processes, what it decides.
"""

import sys
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from .theme import THEME, format_size, format_time


@dataclass
class LayerActivation:
    """Activation values for a single layer."""
    name: str
    layer_type: str  # linear, relu, softmax, etc.
    values: List[float]
    pre_activation: Optional[List[float]] = None
    annotations: Optional[List[str]] = None  # e.g., "killed by ReLU"


@dataclass
class Frame:
    """A single inference frame with full activation trace."""
    frame_id: int
    input_values: List[float]
    layers: List[LayerActivation]
    output_values: List[float]
    output_class: int
    output_confidence: float
    ground_truth: Optional[int] = None
    correct: Optional[bool] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrameReplayViewer:
    """
    Frame-by-frame neural network decision viewer.

    Controls:
    - Left/Right: Navigate frames
    - a/d: Jump 10 frames
    - Space: Play/Pause
    - j: Jump to frame
    - /: Search
    - q: Quit
    """

    def __init__(self, color: bool = True):
        self.color = color
        self.theme = THEME
        self.frames: List[Frame] = []
        self.current_frame: int = 0
        self.playing: bool = False
        self.playback_speed: float = 1.0
        self._width = 72

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
            THEME.cold: "\033[90m",
            THEME.cool: "\033[94m",
            THEME.warm: "\033[96m",
            THEME.hot: "\033[91m",
        }
        code = colors.get(color, "\033[37m")
        return f"{code}{text}\033[0m"

    def _heat_color(self, value: float) -> str:
        """Get color based on activation intensity."""
        if value < 0.01:
            return THEME.cold
        elif value < 0.3:
            return THEME.cool
        elif value < 0.7:
            return THEME.warm
        else:
            return THEME.hot

    def _activation_bar(self, value: float, width: int = 30, annotation: str = "") -> str:
        """Generate a heat-mapped activation bar."""
        filled = int(abs(value) * width)
        filled = min(filled, width)
        empty = width - filled

        color = self._heat_color(abs(value))
        bar = self.theme.bar_full * filled + self.theme.bar_one * empty

        ann = f"  {self._c(annotation, THEME.dim)}" if annotation else ""
        return f"{self._c(bar, color)}{ann}"

    def _box(self, title: str = "") -> Tuple[str, str, str]:
        """Get box drawing characters."""
        t = self.theme
        if title:
            top = f"{t.box_tl}{t.box_h} {self._c(title, THEME.header)} {t.box_h * (self._width - len(title) - 5)}{t.box_tr}"
        else:
            top = f"{t.box_tl}{t.box_h * (self._width - 2)}{t.box_tr}"
        bottom = f"{t.box_bl}{t.box_h * (self._width - 2)}{t.box_br}"
        return top, t.box_v, bottom

    def clear_screen(self) -> None:
        """Clear terminal."""
        print("\033[2J\033[H", end="")

    def load_frames(self, frames: List[Frame]) -> None:
        """Load frames for replay."""
        self.frames = frames
        self.current_frame = 0

    def render_frame(self, frame: Frame) -> List[str]:
        """Render a single frame visualization."""
        lines = []
        t = self.theme

        # Header
        frame_str = f"Frame {frame.frame_id}"
        total_str = f"{len(self.frames)}"
        header = f"{t.box_tl}{t.box_h} {self._c('Frame Replay', THEME.header)} {t.box_h * 30} {self._c(frame_str, THEME.accent)}/{total_str} {t.box_h}{t.box_tr}"
        lines.append(header)
        lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Input Section
        lines.append(f"{t.box_v}  {self._c('Input', THEME.header)}{' ' * (self._width - 9)}{t.box_v}")
        for i, val in enumerate(frame.input_values[:8]):  # Max 8 inputs shown
            bar = self._activation_bar(val, 40)
            line = f"{t.box_v}  x[{i}]: {val:5.2f}  {bar}"
            padding = self._width - len(line.replace('\033[90m', '').replace('\033[91m', '').replace('\033[94m', '').replace('\033[96m', '').replace('\033[97m', '').replace('\033[0m', '')) - 1
            # Simplified padding
            lines.append(f"{t.box_v}  x[{i}]: {self._c(f'{val:5.2f}', THEME.text)}  {bar}{' ' * 5}{t.box_v}")

        if len(frame.input_values) > 8:
            lines.append(f"{t.box_v}  {self._c(f'... and {len(frame.input_values) - 8} more', THEME.muted)}{' ' * 40}{t.box_v}")

        lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Arrow
        arrow_line = f"{t.box_v}{' ' * 35}{self._c('│', THEME.muted)}{' ' * 34}{t.box_v}"
        lines.append(arrow_line)
        arrow_line = f"{t.box_v}{' ' * 35}{self._c('▼', THEME.muted)}{' ' * 34}{t.box_v}"
        lines.append(arrow_line)
        lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Hidden Layers
        for layer in frame.layers:
            layer_title = f"{layer.name} ({layer.layer_type})"
            lines.append(f"{t.box_v}  {self._c(layer_title, THEME.header)}{' ' * (self._width - len(layer_title) - 5)}{t.box_v}")

            for i, val in enumerate(layer.values[:6]):  # Max 6 values per layer
                ann = ""
                if layer.annotations and i < len(layer.annotations) and layer.annotations[i]:
                    ann = layer.annotations[i]

                bar = self._activation_bar(val, 30, ann)
                val_str = f"h[{i}]: {val:5.2f}"
                lines.append(f"{t.box_v}  {self._c(val_str, THEME.text)}  {bar}{' ' * 2}{t.box_v}")

            if len(layer.values) > 6:
                lines.append(f"{t.box_v}  {self._c(f'... and {len(layer.values) - 6} more', THEME.muted)}{' ' * 43}{t.box_v}")

            lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

            # Arrow between layers
            arrow_line = f"{t.box_v}{' ' * 35}{self._c('▼', THEME.muted)}{' ' * 34}{t.box_v}"
            lines.append(arrow_line)
            lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Output Section
        lines.append(f"{t.box_v}  {self._c('Output (Softmax)', THEME.header)}{' ' * (self._width - 20)}{t.box_v}")

        for i, val in enumerate(frame.output_values):
            is_selected = (i == frame.output_class)
            pct = f"{val * 100:.1f}%"
            bar = self._activation_bar(val, 30)

            if is_selected:
                marker = self._c(" ◀── SELECTED", THEME.success)
                lines.append(f"{t.box_v}  class_{i}: {self._c(f'{val:5.2f}', THEME.header)}  {bar}{marker}{t.box_v}")
            else:
                lines.append(f"{t.box_v}  class_{i}: {self._c(f'{val:5.2f}', THEME.text)}  {bar}  {pct}{' ' * 3}{t.box_v}")

        lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Decision Summary
        decision = f"class_{frame.output_class}"
        confidence = f"{frame.output_confidence * 100:.1f}%"
        lines.append(f"{t.box_v}  Decision: {self._c(decision, THEME.header)}{' ' * 20}Confidence: {self._c(confidence, THEME.accent)}{' ' * 7}{t.box_v}")

        if frame.ground_truth is not None:
            gt = f"class_{frame.ground_truth}"
            if frame.correct:
                status = self._c(f"{t.icon_success} CORRECT", THEME.success)
            else:
                status = self._c(f"{t.icon_error} INCORRECT", THEME.error)
            lines.append(f"{t.box_v}  Ground Truth: {self._c(gt, THEME.muted)}{' ' * 16}{status}{' ' * 10}{t.box_v}")

        lines.append(f"{t.box_v}{' ' * (self._width - 2)}{t.box_v}")

        # Controls
        lines.append(f"{t.box_t_right}{t.box_h * (self._width - 2)}{t.box_t_left}")
        controls = f"[{self._c('←', THEME.accent)}] prev  [{self._c('→', THEME.accent)}] next  [{self._c('space', THEME.accent)}] play/pause  [{self._c('j', THEME.accent)}] jump  [{self._c('q', THEME.accent)}] quit"
        lines.append(f"{t.box_v}  {controls}{' ' * 3}{t.box_v}")
        lines.append(f"{t.box_bl}{t.box_h * (self._width - 2)}{t.box_br}")

        return lines

    def render(self) -> None:
        """Render current frame."""
        if not self.frames:
            print("No frames loaded.")
            return

        frame = self.frames[self.current_frame]
        lines = self.render_frame(frame)

        self.clear_screen()
        for line in lines:
            print(line)

    def next_frame(self) -> None:
        """Move to next frame."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1

    def prev_frame(self) -> None:
        """Move to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1

    def jump_frames(self, delta: int) -> None:
        """Jump by delta frames."""
        new_pos = self.current_frame + delta
        self.current_frame = max(0, min(new_pos, len(self.frames) - 1))

    def goto_frame(self, frame_id: int) -> None:
        """Go to specific frame."""
        self.current_frame = max(0, min(frame_id, len(self.frames) - 1))


def demo_replay():
    """Demo the frame replay viewer."""
    viewer = FrameReplayViewer(color=True)

    # Create sample frames
    frames = []
    for i in range(10):
        # Simulate different inputs and activations
        base = i * 0.1
        frame = Frame(
            frame_id=i + 1,
            input_values=[0.72 + base * 0.1, 0.31 - base * 0.05, 0.89 - base * 0.1, 0.15 + base * 0.2],
            layers=[
                LayerActivation(
                    name="Hidden Layer",
                    layer_type="Linear + ReLU",
                    values=[0.64 + base * 0.1, 0.91 - base * 0.15, max(0, 0.1 - base * 0.2), 0.43 + base * 0.1],
                    annotations=["", "", "killed by ReLU" if base > 0.3 else "", ""]
                ),
            ],
            output_values=[0.10 + base * 0.05, 0.90 - base * 0.05],
            output_class=1 if (0.90 - base * 0.05) > 0.5 else 0,
            output_confidence=max(0.90 - base * 0.05, 0.10 + base * 0.05),
            ground_truth=1,
            correct=(1 if (0.90 - base * 0.05) > 0.5 else 0) == 1
        )
        frames.append(frame)

    viewer.load_frames(frames)

    # Interactive demo - show a few frames
    print("\n  FRAME REPLAY DEMO - Press Enter to advance frames, 'q' to quit\n")
    time.sleep(1)

    for i in range(min(5, len(frames))):
        viewer.goto_frame(i)
        viewer.render()

        if i < 4:
            print(f"\n  {viewer._c('Advancing to next frame...', THEME.muted)}\n")
            time.sleep(1.5)

    print(f"\n  {viewer._c('Demo complete. In production, use arrow keys to navigate.', THEME.info)}\n")


if __name__ == "__main__":
    demo_replay()
