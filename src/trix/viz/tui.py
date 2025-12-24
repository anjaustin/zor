"""
TUI: Terminal User Interface.

HTOP for self-repairing computation.
Calm by default. Alerts surface problems.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import sys
import time
import threading

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import BarColumn, Progress
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .observer import SDPMXObserver


# Health color mapping
def health_color(value: float) -> str:
    """Map health value to color."""
    if value >= 0.9:
        return "green"
    elif value >= 0.7:
        return "yellow"
    elif value >= 0.5:
        return "orange1"
    else:
        return "red"


def make_bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Create a progress bar string."""
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def make_histogram(data: List[float], width: int = 8) -> str:
    """Create a horizontal histogram from percentages."""
    if not data:
        return ""

    chars = "░▏▎▍▌▋▊▉█"
    result = []

    for i, val in enumerate(data):
        bar_len = int(val * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        result.append(f"{i} {bar} {val*100:4.1f}%")

    return "\n".join(result)


class SimpleTUI:
    """
    Simple terminal display (no Rich dependency).

    Single-line output, minimal but functional.
    """

    def __init__(self, observer: SDPMXObserver):
        self.observer = observer
        self._stop = threading.Event()

    def format_line(self) -> str:
        """Format single status line."""
        s = self.observer.summary

        health = s.get('health', 0)
        trend = s.get('health_trend', '→')
        step = s.get('step', 0)
        sps = s.get('steps_per_sec', 0)
        mask = s.get('mask_density', 0)

        bar = make_bar(health, width=10)

        return (
            f"\rStep: {step:>8} | "
            f"Health: {bar} {health*100:5.1f}% {trend} | "
            f"Mask: {mask*100:4.1f}% | "
            f"{sps:.1f}/s"
        )

    def update(self) -> None:
        """Print single update."""
        sys.stdout.write(self.format_line())
        sys.stdout.flush()

    def run(self, refresh_rate: float = 0.5) -> None:
        """Run display loop."""
        try:
            while not self._stop.is_set():
                self.update()
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            print()  # Newline at end

    def stop(self) -> None:
        self._stop.set()


class RichTUI:
    """
    Rich terminal UI with full dashboard.

    Requires `pip install rich`.
    """

    def __init__(self, observer: SDPMXObserver):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required: pip install rich")

        self.observer = observer
        self.console = Console()
        self._stop = threading.Event()

    def make_header(self, summary: Dict[str, Any]) -> Panel:
        """Create header panel with key metrics."""
        health = summary.get('health', 0)
        trend = summary.get('health_trend', '→')
        status = summary.get('health_status', 'unknown')
        step = summary.get('step', 0)
        sps = summary.get('steps_per_sec', 0)
        loss = summary.get('loss')

        color = health_color(health)
        bar = make_bar(health, width=20)

        text = Text()
        text.append(f"Health: ")
        text.append(f"{bar} ", style=color)
        text.append(f"{health*100:5.1f}% ", style=f"bold {color}")
        text.append(f"{trend} ", style=color)
        text.append(f"[{status}]", style=f"dim {color}")
        text.append(f"  │  ")
        text.append(f"Step: {step:,}", style="cyan")
        text.append(f"  │  ")
        text.append(f"{sps:.1f}/s", style="dim")
        if loss is not None:
            text.append(f"  │  ")
            text.append(f"Loss: {loss:.4f}", style="magenta")

        return Panel(text, title="SDPMX Observer", border_style="blue")

    def make_metrics(self, summary: Dict[str, Any]) -> Panel:
        """Create metrics panel with sparklines."""
        mask = summary.get('mask_density', 0)
        intervention = summary.get('intervention', 0)

        # Get sparklines
        health_spark = self.observer.get_sparkline('health', 40)
        mask_spark = self.observer.get_sparkline('mask_density', 40)
        loss_spark = self.observer.get_sparkline('loss', 40)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", width=12)
        table.add_column("Sparkline", width=42)
        table.add_column("Current", width=10)

        table.add_row(
            Text("Health", style="green"),
            Text(health_spark, style="green"),
            Text(f"{summary.get('health', 0)*100:.1f}%", style="bold green"),
        )
        table.add_row(
            Text("Mask", style="yellow"),
            Text(mask_spark, style="yellow"),
            Text(f"{mask*100:.1f}%", style="yellow"),
        )
        if loss_spark:
            table.add_row(
                Text("Loss", style="magenta"),
                Text(loss_spark, style="magenta"),
                Text(f"{summary.get('loss', 0):.4f}" if summary.get('loss') else "—", style="magenta"),
            )

        return Panel(table, title="Metrics", border_style="dim")

    def make_subspaces(self, summary: Dict[str, Any]) -> Panel:
        """Create subspace routing panel."""
        histogram = summary.get('subspace_histogram', [])

        lines = []
        for i, val in enumerate(histogram):
            bar_width = 15
            bar_len = int(val * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)

            if val == 0:
                style = "dim red"
                label = " (dead)"
            elif val > 0.3:
                style = "cyan"
                label = ""
            else:
                style = "dim cyan"
                label = ""

            lines.append(Text.assemble(
                (f"{i} ", "dim"),
                (bar, style),
                (f" {val*100:5.1f}%", style),
                (label, "dim red"),
            ))

        content = Text("\n").join(lines) if lines else Text("No data", style="dim")
        return Panel(content, title="Subspace Routing", border_style="dim")

    def make_layout(self) -> Layout:
        """Create full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        layout["body"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="subspaces", ratio=1),
        )

        return layout

    def update_layout(self, layout: Layout) -> None:
        """Update layout with current data."""
        summary = self.observer.summary

        layout["header"].update(self.make_header(summary))
        layout["metrics"].update(self.make_metrics(summary))
        layout["subspaces"].update(self.make_subspaces(summary))
        layout["footer"].update(Text(
            "[q]uit  [p]ause  [+/-] zoom  Press Ctrl+C to exit",
            style="dim",
            justify="center",
        ))

    def run(self, refresh_rate: float = 0.5) -> None:
        """Run live dashboard."""
        layout = self.make_layout()

        try:
            with Live(layout, console=self.console, refresh_per_second=1/refresh_rate) as live:
                while not self._stop.is_set():
                    self.update_layout(layout)
                    time.sleep(refresh_rate)
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        self._stop.set()


def create_tui(observer: SDPMXObserver, rich: bool = True) -> 'SimpleTUI | RichTUI':
    """
    Create appropriate TUI based on available dependencies.

    Args:
        observer: The SDPMX observer to visualize
        rich: Whether to use Rich TUI (falls back to simple if unavailable)
    """
    if rich and RICH_AVAILABLE:
        return RichTUI(observer)
    else:
        return SimpleTUI(observer)


def watch(
    observer: SDPMXObserver,
    refresh_rate: float = 0.5,
    rich: bool = True,
) -> None:
    """
    Convenience function to start watching an observer.

    Usage:
        observer = create_observer(pipeline)
        # Start training in background...
        watch(observer)
    """
    tui = create_tui(observer, rich=rich)
    tui.run(refresh_rate=refresh_rate)
