"""
TRIX Forge CLI Theme - Visual Design System

Color semantics, typography, and visual constants.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Visual theme for TRIX Forge CLI."""

    # Status colors (always paired with icons)
    success: str = "#00D26A"      # Bright green
    error: str = "#FF6B6B"        # Soft red
    warning: str = "#FFA94D"      # Orange
    info: str = "#4DABF7"         # Sky blue
    progress: str = "#FFD93D"     # Amber
    muted: str = "#6C757D"        # Gray

    # Data colors (heat map)
    cold: str = "#2E3440"         # Dark gray (0.0)
    cool: str = "#5E81AC"         # Muted blue (0.5)
    warm: str = "#88C0D0"         # Cyan (1.0)
    hot: str = "#BF616A"          # Warm red (attention)

    # Structure colors
    border: str = "#4C566A"       # Subtle gray
    header: str = "#ECEFF4"       # Bright white
    text: str = "#D8DEE9"         # Soft white
    dim: str = "#6C757D"          # Dim gray
    accent: str = "#81A1C1"       # Nord blue

    # Icons (always paired with color)
    icon_success: str = "\u2713"  # ✓
    icon_error: str = "\u2717"    # ✗
    icon_warning: str = "\u26A0"  # ⚠
    icon_info: str = "\u2139"     # ℹ
    icon_progress: str = "\u25D0" # ◐
    icon_pending: str = "\u00B7"  # ·

    # Progress characters
    bar_full: str = "\u2588"      # █
    bar_three: str = "\u2593"     # ▓
    bar_two: str = "\u2592"       # ▒
    bar_one: str = "\u2591"       # ░
    bar_empty: str = "\u2591"     # ░

    # Spinners
    spinner_frames: tuple = ("\u280B", "\u2819", "\u2839", "\u2838",
                             "\u283C", "\u2834", "\u2826", "\u2827",
                             "\u2807", "\u280F")  # Braille spinner

    # Box drawing
    box_tl: str = "\u250C"        # ┌
    box_tr: str = "\u2510"        # ┐
    box_bl: str = "\u2514"        # └
    box_br: str = "\u2518"        # ┘
    box_h: str = "\u2500"         # ─
    box_v: str = "\u2502"         # │
    box_cross: str = "\u253C"     # ┼
    box_t_down: str = "\u252C"    # ┬
    box_t_up: str = "\u2534"      # ┴
    box_t_right: str = "\u251C"   # ├
    box_t_left: str = "\u2524"    # ┤

    # Double box
    dbox_tl: str = "\u2554"       # ╔
    dbox_tr: str = "\u2557"       # ╗
    dbox_bl: str = "\u255A"       # ╚
    dbox_br: str = "\u255D"       # ╝
    dbox_h: str = "\u2550"        # ═
    dbox_v: str = "\u2551"        # ║

    # Arrows
    arrow_right: str = "\u2192"   # →
    arrow_left: str = "\u2190"    # ←
    arrow_up: str = "\u2191"      # ↑
    arrow_down: str = "\u2193"    # ↓
    arrow_flow: str = "\u2500\u2192"  # ─→

    def progress_bar(self, value: float, width: int = 40) -> str:
        """Generate a progress bar string."""
        filled = int(value * width)
        empty = width - filled
        return self.bar_full * filled + self.bar_one * empty

    def heat_bar(self, value: float, width: int = 24) -> str:
        """Generate a heat-mapped bar for activation values."""
        filled = int(value * width)
        empty = width - filled
        return self.bar_full * filled + self.bar_one * empty

    def status_icon(self, status: str) -> str:
        """Get icon for a status string."""
        icons = {
            "success": self.icon_success,
            "complete": self.icon_success,
            "done": self.icon_success,
            "pass": self.icon_success,
            "error": self.icon_error,
            "fail": self.icon_error,
            "failed": self.icon_error,
            "warning": self.icon_warning,
            "warn": self.icon_warning,
            "info": self.icon_info,
            "progress": self.icon_progress,
            "running": self.icon_progress,
            "in_progress": self.icon_progress,
            "pending": self.icon_pending,
            "waiting": self.icon_pending,
        }
        return icons.get(status.lower(), self.icon_pending)


# Global theme instance
THEME = Theme()


# ASCII Art Logo
LOGO = """
\u2580\u2588\u2580 \u2588\u2580\u2588 \u2588 \u2580\u2584\u2580   \u2588\u2580\u2580 \u2588\u2580\u2588 \u2588\u2580\u2588 \u2588\u2580\u2580 \u2588\u2580\u2580
 \u2588  \u2588\u2580\u2584 \u2588 \u2588 \u2588   \u2588\u2580  \u2588 \u2588 \u2588\u2580\u2584 \u2588 \u2588 \u2588\u2580\u2580
 \u2580  \u2580 \u2580 \u2580 \u2580 \u2580   \u2580   \u2580\u2580\u2580 \u2580 \u2580 \u2580\u2580\u2580 \u2580\u2580\u2580
""".strip()

LOGO_SMALL = "\u2580\u2588\u2580\u2588\u2580\u2588 \u2588\u2580\u2580\u2588\u2580\u2588\u2588\u2580\u2580\u2588\u2580\u2580"


# Pipeline stages
PIPELINE_STAGES = ["Parse", "Validate", "Optimize", "Emit", "Package"]


def format_size(bytes_val: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}" if bytes_val != int(bytes_val) else f"{int(bytes_val)} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def format_time(seconds: float) -> str:
    """Format time to human readable."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}\u03BCs"
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"
