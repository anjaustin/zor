"""
SIMPLE Mode Renderer - Clean, minimal output for everyone.

This is the default mode. Fast, clear, works everywhere.
"""

import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Callable
from .theme import THEME, LOGO, format_size, format_time, PIPELINE_STAGES


@dataclass
class BuildResult:
    """Result of a build operation."""
    success: bool
    output_path: str
    output_size: int
    target: str
    target_desc: str
    build_time: float
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None


class SimpleRenderer:
    """
    SIMPLE mode renderer - clean progress and results.

    Optimized for:
    - Maximum clarity
    - Minimum overhead
    - Universal terminal compatibility
    """

    def __init__(self, color: bool = True, quiet: bool = False):
        self.color = color
        self.quiet = quiet
        self.theme = THEME
        self._start_time: Optional[float] = None

    def _c(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if not self.color:
            return text
        # Convert hex to ANSI 256 color (approximate)
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
        }
        code = colors.get(color, "\033[37m")
        return f"{code}{text}\033[0m"

    def logo(self) -> None:
        """Print the TRIX Forge logo."""
        if self.quiet:
            return
        print()
        for line in LOGO.split("\n"):
            print(f"   {self._c(line, THEME.accent)}")
        print(f"   {self._c('v1.0', THEME.muted)}")
        print()

    def build_start(self, model: str, target: str) -> None:
        """Show build starting message."""
        if self.quiet:
            return
        self._start_time = time.time()
        print(f"   Building {self._c(model, THEME.header)} {self._c('→', THEME.muted)} {self._c(target, THEME.accent)}")
        print()

    def progress(self, value: float, stage: Optional[str] = None) -> None:
        """Update progress bar."""
        if self.quiet:
            return

        width = 40
        filled = int(value * width)
        bar = self.theme.bar_full * filled + self.theme.bar_one * (width - filled)
        percent = int(value * 100)

        status = f" {stage}" if stage else ""
        line = f"   [{self._c(bar, THEME.accent)}] {percent}%{status}"

        # Overwrite line
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()

        if value >= 1.0:
            print()  # Newline when complete

    def build_success(self, result: BuildResult) -> None:
        """Show successful build result."""
        if self.quiet:
            # Machine-readable output
            print(f"OK {result.output_path} {result.output_size}")
            return

        print()
        print(f"   {self._c(self.theme.icon_success, THEME.success)} Output: {self._c(result.output_path, THEME.header)} ({format_size(result.output_size)})")
        print(f"   {self._c(self.theme.icon_success, THEME.success)} Target: {self._c(result.target, THEME.accent)} ({result.target_desc})")
        print(f"   {self._c(self.theme.icon_success, THEME.success)} Time:   {format_time(result.build_time)}")
        print()
        print(f"   Run with:")
        print(f"   {self._c('$', THEME.muted)} ./{result.output_path} input.bin")
        print()

    def build_error(self, error: str, location: Optional[str] = None,
                    context: Optional[List[str]] = None,
                    suggestions: Optional[List[str]] = None) -> None:
        """Show build error with teaching context."""
        if self.quiet:
            print(f"ERROR {error}")
            return

        print()
        print(f"   {self._c(self.theme.icon_error + ' Build Failed', THEME.error)}")
        print()

        # Error message
        print(f"   {self._c('Error:', THEME.error)} {error}")
        print()

        # Location if available
        if location:
            print(f"   {self._c('Location:', THEME.muted)} {location}")
            print()

        # Context (code snippet)
        if context:
            for line in context:
                if "← error" in line or "← here" in line:
                    print(f"   {self._c(line, THEME.error)}")
                else:
                    print(f"   {self._c(line, THEME.muted)}")
            print()

        # Suggestions
        if suggestions:
            print(f"   {self._c('Suggestions:', THEME.info)}")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
            print()

    def info(self, message: str) -> None:
        """Show info message."""
        if self.quiet:
            return
        print(f"   {self._c(self.theme.icon_info, THEME.info)} {message}")

    def warning(self, message: str) -> None:
        """Show warning message."""
        if self.quiet:
            print(f"WARN {message}")
            return
        print(f"   {self._c(self.theme.icon_warning, THEME.warning)} {message}")

    def targets_list(self, targets: List[dict]) -> None:
        """Display available targets."""
        if self.quiet:
            for t in targets:
                print(t["id"])
            return

        print()
        print(f"   {self._c('Available Targets', THEME.header)}")
        print(f"   {self.theme.box_h * 50}")
        print()

        for t in targets:
            print(f"   {self._c(t['id'], THEME.accent):20} {t['description']}")
        print()

    def inspect_model(self, info: dict) -> None:
        """Display model inspection results."""
        if self.quiet:
            import json
            print(json.dumps(info))
            return

        print()
        print(f"   {self._c('Model: ' + info.get('name', 'unnamed'), THEME.header)}")
        print(f"   {self.theme.box_h * 50}")
        print()

        # Shapes
        shapes = info.get("shapes", [])
        print(f"   {self._c('Shapes:', THEME.muted)} {len(shapes)}")
        for s in shapes[:5]:  # Show first 5
            print(f"      {self.theme.icon_pending} {s}")
        if len(shapes) > 5:
            print(f"      {self._c(f'... and {len(shapes) - 5} more', THEME.muted)}")
        print()

        # Tensors
        tensors = info.get("tensors", {})
        print(f"   {self._c('Tensors:', THEME.muted)} {len(tensors)}")

        # Levers
        levers = info.get("levers", {})
        print(f"   {self._c('Levers:', THEME.muted)} {len(levers)}")
        for name, lever in list(levers.items())[:3]:
            print(f"      {name}: {lever.get('value', lever.get('default'))}")
        print()


def demo_simple():
    """Demo the simple renderer."""
    renderer = SimpleRenderer(color=True)

    # Show logo
    renderer.logo()

    # Simulate build
    renderer.build_start("tiny_mlp.trix", "arm64-linux")

    for i in range(101):
        stage = PIPELINE_STAGES[min(i // 20, 4)]
        renderer.progress(i / 100, stage)
        time.sleep(0.01)

    # Show success
    result = BuildResult(
        success=True,
        output_path="tiny_mlp.nge",
        output_size=1847,
        target="arm64-linux",
        target_desc="Raspberry Pi 4+, Jetson Nano",
        build_time=0.18
    )
    renderer.build_success(result)


if __name__ == "__main__":
    demo_simple()
