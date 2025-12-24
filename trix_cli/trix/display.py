"""
Display utilities - pretty output for the REPL.
"""
import sys
import time
from typing import Optional


# ANSI colors (with fallback for non-TTY)
def supports_color() -> bool:
    """Check if terminal supports colors."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


if supports_color():
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    RESET = "\033[0m"
else:
    BOLD = DIM = GREEN = YELLOW = BLUE = MAGENTA = CYAN = RED = RESET = ""


BANNER = f"""{CYAN}
  ████████╗██████╗ ██╗██╗  ██╗
  ╚══██╔══╝██╔══██╗██║╚██╗██╔╝
     ██║   ██████╔╝██║ ╚███╔╝
     ██║   ██╔══██╗██║ ██╔██╗
     ██║   ██║  ██║██║██╔╝ ██╗
     ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{RESET}
  {BOLD}Frozen Shape Foundry{RESET} v1.0

  Type '{GREEN}demo{RESET}' to see the magic, or '{GREEN}help{RESET}' for commands.
"""


def print_banner():
    """Print the welcome banner."""
    print(BANNER)


def print_shape_info(shape) -> str:
    """Format shape information for display."""
    status = f"{GREEN}validated{RESET}" if shape._validated else f"{DIM}not validated{RESET}"
    arity_str = "unary" if shape.arity == 1 else "binary"

    lines = [
        f"",
        f"  {BOLD}{shape.name.upper()}{RESET}",
        f"  {DIM}{'─' * 40}{RESET}",
        f"  Type: {arity_str} ({shape.bits}-bit)",
        f"  Polynomial: {CYAN}{shape.polynomial}{RESET}",
        f"  Status: {status}",
    ]

    if shape._validated and shape._validation_result:
        passed, total = shape._validation_result
        pct = (passed / total) * 100
        color = GREEN if pct == 100 else RED
        lines.append(f"  Validation: {color}{passed}/{total} ({pct:.0f}%){RESET}")

    lines.append("")
    return "\n".join(lines)


def print_truth_table(shape) -> str:
    """Format truth table for display (binary shapes only)."""
    if shape.arity != 2:
        return f"  (Truth table display for unary shapes not implemented)\n"

    lines = [
        f"",
        f"  {BOLD}Truth Table (1-bit){RESET}",
        f"  ┌───┬───┬────────┐",
        f"  │ a │ b │ result │",
        f"  ├───┼───┼────────┤",
    ]

    for a in [0, 1]:
        for b in [0, 1]:
            r = shape(a, b) & 1  # Show 1-bit result
            lines.append(f"  │ {a} │ {b} │   {r}    │")

    lines.append(f"  └───┴───┴────────┘")
    lines.append("")
    return "\n".join(lines)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a progress bar string."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {bar} {pct*100:5.1f}%"


class ProgressDisplay:
    """Live progress display for validation."""

    def __init__(self, total: int, desc: str = "Validating"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.last_update = 0

    def update(self, current: int, total: int):
        """Update the progress display."""
        now = time.time()
        # Throttle updates to avoid flickering
        if now - self.last_update < 0.05 and current < total:
            return

        self.last_update = now
        pct = current / total if total > 0 else 0
        filled = int(40 * pct)
        bar = "█" * filled + "░" * (40 - filled)

        elapsed = now - self.start_time
        if pct > 0:
            eta = elapsed / pct - elapsed
            eta_str = f"ETA {eta:.1f}s" if eta > 0.5 else ""
        else:
            eta_str = ""

        sys.stdout.write(f"\r  {bar} {pct*100:5.1f}% {eta_str}    ")
        sys.stdout.flush()

        if current >= total:
            sys.stdout.write("\n")


def print_error(msg: str, hint: Optional[str] = None):
    """Print a friendly error message."""
    print(f"\n  {RED}Oops!{RESET} {msg}")
    if hint:
        print(f"\n  {DIM}Hint: {hint}{RESET}")
    print()


def print_success(msg: str):
    """Print a success message."""
    print(f"\n  {GREEN}✓{RESET} {msg}\n")


def print_code(code: str, language: str = "c"):
    """Print formatted code block."""
    print(f"\n{DIM}```{language}{RESET}")
    for line in code.strip().split("\n"):
        print(f"  {line}")
    print(f"{DIM}```{RESET}\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{DIM}{'─' * len(title)}{RESET}")
