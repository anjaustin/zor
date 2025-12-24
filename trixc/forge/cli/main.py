#!/usr/bin/env python3
"""
TRIX Forge CLI - Watch neural networks think.

The command-line interface for TRIX Forge compilation and debugging.

Usage:
    forge build <model> --target <target>   Build model for target
    forge glass <model> [--input <data>]    Watch live event stream
    forge dashboard <model>                 Full TUI experience
    forge replay <trace>                    Frame-by-frame viewer
    forge targets                           List available targets
    forge levers <model>                    Explore model levers
    forge inspect <file>                    Examine model or NGE
    forge help [command]                    Get help
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.theme import THEME, LOGO, format_size, format_time
from ui.simple import SimpleRenderer, BuildResult
from ui.glass import GlassRenderer, Event
from ui.dashboard import Dashboard
from ui.replay import FrameReplayViewer, Frame, LayerActivation
from core.forge import ForgeBackend, BuildConfig
from core.events import EventStream
from core.trace import TraceFile, TraceFrame, TraceMetadata


def cmd_build(args):
    """Execute build command."""
    backend = ForgeBackend()

    # Choose renderer based on mode
    if args.glass:
        renderer = GlassRenderer(color=not args.no_color)
        renderer.set_model_info({"shapes": 4, "tensors": 6, "size_est": "~2 KB"})

        # Set up event callback for live updates
        def on_event(event):
            renderer.add_event(Event(
                id=event["id"],
                type=event["type"],
                source=event["source"],
                message=event["message"],
                frame=event.get("frame", 0),
                timestamp=event.get("timestamp", time.time())
            ))
            stage_map = {"loader": 0, "validator": 1, "optimizer": 2, "emitter": 3, "packager": 4}
            stage = stage_map.get(event["source"], 0)
            is_complete = ".complete" in event["type"] or ".pass" in event["type"]
            renderer.set_stage(stage, completed=is_complete)
            renderer.render_full(args.model, args.target)
            time.sleep(0.15)  # Slight delay for visual effect

        backend.set_event_callback(on_event)
        renderer.render_full(args.model, args.target)
    else:
        renderer = SimpleRenderer(color=not args.no_color, quiet=args.quiet)
        if not args.quiet:
            renderer.logo()
            renderer.build_start(args.model, args.target)

    # Execute build
    config = BuildConfig(
        model_path=args.model,
        target=args.target,
        output_path=args.output,
        glass=args.glass,
        quiet=args.quiet
    )

    result = backend.build(config)

    # Show result
    if not args.glass:
        if result.success:
            # Animate progress
            if not args.quiet:
                for i in range(101):
                    renderer.progress(i / 100)
                    time.sleep(0.005)

            renderer.build_success(BuildResult(
                success=True,
                output_path=result.output_path,
                output_size=result.output_size,
                target=result.target,
                target_desc=result.target_desc,
                build_time=result.build_time
            ))
        else:
            renderer.build_error(
                result.error or "Unknown error",
                suggestions=result.suggestions
            )
            return 1

    return 0


def cmd_targets(args):
    """List available targets."""
    backend = ForgeBackend()
    renderer = SimpleRenderer(color=not args.no_color, quiet=args.quiet)

    targets = backend.get_targets()
    renderer.targets_list(targets)
    return 0


def cmd_dashboard(args):
    """Launch dashboard TUI."""
    dash = Dashboard(color=not args.no_color)

    # Load model info
    backend = ForgeBackend()
    try:
        model_info = backend.load_model(args.model)
        dash.set_model(model_info["name"], model_info["layers"])

        for name, lever in model_info["levers"].items():
            dash.set_lever(name, lever)
    except FileNotFoundError:
        # Demo mode
        dash.set_model("demo_model.trix", [
            {"name": "Input", "shape": "[4]"},
            {"name": "Hidden", "shape": "[8]"},
            {"name": "Output", "shape": "[2]"},
        ])

    # Add sample events
    for i in range(8):
        dash.add_event(127 + i, f"layer[{i % 3}].act", "activation computed")

    # Set sample frame
    dash.set_frame(127, {
        "input": [0.72, 0.31, 0.89, 0.15],
        "activations": {"Hidden Layer": [0.82, 0.51, 0.00]},
        "output": {"class": 1, "confidence": 0.91}
    })

    dash.render()
    print(f"\n  Press any key to exit (keyboard nav in production)...\n")
    return 0


def cmd_replay(args):
    """Launch frame replay viewer."""
    viewer = FrameReplayViewer(color=not args.no_color)

    # Try to load trace file
    if os.path.exists(args.trace):
        trace = TraceFile()
        try:
            trace.load(args.trace)
            # Convert trace frames to viewer frames
            frames = []
            for tf in trace.frames:
                layers = []
                for name, acts in tf.layer_activations.items():
                    layers.append(LayerActivation(
                        name=name,
                        layer_type="hidden",
                        values=acts
                    ))
                frames.append(Frame(
                    frame_id=tf.frame_id,
                    input_values=tf.input_values,
                    layers=layers,
                    output_values=tf.output_values,
                    output_class=tf.output_class,
                    output_confidence=tf.output_confidence,
                    ground_truth=tf.ground_truth,
                    correct=(tf.output_class == tf.ground_truth) if tf.ground_truth is not None else None
                ))
            viewer.load_frames(frames)
        except Exception as e:
            print(f"Error loading trace: {e}")
            return 1
    else:
        # Demo frames
        frames = []
        for i in range(10):
            base = i * 0.1
            frames.append(Frame(
                frame_id=i + 1,
                input_values=[0.72 + base * 0.1, 0.31 - base * 0.05, 0.89 - base * 0.1, 0.15 + base * 0.2],
                layers=[LayerActivation(
                    name="Hidden Layer",
                    layer_type="Linear + ReLU",
                    values=[0.64 + base * 0.1, 0.91 - base * 0.15, max(0, 0.1 - base * 0.2), 0.43 + base * 0.1],
                    annotations=["", "", "killed by ReLU" if base > 0.3 else "", ""]
                )],
                output_values=[0.10 + base * 0.05, 0.90 - base * 0.05],
                output_class=1 if (0.90 - base * 0.05) > 0.5 else 0,
                output_confidence=max(0.90 - base * 0.05, 0.10 + base * 0.05),
                ground_truth=1,
                correct=(1 if (0.90 - base * 0.05) > 0.5 else 0) == 1
            ))
        viewer.load_frames(frames)

    # Show first few frames as demo
    for i in range(min(3, len(viewer.frames))):
        viewer.goto_frame(i)
        viewer.render()
        if i < 2:
            print(f"\n  Advancing frame...\n")
            time.sleep(1.0)

    return 0


def cmd_inspect(args):
    """Inspect model or NGE file."""
    backend = ForgeBackend()
    renderer = SimpleRenderer(color=not args.no_color, quiet=args.quiet)

    try:
        info = backend.inspect(args.file)
        renderer.inspect_model(info)
    except FileNotFoundError as e:
        renderer.build_error(str(e))
        return 1

    return 0


def cmd_demo(args):
    """Run full demo of all modes."""
    print("\n" + "=" * 60)
    print("  TRIX FORGE CLI - FULL DEMO")
    print("=" * 60)

    # Simple mode
    print("\n" + "-" * 60)
    print("  MODE 1: SIMPLE (default)")
    print("-" * 60 + "\n")
    time.sleep(1)

    renderer = SimpleRenderer(color=True)
    renderer.logo()
    renderer.build_start("tiny_mlp.trix", "arm64-linux")

    for i in range(101):
        renderer.progress(i / 100)
        time.sleep(0.008)

    renderer.build_success(BuildResult(
        success=True,
        output_path="tiny_mlp.nge",
        output_size=1847,
        target="arm64-linux",
        target_desc="Raspberry Pi 4+, Jetson Nano",
        build_time=0.18
    ))

    time.sleep(2)

    # Glass mode
    print("\n" + "-" * 60)
    print("  MODE 2: GLASS (--glass flag)")
    print("-" * 60 + "\n")
    time.sleep(1)

    glass = GlassRenderer(color=True)
    glass.set_model_info({"shapes": 4, "tensors": 6, "size_est": "~1.8 KB"})
    glass.set_lever("precision", "fp16")
    glass.set_lever("optimize", "aggressive")
    glass.set_lever("debug_info", "off")
    glass.set_lever("target_arch", "arm64-linux")

    events_data = [
        ("parse.complete", "parse", "4 shapes, 6 tensors loaded"),
        ("validate.pass", "validate", "all shape dimensions compatible"),
        ("optimize.fuse", "optimize", "Linear→ReLU fused (2 ops → 1)"),
        ("emit.complete", "emit", "generated 847 ARM64 instructions"),
        ("package.complete", "package", "output: tiny_mlp.nge (1847 bytes)"),
    ]

    stage_map = {"parse": 0, "validate": 1, "optimize": 2, "emit": 3, "package": 4}

    for i, (event_type, source, message) in enumerate(events_data):
        glass.add_event(Event(id=i+1, type=event_type, source=source, message=message))
        stage_idx = stage_map.get(source, 0)
        glass.set_stage(stage_idx, completed=True)
        glass.render_full("tiny_mlp.trix", "arm64-linux")
        time.sleep(0.4)

    time.sleep(2)

    # Dashboard
    print("\n" + "-" * 60)
    print("  MODE 3: DASHBOARD (full TUI)")
    print("-" * 60 + "\n")
    time.sleep(1)

    dash = Dashboard(color=True)
    dash.set_model("tiny_mlp.trix", [
        {"name": "Input", "shape": "[4]"},
        {"name": "Hidden", "shape": "[8]"},
        {"name": "Output", "shape": "[2]"},
    ])
    for i in range(8):
        dash.add_event(127 + i, f"layer[{i % 3}].act", "activation computed")
    dash.set_lever("precision", {"value": "fp16", "options": ["fp32", "fp16", "fp8"]})
    dash.set_lever("optimize", {"value": "aggressive", "options": ["none", "aggressive"]})
    dash.set_lever("fuse_ops", {"value": "on", "options": ["on", "off"]})
    dash.set_frame(127, {
        "input": [0.72, 0.31, 0.89, 0.15],
        "activations": {"Hidden Layer": [0.82, 0.51, 0.00]},
        "output": {"class": 1, "confidence": 0.91}
    })
    dash.render()

    time.sleep(3)

    # Frame Replay
    print("\n" + "-" * 60)
    print("  MODE 4: FRAME REPLAY (the killer feature)")
    print("-" * 60 + "\n")
    time.sleep(1)

    viewer = FrameReplayViewer(color=True)
    frames = []
    for i in range(5):
        base = i * 0.15
        frames.append(Frame(
            frame_id=i + 1,
            input_values=[0.72 + base * 0.1, 0.31 - base * 0.05, 0.89 - base * 0.1, 0.15 + base * 0.2],
            layers=[LayerActivation(
                name="Hidden Layer",
                layer_type="Linear + ReLU",
                values=[0.64 + base * 0.1, 0.91 - base * 0.15, max(0, 0.15 - base * 0.3), 0.43 + base * 0.1],
                annotations=["", "", "killed by ReLU" if base > 0.3 else "", ""]
            )],
            output_values=[0.10 + base * 0.08, 0.90 - base * 0.08],
            output_class=1 if (0.90 - base * 0.08) > 0.5 else 0,
            output_confidence=max(0.90 - base * 0.08, 0.10 + base * 0.08),
            ground_truth=1,
            correct=(1 if (0.90 - base * 0.08) > 0.5 else 0) == 1
        ))
    viewer.load_frames(frames)

    for i in range(3):
        viewer.goto_frame(i)
        viewer.render()
        time.sleep(1.5)

    # Finale
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"""
  TRIX Forge CLI - Watch neural networks think.

  Features demonstrated:
  {THEME.icon_success} SIMPLE mode   - Clean progress, anyone can use
  {THEME.icon_success} GLASS mode    - Live events, pipeline visibility
  {THEME.icon_success} DASHBOARD     - Full TUI development environment
  {THEME.icon_success} FRAME REPLAY  - See neural decisions frame-by-frame

  "No one's laughing us out of anywhere."
""")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TRIX Forge CLI - Watch neural networks think.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  forge build model.trix --target arm64-linux
  forge build model.trix --target wasm --glass
  forge dashboard model.trix
  forge replay trace.bin
  forge targets
  forge demo
        """
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (machine-readable)")
    parser.add_argument("--version", "-v", action="version", version="TRIX Forge CLI v1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build model for target")
    build_parser.add_argument("model", help="Model file path")
    build_parser.add_argument("--target", "-t", required=True, help="Target platform")
    build_parser.add_argument("--output", "-o", help="Output file path")
    build_parser.add_argument("--glass", "-g", action="store_true", help="Enable GLASS mode")
    build_parser.add_argument("--trace", help="Write trace file")

    # Targets command
    targets_parser = subparsers.add_parser("targets", help="List available targets")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch dashboard TUI")
    dash_parser.add_argument("model", help="Model file path")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Frame-by-frame viewer")
    replay_parser.add_argument("trace", help="Trace file path")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model or NGE")
    inspect_parser.add_argument("file", help="File to inspect")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run full demo")

    args = parser.parse_args()

    # Copy global args to namespace
    if not hasattr(args, 'no_color'):
        args.no_color = False
    if not hasattr(args, 'quiet'):
        args.quiet = False

    # Dispatch to command
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "targets":
        return cmd_targets(args)
    elif args.command == "dashboard":
        return cmd_dashboard(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "demo":
        return cmd_demo(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
