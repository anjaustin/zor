#!/usr/bin/env python3
"""
SDPMX Live Dashboard Demo

Runs training in a background thread while displaying
a live dashboard in the terminal.

Usage:
    python examples/viz/demo_live_dashboard.py

Requirements:
    pip install rich  (for full dashboard)

Press Ctrl+C to stop.
"""

import torch
import threading
import time
from trix.nn import SDPMXPipeline
from trix.viz import create_observer, watch, LogLevel


def training_loop(pipeline, observer, stop_event):
    """Background training loop."""
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.01)

    step = 0
    while not stop_event.is_set():
        # Generate synthetic data
        x = torch.randn(16, 32)
        target = torch.tanh(x)

        # Forward + backward
        optimizer.zero_grad()
        x_new, loss, _ = pipeline.forward_with_loss(x, target)
        loss.backward()
        optimizer.step()

        # Update observer
        observer.set_training_context(loss=loss.item())

        step += 1

        # Slow down a bit for demo
        time.sleep(0.05)


def main():
    print("SDPMX Live Dashboard Demo")
    print("=" * 40)
    print("Starting training in background...")
    print("Dashboard will appear shortly.")
    print("Press Ctrl+C to stop.")
    print("=" * 40)
    print()

    # Create pipeline and observer
    pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
    observer = create_observer(
        pipeline,
        level=LogLevel.TRAJECTORY,
        sample_rate=1,
    )
    observer.attach()

    # Start training in background
    stop_event = threading.Event()
    training_thread = threading.Thread(
        target=training_loop,
        args=(pipeline, observer, stop_event),
        daemon=True,
    )
    training_thread.start()

    try:
        # Show live dashboard (blocks)
        watch(observer, refresh_rate=0.25)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        training_thread.join(timeout=1.0)
        observer.stop()
        print("Done!")


if __name__ == "__main__":
    main()
