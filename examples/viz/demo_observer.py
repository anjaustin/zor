#!/usr/bin/env python3
"""
SDPMX Visualizer Demo

Demonstrates the observer watching a training loop.

Usage:
    python examples/viz/demo_observer.py

Features demonstrated:
    - Observer attachment
    - Training context updates
    - Summary access
    - Sparkline generation
"""

import torch
import torch.nn as nn
from trix.nn import SDPMXPipeline
from trix.viz import create_observer, LogLevel


def main():
    print("=" * 60)
    print("SDPMX Visualizer Demo")
    print("=" * 60)

    # Create pipeline
    pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.01)

    # Create observer
    observer = create_observer(
        pipeline,
        level=LogLevel.TRAJECTORY,
        sample_rate=5,  # Log every 5 steps
    )
    observer.attach()

    print("\nTraining with observer attached...")
    print("-" * 60)

    # Training loop
    num_epochs = 10
    batch_size = 16

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for step in range(20):
            # Generate synthetic data
            x = torch.randn(batch_size, 32)
            target = torch.tanh(x)  # Learn tanh

            # Forward pass
            optimizer.zero_grad()
            x_new, loss, _ = pipeline.forward_with_loss(x, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update observer context
            observer.set_training_context(loss=loss.item())
            epoch_loss += loss.item()

        # End of epoch - show summary
        s = observer.summary
        avg_loss = epoch_loss / 20

        print(f"Epoch {epoch+1:2d}: "
              f"Health={s['health']:.1%} {s['health_trend']} | "
              f"Mask={s['mask_density']:.1%} | "
              f"Loss={avg_loss:.4f} | "
              f"{s['steps_per_sec']:.0f}/s")

    print("-" * 60)

    # Final summary
    print("\nFinal State:")
    s = observer.summary
    print(f"  Total steps:     {s['step']}")
    print(f"  Total observed:  {s['total_observations']}")
    print(f"  Final health:    {s['health']:.2%} ({s['health_status']})")
    print(f"  Health trend:    {s['health_trend']}")
    print(f"  Mask density:    {s['mask_density']:.2%}")

    # Subspace routing
    print("\n  Subspace routing:")
    for i, pct in enumerate(s['subspace_histogram']):
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"    {i}: {bar} {pct:.1%}")

    # Sparklines
    print("\n  Recent history (sparklines):")
    print(f"    Health:  {observer.get_sparkline('health', 40)}")
    print(f"    Mask:    {observer.get_sparkline('mask_density', 40)}")
    print(f"    Loss:    {observer.get_sparkline('loss', 40)}")

    # Clean up
    observer.stop()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
