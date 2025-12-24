"""
Training Integration - Doula in the Training Loop.

Mesa 16.2: The Doula

This module provides utilities for integrating the Doula
into TriX training loops.

Glass box: every step observed, every pattern recorded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass

from .function import DoulaFunction
from .signature import DoulaGuidance


@dataclass
class DoulaTrainingConfig:
    """Configuration for Doula-assisted training."""

    # VDB settings
    vdb_path: Optional[str] = None
    domain: Optional[str] = None

    # Observation frequency
    observe_every: int = 1          # Observe every N steps
    store_every: int = 1            # Store to VDB every N observations
    query_every: int = 100          # Query for guidance every N steps

    # Guidance settings
    apply_guidance: bool = True     # Whether to apply temperature adjustments
    early_stop_on_stable: bool = True  # Stop when training is complete
    min_stable_steps: int = 50      # Steps of stability before stopping

    # Logging
    log_phases: bool = True         # Print phase transitions
    log_emergence: bool = True      # Print emergence events
    log_bottlenecks: bool = True    # Print bottleneck warnings


class DoulaCallback:
    """
    Callback for Doula observation during training.

    Use with any training framework:

        doula_callback = DoulaCallback(config)

        for step, batch in enumerate(dataloader):
            output, states, info = model(batch)
            loss = compute_loss(output, targets)

            # Let Doula observe
            guidance = doula_callback.on_step(
                step=step,
                info=info,
                loss=loss.item(),
            )

            if guidance and guidance.training_complete:
                break
    """

    def __init__(self, config: DoulaTrainingConfig):
        self.config = config
        self.doula = DoulaFunction(
            vdb_path=config.vdb_path,
            domain=config.domain,
        )
        self.step_count = 0
        self.current_phase = 'exploration'
        self.last_guidance: Optional[DoulaGuidance] = None

    def on_step(
        self,
        step: int,
        info: Dict,
        loss: float,
        accuracy: Optional[float] = None,
    ) -> Optional[DoulaGuidance]:
        """
        Called after each training step.

        Args:
            step: Current training step
            info: From model forward pass (contains routing info per layer)
            loss: Current loss value
            accuracy: Optional accuracy metric

        Returns:
            DoulaGuidance if query was performed, else None
        """
        self.step_count = step

        # Observe (aggregate + store)
        if step % self.config.observe_every == 0:
            self._observe(step, info, loss, accuracy)

        # Query for guidance
        guidance = None
        if step % self.config.query_every == 0 and step > 0:
            guidance = self._query_guidance()

        return guidance

    def _observe(
        self,
        step: int,
        info: Dict,
        loss: float,
        accuracy: Optional[float],
    ):
        """Observe routing dynamics from all layers."""
        layers_info = info.get('layers', [])

        for layer_idx, layer_info in enumerate(layers_info):
            # Get mixer routing info
            mixer_info = layer_info.get('mixer', {})
            if not mixer_info:
                continue

            # Aggregate
            signature = self.doula.aggregate(
                routing_info=mixer_info,
                step=step,
                layer=layer_idx,
                loss=loss,
            )

            # Store
            if step % self.config.store_every == 0:
                self.doula.store(signature, loss, accuracy)

    def _query_guidance(self) -> DoulaGuidance:
        """Query for guidance based on recent observations."""
        if not self.doula.history:
            return None

        # Use most recent signature
        current_sig = self.doula.history[-1]
        guidance = self.doula.query(current_sig)

        # Log phase transitions
        if self.config.log_phases and guidance.current_phase != self.current_phase:
            print(f"[Doula] Phase transition: {self.current_phase} → {guidance.current_phase}")
            self.current_phase = guidance.current_phase

        # Log emergence
        if self.config.log_emergence and guidance.emergence_imminent:
            print(f"[Doula] Emergence detected at step {self.step_count}")

        # Log bottlenecks
        if self.config.log_bottlenecks and guidance.bottleneck_detected:
            print(f"[Doula] Bottleneck warning at step {self.step_count}")

        # Log completion
        if guidance.training_complete:
            print(f"[Doula] Training complete at step {self.step_count}")

        self.last_guidance = guidance
        return guidance

    def should_stop(self) -> bool:
        """Check if training should stop."""
        if not self.config.early_stop_on_stable:
            return False
        if self.last_guidance is None:
            return False
        return self.last_guidance.training_complete

    def get_temperature_adjustment(self) -> float:
        """Get suggested temperature adjustment."""
        if not self.config.apply_guidance:
            return 0.0
        if self.last_guidance is None:
            return 0.0
        return self.last_guidance.temperature_adjustment

    def get_priority_shapes(self) -> list:
        """Get suggested priority shapes."""
        if self.last_guidance is None:
            return []
        return self.last_guidance.priority_shapes

    def save(self):
        """Save VDB to disk."""
        self.doula.save()

    def summarize(self) -> Dict:
        """Get summary of observations."""
        return self.doula.summarize()


def train_with_doula(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    config: DoulaTrainingConfig,
    epochs: int = 1,
    device: str = 'cpu',
) -> Dict:
    """
    Training loop with Doula observation.

    This is a reference implementation. Adapt to your framework.

    Args:
        model: TriX model (must return routing info)
        dataloader: Training data
        optimizer: Optimizer
        loss_fn: Loss function
        config: Doula configuration
        epochs: Number of epochs
        device: Device to train on

    Returns:
        Training summary including Doula observations
    """
    model = model.to(device)
    doula_callback = DoulaCallback(config)

    # Set steps per epoch
    steps_per_epoch = len(dataloader)
    doula_callback.doula.set_steps_per_epoch(steps_per_epoch)

    total_steps = 0
    training_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x = batch.to(device)
                y = None

            # Forward
            output, states, info = model(x)

            # Loss
            if y is not None:
                loss = loss_fn(output, y)
            else:
                loss = loss_fn(output)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Doula observation
            guidance = doula_callback.on_step(
                step=total_steps,
                info=info,
                loss=loss.item(),
            )

            # Apply temperature adjustment if suggested
            if guidance and config.apply_guidance:
                temp_adj = guidance.temperature_adjustment
                if hasattr(model, 'temperature'):
                    model.temperature = max(0.1, model.temperature + temp_adj)

            epoch_loss += loss.item()
            total_steps += 1

            # Early stopping
            if doula_callback.should_stop():
                print(f"[Doula] Early stop at epoch {epoch}, step {total_steps}")
                break

        avg_loss = epoch_loss / (batch_idx + 1)
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'phase': doula_callback.current_phase,
        })

        if doula_callback.should_stop():
            break

    # Save and summarize
    doula_callback.save()

    return {
        'history': training_history,
        'doula_summary': doula_callback.summarize(),
        'total_steps': total_steps,
        'final_phase': doula_callback.current_phase,
    }


class DoulaTrainer:
    """
    Wrapper class for training with Doula.

    Alternative to the functional API.

    Usage:
        trainer = DoulaTrainer(model, config)
        trainer.train(dataloader, epochs=10)
        summary = trainer.summarize()
    """

    def __init__(
        self,
        model: nn.Module,
        config: DoulaTrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters())
        self.loss_fn = loss_fn or F.cross_entropy
        self.callback = DoulaCallback(config)
        self.step = 0

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
    ) -> Tuple[float, Optional[DoulaGuidance]]:
        """
        Single training step with Doula observation.

        Args:
            batch: (input, target) tuple

        Returns:
            (loss, guidance) tuple
        """
        x, y = batch
        self.model.train()

        # Forward
        output, states, info = self.model(x)

        # Loss
        loss = self.loss_fn(output.view(-1, output.size(-1)), y.view(-1))

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Doula
        guidance = self.callback.on_step(
            step=self.step,
            info=info,
            loss=loss.item(),
        )

        self.step += 1
        return loss.item(), guidance

    def train(
        self,
        dataloader,
        epochs: int = 1,
    ) -> Dict:
        """
        Full training loop.

        Args:
            dataloader: Training data
            epochs: Number of epochs

        Returns:
            Training summary
        """
        self.callback.doula.set_steps_per_epoch(len(dataloader))
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                loss, guidance = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1

                if self.callback.should_stop():
                    break

            history.append({
                'epoch': epoch,
                'loss': epoch_loss / n_batches,
                'phase': self.callback.current_phase,
            })

            if self.callback.should_stop():
                break

        return {
            'history': history,
            'doula_summary': self.summarize(),
        }

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.callback.should_stop()

    def get_guidance(self) -> Optional[DoulaGuidance]:
        """Get most recent guidance."""
        return self.callback.last_guidance

    def save(self):
        """Save Doula VDB."""
        self.callback.save()

    def summarize(self) -> Dict:
        """Get Doula summary."""
        return self.callback.summarize()
