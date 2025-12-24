"""
Doula - The Birth Attendant for TriX Training.

Mesa 16.2: Glass Box Meta-Learning

The Doula observes how TriX learns - not to embody geometric truth,
but to understand the nature of geometries as they Emerge.

This helps us be better Doulas.

Components:
    - DynamicsSignature: Captures learning dynamics at a moment
    - DoulaEntry: VDB entry with embedding + full signature
    - DoulaGuidance: Suggestions based on past observations
    - PhaseDetector: Recognizes learning phases
    - DoulaVDB: Vector database for dynamics
    - DoulaFunction: Aggregate, store, query
    - DoulaCallback: Training loop integration
    - DoulaTrainer: High-level training wrapper

Usage:
    from trix.doula import DoulaFunction, DoulaTrainingConfig, DoulaCallback

    # Simple: Just observe
    doula = DoulaFunction(vdb_path="training_dynamics.json")

    for step, (x, y) in enumerate(dataloader):
        output, states, info = model(x)
        loss = compute_loss(output, y)

        # Observe each layer
        for layer_idx, layer_info in enumerate(info['layers']):
            signature = doula.aggregate(
                routing_info=layer_info['mixer'],
                step=step,
                layer=layer_idx,
                loss=loss.item(),
            )
            doula.store(signature, loss.item())

    # Save for later analysis
    doula.save()

    # With guidance:
    config = DoulaTrainingConfig(
        vdb_path="dynamics.json",
        domain="6502_alu",
        early_stop_on_stable=True,
    )
    callback = DoulaCallback(config)

    for step, batch in enumerate(dataloader):
        output, states, info = model(batch)
        guidance = callback.on_step(step, info, loss.item())

        if guidance and guidance.training_complete:
            print("Doula says we're done!")
            break

Glass Box Principles:
    - All data accessible (get_all_entries, query_log)
    - All patterns recorded (VDB persists everything)
    - All phases visible (exploration → transition → crystallization → stable)
    - Nothing hidden (even queries are logged)

"We do not teach the Doula to give birth.
 We teach her to witness birth.
 So she may help the next birth come easier."
"""

from .signature import (
    DynamicsSignature,
    DoulaEntry,
    DoulaGuidance,
)

from .phase import PhaseDetector

from .vdb import DoulaVDB

from .function import DoulaFunction

from .training import (
    DoulaTrainingConfig,
    DoulaCallback,
    DoulaTrainer,
    train_with_doula,
)

__all__ = [
    # Signatures
    'DynamicsSignature',
    'DoulaEntry',
    'DoulaGuidance',
    # Phase detection
    'PhaseDetector',
    # VDB
    'DoulaVDB',
    # Main function
    'DoulaFunction',
    # Training integration
    'DoulaTrainingConfig',
    'DoulaCallback',
    'DoulaTrainer',
    'train_with_doula',
]
