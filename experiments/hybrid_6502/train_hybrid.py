#!/usr/bin/env python3
"""
Train Hybrid 6502: Let the Geometry Teach Itself

This experiment tests the hypothesis:
    Given the right frozen shapes, the routing is FORCED.
    A randomly-initialized routing layer will converge to the
    analytically-correct opcode→shape mapping.

If successful, this proves that computation has an inherent geometry.
The 6502's structure is INEVITABLE given the shapes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

from data_generator import (
    Hybrid6502DataGenerator,
    PHASE1_OPCODES,
    get_ground_truth_routing,
    NUM_OPCODES,
)
from hybrid_model import Hybrid6502, compare_routing


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
TAU_START = 2.0
TAU_END = 0.1
TAU_DECAY = 25  # epochs for tau to decay by factor of e
SAMPLES_PER_OPCODE = 5000
SEED = 42


# =============================================================================
# TRAINING
# =============================================================================

def train():
    """Main training loop."""
    print("=" * 70)
    print("HYBRID 6502: LET THE GEOMETRY TEACH ITSELF")
    print("=" * 70)
    print()

    # Set seed
    torch.manual_seed(SEED)

    # Generate data
    print("Generating training data...")
    generator = Hybrid6502DataGenerator(
        num_samples_per_opcode=SAMPLES_PER_OPCODE,
        seed=SEED
    )
    train_data, test_data = generator.generate_train_test_split(test_ratio=0.1)

    opcodes_train, regs_b_train, carry_b_train, regs_a_train, carry_a_train = train_data
    opcodes_test, regs_b_test, carry_b_test, regs_a_test, carry_a_test = test_data

    print(f"  Train: {len(opcodes_train):,} samples")
    print(f"  Test:  {len(opcodes_test):,} samples")
    print()

    # Create dataloader
    train_dataset = TensorDataset(
        opcodes_train, regs_b_train, carry_b_train, regs_a_train, carry_a_train
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(
        opcodes_test, regs_b_test, carry_b_test, regs_a_test, carry_a_test
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = Hybrid6502(num_opcodes=NUM_OPCODES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {model.param_count()}")
    print(f"  (All in routing layer - shapes are frozen)")
    print()

    # Ground truth for comparison
    ground_truth = get_ground_truth_routing()

    # Loss function
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'entropy': [],
        'tau': [],
        'shape_match': [],
    }

    print("Training...")
    print("-" * 70)
    print(f"{'Epoch':>5} {'Loss':>10} {'Acc':>8} {'Entropy':>8} {'Tau':>6} {'Shape':>8}")
    print("-" * 70)

    best_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Temperature annealing (exponential decay)
        tau = max(TAU_END, TAU_START * math.exp(-epoch / TAU_DECAY))

        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            opcode, reg_b, carry_b, reg_a, carry_a = batch

            # Forward pass
            out = model(opcode, reg_b, carry_b, tau=tau, hard=False)

            # Loss: compare predicted registers to actual
            # The output register should match, others should stay same
            loss = mse(out['new_registers'], reg_a)

            # Also loss on carry
            loss = loss + mse(out['carry'], carry_a) * 0.5

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate
        accuracy = evaluate(model, test_loader)
        entropy = model.routing_entropy()

        # Compare to ground truth
        learned = model.get_routing_matrix()
        match = compare_routing(learned, ground_truth)
        shape_match = match.get('shape', 0)

        # Record history
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['entropy'].append(entropy)
        history['tau'].append(tau)
        history['shape_match'].append(shape_match)

        # Print progress
        print(f"{epoch+1:5d} {avg_loss:10.6f} {accuracy:7.2f}% {entropy:8.3f} {tau:6.3f} {shape_match*100:7.1f}%")

        # Track best
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Early stopping if perfect
        if accuracy >= 99.99 and shape_match >= 0.99:
            print(f"\n*** Converged at epoch {epoch+1}! ***")
            break

    print("-" * 70)
    print()

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print()

    model.eval()
    final_accuracy = evaluate(model, test_loader, verbose=True)
    print()

    # Compare routing
    print("ROUTING COMPARISON")
    print("-" * 70)
    learned = model.get_routing_matrix()
    match = compare_routing(learned, ground_truth)

    print(f"Shape routing match:   {match.get('shape', 0)*100:.1f}%")
    print(f"Input A routing match: {match.get('input_a', 0)*100:.1f}%")
    print(f"Input B routing match: {match.get('input_b', 0)*100:.1f}%")
    print(f"Output routing match:  {match.get('output', 0)*100:.1f}%")
    print(f"Uses carry match:      {match.get('uses_carry', 0)*100:.1f}%")
    print()

    # Detailed routing analysis
    print("LEARNED SHAPE ROUTING")
    print("-" * 70)
    print(f"{'Op':>3} {'Name':>4} {'Learned':>8} {'Correct':>8} {'Match':>6}")
    print("-" * 70)

    all_match = True
    for opcode_id in range(NUM_OPCODES):
        spec = PHASE1_OPCODES[opcode_id]
        learned_shape = learned['shape'][opcode_id].item()
        correct_shape = ground_truth['shape'][opcode_id].item()
        is_match = learned_shape == correct_shape
        if not is_match:
            all_match = False
        match_str = "OK" if is_match else "MISS"
        print(f"{opcode_id:3d} {spec['name']:>4} {learned_shape:8d} {correct_shape:8d} {match_str:>6}")

    print("-" * 70)
    print()

    # Final verdict
    print("=" * 70)
    if all_match and final_accuracy >= 99.0:
        print("SUCCESS: THE GEOMETRY TAUGHT ITSELF")
        print()
        print("The routing converged to the analytically-correct mapping.")
        print("This proves: Computation is geometry. The shapes force the routing.")
    elif final_accuracy >= 99.0:
        print("PARTIAL SUCCESS: High accuracy but routing differs")
        print()
        print("This may indicate degeneracy - multiple valid routings exist.")
    else:
        print("NEEDS MORE TRAINING")
        print()
        print("The routing has not converged. Consider:")
        print("  - More epochs")
        print("  - Different learning rate")
        print("  - Slower temperature annealing")
    print("=" * 70)

    return model, history


def evaluate(model, loader, verbose=False):
    """Evaluate model with hard routing."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            opcode, reg_b, carry_b, reg_a, carry_a = batch

            out = model(opcode, reg_b, carry_b, tau=0.1, hard=True)

            # Check if all register bits match
            pred = (out['new_registers'] > 0.5).float()
            target = reg_a

            # Match if all bits of all registers are correct
            batch_correct = (pred == target).all(dim=-1).all(dim=-1).sum().item()
            correct += batch_correct
            total += pred.shape[0]

    accuracy = correct / total * 100

    if verbose:
        print(f"Test Accuracy (hard routing): {accuracy:.2f}%")
        print(f"  {correct:,} / {total:,} samples fully correct")

    model.train()
    return accuracy


if __name__ == "__main__":
    model, history = train()

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'routing': model.get_routing_matrix(),
    }, results_dir / "hybrid_6502_trained.pt")

    print(f"\nResults saved to {results_dir / 'hybrid_6502_trained.pt'}")
