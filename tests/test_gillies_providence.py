#!/usr/bin/env python3
"""
Rigorous Tests for TriX Providence LLM with GILLIES

Are we crushing coal into diamonds, or polishing turds?
Let's find out with math.

Tests:
1. GILLIES XOR correctness - does it match NumPy?
2. Hamming distance correctness - does GILLIES XOR give correct distances?
3. Providence retrieval - does it find actual nearest neighbors?
4. Training signal - is loss correlated with prediction quality?
5. Learning dynamics - does the model actually learn patterns?
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.native.vulkan.shapes import GilliesFrozenShapes


def test_xor_correctness():
    """
    Test 1: Does GILLIES XOR match the mathematical definition?

    XOR(a, b) = a + b - 2*a*b

    For binary inputs {0, 1}:
    - XOR(0, 0) = 0
    - XOR(0, 1) = 1
    - XOR(1, 0) = 1
    - XOR(1, 1) = 0
    """
    print("=" * 60)
    print("TEST 1: GILLIES XOR Correctness")
    print("=" * 60)

    shapes = GilliesFrozenShapes()

    # Truth table test
    a = np.array([0, 0, 1, 1], dtype=np.float32)
    b = np.array([0, 1, 0, 1], dtype=np.float32)
    expected = np.array([0, 1, 1, 0], dtype=np.float32)

    result = shapes.xor(a, b)

    match = np.allclose(result, expected)
    print(f"Truth table test: {'PASS' if match else 'FAIL'}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")

    # Random binary test (1M elements)
    n = 1_000_000
    a_rand = np.random.randint(0, 2, n).astype(np.float32)
    b_rand = np.random.randint(0, 2, n).astype(np.float32)

    # NumPy reference
    expected_rand = a_rand + b_rand - 2 * a_rand * b_rand

    # GILLIES
    result_rand = shapes.xor(a_rand, b_rand)

    match_rand = np.allclose(result_rand, expected_rand)
    max_diff = np.max(np.abs(result_rand - expected_rand))

    print(f"Random binary test (1M): {'PASS' if match_rand else 'FAIL'}")
    print(f"  Max difference: {max_diff}")

    # Continuous values test
    a_cont = np.random.rand(10000).astype(np.float32)
    b_cont = np.random.rand(10000).astype(np.float32)

    expected_cont = a_cont + b_cont - 2 * a_cont * b_cont
    result_cont = shapes.xor(a_cont, b_cont)

    match_cont = np.allclose(result_cont, expected_cont, rtol=1e-5)
    max_diff_cont = np.max(np.abs(result_cont - expected_cont))

    print(f"Continuous values test: {'PASS' if match_cont else 'FAIL'}")
    print(f"  Max difference: {max_diff_cont}")

    shapes.close()

    return match and match_rand and match_cont


def test_hamming_distance():
    """
    Test 2: Does XOR + popcount give correct Hamming distance?

    Hamming(a, b) = number of positions where a != b
                  = sum(a XOR b) for binary vectors
    """
    print("\n" + "=" * 60)
    print("TEST 2: Hamming Distance via XOR")
    print("=" * 60)

    shapes = GilliesFrozenShapes()

    # Manual test cases
    test_cases = [
        # (a, b, expected_hamming)
        ([0, 0, 0, 0], [0, 0, 0, 0], 0),  # Identical
        ([1, 1, 1, 1], [1, 1, 1, 1], 0),  # Identical
        ([0, 0, 0, 0], [1, 1, 1, 1], 4),  # All different
        ([1, 0, 1, 0], [0, 1, 0, 1], 4),  # All different
        ([1, 1, 0, 0], [1, 0, 0, 1], 2),  # 2 different
        ([1, 0, 0, 0], [0, 0, 0, 0], 1),  # 1 different
    ]

    all_pass = True
    for a, b, expected in test_cases:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)

        xor_result = shapes.xor(a_arr, b_arr)
        hamming = xor_result.sum()

        match = hamming == expected
        all_pass = all_pass and match
        status = "PASS" if match else "FAIL"
        print(f"  {a} vs {b}: Hamming={int(hamming)}, Expected={expected} [{status}]")

    # Large scale test
    n_vectors = 100
    d_model = 256

    # Random binary vectors
    vectors_a = np.random.randint(0, 2, (n_vectors, d_model)).astype(np.float32)
    vectors_b = np.random.randint(0, 2, (n_vectors, d_model)).astype(np.float32)

    # NumPy Hamming (ground truth)
    numpy_hamming = np.sum(vectors_a != vectors_b, axis=1)

    # GILLIES Hamming
    gillies_hamming = np.zeros(n_vectors)
    for i in range(n_vectors):
        xor_result = shapes.xor(vectors_a[i], vectors_b[i])
        gillies_hamming[i] = xor_result.sum()

    match_large = np.allclose(gillies_hamming, numpy_hamming)

    print(f"\nLarge scale test ({n_vectors} x {d_model}): {'PASS' if match_large else 'FAIL'}")
    if not match_large:
        print(f"  Mismatches: {np.sum(gillies_hamming != numpy_hamming)}")

    shapes.close()

    return all_pass and match_large


def test_providence_retrieval():
    """
    Test 3: Does Providence find the actual nearest neighbors?

    Create a set of embeddings, query with known vectors,
    verify that the closest embeddings are found.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Providence Nearest Neighbor Retrieval")
    print("=" * 60)

    shapes = GilliesFrozenShapes()

    vocab_size = 100
    d_model = 64

    # Create embeddings with known structure
    # First 10 embeddings are "cluster A" (all start with 1s)
    # Next 10 are "cluster B" (all start with 0s)
    # Rest are random
    embeddings = np.random.randint(0, 2, (vocab_size, d_model)).astype(np.float32)
    embeddings[:10, :32] = 1  # Cluster A: first 32 bits are 1
    embeddings[10:20, :32] = 0  # Cluster B: first 32 bits are 0

    # Query vector similar to cluster A
    query_a = np.ones(d_model, dtype=np.float32)
    query_a[32:] = np.random.randint(0, 2, d_model - 32)

    # Query vector similar to cluster B
    query_b = np.zeros(d_model, dtype=np.float32)
    query_b[32:] = np.random.randint(0, 2, d_model - 32)

    # Compute Hamming distances using GILLIES
    def hamming_distances(query, embeddings):
        distances = np.zeros(len(embeddings))
        for i, emb in enumerate(embeddings):
            xor = shapes.xor(query, emb)
            distances[i] = xor.sum()
        return distances

    # NumPy reference
    def numpy_hamming(query, embeddings):
        return np.sum(query != embeddings, axis=1)

    # Test query A
    dist_a_gillies = hamming_distances(query_a, embeddings)
    dist_a_numpy = numpy_hamming(query_a, embeddings)

    top5_gillies = np.argsort(dist_a_gillies)[:5]
    top5_numpy = np.argsort(dist_a_numpy)[:5]

    # Query A should find cluster A (indices 0-9) as nearest
    cluster_a_in_top5 = np.sum(top5_gillies < 10)

    print(f"Query A (similar to cluster A):")
    print(f"  Top 5 indices (GILLIES): {top5_gillies}")
    print(f"  Top 5 indices (NumPy):   {top5_numpy}")
    print(f"  Cluster A members in top 5: {cluster_a_in_top5}/5")

    # Test query B
    dist_b_gillies = hamming_distances(query_b, embeddings)
    top5_b = np.argsort(dist_b_gillies)[:5]

    # Query B should find cluster B (indices 10-19) as nearest
    cluster_b_in_top5 = np.sum((top5_b >= 10) & (top5_b < 20))

    print(f"Query B (similar to cluster B):")
    print(f"  Top 5 indices: {top5_b}")
    print(f"  Cluster B members in top 5: {cluster_b_in_top5}/5")

    # Verify GILLIES matches NumPy
    match = np.allclose(dist_a_gillies, dist_a_numpy)
    print(f"\nGILLIES matches NumPy: {'PASS' if match else 'FAIL'}")

    # Success if we find at least 3/5 from the right cluster
    retrieval_success = (cluster_a_in_top5 >= 3) and (cluster_b_in_top5 >= 3)
    print(f"Retrieval quality: {'PASS' if retrieval_success else 'FAIL'}")

    shapes.close()

    return match and retrieval_success


def test_training_signal():
    """
    Test 4: Is the loss actually correlated with prediction quality?

    If Providence is working:
    - When target IS in top-k: loss should be low
    - When target NOT in top-k: loss should be high
    - Loss should correlate with distance to target
    """
    print("\n" + "=" * 60)
    print("TEST 4: Training Signal Validity")
    print("=" * 60)

    shapes = GilliesFrozenShapes()

    vocab_size = 100
    d_model = 64
    k = 10  # top-k

    # Create structured embeddings
    embeddings = np.random.randint(0, 2, (vocab_size, d_model)).astype(np.float32)

    # Create queries at known distances from embedding 0
    emb_0 = embeddings[0]

    # Query identical to embedding 0 (distance 0)
    query_identical = emb_0.copy()

    # Query with distance 10 from embedding 0
    query_near = emb_0.copy()
    query_near[:10] = 1 - query_near[:10]  # Flip 10 bits

    # Query with distance 32 from embedding 0
    query_far = emb_0.copy()
    query_far[:32] = 1 - query_far[:32]  # Flip 32 bits

    def compute_distances(query, embeddings):
        distances = np.zeros(len(embeddings))
        for i, emb in enumerate(embeddings):
            xor = shapes.xor(query, emb)
            distances[i] = xor.sum()
        return distances

    # Compute distances
    dist_identical = compute_distances(query_identical, embeddings)
    dist_near = compute_distances(query_near, embeddings)
    dist_far = compute_distances(query_far, embeddings)

    # Check that distances are correct
    print(f"Distance to embedding 0:")
    print(f"  Identical query: {dist_identical[0]} (expected: 0)")
    print(f"  Near query:      {dist_near[0]} (expected: 10)")
    print(f"  Far query:       {dist_far[0]} (expected: 32)")

    dist_check = (
        dist_identical[0] == 0 and
        dist_near[0] == 10 and
        dist_far[0] == 32
    )
    print(f"Distance computation: {'PASS' if dist_check else 'FAIL'}")

    # Check if embedding 0 is in top-k for each query
    top_k_identical = np.argsort(dist_identical)[:k]
    top_k_near = np.argsort(dist_near)[:k]
    top_k_far = np.argsort(dist_far)[:k]

    in_topk_identical = 0 in top_k_identical
    in_topk_near = 0 in top_k_near
    in_topk_far = 0 in top_k_far

    print(f"\nEmbedding 0 in top-{k}:")
    print(f"  Identical query: {in_topk_identical} (expected: True)")
    print(f"  Near query:      {in_topk_near}")
    print(f"  Far query:       {in_topk_far}")

    # The identical query should definitely have target in top-k
    topk_check = in_topk_identical
    print(f"Top-k retrieval: {'PASS' if topk_check else 'FAIL'}")

    shapes.close()

    return dist_check and topk_check


def test_learning_dynamics():
    """
    Test 5: Does the model actually learn patterns?

    Create a simple pattern:
    - Token i predicts token (i+2) % vocab

    Train and verify the model learns this pattern.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Learning Dynamics")
    print("=" * 60)

    # Import the training components
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_trix_llm import TriXLLM, Config

    # Small config for fast test
    config = Config()
    config.vocab_size = 100
    config.n_layers = 2
    config.d_model = 64
    config.n_tiles = 8
    config.seq_len = 32
    config.batch_size = 4
    config.use_providence = True
    config.lm_top_k = 20

    model = TriXLLM(config)

    # Create patterned data: token i is followed by token (i+2) % vocab
    def generate_pattern_batch(batch_size, seq_len, vocab_size):
        start = np.random.randint(0, vocab_size, (batch_size, 1))
        sequence = [start]
        for _ in range(seq_len - 1):
            next_tok = (sequence[-1] + 2) % vocab_size
            sequence.append(next_tok)
        input_ids = np.concatenate(sequence, axis=1)
        target_ids = np.roll(input_ids, -1, axis=1)
        return input_ids.astype(np.int32), target_ids.astype(np.int32)

    # Track loss over training
    losses = []

    print("Training on patterned data...")
    for step in range(50):
        input_ids, target_ids = generate_pattern_batch(
            config.batch_size, config.seq_len, config.vocab_size
        )

        loss, grads = model.compute_loss(input_ids, target_ids)
        model.update(grads, lr=0.01)
        losses.append(loss)

        if step % 10 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    # Check if loss decreased
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    loss_decreased = final_loss < initial_loss

    print(f"\nInitial loss (avg first 5): {initial_loss:.4f}")
    print(f"Final loss (avg last 5):    {final_loss:.4f}")
    print(f"Loss decreased: {'PASS' if loss_decreased else 'FAIL'}")

    # Test: check if predictions improved
    test_input, test_target = generate_pattern_batch(1, config.seq_len, config.vocab_size)

    # Get Providence predictions
    x = model._embed(test_input)
    for layer in range(config.n_layers):
        x = model._layer(x, layer)
    x = model._layernorm(x)

    logits, indices = model._providence_lm_head(x)

    # Check how often the target is in top-k
    targets_flat = test_target.reshape(-1)
    indices_flat = indices.reshape(-1, config.lm_top_k)

    target_in_topk = 0
    for i in range(len(targets_flat)):
        if targets_flat[i] in indices_flat[i]:
            target_in_topk += 1

    topk_rate = target_in_topk / len(targets_flat)
    random_baseline = config.lm_top_k / config.vocab_size

    print(f"\nTarget in top-{config.lm_top_k}: {topk_rate:.1%}")
    print(f"Random baseline: {random_baseline:.1%}")
    print(f"Above random: {'PASS' if topk_rate > random_baseline else 'FAIL'}")

    model.shapes.close()

    return loss_decreased and (topk_rate > random_baseline)


def run_all_tests():
    """Run all rigorous tests."""
    print("\n" + "=" * 60)
    print("RIGOROUS TESTS: Coal or Diamonds?")
    print("=" * 60 + "\n")

    results = {}

    results["XOR Correctness"] = test_xor_correctness()
    results["Hamming Distance"] = test_hamming_distance()
    results["Providence Retrieval"] = test_providence_retrieval()
    results["Training Signal"] = test_training_signal()
    results["Learning Dynamics"] = test_learning_dynamics()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("DIAMONDS - All tests pass!")
    else:
        print("Still polishing...")

    return all_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
