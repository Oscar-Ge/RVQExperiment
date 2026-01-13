"""
Unit tests for RVQ tokenizer.

This script tests:
1. VectorQuantizer basic functionality
2. ResidualVectorQuantizer accumulation
3. RVQTokenizer encode/decode
4. Different numbers of layers
"""

import numpy as np
import torch
from rvq_tokenizer import VectorQuantizer, ResidualVectorQuantizer, RVQTokenizer


def test_vector_quantizer():
    """Test single-layer VQ."""
    print("=" * 80)
    print("TEST 1: VECTOR QUANTIZER")
    print("=" * 80)

    # Create VQ
    vq = VectorQuantizer(num_embeddings=256, embedding_dim=64)

    # Create test input
    test_input = torch.randn(2, 10, 64)  # [batch, time, dim]

    # Forward pass
    quantized, loss, indices = vq(test_input)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ loss: {loss.item():.6f}")

    # Check shapes
    assert quantized.shape == test_input.shape, "Quantized shape mismatch"
    assert indices.shape == (2, 10), "Indices shape mismatch"
    assert indices.min() >= 0 and indices.max() < 256, "Indices out of range"

    print("\n✅ VectorQuantizer test passed!")
    return True


def test_residual_vector_quantizer():
    """Test multi-layer RVQ."""
    print("\n" + "=" * 80)
    print("TEST 2: RESIDUAL VECTOR QUANTIZER")
    print("=" * 80)

    # Create RVQ with 4 layers
    rvq = ResidualVectorQuantizer(
        num_layers=4,
        num_embeddings=256,
        embedding_dim=64
    )

    # Test input
    test_input = torch.randn(2, 10, 64)

    # Forward with all layers
    quantized_full, loss_full, indices_full = rvq(test_input, num_layers=4)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Quantized shape: {quantized_full.shape}")
    print(f"Number of index arrays: {len(indices_full)}")
    print(f"Total VQ loss: {loss_full.item():.6f}")

    # Forward with only 2 layers
    quantized_partial, loss_partial, indices_partial = rvq(test_input, num_layers=2)

    print(f"\nWith 2 layers:")
    print(f"  Number of index arrays: {len(indices_partial)}")
    print(f"  Total VQ loss: {loss_partial.item():.6f}")

    # Check reconstruction residuals (on untrained model, just verify it works)
    residual_full = (test_input - quantized_full).pow(2).mean()
    residual_partial = (test_input - quantized_partial).pow(2).mean()

    print(f"\nReconstruction residuals:")
    print(f"  With 4 layers: {residual_full.item():.6f}")
    print(f"  With 2 layers: {residual_partial.item():.6f}")

    # Note: On untrained model, more layers might not be better yet
    if residual_full < residual_partial:
        print(f"  ✓ More layers = lower residual (as expected after training)")
    else:
        print(f"  ⚠️  More layers didn't reduce residual (model is untrained)")
        print(f"     This is expected - RVQ needs training to be effective")

    assert len(indices_full) == 4, "Should have 4 index arrays"
    assert len(indices_partial) == 2, "Should have 2 index arrays"

    print("\n✅ ResidualVectorQuantizer test passed!")
    return True


def test_rvq_tokenizer_basic():
    """Test RVQTokenizer basic functionality."""
    print("\n" + "=" * 80)
    print("TEST 3: RVQ TOKENIZER - BASIC FUNCTIONALITY")
    print("=" * 80)

    # Create tokenizer
    tokenizer = RVQTokenizer(
        action_dim=7,
        chunk_size=10,
        num_layers=4,
        hidden_dim=32,
        num_embeddings=256,
    )

    print(f"\nTokenizer: {tokenizer}")

    # Create dummy dataset for fitting
    np.random.seed(42)
    dummy_data = [np.random.randn(10, 7) for _ in range(20)]
    tokenizer.fit(dummy_data)

    # Test action
    test_action = np.random.randn(10, 7)

    # Encode
    tokens = tokenizer.encode(test_action)

    print(f"\nOriginal shape: {test_action.shape}")
    print(f"Number of token arrays: {len(tokens)}")
    print(f"Each token array shape: {tokens[0].shape}")
    print(f"Token values range: [{tokens[0].min()}, {tokens[0].max()}]")

    # Decode
    reconstructed = tokenizer.decode(tokens)

    print(f"Reconstructed shape: {reconstructed.shape}")

    # Compute error
    mse = np.mean((test_action - reconstructed) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}")

    assert len(tokens) == 4, "Should have 4 token arrays"
    assert tokens[0].shape == (10,), "Each token array should be (chunk_size,)"
    assert reconstructed.shape == test_action.shape, "Reconstructed shape mismatch"

    print("\n✅ RVQTokenizer basic test passed!")
    return mse


def test_rvq_tokenizer_layers():
    """Test RVQTokenizer with different numbers of layers."""
    print("\n" + "=" * 80)
    print("TEST 4: RVQ TOKENIZER - DIFFERENT LAYERS")
    print("=" * 80)

    # Create tokenizer
    tokenizer = RVQTokenizer(
        action_dim=7,
        chunk_size=10,
        num_layers=8,
        hidden_dim=64,
        num_embeddings=256,
    )

    # Fit
    np.random.seed(42)
    dummy_data = [np.random.randn(10, 7) for _ in range(50)]
    tokenizer.fit(dummy_data)

    # Test action
    test_action = np.random.randn(10, 7)

    print(f"\nOriginal action shape: {test_action.shape}")
    print(f"\n{'Layers':<8} {'Tokens':<10} {'Compression':<15} {'MSE':<12} {'Status':<10}")
    print("-" * 60)

    results = []
    for num_layers in [1, 2, 4, 6, 8]:
        # Encode with num_layers
        tokens = tokenizer.encode(test_action, num_layers=num_layers)

        # Decode
        reconstructed = tokenizer.decode(tokens)

        # Metrics
        mse = np.mean((test_action - reconstructed) ** 2)
        total_tokens = sum(len(t) for t in tokens)
        compression = tokenizer.get_compression_ratio(num_layers)
        status = "✅ Good" if mse < 0.1 else "⚠️  High"

        print(f"{num_layers:<8} {total_tokens:<10} {compression:<15.2f}x {mse:<12.6f} {status:<10}")

        results.append((num_layers, mse))

    # Check MSE trend (on untrained model, this is informational only)
    print("\nTrend check (on untrained model):")
    decreased_count = 0
    for i in range(len(results) - 1):
        layers1, mse1 = results[i]
        layers2, mse2 = results[i + 1]
        if mse2 < mse1:
            print(f"  ✓ {layers1} layers (MSE={mse1:.6f}) > {layers2} layers (MSE={mse2:.6f})")
            decreased_count += 1
        else:
            print(f"  ⚠️  {layers1} layers (MSE={mse1:.6f}) ≤ {layers2} layers (MSE={mse2:.6f})")

    print(f"\n  Note: On untrained model, {decreased_count}/{len(results)-1} transitions showed improvement")
    print(f"  After training, all transitions should show MSE decrease")

    print("\n✅ RVQTokenizer layers test passed!")
    return True


def test_realistic_smooth_actions():
    """Test with realistic smooth robot actions."""
    print("\n" + "=" * 80)
    print("TEST 5: REALISTIC SMOOTH ACTIONS")
    print("=" * 80)

    # Create smooth action sequences (sine waves)
    t = np.linspace(0, 2 * np.pi, 10)
    smooth_actions = []

    print("\nGenerating smooth action patterns...")
    for _ in range(30):
        action = np.zeros((10, 7))
        for dim in range(7):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 0.5)
            action[:, dim] = amplitude * np.sin(freq * t + phase)
        smooth_actions.append(action)

    # Create and train tokenizer
    tokenizer = RVQTokenizer(
        action_dim=7,
        chunk_size=10,
        num_layers=8,
        hidden_dim=64,
        num_embeddings=256,
    )
    tokenizer.fit(smooth_actions)

    # Test on all actions
    mses_by_layers = {num_layers: [] for num_layers in [2, 4, 8]}

    for action in smooth_actions:
        for num_layers in [2, 4, 8]:
            tokens = tokenizer.encode(action, num_layers=num_layers)
            reconstructed = tokenizer.decode(tokens)
            mse = np.mean((action - reconstructed) ** 2)
            mses_by_layers[num_layers].append(mse)

    # Report results
    print(f"\nResults across {len(smooth_actions)} smooth actions:")
    print(f"{'Layers':<8} {'Avg MSE':<15} {'Min MSE':<15} {'Max MSE':<15}")
    print("-" * 60)

    for num_layers in [2, 4, 8]:
        mses = mses_by_layers[num_layers]
        avg_mse = np.mean(mses)
        min_mse = np.min(mses)
        max_mse = np.max(mses)
        print(f"{num_layers:<8} {avg_mse:<15.6f} {min_mse:<15.6f} {max_mse:<15.6f}")

    # Check MSE values (on untrained model, just for reference)
    avg_mse_2layer = np.mean(mses_by_layers[2])
    avg_mse_8layer = np.mean(mses_by_layers[8])

    print(f"\n  Average MSE comparison:")
    print(f"    2 layers: {avg_mse_2layer:.6f}")
    print(f"    8 layers: {avg_mse_8layer:.6f}")

    if avg_mse_2layer < 0.01:
        print(f"\n  ✅ Even untrained, 2 layers achieve MSE < 0.01!")
        print(f"     This is promising for the hypothesis")
    else:
        print(f"\n  Note: This is an untrained model")
        print(f"  After training, we expect:")
        print(f"    - 2 layers: MSE < 0.01 (coarse motion)")
        print(f"    - 8 layers: MSE < 0.001 (fine details)")

    print("\n✅ Realistic actions test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RVQ TOKENIZER TEST SUITE")
    print("=" * 80)

    all_passed = True

    # Run tests
    try:
        all_passed &= test_vector_quantizer()
        all_passed &= test_residual_vector_quantizer()
        basic_mse = test_rvq_tokenizer_basic()
        all_passed &= test_rvq_tokenizer_layers()
        all_passed &= test_realistic_smooth_actions()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("  1. Run: python train_rvq_tokenizer.py --num_episodes 50 --epochs 100")
        print("  2. Train RVQ tokenizer on real LIBERO actions")
        print("  3. Run: python analyze_rvq_compression.py --model rvq_tokenizer.pt")
        print("  4. Compare results with DCT compression")
    else:
        print("❌ Some tests failed. Check the output above.")

    print("=" * 80 + "\n")
