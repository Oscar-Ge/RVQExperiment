"""
Test the minimal DCT tokenizer with round-trip encoding/decoding.

This script verifies that:
1. The tokenizer can encode and decode actions
2. The reconstruction error is acceptable
3. Compression is achieved
"""

import numpy as np
from minimal_dct_tokenizer import MinimalDCTTokenizer


def test_roundtrip():
    """Test encode-decode gives back similar actions."""
    print("=" * 80)
    print("TESTING DCT TOKENIZER - ROUND-TRIP ENCODING")
    print("=" * 80)

    # Create tokenizer with 4 DCT coefficients (4x compression)
    tokenizer = MinimalDCTTokenizer(num_dct_keep=4)
    print(f"\nTokenizer: {tokenizer}")

    # Create dummy dataset for fitting
    print("\nGenerating dummy dataset for fitting...")
    np.random.seed(42)
    dummy_data = [np.random.randn(16, 7) for _ in range(10)]
    tokenizer.fit(dummy_data)

    # Test action chunk
    print("\nTesting round-trip encoding...")
    test_action = np.random.randn(16, 7)

    # Encode
    tokens = tokenizer.encode(test_action)
    print(f"\n  Original shape: {test_action.shape} ({test_action.size} values)")
    print(f"  Token length: {len(tokens)} tokens")
    print(f"  Compression ratio: {tokenizer.get_compression_ratio():.2f}x")

    # Decode
    reconstructed = tokenizer.decode(tokens)
    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Compute error metrics
    mse = np.mean((test_action - reconstructed) ** 2)
    mae = np.mean(np.abs(test_action - reconstructed))
    max_error = np.max(np.abs(test_action - reconstructed))

    print(f"\n  Reconstruction Metrics:")
    print(f"    MSE (Mean Squared Error): {mse:.6f}")
    print(f"    MAE (Mean Absolute Error): {mae:.6f}")
    print(f"    Max Absolute Error: {max_error:.6f}")

    # Check if reconstruction is good enough
    threshold = 0.1
    if mse < threshold:
        print(f"\n  ✅ PASSED! MSE ({mse:.6f}) < threshold ({threshold})")
    else:
        print(f"\n  ❌ FAILED! MSE ({mse:.6f}) >= threshold ({threshold})")
        print(f"     Reconstruction error too high!")

    return mse < threshold


def test_different_compression_ratios():
    """Test different compression settings."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT COMPRESSION RATIOS")
    print("=" * 80)

    # Create dummy dataset
    np.random.seed(42)
    dummy_data = [np.random.randn(16, 7) for _ in range(10)]
    test_action = np.random.randn(16, 7)

    print(f"\nOriginal size: {test_action.size} values (16 actions × 7 dims)")
    print(f"\n{'DCT Coeffs':<12} {'Tokens':<8} {'Compression':<12} {'MSE':<12} {'Status':<10}")
    print("-" * 60)

    for num_dct_keep in [2, 4, 6, 8, 12, 16]:
        tokenizer = MinimalDCTTokenizer(num_dct_keep=num_dct_keep)
        tokenizer.fit(dummy_data)

        tokens = tokenizer.encode(test_action)
        reconstructed = tokenizer.decode(tokens)
        mse = np.mean((test_action - reconstructed) ** 2)

        compression_ratio = tokenizer.get_compression_ratio()
        status = "✅ Good" if mse < 0.01 else "⚠️  High MSE"

        print(f"{num_dct_keep:<12} {len(tokens):<8} {compression_ratio:<12.2f}x {mse:<12.6f} {status:<10}")


def test_with_realistic_actions():
    """Test with more realistic action patterns."""
    print("\n" + "=" * 80)
    print("TESTING WITH REALISTIC ACTION PATTERNS")
    print("=" * 80)

    # Create realistic action sequences
    # Simulate smooth trajectories (sine waves with different frequencies)
    t = np.linspace(0, 2 * np.pi, 16)
    realistic_actions = []

    print("\nGenerating realistic action patterns...")
    print("  - Smooth reaching motions (low frequency)")
    print("  - Grasping patterns (medium frequency)")
    print("  - Fine manipulation (higher frequency)")

    for _ in range(20):
        action = np.zeros((16, 7))
        for dim in range(7):
            # Mix of different frequency components
            freq = np.random.uniform(0.5, 3.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 1.0)
            action[:, dim] = amplitude * np.sin(freq * t + phase)
        realistic_actions.append(action)

    # Test tokenizer
    tokenizer = MinimalDCTTokenizer(num_dct_keep=4)
    tokenizer.fit(realistic_actions)

    mses = []
    for action in realistic_actions:
        tokens = tokenizer.encode(action)
        reconstructed = tokenizer.decode(tokens)
        mse = np.mean((action - reconstructed) ** 2)
        mses.append(mse)

    avg_mse = np.mean(mses)
    std_mse = np.std(mses)

    print(f"\n  Results across {len(realistic_actions)} action chunks:")
    print(f"    Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"    Min MSE: {min(mses):.6f}")
    print(f"    Max MSE: {max(mses):.6f}")
    print(f"    Compression: {tokenizer.get_compression_ratio():.2f}x")

    if avg_mse < 0.01:
        print(f"\n  ✅ Excellent! Average MSE is very low")
    elif avg_mse < 0.05:
        print(f"\n  ✅ Good! Average MSE is acceptable")
    else:
        print(f"\n  ⚠️  Warning: Average MSE might be too high for some tasks")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("π0-FAST DCT TOKENIZER TEST SUITE")
    print("=" * 80)

    # Run all tests
    all_passed = True

    # Test 1: Basic round-trip
    passed = test_roundtrip()
    all_passed = all_passed and passed

    # Test 2: Different compression ratios
    test_different_compression_ratios()

    # Test 3: Realistic actions
    test_with_realistic_actions()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if all_passed:
        print("✅ All critical tests passed!")
        print("\nNext steps:")
        print("  1. Run analyze_libero_actions.py to test on real π0.5 actions")
        print("  2. Check if compression works well on actual robot tasks")
        print("  3. Analyze MSE across different task phases (reach, grasp, etc.)")
    else:
        print("❌ Some tests failed. Check the output above.")

    print("=" * 80 + "\n")
