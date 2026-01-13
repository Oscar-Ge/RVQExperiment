import numpy as np
from scipy.fft import dct, idct  # scipy >= 1.4.0 required


class MinimalDCTTokenizer:
    """
    Simplest possible DCT tokenizer - no BPE, just DCT + quantization.

    This tokenizer compresses robot action sequences using DCT (Discrete Cosine Transform)
    and quantization. It's designed to prove that DCT can compress LIBERO actions without
    losing task-critical information.

    CRITICAL FIX: Properly handles DCT coefficient range (not [-1, 1]!)
    """

    def __init__(self, action_dim=7, chunk_size=16, num_dct_keep=4, num_bins=256):
        """
        Initialize the DCT tokenizer.

        Args:
            action_dim: Dimensionality of actions (7 for LIBERO: x,y,z, roll,pitch,yaw, gripper)
            chunk_size: Number of actions per chunk (π0.5-libero uses 10)
            num_dct_keep: How many DCT coefficients to keep (controls compression ratio)
            num_bins: Quantization bins per coefficient (256 = uint8)
        """
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_dct_keep = num_dct_keep
        self.num_bins = num_bins

        # Action normalization stats (will be set during fit())
        self.action_min = None
        self.action_max = None

        # DCT coefficient range stats (will be set during fit())
        self.coeff_min = None
        self.coeff_max = None

    def fit(self, actions_dataset):
        """
        Compute normalization statistics from dataset.

        This function:
        1. Computes action min/max for normalizing actions to [-1, 1]
        2. Computes DCT coefficient min/max for proper quantization

        Args:
            actions_dataset: List of [chunk_size, action_dim] numpy arrays
        """
        # Step 1: Compute action statistics (for normalizing actions)
        all_actions = np.concatenate(actions_dataset, axis=0)
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)

        print(f"Fitted tokenizer on {len(all_actions)} actions")
        print(f"  Action range: [{self.action_min.min():.4f}, {self.action_max.max():.4f}]")

        # Step 2: Compute DCT coefficient statistics
        # Reshape actions into chunks
        num_chunks = len(all_actions) // self.chunk_size
        if num_chunks == 0:
            print("  Warning: Not enough actions to form a complete chunk, using dataset directly")
            reshaped_actions = np.array(actions_dataset)
        else:
            # Truncate to multiple of chunk_size
            truncated_actions = all_actions[:num_chunks * self.chunk_size]
            reshaped_actions = truncated_actions.reshape(num_chunks, self.chunk_size, self.action_dim)

        # Normalize actions to [-1, 1]
        actions_norm = 2 * (reshaped_actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1
        actions_norm = np.clip(actions_norm, -1, 1)

        # Compute DCT coefficients for all chunks and dimensions
        all_coeffs = []
        for chunk_idx in range(reshaped_actions.shape[0]):
            for dim in range(self.action_dim):
                coeffs = dct(actions_norm[chunk_idx, :, dim], type=2, norm='ortho')
                all_coeffs.extend(coeffs.tolist())

        all_coeffs = np.array(all_coeffs)

        # Record coefficient range with a small margin for safety
        margin = 0.1
        self.coeff_min = all_coeffs.min() - margin
        self.coeff_max = all_coeffs.max() + margin

        print(f"  DCT Coeff range: [{self.coeff_min:.4f}, {self.coeff_max:.4f}]")
        print(f"  → DC component is NOT in [-1, 1]! (This was the bug)")

    def encode(self, actions):
        """
        Encode actions into discrete tokens.

        For each action dimension:
        1. Normalize actions to [-1, 1]
        2. Apply DCT along time axis
        3. Keep top num_dct_keep coefficients
        4. Quantize to num_bins using ACTUAL coefficient range

        Args:
            actions: [chunk_size, action_dim] numpy array

        Returns:
            tokens: List of integers [num_dct_keep * action_dim]
        """
        if actions.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected actions shape ({self.chunk_size}, {self.action_dim}), "
                f"got {actions.shape}"
            )

        if self.action_min is None or self.coeff_min is None:
            raise ValueError("Tokenizer must be fitted before encoding. Call fit() first.")

        # Normalize actions to [-1, 1]
        actions_norm = 2 * (actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1
        actions_norm = np.clip(actions_norm, -1, 1)

        tokens = []
        for dim in range(self.action_dim):
            # DCT along time axis
            coeffs = dct(actions_norm[:, dim], type=2, norm='ortho')

            # Keep only top-k coefficients
            coeffs_keep = coeffs[:self.num_dct_keep]

            # Quantize using ACTUAL coefficient range (not [-1, 1]!)
            # Normalize coefficients to [0, 1]
            coeffs_norm = (coeffs_keep - self.coeff_min) / (self.coeff_max - self.coeff_min + 1e-8)
            coeffs_norm = np.clip(coeffs_norm, 0, 1)

            # Quantize to [0, num_bins-1]
            coeffs_quantized = np.round(coeffs_norm * (self.num_bins - 1)).astype(int)
            coeffs_quantized = np.clip(coeffs_quantized, 0, self.num_bins - 1)

            tokens.extend(coeffs_quantized.tolist())

        return tokens

    def decode(self, tokens):
        """
        Decode tokens back into actions.

        Args:
            tokens: List of integers [num_dct_keep * action_dim]

        Returns:
            actions: [chunk_size, action_dim] numpy array
        """
        if len(tokens) != self.num_dct_keep * self.action_dim:
            raise ValueError(
                f"Expected {self.num_dct_keep * self.action_dim} tokens, got {len(tokens)}"
            )

        if self.action_min is None or self.coeff_min is None:
            raise ValueError("Tokenizer must be fitted before decoding. Call fit() first.")

        tokens = np.array(tokens)
        actions_reconstructed = np.zeros((self.chunk_size, self.action_dim))

        for dim in range(self.action_dim):
            # Extract quantized coefficients for this dimension
            start_idx = dim * self.num_dct_keep
            end_idx = start_idx + self.num_dct_keep
            coeffs_quantized = tokens[start_idx:end_idx]

            # Dequantize: [0, num_bins-1] → [0, 1] → [coeff_min, coeff_max]
            coeffs_norm = coeffs_quantized / (self.num_bins - 1)
            coeffs_keep = coeffs_norm * (self.coeff_max - self.coeff_min) + self.coeff_min

            # Pad with zeros for discarded high-frequency coefficients
            coeffs_full = np.zeros(self.chunk_size)
            coeffs_full[:self.num_dct_keep] = coeffs_keep

            # Inverse DCT
            actions_norm = idct(coeffs_full, type=2, norm='ortho')

            # Denormalize back to original action range
            actions_reconstructed[:, dim] = (actions_norm + 1) / 2 * \
                (self.action_max[dim] - self.action_min[dim]) + self.action_min[dim]

        return actions_reconstructed

    def get_compression_ratio(self):
        """
        Calculate the compression ratio.

        Returns:
            float: Compression ratio (original_size / compressed_size)
        """
        original_size = self.chunk_size * self.action_dim
        compressed_size = self.num_dct_keep * self.action_dim
        return original_size / compressed_size

    def __repr__(self):
        coeff_range = f"[{self.coeff_min:.2f}, {self.coeff_max:.2f}]" if self.coeff_min is not None else "not fitted"
        return (
            f"MinimalDCTTokenizer("
            f"action_dim={self.action_dim}, "
            f"chunk_size={self.chunk_size}, "
            f"num_dct_keep={self.num_dct_keep}, "
            f"num_bins={self.num_bins}, "
            f"coeff_range={coeff_range}, "
            f"compression_ratio={self.get_compression_ratio():.2f}x)"
        )
