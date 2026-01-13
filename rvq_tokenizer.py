"""
RVQ (Residual Vector Quantization) Tokenizer for Robot Actions.

This implements a VQ-VAE with Residual Quantization for compressing robot action
sequences into hierarchical discrete tokens.

Key concepts:
- Layer 1-2: Capture coarse motion (low frequency)
- Layer 3-8: Capture fine corrections (high frequency)
- Training: Residual dropout to force early layers to encode more information
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single-layer Vector Quantizer with EMA updates.

    Args:
        num_embeddings: Size of the codebook
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss
        decay: EMA decay for codebook updates
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # EMA parameters
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())
        self.decay = decay

    def forward(self, inputs):
        """
        Args:
            inputs: [B, T, D] where D = embedding_dim

        Returns:
            quantized: [B, T, D] quantized vectors
            loss: VQ loss (codebook + commitment)
            encoding_indices: [B, T] discrete codes
        """
        # Flatten to [B*T, D]
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances to codebook entries
        # distances: [B*T, num_embeddings]
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Get nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # [B*T]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [B*T, K]

        # Quantize
        quantized = F.embedding(encoding_indices, self.embedding.weight)  # [B*T, D]
        quantized = quantized.reshape(input_shape)  # [B, T, D]

        # Update codebook with EMA (only during training)
        if self.training:
            with torch.no_grad():
                # Update cluster sizes
                self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                        (1 - self.decay) * torch.sum(encodings, dim=0)

                # Laplace smoothing
                n = torch.sum(self._ema_cluster_size)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + 1e-5)
                    / (n + self.num_embeddings * 1e-5) * n
                )

                # Update embeddings
                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw
                self.embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Reshape indices to [B, T]
        encoding_indices = encoding_indices.reshape(input_shape[0], input_shape[1])

        return quantized, loss, encoding_indices


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-layer Residual Vector Quantizer.

    Each layer quantizes the residual from previous layers.
    """

    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_layers = num_layers

        # Create VQ layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_layers)
        ])

    def forward(self, inputs, num_layers=None, residual_dropout_prob=0.0):
        """
        Args:
            inputs: [B, T, D]
            num_layers: How many layers to use (default: all)
            residual_dropout_prob: Probability of dropping each residual layer during training

        Returns:
            quantized: [B, T, D] cumulative quantized output
            total_loss: Sum of VQ losses
            all_indices: List of [B, T] encoding indices for each layer
        """
        if num_layers is None:
            num_layers = self.num_layers

        residual = inputs
        quantized_sum = 0
        total_loss = 0
        all_indices = []

        for i in range(num_layers):
            # Quantize residual
            quantized, loss, indices = self.vq_layers[i](residual)

            # Residual dropout (only during training)
            if self.training and i > 0 and torch.rand(1).item() < residual_dropout_prob:
                # Skip this layer (force earlier layers to encode more)
                continue

            # Accumulate
            quantized_sum = quantized_sum + quantized
            total_loss = total_loss + loss
            all_indices.append(indices)

            # Update residual
            residual = residual - quantized

        return quantized_sum, total_loss, all_indices

    def encode(self, inputs, num_layers=None):
        """Encode to discrete tokens (inference mode)."""
        with torch.no_grad():
            _, _, all_indices = self.forward(inputs, num_layers)
        return all_indices

    def decode_indices(self, all_indices):
        """Decode from discrete tokens."""
        with torch.no_grad():
            quantized_sum = 0
            for i, indices in enumerate(all_indices):
                quantized = F.embedding(indices, self.vq_layers[i].embedding.weight)
                quantized_sum = quantized_sum + quantized
        return quantized_sum


class RVQTokenizer(nn.Module):
    """
    Complete RVQ-based action tokenizer with encoder and decoder.

    Architecture:
        Encoder: (T, action_dim) → (T, hidden_dim)
        RVQ: (T, hidden_dim) → num_layers discrete codes
        Decoder: (T, hidden_dim) → (T, action_dim)
    """

    def __init__(
        self,
        action_dim=7,
        chunk_size=10,
        num_layers=8,
        hidden_dim=64,
        num_embeddings=256,
        commitment_cost=0.25,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings

        # Action normalization stats
        self.register_buffer('action_min', torch.zeros(action_dim))
        self.register_buffer('action_max', torch.ones(action_dim))
        self.register_buffer('is_fitted', torch.tensor(False))

        # Encoder: (T, action_dim) → (T, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # RVQ
        self.rvq = ResidualVectorQuantizer(
            num_layers=num_layers,
            num_embeddings=num_embeddings,
            embedding_dim=hidden_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder: (T, hidden_dim) → (T, action_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def fit(self, actions_dataset):
        """
        Compute normalization statistics from dataset.

        Args:
            actions_dataset: List of [chunk_size, action_dim] numpy arrays
        """
        all_actions = np.concatenate(actions_dataset, axis=0)
        self.action_min = torch.from_numpy(all_actions.min(axis=0)).float()
        self.action_max = torch.from_numpy(all_actions.max(axis=0)).float()
        self.is_fitted = torch.tensor(True)

        print(f"Fitted RVQ tokenizer on {len(all_actions)} actions")
        print(f"  Action range: [{self.action_min.min():.4f}, {self.action_max.max():.4f}]")

    def normalize_actions(self, actions):
        """Normalize actions to [-1, 1]."""
        # Ensure action_min/max are on the same device as actions
        action_min = self.action_min.to(actions.device)
        action_max = self.action_max.to(actions.device)
        actions_norm = 2 * (actions - action_min) / (action_max - action_min + 1e-8) - 1
        return torch.clamp(actions_norm, -1, 1)

    def denormalize_actions(self, actions_norm):
        """Denormalize actions from [-1, 1] to original range."""
        # Ensure action_min/max are on the same device as actions_norm
        action_min = self.action_min.to(actions_norm.device)
        action_max = self.action_max.to(actions_norm.device)
        return (actions_norm + 1) / 2 * (action_max - action_min) + action_min

    def forward(self, actions, num_layers=None, residual_dropout_prob=0.0):
        """
        Forward pass with reconstruction.

        Args:
            actions: [B, T, action_dim] or numpy array
            num_layers: How many RVQ layers to use
            residual_dropout_prob: Dropout probability for residual layers

        Returns:
            reconstructed: [B, T, action_dim] reconstructed actions
            vq_loss: VQ loss
            all_indices: List of discrete codes
        """
        # Convert to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        # Add batch dim if needed
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)

        # Move to same device as model
        actions = actions.to(next(self.parameters()).device)

        # Normalize
        actions_norm = self.normalize_actions(actions)

        # Encode
        encoded = self.encoder(actions_norm)  # [B, T, hidden_dim]

        # Quantize with RVQ
        quantized, vq_loss, all_indices = self.rvq(
            encoded, num_layers, residual_dropout_prob
        )

        # Decode
        decoded_norm = self.decoder(quantized)  # [B, T, action_dim]

        # Denormalize
        reconstructed = self.denormalize_actions(decoded_norm)

        return reconstructed, vq_loss, all_indices

    def encode(self, actions, num_layers=None):
        """
        Encode actions to discrete tokens.

        Args:
            actions: [chunk_size, action_dim] numpy array
            num_layers: How many layers to use (default: all)

        Returns:
            tokens: List of [chunk_size] arrays, one per layer
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")

        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()

            # Add batch dim
            if len(actions.shape) == 2:
                actions = actions.unsqueeze(0)

            actions = actions.to(next(self.parameters()).device)

            # Normalize and encode
            actions_norm = self.normalize_actions(actions)
            encoded = self.encoder(actions_norm)

            # Get discrete codes
            all_indices = self.rvq.encode(encoded, num_layers)

            # Convert to numpy
            tokens = [indices[0].cpu().numpy() for indices in all_indices]

        return tokens

    def decode(self, tokens):
        """
        Decode tokens to actions.

        Args:
            tokens: List of [chunk_size] arrays (discrete codes)

        Returns:
            actions: [chunk_size, action_dim] numpy array
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding")

        self.eval()
        with torch.no_grad():
            # Convert tokens to tensor
            device = next(self.parameters()).device
            indices_list = [torch.from_numpy(t).long().unsqueeze(0).to(device) for t in tokens]

            # Decode from indices
            quantized = self.rvq.decode_indices(indices_list)

            # Decode to actions
            decoded_norm = self.decoder(quantized)
            reconstructed = self.denormalize_actions(decoded_norm)

            # Remove batch dim and convert to numpy
            actions = reconstructed[0].cpu().numpy()

        return actions

    def get_compression_ratio(self, num_layers=None):
        """
        Calculate compression ratio.

        Original: chunk_size × action_dim × 32 bits (float32)
        Compressed: chunk_size × num_layers × 8 bits (uint8 for 256 codebook)
        """
        if num_layers is None:
            num_layers = self.num_layers

        original_bits = self.chunk_size * self.action_dim * 32
        compressed_bits = self.chunk_size * num_layers * 8
        return original_bits / compressed_bits

    def __repr__(self):
        return (
            f"RVQTokenizer("
            f"action_dim={self.action_dim}, "
            f"chunk_size={self.chunk_size}, "
            f"num_layers={self.num_layers}, "
            f"hidden_dim={self.hidden_dim}, "
            f"codebook_size={self.num_embeddings}, "
            f"compression_ratio={self.get_compression_ratio():.2f}x)"
        )
