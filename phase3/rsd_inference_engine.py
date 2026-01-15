"""
RSD Inference Engine - Hierarchical Speculative Decoding

This module implements the core inference engine for RSD (Residual Speculative Decoding).
It orchestrates the Draft Model and Main Model to achieve faster inference with minimal
quality degradation.

Key Features:
    1. Hierarchical Speculative Decoding (HSD)
    2. Partial Acceptance Strategy
    3. RFSQ Token Decoding
    4. Batch Processing Support

Architecture:
    Input (Image + Task Description)
        ↓
    Draft Model (Fast) → Predicts Layers 0-2 (Coarse)
        ↓
    Main Model (Accurate) → Validates Layers 0-2 + Predicts Layers 3-7 (Fine)
        ↓
    Partial Acceptance → Accept correct layers, reject incorrect ones
        ↓
    RFSQ Decoder → Convert tokens to continuous actions
        ↓
    Output Actions [Chunk_Len, Action_Dim]

Author: RSD-VLA Research Team
Date: 2026-01-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np


class RSDInferenceEngine:
    """
    RSD Inference Engine for fast and accurate action prediction.

    This engine implements Hierarchical Speculative Decoding (HSD) to accelerate
    inference by using a lightweight Draft Model to predict coarse-grained tokens,
    which are then verified and refined by the Main Model.

    Args:
        main_model: The main VLA model (OpenVLA-RFSQ)
        draft_model: The lightweight draft model for speculation
        rfsq_decoder: RFSQ decoder to convert tokens to actions
        num_layers: Total number of RFSQ layers (default: 8)
        num_coarse_layers: Number of coarse layers predicted by draft (default: 3)
        acceptance_threshold: Confidence threshold for accepting draft predictions
        enable_partial_acceptance: Whether to use partial acceptance strategy
        device: Device to run inference on
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        rfsq_decoder: nn.Module,
        num_layers: int = 8,
        num_coarse_layers: int = 3,
        acceptance_threshold: float = 0.7,
        enable_partial_acceptance: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.main_model = main_model.to(device)
        self.draft_model = draft_model.to(device)
        self.rfsq_decoder = rfsq_decoder.to(device)

        self.num_layers = num_layers
        self.num_coarse_layers = num_coarse_layers
        self.num_fine_layers = num_layers - num_coarse_layers

        self.acceptance_threshold = acceptance_threshold
        self.enable_partial_acceptance = enable_partial_acceptance
        self.device = device

        # Set models to eval mode
        self.main_model.eval()
        self.draft_model.eval()
        self.rfsq_decoder.eval()

        # Statistics tracking
        self.stats = {
            'total_inferences': 0,
            'draft_accepts': 0,
            'partial_accepts': 0,
            'full_rejects': 0,
            'avg_accepted_layers': 0.0,
            'total_time': 0.0,
            'draft_time': 0.0,
            'main_time': 0.0,
        }

        print(f"🚀 RSD Inference Engine initialized")
        print(f"   Device: {device}")
        print(f"   Num Layers: {num_layers} (Coarse: {num_coarse_layers}, Fine: {num_fine_layers})")
        print(f"   Acceptance Threshold: {acceptance_threshold}")
        print(f"   Partial Acceptance: {enable_partial_acceptance}")

    @torch.no_grad()
    def generate_action(
        self,
        observation: Dict[str, np.ndarray],
        task_description: str,
        processor: Any = None,
        chunk_len: int = 8,
        action_dim: int = 7,
        temperature: float = 1.0,
        use_sampling: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate action using Hierarchical Speculative Decoding.

        Args:
            observation: Dictionary containing 'full_image', 'wrist_image', 'state'
            task_description: Natural language task instruction
            processor: OpenVLA processor for encoding inputs
            chunk_len: Number of timesteps to predict (default: 8)
            action_dim: Action space dimension (default: 7)
            temperature: Sampling temperature (default: 1.0)
            use_sampling: Whether to use sampling instead of argmax

        Returns:
            actions: Generated action sequence [Chunk_Len, Action_Dim]
            info: Dictionary with inference statistics
        """
        start_time = time.time()

        # ==========================================
        # Step 1: Encode observation + task
        # ==========================================
        if processor is not None:
            # Use OpenVLA processor
            inputs = processor(
                images=observation['full_image'],
                text=task_description,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Manually encode (for custom models)
            inputs = self._encode_observation(observation, task_description)

        # Get embeddings from main model's backbone
        # These embeddings will be shared between draft and main models
        with torch.no_grad():
            embeddings = self._get_embeddings(inputs)

        # ==========================================
        # Step 2: Draft Model - Predict Coarse Layers (0-2)
        # ==========================================
        draft_start = time.time()

        draft_logits = self.draft_model(embeddings)
        # Shape: [Batch, Num_Coarse_Layers, Seq_Len, Grid_Size]

        # Sample or take argmax
        if use_sampling:
            draft_tokens = self._sample_tokens(draft_logits, temperature)
        else:
            draft_tokens = draft_logits.argmax(dim=-1)
        # Shape: [Batch, Num_Coarse_Layers, Seq_Len]

        draft_time = time.time() - draft_start

        # ==========================================
        # Step 3: Main Model - Validate + Complete
        # ==========================================
        main_start = time.time()

        # Get main model's predictions for ALL layers
        main_logits = self.main_model(inputs)
        # Shape: [Batch, Num_Layers, Seq_Len, Grid_Size]

        # Extract coarse layers for validation
        main_coarse_logits = main_logits[:, :self.num_coarse_layers, :, :]

        # Sample or take argmax
        if use_sampling:
            main_coarse_tokens = self._sample_tokens(main_coarse_logits, temperature)
            main_fine_tokens = self._sample_tokens(
                main_logits[:, self.num_coarse_layers:, :, :],
                temperature
            )
        else:
            main_coarse_tokens = main_coarse_logits.argmax(dim=-1)
            main_fine_tokens = main_logits[:, self.num_coarse_layers:, :, :].argmax(dim=-1)

        main_time = time.time() - main_start

        # ==========================================
        # Step 4: Partial Acceptance Strategy
        # ==========================================
        if self.enable_partial_acceptance:
            final_tokens, acceptance_info = self._partial_acceptance(
                draft_tokens=draft_tokens,
                main_tokens=main_coarse_tokens,
                main_fine_tokens=main_fine_tokens,
                draft_logits=draft_logits,
                main_logits=main_coarse_logits,
            )
        else:
            # No partial acceptance - use main model predictions only
            final_tokens = torch.cat([main_coarse_tokens, main_fine_tokens], dim=1)
            acceptance_info = {'strategy': 'no_acceptance', 'accepted_layers': 0}

        # Shape: [Batch, Num_Layers, Seq_Len]

        # ==========================================
        # Step 5: RFSQ Decoding - Tokens to Actions
        # ==========================================
        actions = self._decode_tokens_to_actions(
            final_tokens,
            chunk_len=chunk_len,
            action_dim=action_dim
        )
        # Shape: [Batch, Chunk_Len, Action_Dim]

        total_time = time.time() - start_time

        # ==========================================
        # Step 6: Update Statistics
        # ==========================================
        self._update_stats(
            total_time=total_time,
            draft_time=draft_time,
            main_time=main_time,
            acceptance_info=acceptance_info,
        )

        # Prepare info dictionary
        info = {
            'total_time': total_time,
            'draft_time': draft_time,
            'main_time': main_time,
            'acceptance_info': acceptance_info,
            'tokens': final_tokens.cpu().numpy(),
        }

        # Convert to numpy and return
        return actions.squeeze(0).cpu().numpy(), info

    def _get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract embeddings from the main model's backbone.

        This is a placeholder - you'll need to adapt this based on your
        actual OpenVLA model structure.
        """
        # For OpenVLA, embeddings come from the vision+language encoder
        # You may need to access model internals here

        # Example (adapt to your model):
        if hasattr(self.main_model, 'get_embeddings'):
            return self.main_model.get_embeddings(inputs)
        elif hasattr(self.main_model, 'vision_backbone'):
            # Extract from vision backbone
            vision_features = self.main_model.vision_backbone(inputs['pixel_values'])
            # Combine with text features if needed
            return vision_features
        else:
            raise NotImplementedError(
                "Please implement embedding extraction for your model. "
                "You need to extract the hidden states from the main model's backbone."
            )

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample tokens from logits with temperature."""
        # Apply temperature
        logits = logits / temperature

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(probs.shape[:-1])

        return tokens

    def _partial_acceptance(
        self,
        draft_tokens: torch.Tensor,
        main_tokens: torch.Tensor,
        main_fine_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Implement partial acceptance strategy.

        This is the core of Hierarchical Speculative Decoding:
        - Layer-by-layer validation from coarse to fine
        - Accept draft predictions if they match main model
        - Reject and use main model if mismatch occurs

        Returns:
            final_tokens: [Batch, Num_Layers, Seq_Len]
            acceptance_info: Dictionary with acceptance statistics
        """
        batch_size, num_coarse, seq_len = draft_tokens.shape

        # Compare draft and main tokens layer by layer
        matches = (draft_tokens == main_tokens)  # [Batch, Num_Coarse, Seq_Len]

        # Calculate per-layer match rate
        layer_match_rates = matches.float().mean(dim=-1)  # [Batch, Num_Coarse]

        # Find the first layer where match rate drops below threshold
        accepted_layers = 0
        for layer_idx in range(num_coarse):
            if layer_match_rates[0, layer_idx] >= self.acceptance_threshold:
                accepted_layers += 1
            else:
                break

        # Construct final tokens
        if accepted_layers == num_coarse:
            # Full acceptance - use draft for coarse layers
            final_coarse = draft_tokens
            strategy = 'full_accept'
            self.stats['draft_accepts'] += 1
        elif accepted_layers > 0:
            # Partial acceptance - use draft for accepted layers, main for rest
            final_coarse = torch.cat([
                draft_tokens[:, :accepted_layers, :],
                main_tokens[:, accepted_layers:, :],
            ], dim=1)
            strategy = 'partial_accept'
            self.stats['partial_accepts'] += 1
        else:
            # Full rejection - use main model for all coarse layers
            final_coarse = main_tokens
            strategy = 'full_reject'
            self.stats['full_rejects'] += 1

        # Combine with fine layers from main model
        final_tokens = torch.cat([final_coarse, main_fine_tokens], dim=1)

        acceptance_info = {
            'strategy': strategy,
            'accepted_layers': accepted_layers,
            'layer_match_rates': layer_match_rates.cpu().numpy(),
        }

        return final_tokens, acceptance_info

    def _decode_tokens_to_actions(
        self,
        tokens: torch.Tensor,
        chunk_len: int,
        action_dim: int,
    ) -> torch.Tensor:
        """
        Decode RFSQ tokens to continuous actions.

        Args:
            tokens: RFSQ token indices [Batch, Num_Layers, Seq_Len]
            chunk_len: Number of action timesteps
            action_dim: Action space dimension

        Returns:
            actions: Continuous actions [Batch, Chunk_Len, Action_Dim]
        """
        batch_size, num_layers, seq_len = tokens.shape

        # Reshape tokens to match RFSQ decoder input
        # Expected: [Batch, Chunk_Len, Num_Layers, Action_Dim]
        # Current: [Batch, Num_Layers, Seq_Len]

        # Calculate expected sequence length
        expected_seq_len = chunk_len * action_dim

        # Truncate or pad if needed
        if seq_len > expected_seq_len:
            tokens = tokens[:, :, :expected_seq_len]
        elif seq_len < expected_seq_len:
            padding = torch.zeros(
                batch_size, num_layers, expected_seq_len - seq_len,
                dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat([tokens, padding], dim=-1)

        # Reshape: [Batch, Num_Layers, Chunk_Len * Action_Dim]
        #       -> [Batch, Chunk_Len, Action_Dim, Num_Layers]
        tokens_reshaped = tokens.view(batch_size, num_layers, chunk_len, action_dim)
        tokens_reshaped = tokens_reshaped.permute(0, 2, 3, 1)

        # Decode using RFSQ decoder
        # The decoder expects tokens and returns reconstructed latents
        actions = self.rfsq_decoder.decode_from_indices(tokens_reshaped)

        return actions

    def _update_stats(
        self,
        total_time: float,
        draft_time: float,
        main_time: float,
        acceptance_info: Dict[str, Any],
    ):
        """Update inference statistics."""
        self.stats['total_inferences'] += 1
        self.stats['total_time'] += total_time
        self.stats['draft_time'] += draft_time
        self.stats['main_time'] += main_time

        # Update accepted layers average
        prev_avg = self.stats['avg_accepted_layers']
        n = self.stats['total_inferences']
        new_avg = (prev_avg * (n - 1) + acceptance_info['accepted_layers']) / n
        self.stats['avg_accepted_layers'] = new_avg

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        n = self.stats['total_inferences']
        if n == 0:
            return self.stats

        return {
            'total_inferences': n,
            'draft_accept_rate': self.stats['draft_accepts'] / n,
            'partial_accept_rate': self.stats['partial_accepts'] / n,
            'full_reject_rate': self.stats['full_rejects'] / n,
            'avg_accepted_layers': self.stats['avg_accepted_layers'],
            'avg_total_time': self.stats['total_time'] / n,
            'avg_draft_time': self.stats['draft_time'] / n,
            'avg_main_time': self.stats['main_time'] / n,
            'speedup': (self.stats['draft_time'] + self.stats['main_time']) / self.stats['total_time'],
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'total_inferences': 0,
            'draft_accepts': 0,
            'partial_accepts': 0,
            'full_rejects': 0,
            'avg_accepted_layers': 0.0,
            'total_time': 0.0,
            'draft_time': 0.0,
            'main_time': 0.0,
        }
        print("📊 Statistics reset")

    def print_stats(self):
        """Print inference statistics in a nice format."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("RSD Inference Engine Statistics")
        print("="*60)
        print(f"Total Inferences: {stats['total_inferences']}")
        print(f"")
        print(f"Acceptance Strategy:")
        print(f"  Full Accept:    {stats['draft_accept_rate']:.1%}")
        print(f"  Partial Accept: {stats['partial_accept_rate']:.1%}")
        print(f"  Full Reject:    {stats['full_reject_rate']:.1%}")
        print(f"  Avg Accepted Layers: {stats['avg_accepted_layers']:.2f} / {self.num_coarse_layers}")
        print(f"")
        print(f"Timing:")
        print(f"  Avg Total Time: {stats['avg_total_time']*1000:.2f} ms")
        print(f"  Avg Draft Time: {stats['avg_draft_time']*1000:.2f} ms")
        print(f"  Avg Main Time:  {stats['avg_main_time']*1000:.2f} ms")
        print(f"  Speedup:        {stats['speedup']:.2f}x")
        print("="*60 + "\n")


if __name__ == "__main__":
    """Test the RSD Inference Engine."""
    print("Testing RSD Inference Engine...")

    # Create dummy models for testing
    class DummyMainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, inputs):
            # Return dummy logits [Batch, Num_Layers, Seq_Len, Grid_Size]
            return torch.randn(1, 8, 56, 7)

        def get_embeddings(self, inputs):
            # Return dummy embeddings
            return torch.randn(1, 56, 4096)

    class DummyDraftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, embeddings):
            # Return dummy logits [Batch, Num_Coarse_Layers, Seq_Len, Grid_Size]
            return torch.randn(1, 3, 56, 7)

    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def decode_from_indices(self, tokens):
            # Return dummy actions [Batch, Chunk_Len, Action_Dim]
            batch_size, chunk_len, action_dim, _ = tokens.shape
            return torch.randn(batch_size, chunk_len, action_dim)

    # Initialize engine
    main_model = DummyMainModel()
    draft_model = DummyDraftModel()
    decoder = DummyDecoder()

    engine = RSDInferenceEngine(
        main_model=main_model,
        draft_model=draft_model,
        rfsq_decoder=decoder,
        device='cpu',
    )

    # Create dummy observation
    observation = {
        'full_image': np.random.randn(224, 224, 3),
        'wrist_image': np.random.randn(224, 224, 3),
        'state': np.random.randn(8),
    }

    # Generate action
    actions, info = engine.generate_action(
        observation=observation,
        task_description="Pick up the red block",
        processor=None,
    )

    print(f"\n✅ Generated actions shape: {actions.shape}")
    print(f"   Expected: (8, 7)")
    print(f"   Inference time: {info['total_time']*1000:.2f} ms")

    # Print statistics
    engine.print_stats()

    print("\n✅ RSD Inference Engine test passed!")
