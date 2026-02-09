"""
Checkpoint manager for saving and loading model checkpoints.
Keeps N best checkpoints and auto-cleans old ones.
"""
import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup."""

    def __init__(self, checkpoint_dir: str, keep_n_best: int = 3, metric_name: str = "loss", mode: str = "min"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_best: Number of best checkpoints to keep
            metric_name: Metric name to track for best checkpoints
            mode: "min" or "max" - whether lower or higher metric is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_best = keep_n_best
        self.metric_name = metric_name
        self.mode = mode

        # Track saved checkpoints: [(path, metric_value), ...]
        self.checkpoints: List[Tuple[Path, float]] = []
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"

        # Load existing metadata if available
        self._load_metadata()

    def _load_metadata(self):
        """Load checkpoint metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.checkpoints = [(Path(p), v) for p, v in metadata.get("checkpoints", [])]

    def _save_metadata(self):
        """Save checkpoint metadata to file."""
        metadata = {
            "checkpoints": [(str(p), v) for p, v in self.checkpoints],
            "keep_n_best": self.keep_n_best,
            "metric_name": self.metric_name,
            "mode": self.mode
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            extra_state: Optional extra state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        # Create filename
        metric_value = metrics.get(self.metric_name, float('inf') if self.mode == "min" else float('-inf'))
        filename = f"checkpoint_epoch{epoch:04d}_step{step:06d}_{self.metric_name}{metric_value:.6f}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Track this checkpoint
        self.checkpoints.append((checkpoint_path, metric_value))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Save metadata
        self._save_metadata()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove checkpoints beyond keep_n_best."""
        if len(self.checkpoints) <= self.keep_n_best:
            return

        # Sort by metric (best first)
        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x[1], reverse=reverse)

        # Remove worst checkpoints
        to_remove = self.checkpoints[self.keep_n_best:]
        self.checkpoints = self.checkpoints[:self.keep_n_best]

        for checkpoint_path, _ in to_remove:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Removed old checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint dictionary with epoch, step, metrics, etc.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
        print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load the best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint dictionary
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints found")

        # Best checkpoint is first after sorting
        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x[1], reverse=reverse)
        best_path, best_value = self.checkpoints[0]

        print(f"Loading best checkpoint: {best_path} ({self.metric_name}={best_value:.6f})")
        return self.load_checkpoint(str(best_path), model, optimizer)

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint without loading it."""
        if not self.checkpoints:
            return None

        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x[1], reverse=reverse)
        return self.checkpoints[0][0]

    def list_checkpoints(self) -> List[Tuple[Path, float]]:
        """Get list of all checkpoints sorted by metric."""
        reverse = (self.mode == "max")
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[1], reverse=reverse)
        return sorted_checkpoints
