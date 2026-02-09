"""
Simple file-based experiment logger to replace Orchestra SDK.
Logs metrics to JSON and text files for easy tracking on SLURM.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Simple experiment logger that writes to JSON and text files."""

    def __init__(self, name: str, log_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment logger.

        Args:
            name: Experiment name
            log_dir: Directory to save logs
            config: Optional config dict to log
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / f"{name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        self.summary_file = self.exp_dir / "summary.json"
        self.log_file = self.exp_dir / "log.txt"

        # Initialize
        self.config = config or {}
        self.metrics_history = []
        self.start_time = datetime.now()

        # Log config
        self.log_text(f"=== Experiment: {name} ===")
        self.log_text(f"Started at: {self.start_time}")
        self.log_text(f"Config: {json.dumps(self.config, indent=2)}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics at a given step.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics
        }

        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Store in memory
        self.metrics_history.append(entry)

        # Log to text file
        if step is not None:
            msg = f"[Step {step}] " + " | ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                                   for k, v in metrics.items())
        else:
            msg = " | ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in metrics.items())
        self.log_text(msg)

    def log_text(self, message: str):
        """
        Log a text message.

        Args:
            message: Text to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(full_msg + '\n')

        # Also print to console
        print(full_msg)

    def finish(self, status: str = "completed"):
        """
        Finalize the experiment and save summary.

        Args:
            status: Final status (completed, failed, etc.)
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            "name": self.name,
            "status": status,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_steps": len(self.metrics_history),
            "metrics_file": str(self.metrics_file),
            "log_file": str(self.log_file)
        }

        # Compute final metrics (last value of each metric)
        if self.metrics_history:
            final_metrics = {}
            for key in self.metrics_history[-1].keys():
                if key not in ["timestamp", "step"]:
                    final_metrics[key] = self.metrics_history[-1][key]
            summary["final_metrics"] = final_metrics

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log_text(f"=== Experiment Finished ===")
        self.log_text(f"Status: {status}")
        self.log_text(f"Duration: {duration:.2f}s")
        self.log_text(f"Summary saved to: {self.summary_file}")

    def get_exp_dir(self) -> Path:
        """Get the experiment directory path."""
        return self.exp_dir
