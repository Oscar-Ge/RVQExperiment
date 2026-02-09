"""
Video recorder for LIBERO episodes.
Records RGB frames and saves as MP4 videos for failure analysis.
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
import imageio


class VideoRecorder:
    """Records episodes as MP4 videos."""

    def __init__(self, output_dir: str, fps: int = 20):
        """
        Initialize video recorder.

        Args:
            output_dir: Directory to save videos
            fps: Frames per second for output videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        # Current recording state
        self.frames: List[np.ndarray] = []
        self.recording = False
        self.current_name = None

    def start_recording(self, name: str):
        """
        Start recording a new episode.

        Args:
            name: Name identifier for this recording
        """
        self.frames = []
        self.recording = True
        self.current_name = name

    def add_frame(self, rgb_array: np.ndarray):
        """
        Add a frame to the current recording.

        Args:
            rgb_array: RGB image array (H, W, 3) with values 0-255
        """
        if not self.recording:
            raise RuntimeError("Cannot add frame: not currently recording")

        # Ensure uint8
        if rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

        self.frames.append(rgb_array)

    def save_video(self, filename: str, overwrite: bool = False) -> Optional[Path]:
        """
        Save the current recording as a video file.

        Args:
            filename: Output filename (should end with .mp4)
            overwrite: Whether to overwrite existing files

        Returns:
            Path to saved video, or None if no frames
        """
        if not self.frames:
            print(f"Warning: No frames to save for {filename}")
            return None

        # Add .mp4 extension if not present
        if not filename.endswith('.mp4'):
            filename = filename + '.mp4'

        output_path = self.output_dir / filename

        # Check if file exists
        if output_path.exists() and not overwrite:
            print(f"Warning: {output_path} already exists, skipping")
            return None

        # Save video
        try:
            imageio.mimsave(output_path, self.frames, fps=self.fps, codec='libx264')
            print(f"Saved video: {output_path} ({len(self.frames)} frames)")
        except Exception as e:
            print(f"Error saving video {output_path}: {e}")
            return None

        # Reset recording state
        self._reset()

        return output_path

    def discard(self):
        """Discard the current recording without saving."""
        if self.recording:
            print(f"Discarded recording: {self.current_name} ({len(self.frames)} frames)")
        self._reset()

    def _reset(self):
        """Reset internal state."""
        self.frames = []
        self.recording = False
        self.current_name = None

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording

    def get_frame_count(self) -> int:
        """Get number of frames in current recording."""
        return len(self.frames)
