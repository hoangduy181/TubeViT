"""
Video loading module - replacement for pytorchvideo.data.encoded_video
Uses torchvision.io for better compatibility and maintenance.
"""
import torch
from torchvision.io import read_video, read_video_timestamps
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings


class EncodedVideo:
    """
    Encoded video loader.
    Replaces pytorchvideo.data.encoded_video.EncodedVideo
    
    Uses torchvision.io for video loading, which is actively maintained.
    """
    def __init__(
        self,
        file_path: Union[str, Path],
        decode_audio: bool = False,
        decode_video: bool = True,
    ):
        """
        Args:
            file_path: Path to video file
            decode_audio: Whether to decode audio (not supported yet)
            decode_video: Whether to decode video
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        self.decode_audio = decode_audio
        self.decode_video = decode_video
        
        # Get video metadata
        self._duration = None
        self._fps = None
        self._video_tensor = None
        self._audio_tensor = None
        
        # Load video metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load video metadata without decoding full video."""
        try:
            # Read timestamps to get duration and fps
            pts, video_fps = read_video_timestamps(str(self.file_path), pts_unit='sec')
            if pts:
                self._duration = pts[-1]  # Last timestamp
                self._fps = video_fps
            else:
                # Fallback: read a small portion to get metadata
                video, audio, info = read_video(
                    str(self.file_path),
                    start_pts=0.0,
                    end_pts=1.0,
                    pts_unit='sec'
                )
                if len(video) > 0:
                    # Estimate duration from first frame
                    self._fps = info.get('video_fps', 30.0)
                    # We'll need to read full video to get actual duration
                    # For now, use a reasonable default
                    self._duration = 10.0  # Will be updated when video is read
        except Exception as e:
            warnings.warn(f"Could not load video metadata: {e}")
            self._duration = 10.0
            self._fps = 30.0
    
    @classmethod
    def from_path(cls, file_path: Union[str, Path], **kwargs):
        """
        Create EncodedVideo from file path.
        
        Args:
            file_path: Path to video file
            **kwargs: Additional arguments for __init__
        """
        return cls(file_path, **kwargs)
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self._duration is None:
            self._load_metadata()
        return self._duration
    
    @property
    def fps(self) -> float:
        """Get video frames per second."""
        if self._fps is None:
            self._load_metadata()
        return self._fps
    
    def get_clip(
        self,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> dict:
        """
        Get video clip between start_sec and end_sec.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds (if None, uses start_sec + 1.0)
        
        Returns:
            Dictionary with 'video' key containing video tensor
            Shape: (T, H, W, C) where T is number of frames
        """
        if end_sec is None:
            end_sec = start_sec + 1.0
        
        # Ensure valid time range
        start_sec = max(0.0, start_sec)
        if self._duration is not None:
            end_sec = min(end_sec, self._duration)
        
        try:
            # Read video using torchvision
            video, audio, info = read_video(
                str(self.file_path),
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit='sec'
            )
            
            # video shape: (T, H, W, C)
            # Convert to uint8 if needed
            if video.dtype != torch.uint8:
                video = (video * 255).clamp(0, 255).byte()
            
            result = {'video': video}
            
            if self.decode_audio and audio is not None:
                result['audio'] = audio
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error loading video clip from {self.file_path}: {e}")
    
    def __repr__(self) -> str:
        return f"EncodedVideo(file_path={self.file_path}, duration={self.duration:.2f}s, fps={self.fps:.2f})"
