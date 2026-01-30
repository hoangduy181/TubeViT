import os
import warnings
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torchvision.datasets import UCF101, Kinetics


def _ensure_uint8_video(video: Tensor) -> Tensor:
    """Convert video to uint8 if needed. ToTensorVideo() expects uint8 input."""
    if video.dtype == torch.uint8:
        return video
    if video.dtype in (torch.float32, torch.float64):
        # Float video: either [0, 1] or [0, 255]; clamp and convert
        if video.max() <= 1.0:
            video = (video * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            video = video.clamp(0, 255).to(torch.uint8)
        return video
    return video


def _ensure_thwc(video: Tensor) -> Tensor:
    """Ensure video is (T, H, W, C). Some backends return (H, W, T, C); convert so Resize doesn't touch channels."""
    if video.dim() != 4:
        return video
    # (H, W, T, C): last dim is 3 (C), third is small (T), first two are spatial
    if (video.shape[3] == 3 and video.shape[2] in (8, 16, 32, 64, 128) and
            video.shape[0] >= 100 and video.shape[1] >= 100):
        return video.permute(2, 0, 1, 3)  # (H, W, T, C) -> (T, H, W, C)
    return video

# Suppress pts_unit warning from torchvision video reading
warnings.filterwarnings('ignore', message=".*pts_unit.*", category=UserWarning)

# Fix PyAV compatibility issue: patch torchvision's video reading
# Newer PyAV versions use av.error.Error instead of av.AVError
try:
    from torchvision import io
    import av.error
    
    # Store original function
    _original_read_video_timestamps = io.read_video_timestamps
    
    def _patched_read_video_timestamps(filename, start_pts=0, end_pts=None, pts_unit='pts'):
        """Patched version that handles PyAV compatibility."""
        try:
            return _original_read_video_timestamps(filename, start_pts, end_pts, pts_unit)
        except AttributeError as e:
            # Handle case where av.AVError doesn't exist
            if "'av' has no attribute 'AVError'" in str(e) or "module 'av' has no attribute 'AVError'" in str(e):
                # The actual error was likely an InvalidDataError
                # Re-raise as RuntimeError so it can be caught by our error handling
                raise RuntimeError(f"Invalid video data in {filename}: moov atom not found or corrupted")
            raise
    
    # Apply patch
    io.read_video_timestamps = _patched_read_video_timestamps
except Exception:
    # If patching fails, continue without patch
    # Error handling in __getitem__ will catch corrupted videos
    pass


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        # Add logging to debug dataset initialization
        root = kwargs.get('root', args[0] if args else None)
        annotation_path = kwargs.get('annotation_path', args[1] if len(args) > 1 else None)
        train = kwargs.get('train', True)
        
        print(f"✨ [MyUCF101] Initializing dataset...")
        print(f"  Root: {root}")
        print(f"  Annotation path: {annotation_path}")
        print(f"  Train: {train}")
        
        # Check if annotation_path is a file or directory
        # Torchvision expects annotation_path to be a directory containing trainlist01.txt, etc.
        if annotation_path and os.path.exists(annotation_path):
            if os.path.isfile(annotation_path):
                # If it's a file, use its directory
                annotation_dir = os.path.dirname(annotation_path)
                print(f"  Annotation path is a file, using directory: {annotation_dir}")
                kwargs['annotation_path'] = annotation_dir
            elif os.path.isdir(annotation_path):
                # If it's already a directory, check for required files
                trainlist = os.path.join(annotation_path, 'trainlist01.txt')
                testlist = os.path.join(annotation_path, 'testlist01.txt')
                if os.path.exists(trainlist):
                    print(f"  Found trainlist01.txt in annotation directory")
                if os.path.exists(testlist):
                    print(f"  Found testlist01.txt in annotation directory")
        
        # Check if root contains train/val subdirectories
        if root and os.path.exists(root):
            train_dir = os.path.join(root, 'train')
            val_dir = os.path.join(root, 'val')
            
            if os.path.exists(train_dir) and os.path.exists(val_dir):
                print(f"  Found train/val subdirectories in root")
                # Adjust root to point to train or val directory
                if train:
                    new_root = train_dir
                    print(f"  Using train directory: {new_root}")
                else:
                    new_root = val_dir
                    print(f"  Using val directory: {new_root}")
                
                # Check if videos exist in the adjusted root
                class_dirs = [d for d in os.listdir(new_root) if os.path.isdir(os.path.join(new_root, d))]
                avi_files = []
                for class_dir in class_dirs[:3]:  # Check first 3 classes
                    class_path = os.path.join(new_root, class_dir)
                    if os.path.exists(class_path):
                        avis = [f for f in os.listdir(class_path) if f.endswith('.avi')]
                        if avis:
                            avi_files.extend(avis[:2])  # Get first 2 videos
                
                print(f"  Found {len(class_dirs)} class directories")
                print(f"  Sample videos found: {len(avi_files)} (showing first few)")
                if avi_files and class_dirs:
                    print(f"    Example: {class_dirs[0]}/{avi_files[0]}")
                
                # Update root in kwargs
                kwargs['root'] = new_root
        
        try:
            super().__init__(*args, **kwargs)
            print(f"  ✓ Dataset initialized successfully")
            print(f"  Total samples: {len(self.samples) if hasattr(self, 'samples') else 'unknown'}")
            
            # Note: pts_unit warning is suppressed at module level
            # The VideoClips object is created internally by torchvision
            # and we cannot directly modify its pts_unit parameter
        except Exception as e:
            print(f"  ✗ Error initializing dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get video clip with error handling for corrupted videos.
        
        If a video is corrupted, tries to get the next valid video.
        This handles cases where videos have missing moov atoms or other corruption issues.
        """
        max_retries = 10  # Maximum number of retries to find a valid video
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                label = self.samples[self.indices[video_idx]][1]
                
                # Check if video is valid (has frames)
                if video is None or video.numel() == 0:
                    raise ValueError(f"Empty video at index {idx}")
                
                video = _ensure_uint8_video(video)
                video = _ensure_thwc(video)
                if self.transform is not None:
                    video = self.transform(video)
                
                return video, label
                
            except Exception as e:
                # Handle various video reading errors
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    'moov atom not found',
                    'invalid data',
                    'av.error',
                    'attributeerror',
                    'corrupted',
                    'invalid file',
                ]):
                    # Corrupted video - try next index
                    retry_count += 1
                    idx = (idx + 1) % len(self)
                    if retry_count < max_retries:
                        continue
                    else:
                        # If we've tried many times, return a dummy video or raise
                        print(f"Warning: Could not find valid video after {max_retries} retries. "
                              f"Last error: {e}")
                        raise RuntimeError(
                            f"Failed to load video after {max_retries} retries. "
                            f"Dataset may have too many corrupted videos. "
                            f"Last error: {e}"
                        )
                else:
                    # Re-raise if it's a different error
                    raise


class MyKinetics(Kinetics):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        # Add logging to debug dataset initialization
        root = kwargs.get('root', args[0] if args else None)
        split = kwargs.get('split', 'train')
        num_classes = kwargs.get('num_classes', '400')
        
        print(f"✨ [MyKinetics] Initializing dataset...")
        print(f"  Root: {root}")
        print(f"  Split: {split}")
        print(f"  Num classes: {num_classes}")
        
        # Ensure num_classes is a string (required by torchvision)
        if isinstance(num_classes, int):
            num_classes = str(num_classes)
        kwargs['num_classes'] = num_classes
        
        # Check if root contains train/val subdirectories
        if root and os.path.exists(root):
            split_dir = os.path.join(root, split)
            if os.path.exists(split_dir):
                print(f"  Found {split} subdirectory in root")
                # Check if class directories exist
                class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                video_files = []
                for class_dir in class_dirs[:3]:  # Check first 3 classes
                    class_path = os.path.join(split_dir, class_dir)
                    if os.path.exists(class_path):
                        videos = [f for f in os.listdir(class_path) 
                                 if f.endswith(('.mp4', '.avi', '.webm'))]
                        if videos:
                            video_files.extend(videos[:2])  # Get first 2 videos
                
                print(f"  Found {len(class_dirs)} class directories")
                print(f"  Sample videos found: {len(video_files)} (showing first few)")
                if video_files and class_dirs:
                    print(f"    Example: {split}/{class_dirs[0]}/{video_files[0]}")
        
        try:
            super().__init__(*args, **kwargs)
            print(f"  ✓ Dataset initialized successfully")
            print(f"  Total samples: {len(self.video_clips) if hasattr(self, 'video_clips') else 'unknown'}")
            if hasattr(self, 'classes'):
                print(f"  Number of classes: {len(self.classes)}")
        except Exception as e:
            print(f"  ✗ Error initializing dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Get video clip with error handling for corrupted videos.
        
        If a video is corrupted, tries to get the next valid video.
        This handles cases where videos have missing moov atoms or other corruption issues.
        """
        max_retries = 10  # Maximum number of retries to find a valid video
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                video, audio, label = super().__getitem__(idx)
                
                # Check if video is valid (has frames)
                if video is None or video.numel() == 0:
                    raise ValueError(f"Empty video at index {idx}")
                
                video = _ensure_uint8_video(video)
                video = _ensure_thwc(video)
                if self.transform is not None:
                    video = self.transform(video)
                
                return video, label
                
            except Exception as e:
                # Handle various video reading errors
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    'moov atom not found',
                    'invalid data',
                    'av.error',
                    'attributeerror',
                    'corrupted',
                    'invalid file',
                ]):
                    # Corrupted video - try next index
                    retry_count += 1
                    idx = (idx + 1) % len(self)
                    if retry_count < max_retries:
                        continue
                    else:
                        # If we've tried many times, return a dummy video or raise
                        print(f"Warning: Could not find valid video after {max_retries} retries. "
                              f"Last error: {e}")
                        raise RuntimeError(
                            f"Failed to load video after {max_retries} retries. "
                            f"Dataset may have too many corrupted videos. "
                            f"Last error: {e}"
                        )
                else:
                    # Re-raise if it's a different error
                    raise


def get_dataset(dataset_name: str, **kwargs) -> Union[MyUCF101, MyKinetics]:
    """
    Factory function to get dataset instance.
    
    Args:
        dataset_name: 'ucf101', 'kinetics400', 'kinetics600', 'kinetics700', 
                     'k400', 'k600', 'k700', 'kinetics-400', etc.
        **kwargs: Dataset-specific parameters
        
    Returns:
        Dataset instance (MyUCF101 or MyKinetics)
        
    Examples:
        >>> # UCF101
        >>> dataset = get_dataset('ucf101', root='...', annotation_path='...', train=False)
        
        >>> # Kinetics-400
        >>> dataset = get_dataset('kinetics400', root='...', split='val', frames_per_clip=32)
    """
    dataset_name = dataset_name.lower().replace('-', '').replace('_', '')
    
    if dataset_name == 'ucf101':
        return MyUCF101(**kwargs)
    elif dataset_name in ['kinetics400', 'k400']:
        # num_classes must be string for Kinetics!
        kwargs['num_classes'] = '400'
        return MyKinetics(**kwargs)
    elif dataset_name in ['kinetics600', 'k600']:
        kwargs['num_classes'] = '600'
        return MyKinetics(**kwargs)
    elif dataset_name in ['kinetics700', 'k700']:
        kwargs['num_classes'] = '700'
        return MyKinetics(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: 'ucf101', 'kinetics400'/'k400', 'kinetics600'/'k600', 'kinetics700'/'k700'"
        )
