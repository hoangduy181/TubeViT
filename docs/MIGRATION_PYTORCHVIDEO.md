# Migration from pytorchvideo to Internal Modules

## Overview

This project has migrated from `pytorchvideo` (which is outdated and not actively maintained) to internal modules built on top of `torchvision` and standard PyTorch. This ensures better compatibility, maintenance, and future-proofing.

## What Changed

### Before (pytorchvideo)
```python
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, ShortSideScale
```

### After (Internal Modules)
```python
from tubevit.transforms import Normalize, Permute, RandAugment
from tubevit.video_loader import EncodedVideo
from tubevit.transforms import ApplyTransformToKey, UniformTemporalSubsample, ShortSideScale
```

## New Module Structure

### `tubevit/transforms.py`
Contains video transform classes:
- **Normalize**: Normalize video tensors with mean/std
- **Permute**: Permute tensor dimensions
- **RandAugment**: Random augmentation (placeholder for future implementation)
- **UniformTemporalSubsample**: Uniformly subsample frames
- **ShortSideScale**: Scale video short side to target size
- **ApplyTransformToKey**: Apply transform to dictionary key

### `tubevit/video_loader.py`
Contains video loading utilities:
- **EncodedVideo**: Load and decode video files using `torchvision.io`
  - Uses `torchvision.io.read_video()` for better compatibility
  - Supports `get_clip()` method for extracting video segments
  - Provides `duration` and `fps` properties

## Benefits

1. **Better Maintenance**: Uses actively maintained `torchvision` instead of outdated `pytorchvideo`
2. **Compatibility**: Works with latest PyTorch and torchvision versions
3. **Customization**: Easy to extend and modify for project-specific needs
4. **No External Dependency**: Removes dependency on unmaintained package
5. **Future-Proof**: Can be updated as PyTorch ecosystem evolves

## Migration Status

✅ **Completed:**
- `scripts/train.py` - Updated to use internal transforms
- `scripts/inference.py` - Updated to use internal video loader and transforms
- `scripts/evaluate.py` - Updated to use internal transforms
- `scripts/visualise_dataset.py` - Updated to use internal transforms

## Usage Examples

### Video Loading
```python
from tubevit.video_loader import EncodedVideo

# Load video
video = EncodedVideo.from_path("path/to/video.mp4")
print(f"Duration: {video.duration}s, FPS: {video.fps}")

# Get clip
clip = video.get_clip(start_sec=0.0, end_sec=2.0)
video_tensor = clip['video']  # Shape: (T, H, W, C)
```

### Video Transforms
```python
from tubevit.transforms import Normalize, Permute, UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose

transform = Compose([
    UniformTemporalSubsample(num_samples=32),
    ShortSideScale(size=224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Notes

- **RandAugment**: Currently a placeholder (returns identity). Can be extended with actual augmentation operations if needed.
- **Video Format**: Uses `torchvision.io` which supports common video formats (MP4, AVI, etc.)
- **Tensor Shapes**: Maintains compatibility with existing code expectations

## Removing pytorchvideo

The `pytorchvideo` package is no longer required and has been removed from `requirements.txt`. If you have it installed, you can safely uninstall it:

```bash
pip uninstall pytorchvideo
```

## Future Enhancements

Potential improvements that can be added:
- Additional video transforms (temporal cropping, frame interpolation, etc.)
- Better error handling and video format support
- GPU-accelerated video processing
