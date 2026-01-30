"""
Video transforms module - replacement for pytorchvideo.transforms
Uses torchvision and standard PyTorch for better compatibility and maintenance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional, Callable, Any

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Normalize(nn.Module):
    """
    Normalize video tensor with mean and std.
    Replaces pytorchvideo.transforms.Normalize
    
    Args:
        mean: Mean values for each channel (e.g., [0.485, 0.456, 0.406])
        std: Std values for each channel (e.g., [0.229, 0.224, 0.225])
    """
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(3, 1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (T, C, H, W), (C, T, H, W), (T, H, W, C), (H, W, C, T), or (B, C, T, H, W)
        Returns:
            Normalized tensor (same layout as input)
        """
        if x.dim() == 4:
            # Find which dimension has C=3 and broadcast mean/std there
            for c_dim in range(4):
                if x.shape[c_dim] == 3:
                    # Broadcast (3, 1, 1, 1) to the right place: insert 1s for other dims
                    view = [1] * 4
                    view[c_dim] = 3
                    mean = self.mean.view(view)
                    std = self.std.view(view)
                    return (x - mean) / std
            # Fallback: (H, W, T, C) was resized and C=3 was lost -> (H, W, T, H). Not fixable here.
            raise ValueError(
                f"Normalize: no dimension has size 3 for shape {x.shape}. "
                "Ensure video is (T, H, W, C) before ToTensorVideo/Resize (dataset should use _ensure_thwc)."
            )
        elif x.dim() == 5:
            # (B, C, T, H, W) - channel at dim 1
            mean = self.mean.unsqueeze(0)  # (1, 3, 1, 1, 1)
            std = self.std.unsqueeze(0)
            return (x - mean) / std
        else:
            raise ValueError(f"Normalize: expected 4D or 5D tensor, got {x.dim()}D")


class Permute(nn.Module):
    """
    Permute tensor dimensions.
    Replaces pytorchvideo.transforms.Permute
    
    Args:
        dims: Tuple of dimension indices to permute to
    """
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class RandAugment(nn.Module):
    """
    Random augmentation for video.
    Implementation based on RandAugment paper: https://arxiv.org/abs/1909.13719
    Compatible with pytorchvideo.transforms.RandAugment API.
    Applies random augmentations consistently across all frames in a video.
    
    Args:
        magnitude: Magnitude of augmentations (0-10, default: 9)
        num_layers: Number of augmentation operations to apply (default: 2)
        prob: Probability of applying each transform (default: 0.5)
        sampling_type: Sampling method for magnitude ('gaussian' or 'uniform', default: 'gaussian')
        sampling_std: Standard deviation for gaussian sampling (default: 0.5)
    """
    # Augmentation operations matching pytorchvideo (excluding identity)
    AUGMENT_OPS = [
        'autocontrast',
        'equalize',
        'invert',
        'rotate',
        'posterize',
        'solarize',
        'color',
        'contrast',
        'brightness',
        'sharpness',
        'shear_x',
        'shear_y',
        'translate_x',
        'translate_y',
    ]
    
    # Max parameters for each transform (matching pytorchvideo exactly)
    # Format: (min, max) where magnitude interpolates from min (magnitude=0) to max (magnitude=10)
    # or None for operations without magnitude
    TRANSFORM_MAX_PARAMS = {
        'autocontrast': None,
        'equalize': None,
        'invert': None,
        'rotate': (0, 30),  # degrees: 0° (no rotation) to 30° (max rotation)
        'posterize': (4, 4),  # bits: always 4 bits (special case in pytorchvideo)
        'solarize': (1, 1),  # threshold: always 1.0 (special case in pytorchvideo)
        'color': (1, 0.9),  # saturation: 1.0 (no change) to 0.9 (desaturated)
        'contrast': (1, 0.9),  # contrast: 1.0 (no change) to 0.9 (low contrast)
        'brightness': (1, 0.9),  # brightness: 1.0 (no change) to 0.9 (darker)
        'sharpness': (1, 0.9),  # sharpness: 1.0 (no change) to 0.9 (less sharp)
        'shear_x': (0, 0.3),  # shear: 0 (no shear) to 0.3 (max shear)
        'shear_y': (0, 0.3),
        'translate_x': (0, 0.45),  # translate: 0 (no translate) to 0.45 (max translate)
        'translate_y': (0, 0.45),
    }
    
    def __init__(
        self,
        magnitude: int = 9,
        num_layers: int = 2,
        prob: float = 0.5,
        sampling_type: str = 'gaussian',
        sampling_std: float = 0.5,
    ):
        super().__init__()
        self.magnitude = magnitude
        self.num_layers = num_layers
        self.prob = prob
        assert sampling_type in ['gaussian', 'uniform'], "sampling_type must be 'gaussian' or 'uniform'"
        self.sampling_type = sampling_type
        self.sampling_std = sampling_std
    
    def _sample_magnitude(self, max_params: Tuple[float, float] = None) -> float:
        """
        Sample magnitude value based on sampling_type.
        
        Args:
            max_params: (min, max) tuple for the transform, or None
        Returns:
            Sampled magnitude value (0.0-1.0 normalized)
        """
        if max_params is None:
            return 0.0  # Operations without magnitude
        
        # Normalize magnitude (0-10) to (0.0-1.0)
        base_magnitude = self.magnitude / 10.0
        
        if self.sampling_type == 'gaussian':
            # Sample from gaussian distribution
            sampled = torch.normal(mean=base_magnitude, std=self.sampling_std)
            sampled = torch.clamp(sampled, 0.0, 1.0).item()
        else:  # uniform
            # Sample uniformly around base magnitude
            min_val = max(0.0, base_magnitude - self.sampling_std)
            max_val = min(1.0, base_magnitude + self.sampling_std)
            sampled = torch.rand(1).item() * (max_val - min_val) + min_val
        
        return sampled
    
    def _apply_op(self, x: torch.Tensor, op_name: str, magnitude: float) -> torch.Tensor:
        """
        Apply a single augmentation operation.
        
        Args:
            x: Video tensor of shape (T, C, H, W)
            op_name: Name of the operation to apply
            magnitude: Magnitude parameter (0.0-1.0 normalized)
        Returns:
            Augmented tensor with same shape
        """
        # Ensure input is in [0, 1] range for most operations
        if x.max() > 1.0:
            x = x.clamp(0.0, 1.0)
        
        # Get max parameters for this operation
        max_params = self.TRANSFORM_MAX_PARAMS.get(op_name)
        
        # Convert normalized magnitude to actual parameter value
        if max_params is None:
            # Operations without magnitude (autocontrast, equalize, invert)
            actual_magnitude = 0.0
        else:
            min_val, max_val = max_params
            # Interpolate from normalized magnitude (0.0-1.0) to actual range
            actual_magnitude = min_val + magnitude * (max_val - min_val)
        
        # Apply operation based on name
        op_map = {
            'autocontrast': self._autocontrast,
            'equalize': self._equalize,
            'invert': self._invert,
            'rotate': lambda x, m: self._rotate(x, m),
            'posterize': lambda x, m: self._posterize(x, m),
            'solarize': lambda x, m: self._solarize(x, m),
            'color': lambda x, m: self._color(x, m),
            'contrast': lambda x, m: self._contrast(x, m),
            'brightness': lambda x, m: self._brightness(x, m),
            'sharpness': lambda x, m: self._sharpness(x, m),
            'shear_x': lambda x, m: self._shear_x(x, m),
            'shear_y': lambda x, m: self._shear_y(x, m),
            'translate_x': lambda x, m: self._translate_x(x, m),
            'translate_y': lambda x, m: self._translate_y(x, m),
        }
        
        if op_name in op_map:
            return op_map[op_name](x, actual_magnitude)
        else:
            return x
    
    def _autocontrast(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """
        Autocontrast operation - normalize each channel to [0, 1] based on min/max.
        Args:
            x: Video tensor of shape (T, C, H, W)
            magnitude: Magnitude parameter (0.0-1.0), not used but required for signature
        """
        if x.dim() == 4 and x.shape[1] == 3:  # (T, C, H, W) with C=3
            T, C, H, W = x.shape
            for c in range(C):
                channel = x[:, c, :, :]  # (T, H, W)
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    x[:, c, :, :] = (channel - min_val) / (max_val - min_val)
        return x
    
    def _equalize(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Histogram equalization (simplified)."""
        # Simplified version - full histogram equalization is complex
        # For now, use a contrast enhancement approximation
        return self._autocontrast(x, magnitude)
    
    def _invert(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Invert pixel values."""
        return 1.0 - x
    
    def _rotate(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Rotate by angle in degrees (0 to max_angle)."""
        angle = magnitude  # magnitude is already in degrees (0-30)
        if abs(angle) < 0.1:
            return x
        
        # Reshape for rotation: need (N, C, H, W) for F.affine_grid
        original_shape = x.shape
        if x.dim() == 4:
            if x.shape[0] == 3:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
            # Now x is (T, C, H, W)
            T, C, H, W = x.shape
            x = x.contiguous().view(T * C, 1, H, W)  # (T*C, 1, H, W) for rotation
            
            # Create rotation matrix
            angle_rad = angle * 3.14159 / 180.0
            cos_a = torch.cos(torch.tensor(angle_rad))
            sin_a = torch.sin(torch.tensor(angle_rad))
            
            # Rotation matrix
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], dtype=x.dtype, device=x.device).unsqueeze(0)
            
            # Apply rotation to each frame
            grid = F.affine_grid(theta.expand(T * C, -1, -1), x.shape, align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            
            x = x.view(T, C, H, W)
            if original_shape[0] == 3:
                x = x.permute(1, 0, 2, 3)  # Back to (C, T, H, W)
        
        return x
    
    def _posterize(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Reduce number of bits per pixel."""
        # magnitude is always 4 in pytorchvideo (special case)
        # But we'll use it to interpolate from 8 bits (no change) to 4 bits (max change)
        # Convert magnitude (4) to bit value: when magnitude=4, use 4 bits
        bits = int(magnitude)
        bits = max(1, min(8, bits))
        x = (x * 255.0).int()
        shift = 8 - bits
        x = (x >> shift) << shift
        return (x.float() / 255.0).clamp(0.0, 1.0)
    
    def _solarize(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Invert pixels above threshold."""
        # magnitude is always 1.0 in pytorchvideo (special case)
        # But we'll use it as threshold: invert pixels above this value
        threshold = magnitude
        return torch.where(x > threshold, 1.0 - x, x)
    
    def _color(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust color saturation."""
        # magnitude is the factor (1.0 = no change, 0.1 = desaturated)
        # Convert to grayscale and blend
        gray = x.mean(dim=-3, keepdim=True) if x.shape[-3] == 3 else x.mean(dim=1, keepdim=True)
        factor = magnitude
        if x.shape[-3] == 3:  # (C, T, H, W) or (T, C, H, W) with C=3
            if x.shape[0] == 3:
                x = x * factor + gray * (1 - factor)
            else:
                x = x * factor + gray * (1 - factor)
        return x.clamp(0.0, 1.0)
    
    def _contrast(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust contrast."""
        # magnitude is the factor (1.0 = no change, 0.1 = low contrast)
        mean = x.mean()
        factor = magnitude
        return ((x - mean) * factor + mean).clamp(0.0, 1.0)
    
    def _brightness(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust brightness."""
        # magnitude is the factor (1.0 = no change, 0.1 = dark)
        factor = magnitude
        return (x * factor).clamp(0.0, 1.0)
    
    def _sharpness(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust sharpness (simplified - uses blur approximation)."""
        # Simplified: blend with slightly blurred version
        if x.dim() == 4:
            # Apply simple blur kernel
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device) / 16.0
            kernel = kernel.view(1, 1, 3, 3)
            
            original_shape = x.shape
            if x.shape[0] == 3:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
            
            T, C, H, W = x.shape
            x_flat = x.contiguous().view(T * C, 1, H, W)
            blurred = F.conv2d(x_flat, kernel.expand(1, -1, -1, -1), padding=1)
            blurred = blurred.view(T, C, H, W)
            
            if original_shape[0] == 3:
                blurred = blurred.permute(1, 0, 2, 3)
                x = x.permute(1, 0, 2, 3)
            
            factor = magnitude  # Already in correct range (1.0 = no change, 0.1 = blurry)
            x = x * factor + blurred * (1 - factor)
        
        return x.clamp(0.0, 1.0)
    
    def _shear_x(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Shear along X axis (0 to max_shear)."""
        # magnitude is already in range (0 to 0.3)
        shear = magnitude
        return self._apply_affine(x, [[1, shear, 0], [0, 1, 0]])
    
    def _shear_y(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Shear along Y axis (0 to max_shear)."""
        # magnitude is already in range (0 to 0.3)
        shear = magnitude
        return self._apply_affine(x, [[1, 0, 0], [shear, 1, 0]])
    
    def _translate_x(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Translate along X axis (0 to max_translate)."""
        # magnitude is already in range (0 to 0.45)
        translate = magnitude
        pixels = int(translate * x.shape[-1])
        return self._apply_affine(x, [[1, 0, pixels], [0, 1, 0]])
    
    def _translate_y(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Translate along Y axis (0 to max_translate)."""
        # magnitude is already in range (0 to 0.45)
        translate = magnitude
        pixels = int(translate * x.shape[-2])
        return self._apply_affine(x, [[1, 0, 0], [0, 1, pixels]])
    
    def _apply_affine(self, x: torch.Tensor, matrix: List[List[float]]) -> torch.Tensor:
        """Apply affine transformation."""
        if x.dim() == 4:
            original_shape = x.shape
            if x.shape[0] == 3:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
            
            T, C, H, W = x.shape
            x = x.contiguous().view(T * C, 1, H, W)
            
            theta = torch.tensor([matrix], dtype=x.dtype, device=x.device)
            theta = theta.expand(T * C, -1, -1)
            
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            
            x = x.view(T, C, H, W)
            if original_shape[0] == 3:
                x = x.permute(1, 0, 2, 3)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to video.
        
        Args:
            x: Video tensor of shape (T, C, H, W) - expected from Permute transform
        Returns:
            Augmented tensor with same shape
        """
        if not self.training:
            return x
        
        # Randomly select operations (without replacement)
        num_ops_to_apply = min(self.num_layers, len(self.AUGMENT_OPS))
        if num_ops_to_apply == 0:
            return x
        
        # Get random permutation and select first num_layers operations
        ops_indices = torch.randperm(len(self.AUGMENT_OPS), device=x.device)[:num_ops_to_apply]
        
        # Apply each operation sequentially with probability check
        for op_idx in ops_indices:
            op_name = self.AUGMENT_OPS[op_idx]
            
            # Apply with probability prob (matching pytorchvideo behavior)
            if torch.rand(1, device=x.device).item() > self.prob:
                continue
            
            # Sample magnitude for this operation
            max_params = self.TRANSFORM_MAX_PARAMS.get(op_name)
            sampled_magnitude = self._sample_magnitude(max_params)
            
            x = self._apply_op(x, op_name, sampled_magnitude)
        
        return x


class UniformTemporalSubsample(nn.Module):
    """
    Uniformly subsample frames from video.
    Replaces pytorchvideo.transforms.UniformTemporalSubsample
    
    Args:
        num_samples: Number of frames to sample
    """
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (C, T, H, W) or (T, H, W, C)
        Returns:
            Subsampled tensor with num_samples frames
        """
        if x.dim() == 4:
            if x.shape[0] == 3:  # (C, T, H, W)
                T = x.shape[1]
                indices = torch.linspace(0, T - 1, self.num_samples).long()
                return x[:, indices]
            else:  # (T, H, W, C)
                T = x.shape[0]
                indices = torch.linspace(0, T - 1, self.num_samples).long()
                return x[indices]
        else:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ShortSideScale(nn.Module):
    """
    Scale video so the short side has the given size.
    Replaces pytorchvideo.transforms.ShortSideScale
    
    Args:
        size: Target size for the short side
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (C, T, H, W) or (T, H, W, C)
        Returns:
            Scaled tensor
        """
        if x.dim() == 4:
            if x.shape[0] == 3:  # (C, T, H, W)
                C, T, H, W = x.shape
                # Reshape to (T, C, H, W) for processing
                x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
                # Get short side
                short_side = min(H, W)
                scale = self.size / short_side
                new_H, new_W = int(H * scale), int(W * scale)
                # Resize each frame
                x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
                # Reshape back
                x = x.permute(1, 0, 2, 3)  # (C, T, H, W)
                return x
            else:  # (T, H, W, C)
                T, H, W, C = x.shape
                short_side = min(H, W)
                scale = self.size / short_side
                new_H, new_W = int(H * scale), int(W * scale)
                # Reshape to (T*C, H, W) for batch processing
                x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
                x = x.view(-1, H, W)  # (T*C, H, W) - flatten for interpolation
                x = x.unsqueeze(1)  # (T*C, 1, H, W)
                x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
                x = x.squeeze(1)  # (T*C, new_H, new_W)
                x = x.view(T, C, new_H, new_W)  # (T, C, new_H, new_W)
                x = x.permute(0, 2, 3, 1)  # (T, new_H, new_W, C)
                return x
        else:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ApplyTransformToKey(nn.Module):
    """
    Apply transform to a specific key in a dictionary.
    Replaces pytorchvideo.transforms.ApplyTransformToKey
    
    Args:
        key: Dictionary key to apply transform to
        transform: Transform to apply
    """
    def __init__(self, key: str, transform: Callable):
        super().__init__()
        self.key = key
        self.transform = transform
    
    def forward(self, x: dict) -> dict:
        """
        Args:
            x: Dictionary with video data
        Returns:
            Dictionary with transformed data
        """
        if self.key in x:
            x[self.key] = self.transform(x[self.key])
        return x
    
    def __call__(self, x: dict) -> dict:
        return self.forward(x)
