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
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (C, T, H, W) or (B, C, T, H, W)
        Returns:
            Normalized tensor
        """
        if x.dim() == 4:  # (C, T, H, W)
            x = x.unsqueeze(0)  # Add batch dimension
            x = (x - self.mean) / self.std
            return x.squeeze(0)
        else:  # (B, C, T, H, W)
            return (x - self.mean) / self.std


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
    Applies random augmentations consistently across all frames in a video.
    
    Args:
        num_ops: Number of augmentation operations to apply (default: 2)
        magnitude: Magnitude of augmentations (0-10, default: 9)
        num_layers: Alias for num_ops (for compatibility with pytorchvideo API)
    """
    # Augmentation operations from AutoAugment/RandAugment
    AUGMENT_OPS = [
        'identity',
        'autocontrast',
        'equalize',
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
    
    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_layers: int = 2):
        super().__init__()
        # num_layers is an alias for num_ops (pytorchvideo compatibility)
        self.num_ops = num_layers if num_layers != 2 else num_ops
        self.magnitude = magnitude
        self._magnitude_range = (0.0, 1.0)  # Normalized magnitude range
    
    def _get_magnitude(self, level: float) -> float:
        """Convert magnitude level (0-10) to actual augmentation parameter."""
        # Map magnitude (0-10) to level (0.0-1.0)
        level = level / 10.0
        return level
    
    def _apply_op(self, x: torch.Tensor, op_name: str, magnitude: float) -> torch.Tensor:
        """
        Apply a single augmentation operation.
        
        Args:
            x: Video tensor of shape (T, C, H, W)
            op_name: Name of the operation to apply
            magnitude: Magnitude parameter (0.0-1.0)
        Returns:
            Augmented tensor with same shape
        """
        # Ensure input is in [0, 1] range for most operations
        if x.max() > 1.0:
            x = x.clamp(0.0, 1.0)
        
        # Skip identity operation
        if op_name == 'identity':
            return x
        
        # Apply operation based on name
        op_map = {
            'autocontrast': self._autocontrast,
            'equalize': self._equalize,
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
            return op_map[op_name](x, magnitude)
        else:
            return x
    
    def _autocontrast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autocontrast operation - normalize each channel to [0, 1] based on min/max.
        Args:
            x: Video tensor of shape (T, C, H, W)
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
    
    def _equalize(self, x: torch.Tensor) -> torch.Tensor:
        """Histogram equalization (simplified)."""
        # Simplified version - full histogram equalization is complex
        # For now, use a contrast enhancement approximation
        return self._autocontrast(x)
    
    def _rotate(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Rotate by angle in degrees."""
        angle = (magnitude * 2.0 - 1.0) * 30.0  # -30 to 30 degrees
        if abs(angle) < 0.1:
            return x
        
        # Reshape for rotation: need (N, C, H, W) for F.affine_grid
        original_shape = x.shape
        if x.dim() == 4:
            if x.shape[0] == 3:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
            # Now x is (T, C, H, W)
            T, C, H, W = x.shape
            x = x.view(T * C, 1, H, W)  # (T*C, 1, H, W) for rotation
            
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
        bits = int(8 - magnitude * 4)  # 4-8 bits
        bits = max(1, min(8, bits))
        x = (x * 255.0).int()
        shift = 8 - bits
        x = (x >> shift) << shift
        return (x.float() / 255.0).clamp(0.0, 1.0)
    
    def _solarize(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Invert pixels above threshold."""
        threshold = 1.0 - magnitude
        return torch.where(x > threshold, 1.0 - x, x)
    
    def _color(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust color saturation."""
        # Convert to grayscale and blend
        gray = x.mean(dim=-3, keepdim=True) if x.shape[-3] == 3 else x.mean(dim=1, keepdim=True)
        factor = 1.0 + (magnitude * 2.0 - 1.0) * 0.9  # 0.1 to 1.9
        if x.shape[-3] == 3:  # (C, T, H, W) or (T, C, H, W) with C=3
            if x.shape[0] == 3:
                x = x * factor + gray * (1 - factor)
            else:
                x = x * factor + gray * (1 - factor)
        return x.clamp(0.0, 1.0)
    
    def _contrast(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust contrast."""
        mean = x.mean()
        factor = 1.0 + (magnitude * 2.0 - 1.0) * 0.9  # 0.1 to 1.9
        return ((x - mean) * factor + mean).clamp(0.0, 1.0)
    
    def _brightness(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Adjust brightness."""
        factor = 1.0 + (magnitude * 2.0 - 1.0) * 0.9  # 0.1 to 1.9
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
                x = x.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
            
            T, C, H, W = x.shape
            x_flat = x.view(T * C, 1, H, W)
            blurred = F.conv2d(x_flat, kernel.expand(1, -1, -1, -1), padding=1)
            blurred = blurred.view(T, C, H, W)
            
            if original_shape[0] == 3:
                blurred = blurred.permute(1, 0, 2, 3)
                x = x.permute(1, 0, 2, 3)
            
            factor = 1.0 + (magnitude * 2.0 - 1.0) * 0.9
            x = x * factor + blurred * (1 - factor)
        
        return x.clamp(0.0, 1.0)
    
    def _shear_x(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Shear along X axis."""
        shear = (magnitude * 2.0 - 1.0) * 0.3  # -0.3 to 0.3
        return self._apply_affine(x, [[1, shear, 0], [0, 1, 0]])
    
    def _shear_y(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Shear along Y axis."""
        shear = (magnitude * 2.0 - 1.0) * 0.3  # -0.3 to 0.3
        return self._apply_affine(x, [[1, 0, 0], [shear, 1, 0]])
    
    def _translate_x(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Translate along X axis."""
        translate = (magnitude * 2.0 - 1.0) * 0.3  # -0.3 to 0.3
        pixels = int(translate * x.shape[-1])
        return self._apply_affine(x, [[1, 0, pixels], [0, 1, 0]])
    
    def _translate_y(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """Translate along Y axis."""
        translate = (magnitude * 2.0 - 1.0) * 0.3  # -0.3 to 0.3
        pixels = int(translate * x.shape[-2])
        return self._apply_affine(x, [[1, 0, 0], [0, 1, pixels]])
    
    def _apply_affine(self, x: torch.Tensor, matrix: List[List[float]]) -> torch.Tensor:
        """Apply affine transformation."""
        if x.dim() == 4:
            original_shape = x.shape
            if x.shape[0] == 3:  # (C, T, H, W)
                x = x.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
            
            T, C, H, W = x.shape
            x = x.view(T * C, 1, H, W)
            
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
        
        # Normalize magnitude
        magnitude = self._get_magnitude(self.magnitude)
        
        # Randomly select operations (without replacement)
        num_ops_to_apply = min(self.num_ops, len(self.AUGMENT_OPS))
        if num_ops_to_apply == 0:
            return x
        
        # Get random permutation and select first num_ops
        ops_indices = torch.randperm(len(self.AUGMENT_OPS), device=x.device)[:num_ops_to_apply]
        
        # Apply each operation sequentially
        for op_idx in ops_indices:
            op_name = self.AUGMENT_OPS[op_idx]
            # Random magnitude for this operation (slight variation around base magnitude)
            op_magnitude = magnitude * (0.8 + 0.4 * torch.rand(1, device=x.device).item())
            x = self._apply_op(x, op_name, op_magnitude)
        
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
