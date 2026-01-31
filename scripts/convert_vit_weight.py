import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights

from tubevit.model import TubeViT

# Default output directory for pretrained weights
DEFAULT_WEIGHTS_DIR = Path("pretrained_weights")


def generate_weight_filename(num_classes: int, frames_per_clip: int, video_size: tuple) -> str:
    """Generate informative filename for pretrained weights."""
    return f"init_weight_vitb16_nc{num_classes}_f{frames_per_clip}_v{video_size[0]}x{video_size[1]}.pt"


@click.command()
@click.option("-nc", "--num-classes", type=int, default=101, help="num of classes of dataset.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="video frame size (H, W).")
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    default=None,
    help="Output path for model weights. If not specified, auto-generates name in pretrained_weights/ folder.",
)
@click.option(
    "-d",
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for weights. Defaults to 'pretrained_weights/'.",
)
def main(num_classes, frames_per_clip, video_size, output_path, output_dir):
    """
    Convert ViT-B/16 ImageNet pretrained weights to TubeViT format.
    
    This script:
    1. Downloads ViT-B/16 weights from torchvision
    2. Inflates the patch embedding for video (2D -> 3D)
    3. Removes mismatched layers (pos_embedding, classification head)
    4. Saves weights ready for TubeViT training
    
    Examples:
        # For UCF101 (101 classes):
        python convert_vit_weight.py -nc 101
        
        # For Kinetics-400 (400 classes):
        python convert_vit_weight.py -nc 400
        
        # Custom output path:
        python convert_vit_weight.py -nc 101 -o my_weights.pt
    """
    print(f"\n{'='*60}")
    print("TubeViT Weight Converter")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - num_classes: {num_classes}")
    print(f"  - frames_per_clip: {frames_per_clip}")
    print(f"  - video_size: {video_size}")
    
    x = np.random.random((1, 3, frames_per_clip, video_size[0], video_size[1]))
    x = Tensor(x)
    print(f"  - input shape: {x.shape} (B, C, T, H, W)")

    y = np.random.randint(0, 1, size=(1, num_classes))
    y = Tensor(y)
    print(f"  - output shape: {y.shape} (B, num_classes)")
    print(f"{'='*60}\n")

    print("Creating TubeViT model...")
    model = TubeViT(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )

    print("Downloading ViT-B/16 ImageNet weights...")
    weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)

    # Inflate ViT patch convolution layer weight (2D -> 3D)
    print("Inflating patch embedding for video...")
    conv_proj_weight = weights["conv_proj.weight"]
    conv_proj_weight = F.interpolate(conv_proj_weight, (8, 8), mode="bilinear")
    conv_proj_weight = torch.unsqueeze(conv_proj_weight, dim=2)
    conv_proj_weight = conv_proj_weight.repeat(1, 1, 8, 1, 1)
    conv_proj_weight = conv_proj_weight / 8.0

    # Remove mismatched parameters (will be initialized randomly)
    print("Removing mismatched layers:")
    print("  - encoder.pos_embedding (different sequence length for video)")
    print("  - heads.head.weight (different num_classes)")
    print("  - heads.head.bias (different num_classes)")
    weights.pop("encoder.pos_embedding")
    weights.pop("heads.head.weight")
    weights.pop("heads.head.bias")

    model.load_state_dict(weights, strict=False)
    model.sparse_tubes_tokenizer.conv_proj_weight = torch.nn.Parameter(conv_proj_weight, requires_grad=True)

    # Determine output path
    if output_path is None:
        # Auto-generate filename
        weights_dir = Path(output_dir) if output_dir else DEFAULT_WEIGHTS_DIR
        weights_dir.mkdir(parents=True, exist_ok=True)
        filename = generate_weight_filename(num_classes, frames_per_clip, video_size)
        output_path = weights_dir / filename
    else:
        output_path = Path(output_path)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), output_path)
    
    print(f"\n{'='*60}")
    print("✓ Weights saved successfully!")
    print(f"{'='*60}")
    print(f"Output path: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"\nTo use these weights in training:")
    print(f"  python scripts/train.py --weight-path {output_path} ...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
