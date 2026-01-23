"""
Convert PyTorch Lightning checkpoint (.ckpt) to PyTorch state dict (.pt)

This script extracts the model state dict from a Lightning checkpoint
and saves it as a standard PyTorch .pt file that can be loaded with torch.load().
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import torch
from tubevit.model import TubeViTLightningModule


@click.command()
@click.option(
    "-i",
    "--input-ckpt",
    type=click.Path(exists=True),
    required=True,
    help="Path to input Lightning checkpoint (.ckpt file)",
)
@click.option(
    "-o",
    "--output-pt",
    type=click.Path(),
    required=True,
    help="Path to output PyTorch state dict (.pt file)",
)
@click.option(
    "--model-only",
    is_flag=True,
    default=False,
    help="If set, extract only the model.state_dict() instead of the full LightningModule state",
)
def main(input_ckpt, output_pt, model_only):
    """
    Convert Lightning checkpoint to PyTorch state dict.
    
    Examples:
        # Convert full checkpoint
        python scripts/convert_ckpt_to_pt.py -i models/tubevit_ucf101.ckpt -o models/tubevit_ucf101.pt
        
        # Extract only model weights (not LightningModule wrapper)
        python scripts/convert_ckpt_to_pt.py -i models/tubevit_ucf101.ckpt -o models/tubevit_model.pt --model-only
    """
    print(f"Loading checkpoint: {input_ckpt}")
    
    # Load the checkpoint
    checkpoint = torch.load(input_ckpt, map_location='cpu')
    
    if model_only:
        # Extract only the model state dict (TubeViT model, not LightningModule)
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix from keys (LightningModule wraps the model)
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # Remove 'model.' prefix to get the actual TubeViT model keys
                    new_key = key[6:]  # len('model.') = 6
                    model_state_dict[new_key] = value
                # Skip LightningModule-specific keys (loss_func, optimizer states, etc.)
            
            print(f"Extracted model state dict with {len(model_state_dict)} parameters")
            if model_state_dict:
                print(f"Sample keys: {list(model_state_dict.keys())[:5]}")
            
            torch.save(model_state_dict, output_pt)
            print(f"✓ Saved model state dict to: {output_pt}")
            print(f"  This can be loaded with: model.load_state_dict(torch.load('{output_pt}'))")
        else:
            # Already a state dict format
            torch.save(checkpoint, output_pt)
            print(f"✓ Saved state dict to: {output_pt}")
    else:
        # Save the entire checkpoint (can be loaded with LightningModule.load_from_checkpoint)
        # But save as .pt format
        torch.save(checkpoint, output_pt)
        print(f"✓ Saved full checkpoint to: {output_pt}")
        print(f"  Note: This can still be loaded with LightningModule.load_from_checkpoint()")
    
    # Show file sizes
    input_size = Path(input_ckpt).stat().st_size / (1024**2)  # MB
    output_size = Path(output_pt).stat().st_size / (1024**2)  # MB
    print(f"\nFile sizes:")
    print(f"  Input:  {input_size:.2f} MB")
    print(f"  Output: {output_size:.2f} MB")
    
    # Verify the output can be loaded
    print(f"\nVerifying output file...")
    try:
        loaded = torch.load(output_pt, map_location='cpu')
        if isinstance(loaded, dict):
            if 'state_dict' in loaded:
                print(f"✓ Checkpoint format: Contains 'state_dict' key")
                print(f"  Keys in state_dict: {len(loaded['state_dict'])}")
            else:
                print(f"✓ State dict format: {len(loaded)} parameters")
                print(f"  Sample keys: {list(loaded.keys())[:5]}")
        print("✓ Output file is valid and can be loaded")
    except Exception as e:
        print(f"✗ Error verifying output: {e}")


if __name__ == "__main__":
    main()
