# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tubevit.positional_encoding import get_3d_sincos_pos_embed

def calc_conv_shape(video_shape, kernel_size, stride, offset):
    """Calculate the output shape of convolution (number of tokens in each dimension)"""
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    offset = np.array(offset)
    output = np.floor(
        ((video_shape[[1, 2, 3]] - offset - kernel_size) / stride) + 1
    ).astype(int)  # THW
    return output

def inspect_kernel_pos_embed(video_shape, kernel_idx, kernel_size, stride, offset, embed_dim=768):
    """Detailed inspection of positional embeddings for a single kernel type"""
    print(f"\n{'='*80}")
    print(f"KERNEL {kernel_idx + 1} DETAILED INSPECTION")
    print(f"{'='*80}")
    
    # Calculate tube shape
    tube_shape = calc_conv_shape(video_shape, kernel_size, stride, offset)
    print(f"\n1. CONVOLUTION OUTPUT SHAPE (tube_shape):")
    print(f"   Formula: floor((video_shape - offset - kernel_size) / stride) + 1")
    print(f"   Video shape (CTHW): {video_shape}")
    print(f"   Kernel size (kT, kH, kW): {kernel_size}")
    print(f"   Stride (sT, sH, sW): {stride}")
    print(f"   Offset (oT, oH, oW): {offset}")
    print(f"   → Tube shape (nT, nH, nW): {tuple(tube_shape)}")
    print(f"   → Total tokens: {tube_shape[0] * tube_shape[1] * tube_shape[2]}")
    
    # Calculate actual grid positions (pixel coordinates in video)
    print(f"\n2. GRID POSITIONS (actual pixel coordinates in video):")
    
    # Temporal positions
    t_size = tube_shape[0]
    grid_t = torch.arange(t_size, dtype=torch.float)
    grid_t_positions = grid_t * stride[0] + offset[0] + kernel_size[0] // 2
    print(f"   Temporal positions (T dimension):")
    print(f"     Formula: grid_t * stride[0] + offset[0] + kernel_size[0] // 2")
    print(f"     Positions: {grid_t_positions.tolist()}")
    print(f"     (These are the center pixel positions of each tube in time)")
    
    # Spatial positions
    grid_h_size = tube_shape[1]
    grid_w_size = tube_shape[2]
    grid_h = torch.arange(grid_h_size, dtype=torch.float)
    grid_w = torch.arange(grid_w_size, dtype=torch.float)
    grid_h_positions = grid_h * stride[1] + offset[1] + kernel_size[1] // 2
    grid_w_positions = grid_w * stride[2] + offset[2] + kernel_size[2] // 2
    print(f"\n   Spatial positions (H dimension):")
    print(f"     Formula: grid_h * stride[1] + offset[1] + kernel_size[1] // 2")
    print(f"     Positions: {grid_h_positions.tolist()}")
    print(f"\n   Spatial positions (W dimension):")
    print(f"     Formula: grid_w * stride[2] + offset[2] + kernel_size[2] // 2")
    print(f"     Positions: {grid_w_positions.tolist()}")
    
    # Show example token positions
    print(f"\n   Example token positions (first few):")
    token_idx = 0
    for t_idx in range(min(2, t_size)):
        for h_idx in range(min(2, grid_h_size)):
            for w_idx in range(min(2, grid_w_size)):
                t_pos = grid_t_positions[t_idx].item()
                h_pos = grid_h_positions[h_idx].item()
                w_pos = grid_w_positions[w_idx].item()
                print(f"     Token {token_idx}: (T={t_pos:.1f}, H={h_pos:.1f}, W={w_pos:.1f})")
                token_idx += 1
    
    # Calculate positional embeddings
    print(f"\n3. POSITIONAL EMBEDDING CALCULATION:")
    embed_dim_spatial = embed_dim // 3 * 2  # 512 for embed_dim=768
    embed_dim_temporal = embed_dim // 3      # 256 for embed_dim=768
    print(f"   Embedding dimensions:")
    print(f"     Total: {embed_dim}")
    print(f"     Spatial (H+W): {embed_dim_spatial} (H: {embed_dim_spatial//2}, W: {embed_dim_spatial//2})")
    print(f"     Temporal (T): {embed_dim_temporal}")
    
    # Get the actual embeddings
    pos_embed = get_3d_sincos_pos_embed(
        embed_dim=embed_dim,
        tube_shape=tuple(tube_shape),
        stride=stride,
        offset=offset,
        kernel_size=kernel_size,
    )
    
    print(f"\n4. POSITIONAL EMBEDDING VALUES:")
    print(f"   Shape: {pos_embed.shape} (num_tokens, embed_dim)")
    print(f"   First token embedding (first 10 values): {pos_embed[0, :10].tolist()}")
    print(f"   Min value: {pos_embed.min().item():.4f}")
    print(f"   Max value: {pos_embed.max().item():.4f}")
    print(f"   Mean value: {pos_embed.mean().item():.4f}")
    print(f"   Std value: {pos_embed.std().item():.4f}")
    
    # Show embedding breakdown
    print(f"\n5. EMBEDDING BREAKDOWN (first token):")
    print(f"   Temporal part (first {embed_dim_temporal} dims): {pos_embed[0, :embed_dim_temporal].tolist()[:5]}...")
    print(f"   Spatial part (next {embed_dim_spatial} dims): {pos_embed[0, embed_dim_temporal:embed_dim_temporal+embed_dim_spatial].tolist()[:5]}...")
    
    return pos_embed, tube_shape, grid_t_positions, grid_h_positions, grid_w_positions

def main():
    # Default video shape: (C, T, H, W) = (3, 32, 224, 224)
    video_shape = np.array([3, 32, 224, 224])
    embed_dim = 768
    
    kernel_sizes = (
        (8, 8, 8),      # Kernel 0
        (16, 4, 4),     # Kernel 1
        (4, 12, 12),    # Kernel 2
        (1, 16, 16),    # Kernel 3
    )
    
    strides = (
        (16, 32, 32),
        (6, 32, 32),
        (16, 32, 32),
        (32, 16, 16),
    )
    
    offsets = (
        (0, 0, 0),
        (4, 8, 8),
        (0, 16, 16),
        (0, 0, 0),
    )
    
    print("="*80)
    print("TUBEVIT POSITIONAL EMBEDDING DETAILED INSPECTION")
    print("="*80)
    print(f"\nVideo shape (CTHW): {video_shape}")
    print(f"Embedding dimension: {embed_dim}")
    
    all_pos_embeds = []
    all_tube_shapes = []
    
    # Inspect each kernel
    for i in range(len(kernel_sizes)):
        pos_embed, tube_shape, grid_t, grid_h, grid_w = inspect_kernel_pos_embed(
            video_shape, i, kernel_sizes[i], strides[i], offsets[i], embed_dim
        )
        all_pos_embeds.append(pos_embed)
        all_tube_shapes.append(tube_shape)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal positional embeddings structure:")
    print(f"  1. Class token: 1 token (all zeros)")
    total_tokens = 1
    for i, tube_shape in enumerate(all_tube_shapes):
        num_tokens = tube_shape[0] * tube_shape[1] * tube_shape[2]
        total_tokens += num_tokens
        print(f"  {i+2}. Kernel {i+1}: {num_tokens} tokens (tube_shape: {tuple(tube_shape)})")
    print(f"\n  Total tokens: {total_tokens}")
    print(f"  Final embedding shape: ({total_tokens}, {embed_dim})")
    
    # Visualize
    print(f"\n{'='*80}")
    print("VISUALIZATION")
    print(f"{'='*80}")
    
    # Concatenate all embeddings (including class token)
    class_token = torch.zeros(1, embed_dim)
    all_embeds = torch.cat([class_token] + all_pos_embeds, dim=0)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall embedding heatmap
    ax = axes[0, 0]
    im = ax.imshow(all_embeds.numpy(), aspect='auto', cmap='viridis')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Token Index')
    ax.set_title('All Positional Embeddings (including class token)')
    plt.colorbar(im, ax=ax)
    
    # Show token boundaries
    token_boundaries = [1]  # After class token
    for tube_shape in all_tube_shapes:
        token_boundaries.append(token_boundaries[-1] + tube_shape[0] * tube_shape[1] * tube_shape[2])
    
    for boundary in token_boundaries:
        ax.axhline(y=boundary-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Individual kernel embeddings
    start_idx = 1
    for i, (pos_embed, tube_shape) in enumerate(zip(all_pos_embeds, all_tube_shapes)):
        row = (i + 1) // 2
        col = (i + 1) % 2
        if row < 2 and col < 2:
            ax = axes[row, col]
            im = ax.imshow(pos_embed.numpy(), aspect='auto', cmap='viridis')
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Token Index')
            ax.set_title(f'Kernel {i+1} Embeddings\n(tube_shape: {tuple(tube_shape)})')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    output_file = "positional_embedding_detailed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
