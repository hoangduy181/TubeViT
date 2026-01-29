#!/usr/bin/env python3
"""
Check for corrupted videos in Kinetics dataset and optionally remove them.

This script identifies videos with missing moov atoms or other corruption issues
that cause errors during dataset loading.
"""

import sys
from pathlib import Path
from typing import List

import click
import av


def check_video(video_path: Path) -> tuple[bool, str]:
    """
    Check if a video file is valid.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with av.open(str(video_path), metadata_errors="ignore") as container:
            # Try to access video stream
            if len(container.streams.video) == 0:
                return False, "No video stream found"
            
            # Try to read first packet
            stream = container.streams.video[0]
            container.seek(0)
            next(container.demux(stream))
            
        return True, ""
    except av.error.InvalidDataError as e:
        return False, f"InvalidDataError: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


@click.command()
@click.option("--root", type=click.Path(exists=True), required=True, help="Root directory containing train/val/test folders")
@click.option("--split", type=str, default="train", help="Split to check: train, val, or test")
@click.option("--remove", is_flag=True, default=False, help="Remove corrupted videos (use with caution!)")
@click.option("--output", type=click.Path(), default=None, help="Output file to save list of corrupted videos")
def main(root, split, remove, output):
    """
    Check for corrupted videos in Kinetics dataset.
    """
    root = Path(root)
    split_dir = root / split
    
    if not split_dir.exists():
        print(f"Error: Split directory not found: {split_dir}")
        sys.exit(1)
    
    print(f"Checking videos in: {split_dir}")
    print(f"Remove corrupted: {remove}")
    print()
    
    # Find all video files
    video_files = []
    for ext in ['mp4', 'avi', 'webm']:
        video_files.extend(split_dir.rglob(f"*.{ext}"))
    
    print(f"Found {len(video_files)} video files")
    print()
    
    corrupted = []
    valid = 0
    
    for i, video_path in enumerate(video_files):
        if (i + 1) % 100 == 0:
            print(f"Checked {i + 1}/{len(video_files)} videos... "
                  f"({valid} valid, {len(corrupted)} corrupted)")
        
        is_valid, error = check_video(video_path)
        
        if is_valid:
            valid += 1
        else:
            corrupted.append((str(video_path), error))
            print(f"  ✗ Corrupted: {video_path.name}")
            print(f"    Error: {error}")
            
            if remove:
                try:
                    video_path.unlink()
                    print(f"    → Removed")
                except Exception as e:
                    print(f"    → Failed to remove: {e}")
    
    print()
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Valid: {valid}")
    print(f"  Corrupted: {len(corrupted)}")
    print(f"{'='*60}")
    
    if corrupted and output:
        output_path = Path(output)
        with open(output_path, 'w') as f:
            f.write("corrupted_video,error\n")
            for video_path, error in corrupted:
                f.write(f"{video_path},{error}\n")
        print(f"\nList of corrupted videos saved to: {output_path}")
    
    if corrupted and not remove:
        print(f"\nTo remove corrupted videos, run with --remove flag:")
        print(f"  python scripts/check_corrupted_videos.py --root {root} --split {split} --remove")


if __name__ == "__main__":
    main()
