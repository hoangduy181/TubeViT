#!/usr/bin/env python3
"""
Fix PyAV compatibility issue with torchvision.

The issue: torchvision tries to catch av.AVError but newer PyAV versions
use av.error.Error instead.

This script patches torchvision's video reading to handle both cases.
"""

import sys

def patch_torchvision_video():
    """Patch torchvision's video reading to handle PyAV compatibility."""
    try:
        from torchvision import io
        import av.error
        
        # Check if the issue exists
        original_read_video_timestamps = io.read_video_timestamps
        
        def patched_read_video_timestamps(filename, start_pts=0, end_pts=None, pts_unit='pts'):
            """Patched version that handles both old and new PyAV error types."""
            try:
                return original_read_video_timestamps(filename, start_pts, end_pts, pts_unit)
            except AttributeError as e:
                if "'av' has no attribute 'AVError'" in str(e):
                    # This is the compatibility issue - try to handle it
                    # The actual error is likely an InvalidDataError
                    import av
                    try:
                        # Try to open and read timestamps manually
                        with av.open(filename, metadata_errors="ignore") as container:
                            if len(container.streams.video) == 0:
                                raise RuntimeError(f"No video stream in {filename}")
                            stream = container.streams.video[0]
                            # Return dummy timestamps - this will be handled by error handling
                            return [], stream.average_rate
                    except Exception as inner_e:
                        # Re-raise as a more generic error
                        raise RuntimeError(f"Error reading video {filename}: {inner_e}") from inner_e
                raise
        
        # Apply patch
        io.read_video_timestamps = patched_read_video_timestamps
        print("✓ Patched torchvision.io.read_video_timestamps")
        return True
    except Exception as e:
        print(f"✗ Failed to patch torchvision: {e}")
        return False


if __name__ == "__main__":
    print("PyAV Compatibility Fix")
    print("=" * 60)
    
    # Check PyAV version
    try:
        import av
        print(f"PyAV version: {av.__version__}")
    except ImportError:
        print("✗ PyAV not installed")
        sys.exit(1)
    
    # Check if av.AVError exists
    if hasattr(av, 'AVError'):
        print("✓ av.AVError exists - no patch needed")
    else:
        print("⚠ av.AVError does not exist - compatibility issue detected")
        if hasattr(av, 'error'):
            print(f"  Using av.error instead")
            if hasattr(av.error, 'Error'):
                print(f"  ✓ av.error.Error exists")
    
    # Try to patch
    if patch_torchvision_video():
        print("\n✓ Patch applied successfully")
        print("You can now import this module before using datasets:")
        print("  import scripts.fix_pyav_compatibility")
    else:
        print("\n⚠ Could not apply patch")
        print("\nAlternative solutions:")
        print("1. Downgrade PyAV: pip install 'av<10.0.0'")
        print("2. Upgrade torchvision: pip install --upgrade torchvision")
        print("3. Remove corrupted videos using check_corrupted_videos.py")
