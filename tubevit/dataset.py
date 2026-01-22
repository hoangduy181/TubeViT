import os
import warnings
from typing import Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import UCF101

# Suppress pts_unit warning from torchvision video reading
warnings.filterwarnings('ignore', message=".*pts_unit.*", category=UserWarning)


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        # Add logging to debug dataset initialization
        root = kwargs.get('root', args[0] if args else None)
        annotation_path = kwargs.get('annotation_path', args[1] if len(args) > 1 else None)
        train = kwargs.get('train', True)
        
        print(f"[MyUCF101] Initializing dataset...")
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
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label
