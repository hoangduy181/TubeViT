# Plan: Adding Kinetics Dataset Support (K400, K600, K700)

## Overview
This plan outlines the steps needed to extend the TubeViT codebase to support Kinetics datasets (Kinetics-400, Kinetics-600, Kinetics-700) alongside the existing UCF101 support.

## Executive Summary

**Goal**: Make the evaluation and training scripts dataset-agnostic, supporting both UCF101 and Kinetics datasets.

**Key Changes Required**:
1. ✅ Create `MyKinetics` dataset wrapper class
2. ✅ Create dataset factory function (`get_dataset()`)
3. ✅ Update `evaluate.py` to use factory (make dataset-agnostic)
4. ✅ Update `train.py` to use factory (make dataset-agnostic)
5. ✅ Handle different label loading strategies
6. ✅ Make `annotation_path` optional (not needed for Kinetics)
7. ✅ Create Kinetics config file templates

**Critical Differences**:
- **UCF101**: Requires `annotation_path` (directory with trainlist/testlist.txt), uses `train=True/False`
- **Kinetics**: No `annotation_path` needed, uses `split='train'/'val'`, `num_classes` is a STRING ('400'/'600'/'700')
- **Labels**: UCF101 loads from file before init, Kinetics extracts from dataset after init

## Current State Analysis

### Dataset-Specific Code Locations
1. **Dataset Class**: `tubevit/dataset.py` - Contains `MyUCF101` class
2. **Evaluation Script**: `scripts/evaluate.py` - Hardcoded to use `MyUCF101`
3. **Training Script**: `scripts/train.py` - Likely also hardcoded (needs verification)
4. **Config Files**: `configs/` - UCF101-specific configs
5. **Label Loading**: Hardcoded parsing of `classInd.txt` format

### Key Differences: UCF101 vs Kinetics

| Aspect | UCF101 | Kinetics |
|--------|--------|----------|
| **Torchvision Class** | `torchvision.datasets.UCF101` | `torchvision.datasets.Kinetics` |
| **Annotation Format** | `trainlist01.txt`, `testlist01.txt` | Directory structure (no separate annotation file) |
| **Label File** | `classInd.txt` (index + name) | Class names inferred from directory structure |
| **Num Classes** | 101 | 400/600/700 (passed as string: '400', '600', '700') |
| **Directory Structure** | `root/train/class/video.avi` or `root/class/video.avi` | `root/split/class/video.mp4` (split='train'/'val'/'test') |
| **Video Format** | `.avi` | `.mp4` or `.avi` (configurable via `extensions`) |
| **Annotation Path Param** | Required (`annotation_path`) | Not used (directory structure is annotation) |
| **Metadata File** | `ucf101-val-meta.pickle` | Should be `kinetics400-val-meta.pickle`, etc. |

## Implementation Plan

### Phase 1: Create Generic Dataset Interface ✅ **FOUNDATION**

#### 1.1 Create Dataset Factory/Registry
**File**: `tubevit/dataset.py`

**Action**: 
- Create a dataset factory function that returns the appropriate dataset class based on dataset name
- Support both UCF101 and Kinetics datasets
- Handle dataset-specific initialization parameters

**Implementation**:
```python
def get_dataset(dataset_name: str, **kwargs):
    """
    Factory function to get dataset instance.
    
    Args:
        dataset_name: 'ucf101', 'kinetics400', 'kinetics600', 'kinetics700'
        **kwargs: Dataset-specific parameters
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'ucf101':
        return MyUCF101(**kwargs)
    elif dataset_name in ['kinetics400', 'kinetics-400', 'k400']:
        # num_classes must be string for Kinetics!
        return MyKinetics(num_classes='400', **kwargs)
    elif dataset_name in ['kinetics600', 'kinetics-600', 'k600']:
        return MyKinetics(num_classes='600', **kwargs)
    elif dataset_name in ['kinetics700', 'kinetics-700', 'k700']:
        return MyKinetics(num_classes='700', **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

#### 1.2 Create MyKinetics Dataset Class
**File**: `tubevit/dataset.py`

**Action**:
- Create `MyKinetics` class similar to `MyUCF101`
- Inherit from `torchvision.datasets.Kinetics`
- Handle Kinetics-specific annotation format (CSV)
- Support different Kinetics variants (400/600/700)

**Key Parameters**:
- `num_classes`: '400', '600', or '700' (string, not int!)
- `root`: Dataset root directory (should contain `train/` and `val/` subdirectories)
- `frames_per_clip`: Number of frames per clip
- `split`: 'train', 'val', or 'test'
- `annotation_path`: NOT USED for Kinetics (directory structure is the annotation)
- `output_format`: 'THWC' or 'TCHW' (default 'TCHW', but we use 'THWC')

**Important Notes**: 
- Torchvision's Kinetics class uses **directory structure** as annotation (no separate annotation file)
- Directory structure: `root/train/class_name/video.mp4` or `root/val/class_name/video.mp4`
- `num_classes` must be a **string** ('400', '600', '700'), not an integer
- `annotation_path` parameter is **not used** for Kinetics (unlike UCF101)
- Kinetics automatically appends the `split` to the root path: `root/split/`

### Phase 2: Update Evaluation Script ✅ **CORE FUNCTIONALITY**

#### 2.1 Add Dataset Selection Parameter
**File**: `scripts/evaluate.py`

**Action**:
- Add `--dataset` or `--dataset-name` CLI option
- Default to 'ucf101' for backward compatibility
- Update config file support to include `dataset.name`

**Changes**:
```python
@click.option("--dataset", "--dataset-name", type=str, default="ucf101", 
              help="Dataset name: ucf101, kinetics400, kinetics600, kinetics700")
```

#### 2.2 Make Dataset Instantiation Generic
**File**: `scripts/evaluate.py`

**Current Code** (line ~160):
```python
val_set = MyUCF101(
    root=dataset_root,
    annotation_path=annotation_path,
    ...
)
```

**New Code**:
```python
from tubevit.dataset import get_dataset

dataset_name = get_config_value(merged_config, 'dataset.name', 'ucf101')
dataset_kwargs = {
    'root': dataset_root,
    'frames_per_clip': frames_per_clip,
    'output_format': "THWC",
    'transform': test_transform,
    '_precomputed_metadata': val_precomputed_metadata,
}

# Handle dataset-specific parameters
if dataset_name == 'ucf101':
    dataset_kwargs['annotation_path'] = annotation_path
    dataset_kwargs['train'] = False
elif dataset_name.startswith('kinetics') or dataset_name.startswith('k'):
    # Kinetics uses split parameter, not train/annotation_path
    dataset_kwargs['split'] = 'val'
    # num_classes is handled by get_dataset based on dataset_name

val_set = get_dataset(dataset_name=dataset_name, **dataset_kwargs)
```

#### 2.3 Update Label Loading Logic
**File**: `scripts/evaluate.py`

**Current Code** (line ~142-144):
```python
with open(label_path, "r") as f:
    labels = f.read().splitlines()
    labels = list(map(lambda x: x.split(" ")[-1], labels))
```

**Important**: For Kinetics, we should load labels AFTER dataset initialization to use the `classes` attribute, or make `label_path` optional.

**New Code**: Create a helper function to handle different label formats:
```python
def load_labels(label_path: str, dataset_name: str, dataset_instance=None) -> List[str]:
    """
    Load labels based on dataset format.
    
    Args:
        label_path: Path to label file (optional for Kinetics)
        dataset_name: Name of dataset
        dataset_instance: Optional dataset instance to extract labels from
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'ucf101':
        # UCF101 format: "1 ClassName"
        with open(label_path, "r") as f:
            labels = f.read().splitlines()
            labels = list(map(lambda x: x.split(" ")[-1], labels))
        return labels
    elif dataset_name.startswith('kinetics') or dataset_name.startswith('k'):
        # Kinetics: Try to get labels from dataset instance first (most reliable)
        if dataset_instance is not None and hasattr(dataset_instance, 'classes'):
            return dataset_instance.classes
        
        # Fallback: Load from label_path if provided
        if label_path and os.path.exists(label_path):
            # Option 1: Simple text file, one class per line
            with open(label_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            return labels
        else:
            # Option 2: Extract from dataset directory structure
            # This requires dataset_root to be available
            raise ValueError(
                f"For Kinetics, either provide label_path or ensure dataset is initialized first. "
                f"Label path provided: {label_path}"
            )
    else:
        raise ValueError(f"Unknown dataset format: {dataset_name}")
```

**Note**: For Kinetics, labels can be extracted from the dataset instance's `classes` attribute after initialization, which is more reliable than parsing a file.

#### 2.3b Update Label Loading Order
**File**: `scripts/evaluate.py`

**Action**: Move label loading AFTER dataset initialization for Kinetics, or make it optional:

**New Code**:
```python
# Load labels - handle differently for each dataset
dataset_name = get_config_value(merged_config, 'dataset.name', 'ucf101')

if dataset_name == 'ucf101':
    # UCF101: Load from label_path before dataset init
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))
else:
    # Kinetics: Will load from dataset instance after initialization
    labels = None  # Will be set after dataset init
```

Then after dataset initialization:
```python
# For Kinetics, extract labels from dataset if not loaded
if labels is None:
    if hasattr(val_set, 'classes'):
        labels = val_set.classes
    elif label_path and os.path.exists(label_path):
        # Fallback: load from file
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError(f"Cannot determine labels for {dataset_name}")
```

#### 2.4 Update Parameter Validation
**File**: `scripts/evaluate.py`

**Current Code** (line ~106-114):
```python
# Validate required parameters
if not dataset_root:
    raise ValueError("dataset_root is required...")
if not annotation_path:
    raise ValueError("annotation_path is required...")
if not label_path:
    raise ValueError("label_path is required...")
```

**New Code**: Make validation dataset-aware:
```python
# Validate required parameters
dataset_name = get_config_value(merged_config, 'dataset.name', 'ucf101')

if not dataset_root:
    raise ValueError("dataset_root is required. Provide via --dataset-root or config file (dataset.root)")
if not model_path:
    raise ValueError("model_path is required. Provide via --model-path")

# annotation_path is only required for UCF101
if dataset_name == 'ucf101' and not annotation_path:
    raise ValueError("annotation_path is required for UCF101. Provide via --annotation-path or config file (dataset.annotation_path)")

# label_path is optional for Kinetics (can extract from dataset), but recommended for UCF101
if dataset_name == 'ucf101' and not label_path:
    raise ValueError("label_path is required for UCF101. Provide via --label-path or config file (dataset.label_path)")
```

#### 2.5 Update Metadata File Naming
**File**: `scripts/evaluate.py`

**Current Code** (line ~154):
```python
val_metadata_file = get_config_value(merged_config, 'metadata.val_file', "ucf101-val-meta.pickle")
```

**New Code**:
```python
# Generate dataset-specific metadata filename
dataset_name = get_config_value(merged_config, 'dataset.name', 'ucf101')
default_metadata_file = f"{dataset_name.lower().replace('-', '')}-val-meta.pickle"
val_metadata_file = get_config_value(merged_config, 'metadata.val_file', default_metadata_file)
```

### Phase 3: Update Configuration Files ✅ **CONFIGURATION**

#### 3.1 Create Kinetics Config Templates
**Files**: 
- `configs/kinetics400.yaml`
- `configs/kinetics600.yaml`
- `configs/kinetics700.yaml`

**Structure**:
```yaml
# Dataset Configuration
dataset:
  name: "kinetics400"  # NEW: Dataset identifier
  root: "/path/to/kinetics400"  # Should contain train/ and val/ subdirectories
  # annotation_path: NOT USED for Kinetics (directory structure is annotation)
  label_path: "/path/to/kinetics400/annotations/class_list.txt"  # Optional: class names list
  num_classes: 400

# Training Configuration
training:
  batch_size: 32
  frames_per_clip: 32
  video_size: [224, 224]
  # ... rest same as UCF101

# Metadata files
metadata:
  val_file: "kinetics400-val-meta.pickle"
```

#### 3.2 Update Config Loader
**File**: `utils/config_loader.py` (if exists)

**Action**: Ensure config loader supports `dataset.name` field

### Phase 4: Update Training Script (If Needed) ✅ **TRAINING SUPPORT**

#### 4.1 Verify Training Script
**File**: `scripts/train.py`

**Action**: 
- Check if training script also hardcodes `MyUCF101`
- Apply same changes as evaluation script if needed
- Add dataset selection parameter

### Phase 5: Handle Dataset-Specific Differences ✅ **EDGE CASES**

#### 5.1 Annotation Path Handling
**Issue**: UCF101 uses directory (with trainlist/testlist.txt), Kinetics uses directory structure directly

**Solution**: 
- For UCF101: `annotation_path` is a directory (required)
- For Kinetics: `annotation_path` is **not used** (directory structure is the annotation)
- Update dataset factory to conditionally include `annotation_path` only for UCF101

#### 5.2 Video Format Differences
**Issue**: UCF101 uses `.avi`, Kinetics typically uses `.mp4`

**Solution**: 
- Torchvision's dataset classes handle this automatically
- No code changes needed, but document the difference

#### 5.3 Class Count Validation
**Action**: Add validation to ensure `num_classes` matches dataset:
- UCF101: 101
- Kinetics-400: 400
- Kinetics-600: 600
- Kinetics-700: 700

### Phase 6: Testing & Validation ✅ **QUALITY ASSURANCE**

#### 6.1 Test Dataset Loading
- Test with UCF101 (ensure backward compatibility)
- Test with Kinetics-400
- Test with Kinetics-600
- Test with Kinetics-700

#### 6.2 Test Evaluation Script
- Verify metrics calculation works correctly
- Verify confusion matrix generation
- Verify label mapping

#### 6.3 Test Config File Loading
- Test YAML config files for each dataset
- Test CLI argument overrides

## Implementation Order (Recommended)

1. ✅ **Phase 1.2**: Create `MyKinetics` class (foundation)
2. ✅ **Phase 1.1**: Create dataset factory function
3. ✅ **Phase 2.1-2.2**: Update evaluation script to use factory
4. ✅ **Phase 2.3**: Update label loading logic
5. ✅ **Phase 2.4**: Update metadata file naming
6. ✅ **Phase 3.1**: Create Kinetics config templates
7. ✅ **Phase 4.1**: Update training script (if needed)
8. ✅ **Phase 5**: Handle edge cases
9. ✅ **Phase 6**: Testing

## Files to Modify

### New Files
- `tubevit/dataset.py` - Add `MyKinetics` class and `get_dataset()` function
- `configs/kinetics400.yaml` - Kinetics-400 config template
- `configs/kinetics600.yaml` - Kinetics-600 config template
- `configs/kinetics700.yaml` - Kinetics-700 config template

### Modified Files
- `scripts/evaluate.py` - Make dataset-agnostic
- `scripts/train.py` - Make dataset-agnostic (if needed)
- `utils/config_loader.py` - Support `dataset.name` (if exists)

## Backward Compatibility

✅ **Critical**: All changes must maintain backward compatibility with UCF101:
- Default dataset name should be 'ucf101'
- Existing config files should work without modification
- Existing command-line usage should work unchanged

## Documentation Updates Needed

1. Update `README.md` with Kinetics dataset instructions
2. Add example config files for each Kinetics variant
3. Document annotation format differences
4. Add dataset preparation guide for Kinetics

## Notes

- **Torchvision Kinetics**: Uses directory structure as annotation (`root/split/class/video.mp4`), NOT CSV files
- **Label Files**: Kinetics labels are inferred from directory structure, but we can provide a label_path for class names
- **Metadata Caching**: Different metadata files per dataset prevent conflicts
- **Directory Structure**: Kinetics expects `root/train/` and `root/val/` subdirectories, each containing class-named subdirectories
- **num_classes Parameter**: Must be a STRING ('400', '600', '700'), not an integer for torchvision's Kinetics class

## Questions Resolved

1. ✅ **Kinetics doesn't use CSV annotations** - it uses directory structure directly
2. ✅ **Labels are inferred from directory names** - class names come from subdirectory names
3. ✅ **No annotation_path parameter** - Kinetics uses directory structure only
4. ✅ **Directory structure**: `root/train/class_name/video.mp4` and `root/val/class_name/video.mp4`

## Next Steps After Implementation

1. Test with actual Kinetics-400 dataset
2. Verify performance metrics are correct
3. Document any dataset-specific quirks discovered
4. Create example scripts for each dataset variant
