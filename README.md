# TubeViT

An unofficial implementation of TubeViT
in "[Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning](https://arxiv.org/abs/2212.03229)"

# Spec.

- [x] Fixed Positional embedding
- [ ] Sparse Tube Construction
    - [x] Multi-Tube
    - [x] Interpolated Kernels
    - [x] Space To Depth
    - [ ] config of tubes
- [ ] pipeline
    - [x] training
    - [x] evaluating
    - [x] inference

# Usage

This project is based on `torch~=2.10.0` and [pytorch-lightning](https://github.com/Lightning-AI/lightning)

## Setup

1. Install requirements

    ```commandline
    pip install -r requirements.txt
    ```

2. Download dataset

    - **UCF101**: Download and prepare UCF101 dataset
    - **Kinetics-400/600/700**: Download and prepare Kinetics dataset

## Dataset Preparation

### Handling Corrupted Videos

Kinetics datasets sometimes contain corrupted video files (missing moov atoms, incomplete downloads, etc.). The code includes automatic error handling to skip corrupted videos during data loading.

**If you encounter errors with corrupted videos:**

1. **Check for corrupted videos:**
   ```bash
   python scripts/check_corrupted_videos.py \
       --root /path/to/kinetics400 \
       --split train \
       --output corrupted_videos.txt
   ```

2. **Remove corrupted videos (optional):**
   ```bash
   python scripts/check_corrupted_videos.py \
       --root /path/to/kinetics400 \
       --split train \
       --remove
   ```

3. **PyAV Compatibility Issue:**
   If you see `AttributeError: module 'av' has no attribute 'AVError'`, this is a PyAV version compatibility issue. The code includes an automatic patch, but if issues persist:
   ```bash
   # Option 1: Downgrade PyAV
   pip install 'av<10.0.0'
   
   # Option 2: Upgrade torchvision
   pip install --upgrade torchvision
   ```

### Reorganizing Kinetics Dataset (Flat Structure)

If your Kinetics dataset has videos in flat folders (`train/`, `val/`, `test/`) instead of class-based folders, use the reorganization script:

```bash
python scripts/reorganize_kinetics_flat.py \
    --root /path/to/kinetics400 \
    --annotation-train /path/to/train.csv \
    --annotation-val /path/to/val.csv \
    [--use-symlinks]  # Optional: use symlinks instead of copying (saves disk space)
```

**CSV Format:**
```csv
video_id,class_name,start_time,end_time
Ui0nGr0E3ow,abseiling,304,314
ePgkF2BrL20,archery,2264,2274
```

Or simpler format:
```csv
video_id,class_name
Ui0nGr0E3ow,abseiling
ePgkF2BrL20,archery
```

The script will:
1. Read CSV annotations to map video IDs to classes
2. Create class folders (`train/class_name/`, `val/class_name/`)
3. Move or symlink videos to the correct class folders

After reorganization, your dataset will have the structure:
```
root/
  train/
    abseiling/
      Ui0nGr0E3ow_000304_000314.mp4
      ...
    archery/
      ...
  val/
    abseiling/
      ...
```

## Convert ViT pre-trained weight

Use `convert_vit_weight.py` to convert torch ViT pre-trained weight to TubeVit.

```commandline
python scripts/convert_vit_weight.py --help                                                                              ✔ 
Usage: convert_vit_weight.py [OPTIONS]

Options:
  -nc, --num-classes INTEGER      num of classes of dataset.
  -f, --frames-per-clip INTEGER   frame per clip.
  -v, --video-size <INTEGER INTEGER>...
                                  frame per clip.
  -o, --output-path PATH          output model weight name.
  --help                          Show this message and exit.
```

### Examples

```bash
# UCF101 (101 classes) - saves to pretrained_weights/init_weight_vitb16_nc101_f32_v224x224.pt
python scripts/convert_vit_weight.py -nc 101

# Kinetics-400 (400 classes)
python scripts/convert_vit_weight.py -nc 400

# Custom output path
python scripts/convert_vit_weight.py -nc 101 -o custom/path/weights.pt
```

The script automatically:
1. Downloads ViT-B/16 ImageNet weights from torchvision
2. Inflates the 2D patch embedding to 3D for video
3. Removes mismatched layers (position embedding, classification head)
4. Saves weights to `pretrained_weights/` folder with informative names

### Using Pretrained Weights in Training

```bash
# Specify weight path explicitly
python scripts/train.py --config configs/ucf101.yaml \
    -w pretrained_weights/init_weight_vitb16_nc101_f32_v224x224.pt
```

## Configuration Files

Instead of providing all parameters via command-line, you can use YAML configuration files:

**Available config files:**
- `configs/ucf101.yaml` - UCF101 dataset configuration
- `configs/kinetics400.yaml` - Kinetics-400 dataset configuration
- `configs/kinetics600.yaml` - Kinetics-600 dataset configuration
- `configs/kinetics700.yaml` - Kinetics-700 dataset configuration

```bash
# Training with config file
python scripts/train.py --config configs/ucf101.yaml
python scripts/train.py --config configs/kinetics400.yaml

# Evaluation with config file  
python scripts/evaluate.py --config configs/ucf101.yaml --model-path models/my_model.ckpt
python scripts/evaluate.py --config configs/kinetics400.yaml --model-path models/my_model.ckpt
```

CLI arguments override config file values. See `configs/README.md` for more details.

## Train

`train.py` supports multiple datasets: **UCF101** and **Kinetics-400/600/700**.

### Dataset Requirements

**UCF101:**
- `--dataset-root`: Path to UCF101 dataset root directory
- `--annotation-path`: Path to annotation directory (containing `trainlist01.txt`, `testlist01.txt`)
- `--label-path`: Path to `classInd.txt` file
- Based on [torchvision.datasets.UCF101](https://pytorch.org/vision/main/generated/torchvision.datasets.UCF101.html)

**Kinetics-400/600/700:**
- `--dataset-root`: Path to Kinetics dataset root directory
  - Directory structure: `root/train/class_name/video.mp4` and `root/val/class_name/video.mp4`
- `--annotation-path`: **Not required** for Kinetics (directory structure is the annotation)
- `--label-path`: Optional (labels are extracted from dataset structure if not provided)
- Based on [torchvision.datasets.Kinetics](https://pytorch.org/vision/main/generated/torchvision.datasets.Kinetics.html)

```commandline
python scripts/train.py --help

Usage: train.py [OPTIONS]

Options:
  --dataset, --dataset-name TEXT  Dataset name: ucf101, kinetics400/k400, kinetics600/k600, kinetics700/k700
  -r, --dataset-root PATH         path to dataset.  [required]
  -a, --annotation-path PATH      path to dataset annotations (required for UCF101, not used for Kinetics).
  -nc, --num-classes INTEGER      num of classes of dataset.
  -b, --batch-size INTEGER        batch size.
  -f, --frames-per-clip INTEGER   frame per clip.
  -v, --video-size <INTEGER INTEGER>...
                                  frame per clip.
  --max-epochs INTEGER            max epochs.
  --num-workers INTEGER           Number of DataLoader workers. Defaults to number of CPUs.
  --fast-dev-run
  --seed INTEGER                  random seed.
  --preview-video                 Show input video
  --help                          Show this message and exit.
```

### Examples

**UCF101:**
```commandline
# Using config file (recommended)
python scripts/train.py --config configs/ucf101.yaml

# Using command-line arguments
python scripts/train.py --dataset ucf101 -r path/to/ucf101 -a path/to/annotations

# Mix: config file + override some parameters
python scripts/train.py --config configs/ucf101.yaml --batch-size 64 --max-epochs 20
```

**Kinetics-400:**
```commandline
# Using config file (recommended)
python scripts/train.py --config configs/kinetics400.yaml

# Using command-line arguments
python scripts/train.py --dataset kinetics400 -r /path/to/kinetics400 -nc 400

# Short form dataset name
python scripts/train.py --dataset k400 -r /path/to/kinetics400 -nc 400 -b 32 -f 32

# Mix: config file + override parameters
python scripts/train.py --config configs/kinetics400.yaml --batch-size 64 --max-epochs 20
```

**Kinetics-600/700:**
```commandline
# Kinetics-600
python scripts/train.py --config configs/kinetics600.yaml
# or
python scripts/train.py --dataset k600 -r /path/to/kinetics600 -nc 600

# Kinetics-700
python scripts/train.py --config configs/kinetics700.yaml
# or
python scripts/train.py --dataset k700 -r /path/to/kinetics700 -nc 700
```

**Note:** By default, `--num-workers` automatically uses all available CPUs for optimal data loading performance. You can override this by specifying `--num-workers <number>` if needed.

### Training Speed Optimization

Video training can be slow due to video decoding. Use these options to speed up training:

#### Clip Sampling Options

| Option | Description | Example |
|--------|-------------|---------|
| `--clips-per-video N` | Limit clips per video (default: all) | `--clips-per-video 1` |
| `--step-between-clips N` | Skip N frames between clip starts | `--step-between-clips 100` |
| `--frame-step N` | Load every N-th frame | `--frame-step 2` |

#### Augmentation Options

| Option | Description | Effect |
|--------|-------------|--------|
| `--no-augment` | Disable RandAugment | Faster iteration |
| `--segment-sampling` | Segment-based random temporal sampling | Better coverage |

#### Examples

```bash
# Fastest training (for debugging/testing)
python scripts/train.py --config configs/ucf101.yaml \
    --clips-per-video 1 \
    --no-augment

# Balanced speed + quality
python scripts/train.py --config configs/ucf101.yaml \
    --clips-per-video 3 \
    --segment-sampling

# Fine-grained control
python scripts/train.py --config configs/ucf101.yaml \
    --step-between-clips 100 \
    --frame-step 2
```

#### Speed Comparison

| Configuration | Dataset Size | Speed |
|---------------|--------------|-------|
| Default (all clips) | ~10x videos | Baseline |
| `--clips-per-video 1` | = videos | ~10x faster |
| `--clips-per-video 1 --no-augment` | = videos | ~15x faster |

#### Segment-based Random Sampling (`--segment-sampling`)

When enabled, divides each video into N segments and randomly samples 1 frame from each segment (like TSN). This provides better temporal coverage while maintaining data augmentation:

```
Video: [=========================================]
        |seg1||seg2||seg3||seg4|...|seg32|
          ↓     ↓     ↓     ↓        ↓
        [rand][rand][rand][rand]...[rand]
        
Output: 32 frames spread across entire video
```

**Note:** Segment sampling is only applied during training. Evaluation uses uniform temporal sampling for deterministic results.

## Evaluation

```commandline
python scripts/evaluate.py --help

Usage: evaluate.py [OPTIONS]

Options:
  --dataset, --dataset-name TEXT  Dataset name: ucf101, kinetics400/k400, kinetics600/k600, kinetics700/k700
  -r, --dataset-root PATH         path to dataset.  [required]
  -m, --model-path PATH           path to model weight.  [required]
  -a, --annotation-path PATH      path to dataset annotations (required for UCF101, not used for Kinetics).
  --label-path PATH               path to class labels file (required for UCF101, optional for Kinetics).
  -nc, --num-classes INTEGER      num of classes of dataset.
  -b, --batch-size INTEGER        batch size.
  -f, --frames-per-clip INTEGER   frame per clip.
  -v, --video-size <INTEGER INTEGER>...
                                  frame per clip.
  --num-workers INTEGER           Number of DataLoader workers. Defaults to number of CPUs.
  --seed INTEGER                  random seed.
  --verbose                       Show input video
  --single-clip-per-video         Use only 1 clip per video (faster but less robust). If False, aggregates predictions from multiple clips per video.
  --run-name TEXT                 Name for this evaluation run. If not provided, will be auto-generated.
  --help                          Show this message and exit.
```

### Examples

**UCF101:**
```commandline
# Using config file (recommended)
python scripts/evaluate.py --config configs/ucf101.yaml --model-path path/to/model.ckpt

# Using command-line arguments
python scripts/evaluate.py --dataset ucf101 -r path/to/dataset -a path/to/annotation -m path/to/model.ckpt --label-path path/to/classInd.txt

# Single clip per video (faster evaluation)
python scripts/evaluate.py --config configs/ucf101.yaml --model-path path/to/model.ckpt --single-clip-per-video

# Multiple clips per video with custom run name (default, more robust)
python scripts/evaluate.py --config configs/ucf101.yaml --model-path path/to/model.ckpt --run-name my_evaluation_run
```

**Kinetics-400:**
```commandline
# Using config file (recommended)
python scripts/evaluate.py --config configs/kinetics400.yaml --model-path path/to/model.ckpt

# Using command-line arguments (no annotation-path needed for Kinetics)
python scripts/evaluate.py --dataset kinetics400 -r /path/to/kinetics400 -m path/to/model.ckpt -nc 400

# Short form dataset name
python scripts/evaluate.py --dataset k400 -r /path/to/kinetics400 -m path/to/model.ckpt -nc 400 -b 32

# With optional label file (if you have one)
python scripts/evaluate.py --dataset k400 -r /path/to/kinetics400 -m path/to/model.ckpt --label-path /path/to/class_list.txt -nc 400

# Single clip per video (faster)
python scripts/evaluate.py --config configs/kinetics400.yaml --model-path path/to/model.ckpt --single-clip-per-video
```

**Kinetics-600/700:**
```commandline
# Kinetics-600
python scripts/evaluate.py --config configs/kinetics600.yaml --model-path path/to/model.ckpt
# or
python scripts/evaluate.py --dataset k600 -r /path/to/kinetics600 -m path/to/model.ckpt -nc 600

# Kinetics-700
python scripts/evaluate.py --config configs/kinetics700.yaml --model-path path/to/model.ckpt
# or
python scripts/evaluate.py --dataset k700 -r /path/to/kinetics700 -m path/to/model.ckpt -nc 700
```

**Note:** By default, `--num-workers` automatically uses all available CPUs for optimal data loading performance. You can override this by specifying `--num-workers <number>` if needed.

**Evaluation Modes:**
- **Default (multiple clips per video)**: Aggregates predictions from multiple clips per video by averaging probabilities. This is more robust and provides better accuracy, but is slower (~3-5x slower than single clip mode).
- **Single clip per video (`--single-clip-per-video`)**: Uses only 1 clip per video (the first clip). This is faster and simpler, but less robust as it only samples a single temporal segment from each video. Good for quick evaluation checks or when speed is prioritized.

## Inference

Run inference on a single video file using a trained model.

```commandline
python scripts/inference.py --help

Usage: inference.py [OPTIONS] VIDEO_PATH

Arguments:
  VIDEO_PATH  [required]

Options:
  -m, --model-path PATH          path to model weight.  [required]
  --label-path PATH               path to classInd.txt.  [required]
  -f, --frames-per-clip INTEGER   frame per clip.
  -v, --video-size <INTEGER INTEGER>...
                                  frame per clip.
  --help                          Show this message and exit.
```

### Examples

```commandline
python scripts/inference.py path/to/video.mp4 -m path/to/model.ckpt --label-path path/to/classInd.txt
```

## Visualize Dataset

Visualize samples from the UCF101 dataset to inspect the data and transformations.

```commandline
python scripts/visualise_dataset.py --help

Usage: visualise_dataset.py [OPTIONS]

Options:
  -r, --dataset-root PATH         path to dataset.  [required]
  -a, --annotation-path PATH      path to dataset.  [required]
  --label-path PATH               path to classInd.txt.  [required]
  -b, --batch-size INTEGER        batch size.
  -f, --frames-per-clip INTEGER   frame per clip.
  -v, --video-size <INTEGER INTEGER>...
                                  frame per clip.
  --num-workers INTEGER           Number of DataLoader workers. Defaults to number of CPUs.
  --seed INTEGER                  random seed.
  --help                          Show this message and exit.
```

### Examples

```commandline
python scripts/visualise_dataset.py -r path/to/dataset -a path/to/annotation --label-path path/to/classInd.txt
```

## Visualize Positional Embeddings

Generate a visualization of the 3D positional embeddings used in TubeViT.

```commandline
python scripts/visualise_pos_embed.py
```

This script generates `Position_Embedding.png` showing the positional embedding patterns for different tube configurations.

# Model Architecture

![fig1.png](assets/fig1.png)
![fig2.png](assets/fig2.png)
![fig3.png](assets/fig3.png)

# Positional embedding

![Position_Embedding.png](assets/Position_Embedding.png)

