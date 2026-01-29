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

### Example

Convert ImageNet pre-trained weight to UCF101. `--num-classes` is 101 by default.

```commandline
python scripts/convert_vit_weight.py
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

