# Configuration Files

This directory contains YAML configuration files for TubeViT training and evaluation.

## Usage

Instead of providing all parameters via command-line arguments, you can use a configuration file:

```bash
# Training with config file
python scripts/train.py --config configs/ucf101.yaml

# Evaluation with config file
python scripts/evaluate.py --config configs/ucf101.yaml --model-path models/my_model.ckpt
```

## CLI Arguments Override Config

Command-line arguments always take precedence over config file values:

```bash
# Override batch_size from config
python scripts/train.py --config configs/ucf101.yaml --batch-size 64
```

## Configuration File Structure

Configuration files use YAML format with the following structure:

```yaml
# Dataset Configuration
dataset:
  root: "/path/to/dataset"
  annotation_path: "/path/to/annotations"
  label_path: "/path/to/annotations/classInd.txt"
  num_classes: 101

# Model Configuration
model:
  num_layers: 12
  num_heads: 12
  hidden_dim: 768
  mlp_dim: 3072
  dropout: 0.0
  attention_dropout: 0.0

# Training Configuration
training:
  batch_size: 32
  frames_per_clip: 32
  video_size: [224, 224]
  max_epochs: 10
  num_workers: null  # null = auto-detect CPU count
  seed: 42
  lr: 1e-4
  weight_decay: 0.001
  label_smoothing: 0.0
  precision: "16-mixed"
  limit_val_batches: 0.5

# Data Augmentation
augmentation:
  rand_augment:
    magnitude: 10
    num_layers: 2

# Paths
paths:
  pretrained_weight: "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"

# Metadata files
metadata:
  train_file: "ucf101-train-meta.pickle"
  val_file: "ucf101-val-meta.pickle"
```

## Example Files

- `ucf101.yaml` - Template configuration for UCF101 dataset
- `ucf101_example.yaml` - Example with placeholder paths (copy and update paths)
- `evaluate_example.yaml` - Example config specifically for evaluation

## Evaluation with Config

For evaluation, you still need to provide `--model-path` via CLI since it changes per run:

```bash
# Evaluation with config file
python scripts/evaluate.py --config configs/ucf101.yaml --model-path models/my_model.ckpt

# You can still override other parameters
python scripts/evaluate.py --config configs/ucf101.yaml --model-path models/my_model.ckpt --batch-size 64
```

## Creating Your Own Config

1. Copy `ucf101_example.yaml` to a new file (e.g., `my_config.yaml`)
2. Update all paths to match your setup
3. Adjust hyperparameters as needed
4. Use it: `python scripts/train.py --config configs/my_config.yaml`
