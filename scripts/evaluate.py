# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
import csv
from datetime import datetime

import click
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import seaborn as sns
import torch
from tubevit.transforms import Normalize, Permute

from torch.utils.data import DataLoader, SequentialSampler, Subset, RandomSampler
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score, precision, recall
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from tubevit.dataset import get_dataset, set_debug_shapes
from tubevit.model import TubeViTLightningModule
from utils.constant import IMAGENET_MEAN, IMAGENET_STD
from utils.config_loader import load_config, merge_config_with_args, get_config_value

# Enable Tensor Core optimization for NVIDIA GPUs with Tensor Cores (e.g., A100, V100, etc.)
# 'medium' provides a good balance between performance and precision
# Use 'high' for better precision if needed
torch.set_float32_matmul_precision('medium')


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to YAML configuration file. CLI arguments override config values.")
@click.option("--dataset", "--dataset-name", type=str, default="ucf101", help="Dataset name: ucf101, kinetics400/k400, kinetics600/k600, kinetics700/k700")
@click.option("-r", "--dataset-root", type=click.Path(exists=True), default=None, help="path to dataset.")
@click.option("-m", "--model-path", type=click.Path(exists=True), default=None, help="path to model weight.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), default=None, help="path to dataset annotations (required for UCF101, not used for Kinetics).")
@click.option("--label-path", type=click.Path(exists=True), default=None, help="path to class labels file (required for UCF101, optional for Kinetics).")
@click.option("-nc", "--num-classes", type=int, default=None, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=None, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=None, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=None, help="frame per clip.")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers. Defaults to number of CPUs.")
@click.option("--seed", type=int, default=None, help="random seed.")
@click.option("--verbose", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option("--run-name", type=str, default=None, help="Name for this evaluation run. If not provided, will be auto-generated.")
@click.option("--single-clip-per-video", type=bool, is_flag=True, show_default=True, default=False, help="Use only 1 clip per video (faster but less robust). If False, aggregates predictions from multiple clips per video.")
@click.option("--debug-shapes", type=bool, is_flag=True, show_default=True, default=False, help="Enable debug logging for video tensor shapes through the pipeline.")
def main(
    config,
    dataset,
    dataset_root,
    model_path,
    annotation_path,
    label_path,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    num_workers,
    seed,
    verbose,
    run_name,
    single_clip_per_video,
    debug_shapes,
):
    # Load configuration from file if provided
    cfg = {}
    if config:
        print(f"Loading configuration from: {config}")
        cfg = load_config(config)
        print("✓ Configuration loaded")
    
    # Prepare CLI arguments dictionary
    cli_args = {
        'dataset': dataset,
        'dataset_root': dataset_root,
        'model_path': model_path,
        'annotation_path': annotation_path,
        'label_path': label_path,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'frames_per_clip': frames_per_clip,
        'video_size': video_size,
        'num_workers': num_workers,
        'seed': seed,
        'verbose': verbose,
        'run_name': run_name,
        'single_clip_per_video': single_clip_per_video,
    }
    
    # Merge config with CLI args (CLI args take precedence)
    merged_config = merge_config_with_args(cfg, cli_args)
    
    # Extract values from merged config (support nested structure)
    dataset_name = get_config_value(merged_config, 'dataset') or get_config_value(merged_config, 'dataset.name', 'ucf101')
    dataset_root = get_config_value(merged_config, 'dataset_root') or get_config_value(merged_config, 'dataset.root')
    model_path = get_config_value(merged_config, 'model_path')
    annotation_path = get_config_value(merged_config, 'annotation_path') or get_config_value(merged_config, 'dataset.annotation_path')
    label_path = get_config_value(merged_config, 'label_path') or get_config_value(merged_config, 'dataset.label_path')
    
    # Get num_classes with dataset-specific defaults
    if dataset_name == 'ucf101':
        default_num_classes = 101
    elif '400' in dataset_name or 'k400' in dataset_name:
        default_num_classes = 400
    elif '600' in dataset_name or 'k600' in dataset_name:
        default_num_classes = 600
    elif '700' in dataset_name or 'k700' in dataset_name:
        default_num_classes = 700
    else:
        default_num_classes = 101
    
    num_classes = get_config_value(merged_config, 'num_classes') or get_config_value(merged_config, 'dataset.num_classes', default_num_classes)
    batch_size = get_config_value(merged_config, 'batch_size') or get_config_value(merged_config, 'training.batch_size', 32)
    frames_per_clip = get_config_value(merged_config, 'frames_per_clip') or get_config_value(merged_config, 'training.frames_per_clip', 32)
    video_size = get_config_value(merged_config, 'video_size') or get_config_value(merged_config, 'training.video_size', (224, 224))
    num_workers = get_config_value(merged_config, 'num_workers') or get_config_value(merged_config, 'training.num_workers')
    seed = get_config_value(merged_config, 'seed') or get_config_value(merged_config, 'training.seed', 42)
    run_name = get_config_value(merged_config, 'run_name')
    single_clip_per_video = get_config_value(merged_config, 'single_clip_per_video', False)
    
    # Validate required parameters
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
    
    # Convert video_size tuple if it's a list from YAML
    if isinstance(video_size, list):
        video_size = tuple(video_size)
    pl.seed_everything(seed)

    # Generate run name if not provided
    if run_name is None:
        # Extract model filename without extension
        model_filename = Path(model_path).stem
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{model_filename}_{timestamp}"
    
    # Create results directory structure
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Evaluation Run: {run_name}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}\n")

    # Set num_workers to number of CPUs if not specified
    if num_workers is None:
        num_workers = os.cpu_count() or 0
        print(f"Using {num_workers} DataLoader workers (auto-detected from CPU count)")

    # Load labels - handle differently for each dataset
    # For UCF101: Load from label_path before dataset init
    # For Kinetics: Will extract from dataset instance after initialization
    labels = None
    if dataset_name == 'ucf101':
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = f.read().splitlines()
                labels = list(map(lambda x: x.split(" ")[-1], labels))
        else:
            raise ValueError(f"label_path is required for UCF101: {label_path}")

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            Permute(dims=[1, 0, 2, 3]),  # (T, C, H, W) -> (C, T, H, W) for model
        ]
    )

    # Generate dataset-specific metadata filename
    dataset_name_clean = dataset_name.lower().replace('-', '').replace('_', '')
    default_metadata_file = f"{dataset_name_clean}-val-meta.pickle"
    val_metadata_file = get_config_value(merged_config, 'metadata.val_file', default_metadata_file)
    
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        with open(val_metadata_file, "rb") as f:
            val_precomputed_metadata = pickle.load(f)

    # Prepare dataset kwargs
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
    
    # Enable debug shape logging if requested
    if debug_shapes:
        set_debug_shapes(True)
        print("\n[DEBUG] Shape logging enabled - will show tensor shapes through pipeline")
    
    # Create dataset instance using factory
    print(f"\n{'='*60}")
    print(f"Initializing {dataset_name} dataset...")
    print(f"{'='*60}")
    val_set = get_dataset(dataset_name=dataset_name, **dataset_kwargs)
    
    # For Kinetics, extract labels from dataset if not loaded
    if labels is None:
        if hasattr(val_set, 'classes') and val_set.classes:
            labels = val_set.classes
            print(f"✓ Extracted {len(labels)} class labels from dataset")
        elif label_path and os.path.exists(label_path):
            # Fallback: load from file
            with open(label_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            print(f"✓ Loaded {len(labels)} class labels from file: {label_path}")
        else:
            raise ValueError(
                f"Cannot determine labels for {dataset_name}. "
                f"Either provide --label-path or ensure dataset has classes attribute."
            )

    if not os.path.exists(val_metadata_file):
        with open(val_metadata_file, "wb") as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Always use full validation set for evaluation
    actual_dataset_size = len(val_set)
    print(f"Dataset info:")
    print(f"  Total clips in validation set: {actual_dataset_size}")
    print(f"  Total videos in validation set: {len(val_set.samples)}")
    # print(f"  val_set.indices: {val_set.indices}") # video index
    print(f"  Average clips per video: {actual_dataset_size / len(val_set.samples):.2f}")
    
    # Use RandomSampler to sample a subset of the dataset
    num_samples = len(val_set) // 50
    print(f"  Using RandomSampler with {num_samples} samples (1/50 of dataset)")
    val_sampler = RandomSampler(val_set, num_samples=num_samples)
    
    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        sampler=val_sampler,
    )

    print(f"\n{'='*60}")
    print("Loading first batch to verify shapes...")
    print(f"{'='*60}")
    x, y = next(iter(val_dataloader))
    print(f"\nBatch tensor shape: {x.shape}")
    print(f"  - Expected format: (B, C, T, H, W) = (batch, channels, time, height, width)")
    if x.dim() == 5:
        B, dim1, dim2, dim3, dim4 = x.shape
        if dim1 == 3:
            print(f"  - Detected: (B={B}, C={dim1}, T={dim2}, H={dim3}, W={dim4}) ✓ Correct format!")
        else:
            print(f"  - WARNING: dim1={dim1}, expected C=3. Format may be incorrect!")
            print(f"  - If this fails, check _ensure_thwc and transform pipeline.")
    print(f"Batch labels shape: {y.shape}")
    print(f"{'='*60}\n")

    model = TubeViTLightningModule.load_from_checkpoint(model_path)

    trainer = pl.Trainer(accelerator="auto", default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)
    
    # Use predictions directly
    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])
    
    # Calculate metrics directly from predictions
    acc = accuracy(y_prob, y, task="multiclass", num_classes=num_classes)
    acc_top5 = accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5)
    auroc_score = auroc(y_prob, y, task="multiclass", num_classes=num_classes)
    f1 = f1_score(y_prob, y, task="multiclass", num_classes=num_classes)
    
    # Confusion matrix
    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)
    
    # Simple print statements like evaluate.old.py
    print("accuracy:", acc)
    print("accuracy_top5:", acc_top5)
    print("auroc:", auroc_score)
    print("f1_score:", f1)
    
    # Convert to numpy for processing
    cm_numpy = cm.cpu().numpy() if hasattr(cm, 'cpu') else cm.numpy() if hasattr(cm, 'numpy') else cm
    
    # Normalize confusion matrix to percentages (row-wise normalization)
    # Each row sums to 100% (percentage of predictions for each true class)
    cm_percent = cm_numpy.astype(float)
    row_sums = cm_percent.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_percent = (cm_percent / row_sums) * 100

    # Save confusion matrix as PNG (with raw counts, no annotations)
    cm_count_file = results_dir / "confusion_matrix_count.png"
    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm_numpy, annot=False, fmt="d", xticklabels=labels, yticklabels=labels, 
                     cmap='Blues', cbar_kws={'label': 'Count'})
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix (Count) - {run_name}")
    plt.tight_layout()
    plt.savefig(cm_count_file, dpi=300)
    print(f"✓ Confusion matrix (count PNG) saved to: {cm_count_file}")
    if verbose:
        plt.show()
    else:
        plt.close()
    
    # Save confusion matrix as PNG (with percentages, no annotations)
    cm_percent_file = results_dir / "confusion_matrix_percent.png"
    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm_percent, annot=False, fmt=".2f", xticklabels=labels, yticklabels=labels, 
                     cmap='Blues', cbar_kws={'label': 'Percentage (%)'})
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix (Percentage) - {run_name}")
    plt.tight_layout()
    plt.savefig(cm_percent_file, dpi=300)
    print(f"✓ Confusion matrix (percent PNG) saved to: {cm_percent_file}")
    if verbose:
        plt.show()
    else:
        plt.close()
    
    # Save confusion matrix as CSV (with raw counts)
    cm_count_csv_file = results_dir / "confusion_matrix_count.csv"
    with open(cm_count_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow([''] + labels)
        # Write data rows with counts
        for i, label in enumerate(labels):
            row = [label] + cm_numpy[i].tolist()
            writer.writerow(row)
    print(f"✓ Confusion matrix (count CSV) saved to: {cm_count_csv_file}")
    
    # Save confusion matrix as CSV (with percentages)
    cm_percent_csv_file = results_dir / "confusion_matrix_percent.csv"
    with open(cm_percent_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow([''] + labels)
        # Write data rows with percentages
        for i, label in enumerate(labels):
            row = [label] + [f"{val:.2f}%" for val in cm_percent[i]]
            writer.writerow(row)
    print(f"✓ Confusion matrix (percent CSV) saved to: {cm_percent_csv_file}")
    
    # Save summary with precision and recall
    prec_macro = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    prec_micro = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    rec_macro = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    rec_micro = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    f1_macro = f1_score(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    f1_micro = f1_score(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    
    # Calculate per-class precision and recall
    prec_per_class = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="none")
    rec_per_class = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="none")
    
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"{'='*60}\n")
        f.write("Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_root}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Overall Metrics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total samples evaluated: {len(y)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"\nPrecision:\n")
        f.write(f"  Macro: {prec_macro:.4f} ({prec_macro*100:.2f}%)\n")
        f.write(f"  Micro: {prec_micro:.4f} ({prec_micro*100:.2f}%)\n")
        f.write(f"\nRecall:\n")
        f.write(f"  Macro: {rec_macro:.4f} ({rec_macro*100:.2f}%)\n")
        f.write(f"  Micro: {rec_micro:.4f} ({rec_micro*100:.2f}%)\n")
        f.write(f"\nF1 Score:\n")
        f.write(f"  Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)\n")
        f.write(f"  Micro: {f1_micro:.4f} ({f1_micro*100:.2f}%)\n")
        f.write(f"\n{'='*60}\n")
        f.write("Per-Class Metrics\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 66 + "\n")
        for i in range(num_classes):
            class_name = labels[i] if i < len(labels) else f"Class_{i}"
            prec_val = prec_per_class[i].item()
            rec_val = rec_per_class[i].item()
            f1_val = 2 * (prec_val * rec_val) / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0.0
            f.write(f"{class_name:<30} {prec_val:<12.4f} {rec_val:<12.4f} {f1_val:<12.4f}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
    print(f"\nSaved files:")
    print(f"  ✓ summary.txt: Evaluation summary with precision, recall, and per-class metrics")
    print(f"  ✓ confusion_matrix_count.png: Confusion matrix visualization (counts)")
    print(f"  ✓ confusion_matrix_percent.png: Confusion matrix visualization (percentages)")
    print(f"  ✓ confusion_matrix_count.csv: Confusion matrix in CSV format (counts)")
    print(f"  ✓ confusion_matrix_percent.csv: Confusion matrix in CSV format (percentages)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
