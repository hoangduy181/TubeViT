# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
import json
import csv
import shutil
import time
from datetime import datetime

import click
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import seaborn as sns
import torch
from tubevit.transforms import Normalize

from torch.utils.data import DataLoader, SequentialSampler, Subset, RandomSampler
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score, precision, recall
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from tubevit.dataset import MyUCF101
from tubevit.model import TubeViTLightningModule
from utils.constant import IMAGENET_MEAN, IMAGENET_STD
from utils.config_loader import load_config, merge_config_with_args, get_config_value

# Enable Tensor Core optimization for NVIDIA GPUs with Tensor Cores (e.g., A100, V100, etc.)
# 'medium' provides a good balance between performance and precision
# Use 'high' for better precision if needed
torch.set_float32_matmul_precision('medium')


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to YAML configuration file. CLI arguments override config values.")
@click.option("-r", "--dataset-root", type=click.Path(exists=True), default=None, help="path to dataset.")
@click.option("-m", "--model-path", type=click.Path(exists=True), default=None, help="path to model weight.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), default=None, help="path to dataset.")
@click.option("--label-path", type=click.Path(exists=True), default=None, help="path to classInd.txt.")
@click.option("-nc", "--num-classes", type=int, default=None, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=None, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=None, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=None, help="frame per clip.")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers. Defaults to number of CPUs.")
@click.option("--seed", type=int, default=None, help="random seed.")
@click.option("--verbose", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option("--run-name", type=str, default=None, help="Name for this evaluation run. If not provided, will be auto-generated.")
@click.option("--single-clip-per-video", type=bool, is_flag=True, show_default=True, default=False, help="Use only 1 clip per video (faster but less robust). If False, aggregates predictions from multiple clips per video.")
def main(
    config,
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
):
    # Load configuration from file if provided
    cfg = {}
    if config:
        print(f"Loading configuration from: {config}")
        cfg = load_config(config)
        print("✓ Configuration loaded")
    
    # Prepare CLI arguments dictionary
    cli_args = {
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
    dataset_root = get_config_value(merged_config, 'dataset_root') or get_config_value(merged_config, 'dataset.root')
    model_path = get_config_value(merged_config, 'model_path')
    annotation_path = get_config_value(merged_config, 'annotation_path') or get_config_value(merged_config, 'dataset.annotation_path')
    label_path = get_config_value(merged_config, 'label_path') or get_config_value(merged_config, 'dataset.label_path')
    num_classes = get_config_value(merged_config, 'num_classes') or get_config_value(merged_config, 'dataset.num_classes', 101)
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
    if not annotation_path:
        raise ValueError("annotation_path is required. Provide via --annotation-path or config file (dataset.annotation_path)")
    if not label_path:
        raise ValueError("label_path is required. Provide via --label-path or config file (dataset.label_path)")
    
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

    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_metadata_file = get_config_value(merged_config, 'metadata.val_file', "ucf101-val-meta.pickle")
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        with open(val_metadata_file, "rb") as f:
            val_precomputed_metadata = pickle.load(f)

    val_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=val_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=False,
        output_format="THWC",
        transform=test_transform,
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

    x, y = next(iter(val_dataloader))
    print(x.shape)

    model = TubeViTLightningModule.load_from_checkpoint(model_path)
    model.eval()  # Set to evaluation mode
    
    # Calculate FLOPS using fvcore or thop if available
    flops_count = None
    try:
        try:
            from fvcore.nn import FlopCountMode, flop_count
            # Create a dummy input with the same shape as actual input
            dummy_input = torch.randn(1, *x.shape[1:]).to(next(model.parameters()).device)
            flops_dict, _ = flop_count(model.model, (dummy_input,), mode=FlopCountMode.OPERATION_COUNT)
            flops_count = sum(flops_dict.values())
            print(f"\n{'='*60}")
            print("Model FLOPS Calculation")
            print(f"{'='*60}")
            print(f"FLOPS: {flops_count / 1e9:.2f} GFLOPs")
        except ImportError:
            try:
                from thop import profile, clever_format
                dummy_input = torch.randn(1, *x.shape[1:]).to(next(model.parameters()).device)
                flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
                flops_count = flops
                print(f"\n{'='*60}")
                print("Model FLOPS Calculation")
                print(f"{'='*60}")
                print(f"FLOPS: {flops / 1e9:.2f} GFLOPs")
                print(f"Parameters: {params / 1e6:.2f} M")
            except ImportError:
                print("\nNote: Install 'fvcore' or 'thop' to calculate FLOPS:")
                print("  pip install fvcore")
                print("  or")
                print("  pip install thop")
    except Exception as e:
        print(f"Warning: Could not calculate FLOPS: {e}")

    # Measure inference time and calculate FPS
    print(f"\n{'='*60}")
    print("Starting Evaluation (measuring FPS)...")
    print(f"{'='*60}")
    
    start_time = time.time()
    trainer = pl.Trainer(accelerator="auto", devices=1, default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)
    end_time = time.time()
    
    # Use predictions directly
    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])
    
    # Calculate metrics directly from predictions
    print(f"\n{'='*60}")
    print(f"Calculating Metrics...")
    print(f"{'='*60}")
    
    acc = accuracy(y_prob, y, task="multiclass", num_classes=num_classes)
    prec_macro = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    prec_micro = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    rec_macro = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    rec_micro = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    f1_macro = f1_score(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
    f1_micro = f1_score(y_prob, y, task="multiclass", num_classes=num_classes, average="micro")
    
    # Calculate per-class precision and recall
    prec_per_class = precision(y_prob, y, task="multiclass", num_classes=num_classes, average="none")
    rec_per_class = recall(y_prob, y, task="multiclass", num_classes=num_classes, average="none")
    
    # Confusion matrix
    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)
    
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (macro): {prec_macro:.4f} ({prec_macro*100:.2f}%)")
    print(f"Precision (micro): {prec_micro:.4f} ({prec_micro*100:.2f}%)")
    print(f"Recall (macro): {rec_macro:.4f} ({rec_macro*100:.2f}%)")
    print(f"Recall (micro): {rec_micro:.4f} ({rec_micro*100:.2f}%)")
    print(f"F1 Score (macro): {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"F1 Score (micro): {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    
    # Save simple summary
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
    
    print(f"\n✓ Summary saved to: {summary_file}")

    # Save confusion matrix as PNG
    cm_file = results_dir / "confusion_matrix.png"
    
    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix - {run_name}")
    plt.tight_layout()
    plt.savefig(cm_file, dpi=300)
    print(f"✓ Confusion matrix (PNG) saved to: {cm_file}")
    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    # Save confusion matrix as CSV
    cm_csv_file = results_dir / "confusion_matrix.csv"
    cm_numpy = cm.cpu().numpy() if hasattr(cm, 'cpu') else cm.numpy() if hasattr(cm, 'numpy') else cm
    with open(cm_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow([''] + labels)
        # Write data rows
        for i, label in enumerate(labels):
            row = [label] + cm_numpy[i].tolist()
            writer.writerow(row)
    print(f"✓ Confusion matrix (CSV) saved to: {cm_csv_file}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
    print(f"\nSaved files:")
    print(f"  ✓ summary.txt: Evaluation summary with precision, recall, and per-class metrics")
    print(f"  ✓ confusion_matrix.png: Confusion matrix visualization")
    print(f"  ✓ confusion_matrix.csv: Confusion matrix in CSV format")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
