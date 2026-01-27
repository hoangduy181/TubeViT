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

from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score
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
    print(f"  Evaluating on: {actual_dataset_size} samples (full dataset)")
    print(f"  Dataset has {len(val_set.indices)} clip indices")
    print(f"  Dataset has {len(val_set.samples)} video samples")
    
    # Use SequentialSampler for deterministic evaluation (no shuffling)
    # Always use full dataset - no subsetting
    val_sampler = SequentialSampler(val_set)
    val_dataset = val_set
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # No shuffling for evaluation
        drop_last=False,  # Don't drop last batch in evaluation
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
    trainer = pl.Trainer(accelerator="auto", default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)
    end_time = time.time()
    
    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])
    
    # Calculate performance metrics
    total_time = end_time - start_time
    total_frames = len(y) * frames_per_clip  # Total frames processed
    fps = total_frames / total_time if total_time > 0 else 0
    samples_per_sec = len(y) / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("Performance Metrics")
    print(f"{'='*60}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Total samples processed: {len(y)}")
    print(f"Total frames processed: {total_frames}")
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Samples per second: {samples_per_sec:.2f}")
    print(f"Time per sample: {total_time / len(y) * 1000:.2f} ms" if len(y) > 0 else "N/A")
    
    # Get confidence scores (max probability)
    confidences = torch.max(y_prob, dim=1)[0].cpu().numpy()
    
    # Get video paths from dataset - use actual indices from the dataset
    # The predictions are in the same order as the dataloader iterations
    video_paths = []
    actual_num_samples = len(y)
    actual_dataset_len = len(val_set)  # Full dataset size
    
    print(f"Collecting video paths...")
    print(f"  Predictions: {actual_num_samples}, Dataset size: {actual_dataset_len}")
    
    # Collect paths for all samples (using full dataset)
    for sample_idx in range(actual_num_samples):
        try:
            # Direct access to dataset (no subset, always full dataset)
            if sample_idx >= len(val_set):
                print(f"Warning: sample_idx {sample_idx} >= len(val_set) {len(val_set)}")
                video_paths.append(f"unknown_video_{sample_idx}.avi")
                continue
            
            clip_idx = sample_idx
            
            # Safety check: ensure clip_idx is within bounds of val_set.indices
            if clip_idx >= len(val_set.indices):
                print(f"Warning: clip_idx {clip_idx} >= len(val_set.indices) {len(val_set.indices)} (sample_idx={sample_idx})")
                video_paths.append(f"unknown_video_{sample_idx}.avi")
                continue
            
            # Map clip index to video index using val_set.indices
            video_idx = val_set.indices[clip_idx]
            
            # Safety check: ensure video_idx is within bounds
            if video_idx >= len(val_set.samples):
                print(f"Warning: video_idx {video_idx} >= len(val_set.samples) {len(val_set.samples)}")
                video_paths.append(f"unknown_video_{sample_idx}.avi")
                continue
            
            video_path, _ = val_set.samples[video_idx]
            
            # Convert to absolute path if relative
            if not os.path.isabs(video_path):
                video_path = os.path.join(val_set.root, video_path)
            video_paths.append(video_path)
            
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Warning: Could not get video path for sample {sample_idx}: {e}")
            video_paths.append(f"unknown_video_{sample_idx}.avi")
    
    # Ensure we have the same number of video paths as predictions
    if len(video_paths) != actual_num_samples:
        print(f"Warning: Mismatch between predictions ({actual_num_samples}) and video paths ({len(video_paths)})")
        # Pad or truncate to match
        if len(video_paths) < actual_num_samples:
            video_paths.extend([f"unknown_video_{i}.avi" for i in range(len(video_paths), actual_num_samples)])
        else:
            video_paths = video_paths[:actual_num_samples]
    
    print(f"✓ Collected {len(video_paths)} video paths")

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {len(y)}")
    print(f"Unique classes in evaluation set: {len(torch.unique(y))}")
    
    acc = accuracy(y_prob, y, task="multiclass", num_classes=num_classes)
    acc_top5 = accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5)
    f1 = f1_score(y_prob, y, task="multiclass", num_classes=num_classes)
    
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Top-5 Accuracy: {acc_top5:.4f} ({acc_top5*100:.2f}%)")
    print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    
    # AUROC calculation - only compute if we have enough samples per class
    unique_classes = torch.unique(y)
    auroc_score = None
    if len(unique_classes) >= 2 and len(y) >= 100:  # Need at least 2 classes and 100 samples for meaningful AUROC
        try:
            auroc_score = auroc(y_prob, y, task="multiclass", num_classes=num_classes, average="macro")
            print(f"AUROC (macro): {auroc_score:.4f}")
        except Exception as e:
            print(f"AUROC calculation failed: {e}")
            print("  (This is normal with very small evaluation sets or class imbalance)")
    else:
        print(f"AUROC: Skipped (need at least 2 classes and 100 samples for reliable AUROC)")
        print(f"  Current: {len(unique_classes)} classes, {len(y)} samples")

    # Save metrics to JSON
    metrics = {
        "run_name": run_name,
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(y),
        "unique_classes": len(unique_classes),
        "num_classes": num_classes,
        "accuracy": float(acc.item()),
        "accuracy_top5": float(acc_top5.item()),
        "f1_score": float(f1.item()),
        "auroc_macro": float(auroc_score.item()) if auroc_score is not None else None,
        "performance": {
            "total_inference_time_seconds": total_time,
            "total_frames_processed": total_frames,
            "fps": fps,
            "samples_per_second": samples_per_sec,
            "time_per_sample_ms": total_time / len(y) * 1000 if len(y) > 0 else None,
        },
        "model_complexity": {
            "flops_gflops": flops_count / 1e9 if flops_count is not None else None,
        },
        "evaluation_config": {
            "batch_size": batch_size,
            "frames_per_clip": frames_per_clip,
            "video_size": video_size,
            "num_workers": num_workers,
            "seed": seed,
            "eval_sample_size": actual_dataset_size,  # Always use full dataset
        }
    }
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")

    # Save text summary
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"{'='*60}\n")
        f.write("Evaluation Results Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_root}\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Metrics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total samples evaluated: {len(y)}\n")
        f.write(f"Unique classes in evaluation set: {len(unique_classes)}\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy: {acc_top5:.4f} ({acc_top5*100:.2f}%)\n")
        f.write(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)\n")
        if auroc_score is not None:
            f.write(f"AUROC (macro): {auroc_score:.4f}\n")
        else:
            f.write(f"AUROC: Skipped (need at least 2 classes and 100 samples)\n")
        f.write(f"\n{'='*60}\n")
        f.write("Performance\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total inference time: {total_time:.2f} seconds\n")
        f.write(f"FPS (frames per second): {fps:.2f}\n")
        f.write(f"Samples per second: {samples_per_sec:.2f}\n")
        f.write(f"Time per sample: {total_time / len(y) * 1000:.2f} ms\n" if len(y) > 0 else "N/A\n")
        if flops_count is not None:
            f.write(f"Model FLOPS: {flops_count / 1e9:.2f} GFLOPs\n")
        f.write(f"\n{'='*60}\n")
        f.write("Configuration\n")
        f.write(f"{'='*60}\n")
        for key, value in metrics["evaluation_config"].items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Summary saved to: {summary_file}")

    # Save confusion matrix as PNG
    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)
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
    
    # Save per-sample predictions to txt file
    predictions_file = results_dir / "predictions.txt"
    with open(predictions_file, 'w') as f:
        f.write(f"{'Sample':<60} {'Ground Truth':<20} {'Prediction':<20} {'Confidence':<15} {'Correct':<10}\n")
        f.write("=" * 125 + "\n")
        for i in range(len(y)):
            sample_name = os.path.basename(video_paths[i]) if i < len(video_paths) else f"sample_{i}"
            gt_label = labels[y[i].item()]
            pred_label = labels[y_pred[i].item()]
            confidence = confidences[i]
            is_correct = "✓" if y[i].item() == y_pred[i].item() else "✗"
            f.write(f"{sample_name:<60} {gt_label:<20} {pred_label:<20} {confidence:<15.4f} {is_correct:<10}\n")
    print(f"✓ Per-sample predictions saved to: {predictions_file}")
    
    # Create examples directory
    examples_dir = results_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Convert to numpy for easier indexing
    y_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Find incorrect predictions (highest confidence)
    incorrect_mask = y_np != y_pred_np
    incorrect_indices = torch.where(torch.tensor(incorrect_mask))[0]
    num_incorrect = 0
    if len(incorrect_indices) > 0:
        incorrect_confidences = confidences[incorrect_indices]
        num_incorrect = min(5, len(incorrect_indices))
        top_incorrect_idx = incorrect_indices[torch.argsort(torch.tensor(incorrect_confidences), descending=True)[:num_incorrect]]
        
        print(f"\n{'='*60}")
        print(f"Saving top {num_incorrect} highest confidence incorrect predictions...")
        for i, idx in enumerate(top_incorrect_idx):
            idx = idx.item()
            video_path = video_paths[idx]
            gt_label = labels[y_np[idx]]
            pred_label = labels[y_pred_np[idx]]
            confidence = confidences[idx]
            
            # Create filename: name_label_predict_confidence.avi
            video_name = os.path.basename(video_path)
            video_name_no_ext = os.path.splitext(video_name)[0]
            ext = os.path.splitext(video_path)[1]
            # Sanitize labels for filename (remove spaces, special chars)
            gt_label_safe = gt_label.replace(" ", "_").replace("/", "_")
            pred_label_safe = pred_label.replace(" ", "_").replace("/", "_")
            new_filename = f"{video_name_no_ext}_{gt_label_safe}_{pred_label_safe}_{confidence:.4f}{ext}"
            dest_path = examples_dir / f"high_conf_wrong_{i+1}_{new_filename}"
            
            # Copy video file
            if os.path.exists(video_path):
                shutil.copy2(video_path, dest_path)
                print(f"  {i+1}. {os.path.basename(dest_path)}")
            else:
                print(f"  {i+1}. Warning: Video not found: {video_path}")
    
    # Find correct predictions (lowest confidence)
    correct_mask = y_np == y_pred_np
    correct_indices = torch.where(torch.tensor(correct_mask))[0]
    num_correct = 0
    if len(correct_indices) > 0:
        correct_confidences = confidences[correct_indices]
        num_correct = min(5, len(correct_indices))
        bottom_correct_idx = correct_indices[torch.argsort(torch.tensor(correct_confidences), descending=False)[:num_correct]]
        
        print(f"\n{'='*60}")
        print(f"Saving top {num_correct} lowest confidence correct predictions...")
        for i, idx in enumerate(bottom_correct_idx):
            idx = idx.item()
            video_path = video_paths[idx]
            gt_label = labels[y_np[idx]]
            pred_label = labels[y_pred_np[idx]]
            confidence = confidences[idx]
            
            # Create filename: name_label_predict_confidence.avi
            video_name = os.path.basename(video_path)
            video_name_no_ext = os.path.splitext(video_name)[0]
            ext = os.path.splitext(video_path)[1]
            # Sanitize labels for filename (remove spaces, special chars)
            gt_label_safe = gt_label.replace(" ", "_").replace("/", "_")
            pred_label_safe = pred_label.replace(" ", "_").replace("/", "_")
            new_filename = f"{video_name_no_ext}_{gt_label_safe}_{pred_label_safe}_{confidence:.4f}{ext}"
            dest_path = examples_dir / f"low_conf_right_{i+1}_{new_filename}"
            
            # Copy video file
            if os.path.exists(video_path):
                shutil.copy2(video_path, dest_path)
                print(f"  {i+1}. {os.path.basename(dest_path)}")
            else:
                print(f"  {i+1}. Warning: Video not found: {video_path}")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {results_dir}")
    print(f"{'='*60}")
    print(f"\nSaved files:")
    print(f"  ✓ predictions.txt: Per-sample predictions with ground truth, prediction, confidence")
    print(f"  ✓ confusion_matrix.csv: Confusion matrix in CSV format")
    print(f"  ✓ confusion_matrix.png: Confusion matrix visualization")
    print(f"  ✓ metrics.json: Evaluation metrics in JSON format")
    print(f"  ✓ summary.txt: Text summary of evaluation results")
    print(f"  ✓ examples/: Directory with example videos")
    if len(incorrect_indices) > 0:
        print(f"    - high_conf_wrong_*.avi: Top {num_incorrect} highest confidence incorrect predictions")
    if len(correct_indices) > 0:
        print(f"    - low_conf_right_*.avi: Top {num_correct} lowest confidence correct predictions")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
