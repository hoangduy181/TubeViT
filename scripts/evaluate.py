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

from torch.utils.data import DataLoader, SequentialSampler, Subset
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
    print(f"  Average clips per video: {actual_dataset_size / len(val_set.samples):.2f}")
    
    # Filter to single clip per video if requested
    if single_clip_per_video:
        print(f"  Evaluation mode: Single clip per video (1 clip per video)")
        # Find the first clip index for each video
        video_to_first_clip = {}  # video_idx -> first clip_idx
        for clip_idx in range(actual_dataset_size):
            if clip_idx >= len(val_set.indices):
                continue
            video_idx = val_set.indices[clip_idx]
            if video_idx not in video_to_first_clip:
                video_to_first_clip[video_idx] = clip_idx
        
        # Create list of clip indices to use (first clip of each video)
        clip_indices_to_use = sorted(video_to_first_clip.values())
        print(f"  Using {len(clip_indices_to_use)} clips (1 per video)")
        
        # Create a custom dataset that only returns the first clip of each video
        val_dataset = Subset(val_set, clip_indices_to_use)
    else:
        print(f"  Evaluation mode: Video-level (will aggregate predictions from multiple clips per video)")
        print(f"  Evaluating on: {actual_dataset_size} clips from {len(val_set.samples)} videos (full dataset)")
        val_dataset = val_set
    
    # Use SequentialSampler for deterministic evaluation (no shuffling)
    val_sampler = SequentialSampler(val_dataset)
    
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
    
    # Collect all clip-level predictions
    y_clips = torch.cat([item["y"] for item in predictions])
    y_pred_clips = torch.cat([item["y_pred"] for item in predictions])
    y_prob_clips = torch.cat([item["y_prob"] for item in predictions])
    
    # Calculate performance metrics (clip-level)
    total_time = end_time - start_time
    total_frames = len(y_clips) * frames_per_clip  # Total frames processed
    fps = total_frames / total_time if total_time > 0 else 0
    samples_per_sec = len(y_clips) / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("Performance Metrics (Clip-level)")
    print(f"{'='*60}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Total clips processed: {len(y_clips)}")
    print(f"Total frames processed: {total_frames}")
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Clips per second: {samples_per_sec:.2f}")
    print(f"Time per clip: {total_time / len(y_clips) * 1000:.2f} ms" if len(y_clips) > 0 else "N/A")
    
    actual_num_clips = len(y_clips)
    
    if single_clip_per_video:
        # Single clip per video mode: no aggregation needed
        print(f"\n{'='*60}")
        print("Single clip per video mode (no aggregation)")
        print(f"{'='*60}")
        
        # Get video paths for each clip
        video_paths = []
        for i in range(len(y_clips)):
            try:
                # Get the clip index from the subset
                if isinstance(val_dataset, Subset):
                    clip_idx = val_dataset.indices[i]
                else:
                    clip_idx = i
                
                if clip_idx >= len(val_set.indices):
                    video_paths.append(f"unknown_video_{i}.avi")
                    continue
                
                video_idx = val_set.indices[clip_idx]
                if video_idx >= len(val_set.samples):
                    video_paths.append(f"unknown_video_{i}.avi")
                    continue
                
                video_path, _ = val_set.samples[video_idx]
                if not os.path.isabs(video_path):
                    video_path = os.path.join(val_set.root, video_path)
                video_paths.append(video_path)
            except (IndexError, KeyError, AttributeError) as e:
                video_paths.append(f"unknown_video_{i}.avi")
        
        # Use clip predictions directly (1 clip = 1 video)
        y = y_clips
        y_prob = y_prob_clips
        y_pred = y_pred_clips
        
        print(f"✓ Using {len(y)} video predictions (1 clip per video)")
    else:
        # Multiple clips per video mode: aggregate predictions
        print(f"\n{'='*60}")
        print("Aggregating predictions per video...")
        print(f"{'='*60}")
        
        # Group clips by video index
        video_to_clips = {}  # video_idx -> list of (clip_idx, y, y_prob)
        video_paths_map = {}  # video_idx -> video_path
        
        print(f"Processing {actual_num_clips} clips...")
        
        for clip_idx in range(actual_num_clips):
            try:
                # Map clip index to video index using val_set.indices
                if clip_idx >= len(val_set.indices):
                    print(f"Warning: clip_idx {clip_idx} >= len(val_set.indices) {len(val_set.indices)}")
                    continue
                
                video_idx = val_set.indices[clip_idx]
                
                # Safety check: ensure video_idx is within bounds
                if video_idx >= len(val_set.samples):
                    print(f"Warning: video_idx {video_idx} >= len(val_set.samples) {len(val_set.samples)}")
                    continue
                
                # Store video path (same for all clips from this video)
                if video_idx not in video_paths_map:
                    video_path, _ = val_set.samples[video_idx]
                    # Convert to absolute path if relative
                    if not os.path.isabs(video_path):
                        video_path = os.path.join(val_set.root, video_path)
                    video_paths_map[video_idx] = video_path
                
                # Group clip predictions by video
                if video_idx not in video_to_clips:
                    video_to_clips[video_idx] = []
                video_to_clips[video_idx].append({
                    'clip_idx': clip_idx,
                    'y': y_clips[clip_idx],
                    'y_prob': y_prob_clips[clip_idx],
                })
                
            except (IndexError, KeyError, AttributeError) as e:
                print(f"Warning: Could not process clip {clip_idx}: {e}")
                continue
        
        # Aggregate predictions per video (average probabilities)
        print(f"Aggregating {len(video_to_clips)} videos from {actual_num_clips} clips...")
        
        video_indices = sorted(video_to_clips.keys())
        y_videos = []
        y_prob_videos = []
        video_paths = []
        
        for video_idx in video_indices:
            clips_data = video_to_clips[video_idx]
            
            # Get ground truth label (should be same for all clips from same video)
            gt_label = clips_data[0]['y']
            
            # Average probabilities across all clips from this video
            prob_tensors = [clip['y_prob'] for clip in clips_data]
            avg_prob = torch.stack(prob_tensors).mean(dim=0)
            
            y_videos.append(gt_label)
            y_prob_videos.append(avg_prob)
            video_paths.append(video_paths_map[video_idx])
        
        # Convert to tensors
        y = torch.stack(y_videos)
        y_prob = torch.stack(y_prob_videos)
        y_pred = torch.argmax(y_prob, dim=1)
        
        print(f"✓ Aggregated to {len(y)} video-level predictions")
        print(f"  Average clips per video: {actual_num_clips / len(y):.2f}")
    
    # Get confidence scores (max probability)
    confidences = torch.max(y_prob, dim=1)[0].cpu().numpy()

    print(f"\n{'='*60}")
    eval_mode = "Single clip per video" if single_clip_per_video else "Video-level (aggregated)"
    print(f"Evaluation Results ({eval_mode})")
    print(f"{'='*60}")
    print(f"Total videos evaluated: {len(y)}")
    print(f"Total clips processed: {actual_num_clips}")
    if not single_clip_per_video:
        print(f"Average clips per video: {actual_num_clips / len(y):.2f}")
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
        "evaluation_level": "single_clip" if single_clip_per_video else "video_aggregated",  # Evaluation mode
        "single_clip_per_video": single_clip_per_video,
        "total_videos": len(y),
        "total_clips_processed": actual_num_clips,
        "average_clips_per_video": actual_num_clips / len(y) if len(y) > 0 and not single_clip_per_video else 1.0,
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
            "clips_per_second": samples_per_sec,
            "time_per_clip_ms": total_time / actual_num_clips * 1000 if actual_num_clips > 0 else None,
            "time_per_video_ms": total_time / len(y) * 1000 if len(y) > 0 else None,
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
            "aggregation_method": "none" if single_clip_per_video else "average_probabilities",  # How clips are aggregated per video
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
        eval_mode_desc = "Single clip per video" if single_clip_per_video else "Video-level (aggregated from clips)"
        f.write(f"Evaluation level: {eval_mode_desc}\n")
        f.write(f"Total videos evaluated: {len(y)}\n")
        f.write(f"Total clips processed: {actual_num_clips}\n")
        f.write(f"Average clips per video: {actual_num_clips / len(y):.2f}\n")
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
        f.write(f"Clips per second: {samples_per_sec:.2f}\n")
        f.write(f"Time per clip: {total_time / actual_num_clips * 1000:.2f} ms\n" if actual_num_clips > 0 else "N/A\n")
        f.write(f"Time per video: {total_time / len(y) * 1000:.2f} ms\n" if len(y) > 0 else "N/A\n")
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
    
    # Save per-video predictions to txt file
    predictions_file = results_dir / "predictions.txt"
    with open(predictions_file, 'w') as f:
        f.write(f"{'Video':<60} {'Ground Truth':<20} {'Prediction':<20} {'Confidence':<15} {'Correct':<10}\n")
        f.write("=" * 125 + "\n")
        for i in range(len(y)):
            video_name = os.path.basename(video_paths[i]) if i < len(video_paths) else f"video_{i}"
            gt_label = labels[y[i].item()]
            pred_label = labels[y_pred[i].item()]
            confidence = confidences[i]
            is_correct = "✓" if y[i].item() == y_pred[i].item() else "✗"
            f.write(f"{video_name:<60} {gt_label:<20} {pred_label:<20} {confidence:<15.4f} {is_correct:<10}\n")
    print(f"✓ Per-video predictions saved to: {predictions_file}")
    
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
    print(f"  ✓ predictions.txt: Per-video predictions (aggregated from clips) with ground truth, prediction, confidence")
    print(f"  ✓ confusion_matrix.csv: Confusion matrix in CSV format (video-level)")
    print(f"  ✓ confusion_matrix.png: Confusion matrix visualization (video-level)")
    print(f"  ✓ metrics.json: Evaluation metrics in JSON format")
    print(f"  ✓ summary.txt: Text summary of evaluation results")
    print(f"  ✓ examples/: Directory with example videos")
    if len(incorrect_indices) > 0:
        print(f"    - high_conf_wrong_*.avi: Top {num_incorrect} highest confidence incorrect predictions")
    if len(correct_indices) > 0:
        print(f"    - low_conf_right_*.avi: Top {num_correct} lowest confidence correct predictions")
    if single_clip_per_video:
        print(f"\nNote: Evaluation uses 1 clip per video (faster but less robust).")
        print(f"      {actual_num_clips} clips were processed (1 clip = 1 video).")
    else:
        print(f"\nNote: Evaluation is performed at video-level by aggregating predictions from multiple clips per video.")
        print(f"      {actual_num_clips} clips were processed and aggregated into {len(y)} video-level predictions.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
