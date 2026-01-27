# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
import json
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

# Enable Tensor Core optimization for NVIDIA GPUs with Tensor Cores (e.g., A100, V100, etc.)
# 'medium' provides a good balance between performance and precision
# Use 'high' for better precision if needed
torch.set_float32_matmul_precision('medium')


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-m", "--model-path", type=click.Path(exists=True), required=True, help="path to model weight.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="path to classInd.txt.")
@click.option("-nc", "--num-classes", type=int, default=101, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=32, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers. Defaults to number of CPUs.")
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--verbose", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option("--run-name", type=str, default=None, help="Name for this evaluation run. If not provided, will be auto-generated.")
def main(
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

    val_metadata_file = "ucf101-val-meta.pickle"
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

    # Use a reasonable sample size for evaluation (or use all samples)
    # For proper evaluation, use all validation samples or at least 1000+
    eval_sample_size = min(len(val_set), max(1000, len(val_set) // 10))  # At least 1000 samples or 10% of dataset
    print(f"Evaluating on {eval_sample_size} samples (out of {len(val_set)} total validation samples)")
    
    # Use SequentialSampler for deterministic evaluation (no shuffling)
    # Create a subset by taking first N indices
    from torch.utils.data import Subset
    if eval_sample_size < len(val_set):
        # Create subset with first eval_sample_size samples
        indices = list(range(eval_sample_size))
        val_subset = Subset(val_set, indices)
        val_sampler = None  # No sampler needed, Subset handles indexing
        val_dataset = val_subset
    else:
        # Use full dataset
        val_sampler = SequentialSampler(val_set)
        val_dataset = val_set
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # No shuffling for evaluation
        drop_last=False,  # Don't drop last batch in evaluation
        sampler=val_sampler,  # None if using Subset, SequentialSampler if using full dataset
    )

    x, y = next(iter(val_dataloader))
    print(x.shape)

    model = TubeViTLightningModule.load_from_checkpoint(model_path)

    trainer = pl.Trainer(accelerator="auto", default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)

    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])

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
        "evaluation_config": {
            "batch_size": batch_size,
            "frames_per_clip": frames_per_clip,
            "video_size": video_size,
            "num_workers": num_workers,
            "seed": seed,
            "eval_sample_size": eval_sample_size,
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
        f.write("Configuration\n")
        f.write(f"{'='*60}\n")
        for key, value in metrics["evaluation_config"].items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Summary saved to: {summary_file}")

    # Save confusion matrix
    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)
    cm_file = results_dir / "confusion_matrix.png"
    
    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix - {run_name}")
    plt.tight_layout()
    plt.savefig(cm_file, dpi=300)
    print(f"✓ Confusion matrix saved to: {cm_file}")
    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {results_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
