# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
from datetime import datetime

import click
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import seaborn as sns
import torch
from pytorchvideo.transforms import Normalize
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score, precision, recall
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from tubevit.dataset import MyUCF101
from tubevit.model import TubeViTLightningModule

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).stem
        run_name = f"eval_{model_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Evaluation Run: {run_name}")
    print(f"{'='*60}")

    # Create results directory
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

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
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    # val_sampler = RandomSampler(val_set, num_samples=len(val_set) // 50)
    val_sampler = SequentialSampler(val_set)
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

    trainer = pl.Trainer(accelerator="auto", default_root_dir="lightning_predict_logs")
    predictions = trainer.predict(model, dataloaders=val_dataloader)

    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])

    print("accuracy:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes))
    print("accuracy_top5:", accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5))
    print("auroc:", auroc(y_prob, y, task="multiclass", num_classes=num_classes))
    print("f1_score:", f1_score(y_prob, y, task="multiclass", num_classes=num_classes))

    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)
    
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
    import csv
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
    
    acc = accuracy(y_prob, y, task="multiclass", num_classes=num_classes)
    
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
