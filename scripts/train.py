import sys
from pathlib import Path

from utils.constant import IMAGENET_MEAN, IMAGENET_STD

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import pickle


import click
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from tubevit.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from tubevit.dataset import MyUCF101
from tubevit.model import TubeViTLightningModule
from tubevit.gpu_monitor import GPUMonitorCallback
from lightning.pytorch.callbacks import ModelCheckpoint

# Enable Tensor Core optimization for NVIDIA GPUs with Tensor Cores (e.g., A100, V100, etc.)
# 'medium' provides a good balance between performance and precision
# Use 'high' for better precision if needed
torch.set_float32_matmul_precision('medium')


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-nc", "--num-classes", type=int, default=101, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=32, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--max-epochs", type=int, default=10, help="max epochs.")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers. Defaults to number of CPUs.")
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--preview-video", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
def main(
    dataset_root,
    annotation_path,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    fast_dev_run,
    seed,
    preview_video,
):
    pl.seed_everything(seed)

    # Set num_workers to number of CPUs if not specified
    if num_workers is None:
        num_workers = os.cpu_count() or 0
        print(f"Using {num_workers} DataLoader workers (auto-detected from CPU count)")

    train_transform = T.Compose(
        [
            ToTensorVideo(),  # C, T, H, W
            Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
            RandAugment(magnitude=10, num_layers=2),
            Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    print(f"\n{'='*60}")
    print("Loading Training Dataset")
    print(f"{'='*60}")
    print(f"Dataset root: {dataset_root}")
    print(f"Annotation path: {annotation_path}")
    print(f"Frames per clip: {frames_per_clip}")
    print(f"Video size: {video_size}")
    print(f"Batch size: {batch_size}")
    
    train_metadata_file = "ucf101-train-meta.pickle"
    train_precomputed_metadata = None
    if os.path.exists(train_metadata_file):
        print(f"Loading precomputed metadata from {train_metadata_file}...")
        with open(train_metadata_file, "rb") as f:
            train_precomputed_metadata = pickle.load(f)
        print("✓ Metadata loaded")
    else:
        print(f"No precomputed metadata found. Will create {train_metadata_file} after loading dataset.")

    print("\nInitializing training dataset...")
    train_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=train_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=True,
        output_format="THWC",
        transform=train_transform,
    )
    print(f"✓ Training dataset loaded: {len(train_set)} samples")

    if not os.path.exists(train_metadata_file):
        print(f"Saving metadata to {train_metadata_file}...")
        with open(train_metadata_file, "wb") as f:
            pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Metadata saved")

    print(f"\n{'='*60}")
    print("Loading Validation Dataset")
    print(f"{'='*60}")
    val_metadata_file = "ucf101-val-meta.pickle"
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        print(f"Loading precomputed metadata from {val_metadata_file}...")
        with open(val_metadata_file, "rb") as f:
            val_precomputed_metadata = pickle.load(f)
        print("✓ Metadata loaded")
    else:
        print(f"No precomputed metadata found. Will create {val_metadata_file} after loading dataset.")

    print("\nInitializing validation dataset...")
    val_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=val_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=False,
        output_format="THWC",
        transform=test_transform,
    )
    print(f"✓ Validation dataset loaded: {len(val_set)} samples")

    if not os.path.exists(val_metadata_file):
        print(f"Saving metadata to {val_metadata_file}...")
        with open(val_metadata_file, "wb") as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Metadata saved")

    print(f"\n{'='*60}")
    print("Creating DataLoaders")
    print(f"{'='*60}")
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    print(f"✓ Training DataLoader created: {len(train_dataloader)} batches")

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    print(f"✓ Validation DataLoader created: {len(val_dataloader)} batches")

    print("\nTesting data loading...")
    x, y = next(iter(train_dataloader))
    print(f"✓ Sample batch shape: {x.shape}")
    print(f"✓ Sample labels shape: {y.shape}")

    if preview_video:
        x = x.permute(0, 2, 3, 4, 1)
        fig, axs = plt.subplots(4, 8)
        for i in range(4):
            for j in range(8):
                axs[i][j].imshow(x[0][i * 8 + j])
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
        plt.tight_layout()
        plt.show()

    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        lr=1e-4,
        weight_decay=0.001,
        weight_path="tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt",
        max_epochs=max_epochs,
        # Additional training parameters to save in hparams.yaml
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        video_size=video_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Setup callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        GPUMonitorCallback(log_every_n_steps=50, log_every_n_epochs=1),  # Monitor GPU every 50 steps
        ModelCheckpoint(
            dirpath="./models/checkpoints",
            filename="tubevit-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,  # Save top 3 models
            mode="min",
            save_last=True,  # Always save last checkpoint
            every_n_epochs=1,
        )
    ]
    logger = TensorBoardLogger("logs", name="TubeViT")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed", # Use mixed precision to halve memory usage
        limit_val_batches=0.5
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint("./models/tubevit_ucf101.ckpt")


if __name__ == "__main__":
    main()
