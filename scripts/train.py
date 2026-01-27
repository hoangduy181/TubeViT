# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
from datetime import datetime
from utils.constant import IMAGENET_MEAN, IMAGENET_STD
from utils.config_loader import load_config, merge_config_with_args, get_config_value

# Suppress CUDA/XLA warnings in distributed training
# These warnings occur when multiple processes initialize CUDA libraries
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/XLA warnings
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

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
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to YAML configuration file. CLI arguments override config values.")
@click.option("-r", "--dataset-root", type=click.Path(exists=True), default=None, help="path to dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), default=None, help="path to dataset.")
@click.option("-nc", "--num-classes", type=int, default=None, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=None, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=None, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=None, help="frame per clip.")
@click.option("--max-epochs", type=int, default=None, help="max epochs.")
@click.option("--num-workers", type=int, default=None, help="Number of DataLoader workers. Defaults to number of CPUs.")
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=None, help="random seed.")
@click.option("--preview-video", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option("--run-name", type=str, default=None, help="Name for this training run. If not provided, will be auto-generated.")
def main(
    config,
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
        'annotation_path': annotation_path,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'frames_per_clip': frames_per_clip,
        'video_size': video_size,
        'max_epochs': max_epochs,
        'num_workers': num_workers,
        'fast_dev_run': fast_dev_run,
        'seed': seed,
        'preview_video': preview_video,
        'run_name': run_name,
    }
    
    # Merge config with CLI args (CLI args take precedence)
    merged_config = merge_config_with_args(cfg, cli_args)
    
    # Extract values from merged config (support nested structure)
    dataset_root = get_config_value(merged_config, 'dataset_root') or get_config_value(merged_config, 'dataset.root')
    annotation_path = get_config_value(merged_config, 'annotation_path') or get_config_value(merged_config, 'dataset.annotation_path')
    num_classes = get_config_value(merged_config, 'num_classes') or get_config_value(merged_config, 'dataset.num_classes', 101)
    batch_size = get_config_value(merged_config, 'batch_size') or get_config_value(merged_config, 'training.batch_size', 32)
    frames_per_clip = get_config_value(merged_config, 'frames_per_clip') or get_config_value(merged_config, 'training.frames_per_clip', 32)
    video_size = get_config_value(merged_config, 'video_size') or get_config_value(merged_config, 'training.video_size', (224, 224))
    max_epochs = get_config_value(merged_config, 'max_epochs') or get_config_value(merged_config, 'training.max_epochs', 10)
    num_workers = get_config_value(merged_config, 'num_workers') or get_config_value(merged_config, 'training.num_workers')
    seed = get_config_value(merged_config, 'seed') or get_config_value(merged_config, 'training.seed', 42)
    run_name = get_config_value(merged_config, 'run_name')
    
    # Validate required parameters
    if not dataset_root:
        raise ValueError("dataset_root is required. Provide via --dataset-root or config file (dataset.root)")
    if not annotation_path:
        raise ValueError("annotation_path is required. Provide via --annotation-path or config file (dataset.annotation_path)")
    
    # Convert video_size tuple if it's a list from YAML
    if isinstance(video_size, list):
        video_size = tuple(video_size)
    pl.seed_everything(seed)

    # Generate run name if not provided
    if run_name is None:
        # Generate timestamp-based run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"
    
    # Create directory structure for this run
    models_dir = Path("models") / run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs") / run_name
    
    print(f"\n{'='*60}")
    print(f"Training Run: {run_name}")
    print(f"Models will be saved to: {models_dir}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"Logs will be saved to: {logs_dir}")
    print(f"{'='*60}\n")

    # Set num_workers to number of CPUs if not specified
    if num_workers is None:
        num_workers = os.cpu_count() or 0
        print(f"Using {num_workers} DataLoader workers (auto-detected from CPU count)")

    # Get augmentation parameters from config
    rand_aug_magnitude = get_config_value(merged_config, 'augmentation.rand_augment.magnitude', 10)
    rand_aug_num_layers = get_config_value(merged_config, 'augmentation.rand_augment.num_layers', 2)
    
    train_transform = T.Compose(
        [
            ToTensorVideo(),  # C, T, H, W
            Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
            RandAugment(magnitude=rand_aug_magnitude, num_layers=rand_aug_num_layers),
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
    
    train_metadata_file = get_config_value(merged_config, 'metadata.train_file', "ucf101-train-meta.pickle")
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
    val_metadata_file = get_config_value(merged_config, 'metadata.val_file', "ucf101-val-meta.pickle")
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

    # Get model and training parameters from config
    num_layers = get_config_value(merged_config, 'model.num_layers', 12)
    num_heads = get_config_value(merged_config, 'model.num_heads', 12)
    hidden_dim = get_config_value(merged_config, 'model.hidden_dim', 768)
    mlp_dim = get_config_value(merged_config, 'model.mlp_dim', 3072)
    dropout = get_config_value(merged_config, 'model.dropout', 0.0)
    attention_dropout = get_config_value(merged_config, 'model.attention_dropout', 0.0)
    lr = get_config_value(merged_config, 'training.lr', 1e-4)
    weight_decay = get_config_value(merged_config, 'training.weight_decay', 0.001)
    label_smoothing = get_config_value(merged_config, 'training.label_smoothing', 0.0)
    weight_path = get_config_value(merged_config, 'paths.pretrained_weight', "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt")
    
    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        weight_path=weight_path,
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
            dirpath=str(checkpoints_dir),
            filename="tubevit-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,  # Save top 3 models
            mode="min",
            save_last=True,  # Always save last checkpoint
            every_n_epochs=1,
        )
    ]
    logger = TensorBoardLogger(str(logs_dir.parent), name=run_name)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"\n{'='*60}")
    print("GPU Detection")
    print(f"{'='*60}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {gpu_count}")
    
    if gpu_count > 0:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    
    if gpu_count > 1:
        devices = -1  # Use all GPUs
        strategy = "ddp"  # Distributed Data Parallel
        print(f"\nUsing multi-GPU training: {gpu_count} GPUs with DDP strategy")
    elif gpu_count == 1:
        devices = 1
        strategy = "auto"
        print(f"\nUsing single-GPU training")
    else:
        devices = "auto"
        strategy = "auto"
        print(f"\nUsing CPU training")

    print(f"Trainer setup: accelerator=auto, devices={devices}, strategy={strategy}")
    print(f"{'='*60}\n")
    # Get training precision and limit_val_batches from config
    precision = get_config_value(merged_config, 'training.precision', "16-mixed")
    limit_val_batches = get_config_value(merged_config, 'training.limit_val_batches', 0.5)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
        limit_val_batches=limit_val_batches
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("Training complete")
    
    # Save final checkpoint in the run directory
    final_checkpoint_path = models_dir / f"{run_name}_final.ckpt"
    trainer.save_checkpoint(str(final_checkpoint_path))
    print(f"✓ Final checkpoint saved to: {final_checkpoint_path}")
    print(f"✓ All outputs saved under run: {run_name}")


if __name__ == "__main__":
    main()
