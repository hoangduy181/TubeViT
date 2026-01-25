"""
GPU Monitoring Callback for PyTorch Lightning
Tracks GPU memory usage and utilization during training/validation
"""
import torch
from lightning.pytorch.callbacks import Callback


def print_gpu_usage():
    """
    Utility function to print current GPU usage.
    Can be called manually at any point to check GPU status.
    """
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("\n" + "="*60)
    print("Current GPU Usage")
    print("="*60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
        total_memory = props.total_memory / 1024**3  # GB
        
        allocated_pct = (allocated / total_memory) * 100
        reserved_pct = (reserved / total_memory) * 100
        max_allocated_pct = (max_allocated / total_memory) * 100
        free_pct = 100 - reserved_pct
        
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated_pct:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved_pct:.1f}%)")
        print(f"  Max Allocated: {max_allocated:.2f} GB ({max_allocated_pct:.1f}%)")
        print(f"  Free: {total_memory - reserved:.2f} GB ({free_pct:.1f}%)")
        print()
    
    print("="*60)


class GPUMonitorCallback(Callback):
    """
    Monitor GPU memory usage and utilization during training.
    Logs metrics to TensorBoard and prints to console.
    """
    
    def __init__(self, log_every_n_steps=50, log_every_n_epochs=1):
        """
        Args:
            log_every_n_steps: Log GPU stats every N training steps
            log_every_n_epochs: Log GPU stats every N epochs
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
    
    def _get_gpu_stats(self):
        """Get current GPU memory and utilization stats"""
        if not torch.cuda.is_available():
            return None
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            # Memory stats
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            max_reserved = torch.cuda.max_memory_reserved(i) / 1024**3   # GB
            
            # Get total memory
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            # Calculate percentages
            allocated_pct = (allocated / total_memory) * 100
            reserved_pct = (reserved / total_memory) * 100
            max_allocated_pct = (max_allocated / total_memory) * 100
            
            stats[f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,
                'total_gb': total_memory,
                'allocated_pct': allocated_pct,
                'reserved_pct': reserved_pct,
                'max_allocated_pct': max_allocated_pct,
                'free_gb': total_memory - reserved,
                'free_pct': 100 - reserved_pct,
            }
        
        return stats
    
    def _format_gpu_stats(self, stats, prefix=""):
        """Format GPU stats for printing"""
        if not stats:
            return "No GPU available"
        
        lines = []
        for gpu_name, gpu_stats in stats.items():
            gpu_id = gpu_name.split('_')[1]
            lines.append(
                f"{prefix}GPU {gpu_id}: "
                f"Allocated: {gpu_stats['allocated_gb']:.2f}GB ({gpu_stats['allocated_pct']:.1f}%), "
                f"Reserved: {gpu_stats['reserved_gb']:.2f}GB ({gpu_stats['reserved_pct']:.1f}%), "
                f"Max: {gpu_stats['max_allocated_gb']:.2f}GB ({gpu_stats['max_allocated_pct']:.1f}%), "
                f"Free: {gpu_stats['free_gb']:.2f}GB ({gpu_stats['free_pct']:.1f}%)"
            )
        return "\n".join(lines)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log GPU stats during training"""
        if batch_idx % self.log_every_n_steps == 0:
            stats = self._get_gpu_stats()
            if stats:
                for gpu_name, gpu_stats in stats.items():
                    gpu_id = gpu_name.split('_')[1]
                    # Log to TensorBoard
                    pl_module.log(f"gpu_{gpu_id}/memory_allocated_gb", gpu_stats['allocated_gb'], on_step=True, on_epoch=False)
                    pl_module.log(f"gpu_{gpu_id}/memory_reserved_gb", gpu_stats['reserved_gb'], on_step=True, on_epoch=False)
                    pl_module.log(f"gpu_{gpu_id}/memory_allocated_pct", gpu_stats['allocated_pct'], on_step=True, on_epoch=False)
                    pl_module.log(f"gpu_{gpu_id}/memory_reserved_pct", gpu_stats['reserved_pct'], on_step=True, on_epoch=False)
                    pl_module.log(f"gpu_{gpu_id}/memory_free_pct", gpu_stats['free_pct'], on_step=True, on_epoch=False)
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Print GPU stats at start of training epoch"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n[GPU Monitor] Training Epoch {trainer.current_epoch} Start:")
            print(self._format_gpu_stats(stats, "  "))
            # Reset max memory tracking
            torch.cuda.reset_peak_memory_stats()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print GPU stats at end of training epoch"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n[GPU Monitor] Training Epoch {trainer.current_epoch} End:")
            print(self._format_gpu_stats(stats, "  "))
            # Log max memory to TensorBoard
            for gpu_name, gpu_stats in stats.items():
                gpu_id = gpu_name.split('_')[1]
                pl_module.log(f"gpu_{gpu_id}/max_memory_allocated_gb", gpu_stats['max_allocated_gb'], on_step=False, on_epoch=True)
                pl_module.log(f"gpu_{gpu_id}/max_memory_allocated_pct", gpu_stats['max_allocated_pct'], on_step=False, on_epoch=True)
    
    def on_validation_start(self, trainer, pl_module):
        """Print GPU stats at start of validation"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n[GPU Monitor] Validation Start (Epoch {trainer.current_epoch}):")
            print(self._format_gpu_stats(stats, "  "))
            # Clear cache before validation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def on_validation_end(self, trainer, pl_module):
        """Print GPU stats at end of validation"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n[GPU Monitor] Validation End (Epoch {trainer.current_epoch}):")
            print(self._format_gpu_stats(stats, "  "))
            # Log validation GPU stats directly to logger (can't use pl_module.log in on_validation_end)
            if trainer.logger and hasattr(trainer.logger, "experiment") and trainer.is_global_zero:
                for gpu_name, gpu_stats in stats.items():
                    gpu_id = gpu_name.split('_')[1]
                    trainer.logger.experiment.add_scalar(
                        f"gpu_{gpu_id}/val_memory_allocated_gb",
                        gpu_stats["allocated_gb"],
                        trainer.global_step,
                    )
                    trainer.logger.experiment.add_scalar(
                        f"gpu_{gpu_id}/val_memory_reserved_gb",
                        gpu_stats["reserved_gb"],
                        trainer.global_step,
                    )
    
    def on_train_start(self, trainer, pl_module):
        """Print initial GPU stats at training start"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n{'='*60}")
            print("GPU Information")
            print(f"{'='*60}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"{'='*60}\n")
            
            # Initial stats
            stats = self._get_gpu_stats()
            print("[GPU Monitor] Initial GPU State:")
            print(self._format_gpu_stats(stats, "  "))
    
    def on_train_end(self, trainer, pl_module):
        """Print final GPU stats at training end"""
        stats = self._get_gpu_stats()
        if stats:
            print(f"\n[GPU Monitor] Training Complete - Final GPU State:")
            print(self._format_gpu_stats(stats, "  "))
            # Log final max memory
            for gpu_name, gpu_stats in stats.items():
                gpu_id = gpu_name.split('_')[1]
                print(f"  GPU {gpu_id} Peak Memory: {gpu_stats['max_allocated_gb']:.2f}GB ({gpu_stats['max_allocated_pct']:.1f}%)")
