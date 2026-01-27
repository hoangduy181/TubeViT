# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import time
import torch
from tubevit.video_loader import EncodedVideo
from tubevit.transforms import ApplyTransformToKey, UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo

from tubevit.model import TubeViTLightningModule
from utils.constant import IMAGENET_MEAN, IMAGENET_STD

# Enable Tensor Core optimization for NVIDIA GPUs with Tensor Cores (e.g., A100, V100, etc.)
# 'medium' provides a good balance between performance and precision
# Use 'high' for better precision if needed
torch.set_float32_matmul_precision('medium')


@click.command()
@click.argument("video-path")
@click.option("-m", "--model-path", type=click.Path(exists=True), required=True, help="path to model weight.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="path to classInd.txt.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
def main(
        video_path,
        model_path,
        label_path,
        frames_per_clip,
        video_size,
):
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))

    # Compose video data transforms
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(frames_per_clip),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(
                    mean=IMAGENET_MEAN, 
                    std=IMAGENET_STD
                ),
                ShortSideScale(
                    size=video_size[0]
                ),
                CenterCropVideo(crop_size=video_size)
            ]
        ),
    )

    # Load video
    video = EncodedVideo.from_path(video_path)
    # Get clip
    clip_start_sec = 0.0  # secs
    clip_duration = 2.0  # secs
    duration = video.duration
    video_data = []
    for i in range(10):
        if clip_start_sec + clip_duration * (i + 1) <= duration:
            data = video.get_clip(start_sec=clip_start_sec + clip_duration * i,
                                  end_sec=clip_start_sec + clip_duration * (i + 1))
            data = transform(data)
            video_data.append(data['video'])

    video_data = torch.stack(video_data)
    print(f"Video data shape: {video_data.shape}")
    
    # Load model from checkpoint
    # Works with both .ckpt and .pt files (if .pt was saved as full checkpoint)
    model = TubeViTLightningModule.load_from_checkpoint(model_path)
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates, etc.)
    
    # Calculate FLOPS if possible
    flops_count = None
    try:
        try:
            from fvcore.nn import FlopCountMode, flop_count
            dummy_input = torch.randn(1, *video_data.shape[1:]).to(next(model.parameters()).device)
            flops_dict, _ = flop_count(model.model, (dummy_input,), mode=FlopCountMode.OPERATION_COUNT)
            flops_count = sum(flops_dict.values())
        except ImportError:
            try:
                from thop import profile
                dummy_input = torch.randn(1, *video_data.shape[1:]).to(next(model.parameters()).device)
                flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
                flops_count = flops
            except ImportError:
                pass
    except Exception:
        pass
    
    if flops_count is not None:
        print(f"\n{'='*60}")
        print("Model Complexity")
        print(f"{'='*60}")
        print(f"FLOPS: {flops_count / 1e9:.2f} GFLOPs")
    
    # Measure inference time
    print(f"\n{'='*60}")
    print("Running Inference...")
    print(f"{'='*60}")
    
    # Warmup
    with torch.no_grad():
        _ = model.predict_step(batch=(video_data[:1], None), batch_idx=0)
    
    # Actual inference with timing
    start_time = time.time()
    with torch.no_grad():
        prediction = model.predict_step(batch=(video_data, None), batch_idx=0)
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_frames = video_data.shape[0] * frames_per_clip
    fps = total_frames / inference_time if inference_time > 0 else 0
    
    # Aggregate predictions across all clips and get the final prediction
    # prediction['y_prob'] shape: (num_clips, num_classes)
    aggregated_prob = torch.sum(prediction['y_prob'], dim=0)  # Sum across clips: (num_classes,)
    predicted_class_idx = torch.argmax(aggregated_prob).item()
    predicted_label = labels[predicted_class_idx]
    
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {aggregated_prob[predicted_class_idx].item():.4f}")
    print(f"\n{'='*60}")
    print("Performance")
    print(f"{'='*60}")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Total frames processed: {total_frames}")
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Time per clip: {inference_time / video_data.shape[0] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
