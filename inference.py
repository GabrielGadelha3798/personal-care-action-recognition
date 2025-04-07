import os
import json
import argparse
import torch
import cv2
from collections import defaultdict

from src.video.video_process import (
    classify_video,
    get_consecutive_classes,
    annotate_video_with_classes,
)
from src.train.model import load_new_model
from src.utils.utils import smooth_predictions

CLASS_NAMES = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "BlowDryHair",
    "BrushingTeeth",
    "Haircut"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Video Action Recognition Pipeline")
    
    parser.add_argument('--checkpoint', type=str, 
                        default="./checkpoints/UCF101-filtered-lr0.0001-nobackgroundclass/model_e_10.pth",
                        help="Path to model checkpoint")
    parser.add_argument('--video_path', type=str, 
                        default="./data/Raw/test-vid/test-actions-vid.mp4",
                        help="Input video path")
    parser.add_argument('--output_dir', type=str, 
                        default="./outputs",
                        help="Output directory for results")
    parser.add_argument('--output_name', type=str,
                        default="processed_video",
                        help="Base name for output files")
    parser.add_argument('--window_size', type=int,
                        default=4,
                        help="Smoothing window size (frames)")
    parser.add_argument('--device', type=str,
                        default=None,
                        choices=['cuda', 'cpu'],
                        help="Force computation device (auto-detected if not specified)")

    return parser.parse_args()

def main(args):
    # Configure device
    device = torch.device(args.device if args.device else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_new_model(num_classes=5, pretrained=True, 
                          device=device.type, freeze_params=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    model.to(device)

    # Process video
    raw_classes, raw_probs = classify_video(args.video_path, model, device, single_class=False)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get video metadata
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Create raw video
    raw_consecutive = get_consecutive_classes(raw_classes, CLASS_NAMES)
    raw_video_path, _ = annotate_video_with_classes(
        args.video_path,
        raw_consecutive,
        args.output_dir,
        f"{args.output_name}_raw"
    )

    # Create smoothed video
    smoothed_classes, smoothed_probs = smooth_predictions(
        raw_classes, raw_probs, window_size=args.window_size
    )
    smoothed_consecutive = get_consecutive_classes(
        smoothed_probs.argmax(axis=1), CLASS_NAMES
    )
    smooth_video_path, _ = annotate_video_with_classes(
        args.video_path,
        smoothed_consecutive,
        args.output_dir,
        f"{args.output_name}_smoothed"
    )

    print(f"\nProcessing complete!\nDevice used: {device}")
    print(f"Raw video: {raw_video_path}")
    print(f"Smoothed video: {smooth_video_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)