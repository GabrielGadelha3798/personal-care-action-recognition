import os
import cv2
import json
import torch
import torchvision.io as io
from torchvision.models.video import R3D_18_Weights
from itertools import groupby

def classify_video(video_path, model, device, single_class=True,clip_length=16, step=16):
    # Load the video (output: (T, H, W, C))
    video, _, _ = io.read_video(video_path, pts_unit='sec')
    video = video.permute(0, 3, 1, 2)  

    # Split into clips of 16 frames
    clips = []
    for start in range(0, video.size(0), step):
        end = start + clip_length
        if end > video.size(0):
            # Pad with zeros if needed
            pad = torch.zeros((end - video.size(0), *video.shape[1:]), dtype=video.dtype)
            clip = torch.cat([video[start:], pad], dim=0)
        else:
            clip = video[start:end]
        clips.append(clip)

    # Apply preprocessing transforms
    weights = R3D_18_Weights.DEFAULT
    preprocess = weights.transforms()
    processed_clips = []
    for clip in clips:
        processed_clip = preprocess(clip)
        processed_clips.append(processed_clip)

    processed_clips = torch.stack(processed_clips)

    # Predict with model
    model.eval()
    with torch.no_grad():
        outputs = model(processed_clips.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        if single_class:
            # Average probabilities across all clips
            avg_probs = probs.mean(dim=0)
            predicted_class = avg_probs.argmax().item()
            return predicted_class, avg_probs
        else:
            predicted_classes = probs.argmax(dim=1).cpu().numpy()

            return predicted_classes, probs.cpu().numpy()

def get_consecutive_classes(predicted_classes, class_names):
    """
    Get consecutive classes and their counts from the predicted classes.
    
    Args:
        predicted_classes (List[int]): List of predicted class indices.
        class_names (List[str]): List of class names corresponding to indices.
        
    Returns:
        List[Tuple[str, int]]: List of tuples containing class name and count.
    """
    # Group by consecutive values
    grouped = [(key, len(list(group))) for key, group in groupby(predicted_classes)]
    
    # Convert indices to class names
    result = [(class_names[key], count) for key, count in grouped]
    
    return result



def annotate_video_with_classes(input_video_path, class_list, save_path, file_name):
    # Save actions list to JSON
    config_path = os.path.join(save_path, f"{file_name}_actions_frames.json")
    with open(config_path, 'w') as f:
        json.dump(class_list, f)

   


    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video")

    # Get video properties from input
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")
    # Convert class_list frame count into seconds via fps into a new list
    class_seconds_list = [None] * len(class_list)
    for i in range(len(class_list)):
        class_name, frame_count = class_list[i]
        class_seconds_list[i] = (class_name, round(((frame_count*16) / fps), 2))
    
    # write the class_seconds_list to a JSON file
    config_path = os.path.join(save_path, f"{file_name}_actions_seconds.json")
    with open(config_path, 'w') as f:
        json.dump(class_seconds_list, f)

    # Set up output video writer
    video_path = os.path.join(save_path, f"{file_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_color = (0, 255, 0)  # Green text
    text_position = (50, 100)  # Top-left coordinates (x, y)

    # Calculate class segments
    current_class_index = 0
    frames_remaining = 0
    total_expected_frames = sum(n * 16 for _, n in class_list)
    
    # Warn if frame count mismatch
    if total_frames != total_expected_frames:
        print(f"Warning: Video has {total_frames} frames but class list expects {total_expected_frames}")

    # Process video frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update current class based on frame count
        if frame_count >= sum(n * 16 for _, n in class_list[:current_class_index + 1]):
            current_class_index += 1

        # Get current class name if available
        current_class = class_list[current_class_index][0] if current_class_index < len(class_list) else ""

        # Add text overlay
        cv2.putText(frame, current_class, text_position, 
                   font, font_scale, text_color, font_thickness)

        # Write modified frame
        out.write(frame)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    return video_path, config_path