#dataset/video_frame_extraction.py
import cv2
import numpy as np
import torch
import torchvision.transforms as T

def extract_frames_from_video(video_path, num_frames=4, image_size=128):
    """
    Extracts at least `num_frames` evenly spaced frames from a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
        image_size (int): Target image resolution.

    Returns:
        torch.Tensor: Tensor of extracted frames with shape (num_frames, 3, image_size, image_size).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Debugging: Check if the video file is valid
        if total_frames == 0:
            print(f" Warning: No frames found in video {video_path}. Returning blank tensor.")
            return torch.zeros((num_frames, 3, image_size, image_size))

        # If the video has fewer frames, duplicate existing frames to reach `num_frames`
        if total_frames < num_frames:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f" Warning: Failed to extract frame at index {idx} from {video_path}.")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = transform(torch.tensor(frame).permute(2, 0, 1))  # Apply transformation
            frames.append(frame.unsqueeze(0))

        cap.release()

        # If frames extracted are fewer than `num_frames`, duplicate last frame
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())  # Duplicate last frame

        return torch.cat(frames, dim=0) if frames else torch.zeros((num_frames, 3, image_size, image_size))

    except Exception as e:
        print(f" Error extracting frames from {video_path}: {e}")
        return torch.zeros((num_frames, 3, image_size, image_size))  # Return blank tensor if extraction fails
