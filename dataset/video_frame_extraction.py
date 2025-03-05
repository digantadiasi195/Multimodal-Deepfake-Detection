# dataset/video_frame_extraction.py
import torch
from PIL import Image
import numpy as np

def extract_frames_from_video(video_path=None, num_frames=4, image_size=128, frames=None):
    if frames is None and video_path is None:
        return None
    if frames is not None:
        try:
            frame_tensors = []
            for frame_path in frames:
                img = Image.open(frame_path).convert("RGB").resize((image_size, image_size))
                frame = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
                frame_tensors.append(frame)
            return torch.stack(frame_tensors) if frame_tensors else None
        except Exception as e:
            raise ValueError(f"Error processing uploaded frames: {e}")
    else:
        raise ValueError("cv2 is not available. Please provide pre-extracted frame images via the 'frames' parameter.")
