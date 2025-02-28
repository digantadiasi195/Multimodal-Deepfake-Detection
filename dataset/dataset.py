#dataset/dataset.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from dataset.audio_extraction import extract_audio_from_video
from dataset.video_frame_extraction import extract_frames_from_video

class FakeAVCelebDataset(Dataset):
    def __init__(self, csv_file, base_path, image_size=128, num_frames=4, phase='train'):
        """
        Args:
            csv_file (str): Path to train/test CSV file.
            base_path (str): Root directory containing FakeAVCeleb dataset.
            image_size (int): Image resolution (Default 128x128).
            num_frames (int): Number of frames to extract per video.
            phase (str): 'train' or 'test' mode.
        """
        self.data = pd.read_csv(csv_file)  # Load CSV
        self.base_path = base_path
        self.image_size = image_size
        self.num_frames = num_frames
        self.phase = phase  # Train/Test

    def __getitem__(self, index):
        row = self.data.iloc[index]
        video_path = os.path.join(self.base_path, row['fullpath'])
        label = int(row['label'])

        # Extract frames & audio using the new modular functions
        video_frames = extract_frames_from_video(video_path, self.num_frames, self.image_size)
        audio_waveform = extract_audio_from_video(video_path)

        return video_frames, audio_waveform, label

    def __len__(self):
        return len(self.data)
