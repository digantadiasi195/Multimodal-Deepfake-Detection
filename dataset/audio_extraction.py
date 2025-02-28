#dataset\audio_extraction.py
import os
import sys
import torch
import torchaudio
import moviepy.editor as mp
from torchaudio.transforms import Resample

# Suppress MoviePy logs globally
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

# Define a fixed audio length (in samples) based on 4 seconds at 16kHz
FIXED_AUDIO_LENGTH = 4 * 16000  # 4 seconds at 16kHz

def extract_audio_from_video(video_path, sr=16000):
    """
    Extracts audio from a video file and returns a resampled waveform of fixed length.

    Args:
        video_path (str): Path to the video file.
        sr (int): Target sampling rate (default 16kHz).

    Returns:
        torch.Tensor: Fixed-length extracted waveform.
    """
    try:
        video = mp.VideoFileClip(video_path)

        # Suppress MoviePy-specific logs
        audio_path = video_path.replace(".mp4", ".wav")
        video.audio.write_audiofile(audio_path, fps=sr, verbose=False, logger=None)  # Suppress logs

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = Resample(sample_rate, sr)(waveform)  # Resample to 16kHz
        
        os.remove(audio_path)  # Clean up the temporary file

        # Ensure fixed length
        if waveform.shape[1] > FIXED_AUDIO_LENGTH:
            waveform = waveform[:, :FIXED_AUDIO_LENGTH]  # Truncate
        elif waveform.shape[1] < FIXED_AUDIO_LENGTH:
            padding = FIXED_AUDIO_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # Pad with zeros

        return waveform.squeeze(0)  # Remove extra channel

    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return torch.zeros(FIXED_AUDIO_LENGTH) 

# Restore stdout and stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
