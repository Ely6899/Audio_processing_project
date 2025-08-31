# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import correlate2d
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ConstPaths import RavdessPaths
from Visualizations import plot_mel_spectrogram
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from audio_dataset import EmotionSpecDataset
from correlation_kernel_playground import kernel_list

# Audio params
FREQUENCY_BIN_COUNT = 64
SAMPLE_RATE = 16000
N_FFT = 512
WINDOW_LENGTH = N_FFT
HOP_LENGTH = N_FFT // 2
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)
TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1

# Filter options
EMOTIONS_TO_INCLUDE = ['01', '02', '03', '04', '05', '06', '07', '08']
ACTORS_TO_INCLUDE = [f"{i:02d}" for i in range(1, 25)]
STATEMENTS_TO_INCLUDE = ['01', '02']
REPETITION_TO_INCLUDE = ['01', '02']
INTENSITY_TO_INCLUDE = ['02']

index_emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

label_emotion_mapping = {
    0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised',
}

def wav_indexer(file_name: Path) -> Tuple[str, str]:
    numbers = re.findall(r'\d+', file_name.name.__str__())
    emotion_index = numbers[2]
    actor_number = numbers[-1]
    emotion = index_emotion_mapping[emotion_index]
    return emotion, actor_number

def is_valid_ravdess_file(path: Path) -> bool:
    if "_" in path.stem:
        return False  # augmented file

    parts = path.stem.split("-")
    if len(parts) != 7:
        return False  # malformed filename

    emotion, intensity, statement, repetition, actor = parts[2], parts[3], parts[4], parts[5], parts[6]
    return (
        emotion in EMOTIONS_TO_INCLUDE and
        intensity in INTENSITY_TO_INCLUDE and
        statement in STATEMENTS_TO_INCLUDE and
        repetition in REPETITION_TO_INCLUDE and
        actor in ACTORS_TO_INCLUDE
    )

def build_correlation_kernel(freq_bins = 20, time_bins = 40) -> np.ndarray:
    # Create smooth frequency profile: almost flat, small gentle slope
    freq_profile = np.linspace(1, 0.9, freq_bins)[:, np.newaxis]  # very gentle high→low

    # Smooth time modulation: soft sine wave
    time_profile = np.sin(np.linspace(0, np.pi, time_bins))[np.newaxis, :]

    # Combine profiles to get 2D kernel
    kernel = freq_profile * time_profile  # element-wise multiplication

    # Normalize: zero-mean and unit-norm
    kernel -= kernel.mean()
    kernel /= np.linalg.norm(kernel) + 1e-12

    return kernel

# def extract_important_time_regions(cam_mask: np.ndarray,
#                                    sample_rate: int,
#                                    hop_length: int,
#                                    threshold_quantile: float = 0.9) -> List[Tuple[float, float]]:
#     time_importance = cam_mask.mean(axis=0)
#     threshold = np.quantile(time_importance, threshold_quantile)
#     high_activation = time_importance >= threshold
#
#     regions = []
#     start_idx = None
#     for idx, is_high in enumerate(high_activation):
#         if is_high and start_idx is None:
#             start_idx = idx
#         elif not is_high and start_idx is not None:
#             end_idx = idx
#             start_time = start_idx * hop_length / sample_rate
#             end_time = end_idx * hop_length / sample_rate
#             regions.append((start_time, end_time))
#             start_idx = None
#     if start_idx is not None:
#         end_idx = len(high_activation)
#         start_time = start_idx * hop_length / sample_rate
#         end_time = end_idx * hop_length / sample_rate
#         regions.append((start_time, end_time))
#     return regions
#
# def plot_segmented_line(ax, times: np.ndarray, values: np.ndarray,
#                         highlight_regions: List[Tuple[float, float]],
#                         base_color='gray', highlight_color='crimson', linewidth=2):
#     for i in range(len(times)-1):
#         t0, t1 = times[i], times[i+1]
#         v0, v1 = values[i], values[i+1]
#         mid = 0.5*(t0+t1)
#         color = highlight_color if any(start <= mid <= end for start, end in highlight_regions) else base_color
#         ax.plot([t0, t1], [v0, v1], color=color, linewidth=linewidth)

# Left for hand_picking only!

RECORDINGS_TO_PROCESS_HANDPICKED = []

# For automated picking!
RECORDINGS_TO_PROCESS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path("ResNetWithAttention.pt"), map_location=device, weights_only=False)
model.eval()


## FOR TESTS:
if __name__ == "__main__":
    pass
