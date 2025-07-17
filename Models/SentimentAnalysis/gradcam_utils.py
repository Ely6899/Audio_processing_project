# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
import csv
import os.path
import re
from pathlib import Path
from typing import Tuple, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
from Models.SentimentAnalysis.Visualizations import plot_mel_spectrogram
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS, SAMPLE_RATE
from audio_dataset import EmotionSpecDataset

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
ACTORS_TO_INCLUDE = ['09', '06', '03', '18']
STATEMENTS_TO_INCLUDE = ['02']
INTENSITY_TO_INCLUDE = ['01']
REPETITION_TO_INCLUDE = ['02']

index_emotion_mapping = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
    }

label_emotion_mapping = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised',
}

# truth file index -> prediction index(Lexi) -> list placement -> emotion
# 01(Neutral) -> 5 -> 6 -> fearful
# 02(Calm) ->
# 04(Sad) -> 6 -> 7 -> disgust


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

def extract_important_time_regions(cam_mask: np.ndarray,
                                   sample_rate: int,
                                   hop_length: int,
                                   threshold_quantile: float = 0.9) -> List[Tuple[float, float]]:
    """
    Extract time intervals (in seconds) from a GradCAM mask where activation is high.

    Args:
        cam_mask (np.ndarray): 2D GradCAM activation mask (freq_bins, time_frames)
        sample_rate (int): Audio sample rate
        hop_length (int): Hop length used for spectrogram
        threshold_quantile (float): Threshold percentile to consider as high activation

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    # Step 1: Collapse over frequency axis → importance per time frame
    time_importance = cam_mask.mean(axis=0)  # shape: (time_frames,)

    # Step 2: Threshold based on quantile
    threshold = np.quantile(time_importance, threshold_quantile)
    high_activation = time_importance >= threshold

    # Step 3: Group contiguous high-activation frames
    regions = []
    start_idx = None
    for idx, is_high in enumerate(high_activation):
        if is_high and start_idx is None:
            start_idx = idx
        elif not is_high and start_idx is not None:
            end_idx = idx
            # Convert frame indices to time in seconds
            start_time = start_idx * hop_length / sample_rate
            end_time = end_idx * hop_length / sample_rate
            regions.append((start_time, end_time))
            start_idx = None
    if start_idx is not None:  # handle last region
        end_idx = len(high_activation)
        start_time = start_idx * hop_length / sample_rate
        end_time = end_idx * hop_length / sample_rate
        regions.append((start_time, end_time))

    return regions

def plot_segmented_line(ax, times: np.ndarray, values: np.ndarray,
                        highlight_regions: List[Tuple[float, float]],
                        base_color='gray', highlight_color='crimson', linewidth=2):
    """
    Plot a line on ax where segments inside highlight_regions are in highlight_color.
    """
    for i in range(len(times)-1):
        t0, t1 = times[i], times[i+1]
        v0, v1 = values[i], values[i+1]
        mid = 0.5*(t0+t1)
        color = highlight_color if any(start <= mid <= end for start, end in highlight_regions) else base_color
        ax.plot([t0, t1], [v0, v1], color=color, linewidth=linewidth)


# Left for hand_picking only!
RECORDINGS_TO_PROCESS_HANDPICKED = []

# For automated picking!
RECORDINGS_TO_PROCESS = []


# Set to None if handpicked. Set to False if you want without a csv reference.
# NOTE: It is still required to filter options in the macros above.
PANDAS_FLAG: bool | None = True

if PANDAS_FLAG is True:
    df = pd.read_csv(filepath_or_buffer=Path(os.path.join("attributes", "99_attributes.csv")),
                     usecols=['path'],
                     quoting=csv.QUOTE_NONE,
                     encoding='utf-8',
                     engine='python',
                     dtype=str)

    path_list = df['path']

    for wav in path_list:
        wav =  Path(Path(wav.replace("\\", "/").strip()))

        if is_valid_ravdess_file(wav):
            RECORDINGS_TO_PROCESS.append(wav)

elif PANDAS_FLAG is None:
    RECORDINGS_TO_PROCESS = RECORDINGS_TO_PROCESS_HANDPICKED

else:
    RECORDINGS_TO_PROCESS = sorted([
        wav for wav in RavdessPaths.AUDIO_ORIGINAL_DATA.rglob("*.wav")
        if is_valid_ravdess_file(wav)
    ])

# --------------------------------------------------------------
# 1.  Load model exactly as you do now
# --------------------------------------------------------------
#model = ResNetWithAttention(num_classes=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path("ResNetWithAttention.pt"),
                   map_location=device, weights_only=False)
model.eval()

# --------------------------------------------------------------
# 2.  Loop through each file
# --------------------------------------------------------------
for wav_path in RECORDINGS_TO_PROCESS:
    print(f"Running GradCam, volume and F0 on {wav_path.stem}")
    wav_emotion, actor_index = wav_indexer(wav_path)

    spec_tensor, _ = EmotionSpecDataset({(wav_path, wav_emotion)})[0]
    input_tensor = spec_tensor.unsqueeze(0).to(device)

    # GradCAM
    target_layers = [model.module3.blocks[-1].conv2]
    cam = GradCAM(model=model, target_layers=target_layers)
    pred_idx = model(input_tensor).argmax(dim=1).item()
    target_label = [ClassifierOutputTarget(pred_idx)]
    cam_mask = cam(input_tensor=input_tensor,
                   targets=target_label,
                   aug_smooth=True,
                   eigen_smooth=True)[0]

    # Get high-activation time intervals
    high_activation_regions = extract_important_time_regions(
        cam_mask=cam_mask,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        threshold_quantile=0.94
    )
    print(f"High-activation time regions (s): {high_activation_regions}")

    # Prepare GradCAM overlay
    raw_spec = audio_to_mel_spectrogram(
        file_path=wav_path,
        max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
        normalization_fn=lambda x: x
    ).astype("float32")
    raw_norm = (raw_spec - raw_spec.min()) / (raw_spec.ptp() + 1e-6)
    rgb_base = np.stack([raw_norm]*3, axis=-1).astype(np.float32)
    overlay = show_cam_on_image(rgb_base, cam_mask, use_rgb=True, image_weight=0)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Actor {actor_index} - {wav_emotion}", fontsize=16, y=0.95)

    # Load audio
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    # Subplot 1: Volume (RMS)
    rms = librosa.feature.rms(y=y, frame_length=WINDOW_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
    plot_segmented_line(axes[0], rms_times, rms, high_activation_regions,
                        base_color='lightgray', highlight_color='crimson', linewidth=2)
    axes[0].set_title("Volume (RMS Energy)")
    axes[0].set_ylabel("RMS")
    axes[0].grid(True)

    # Subplot 2: F0 Only
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH
    )
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    plot_segmented_line(axes[1], f0_times, f0, high_activation_regions,
                        base_color='lightgray', highlight_color='black', linewidth=2)
    axes[1].set_title("Fundamental Frequency (F0)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].legend(["F0 (Hz)"], loc='upper right')
    axes[1].grid(True)

    # Subplot 3: Grad-CAM
    axes[2].imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE//2]
    )
    emotion_classified = label_emotion_mapping[pred_idx]
    axes[2].set_title(f"Grad-CAM (Predicted: {emotion_classified})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")

    # Save figure
    save_folder = Path("Benchmark_Results") / f"Actor_{actor_index}"
    save_folder.mkdir(parents=True, exist_ok=True)
    save_name = wav_path.stem + "_subplot.png"
    save_path = save_folder / save_name

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_name}")

