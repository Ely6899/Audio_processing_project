# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
import csv
import os.path
import re
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
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
    print(f"Running GradCam, spectrogram and waveform on {wav_path.stem}")
    wav_emotion, actor_index = wav_indexer(wav_path)

    spec_tensor, _ = EmotionSpecDataset({(wav_path, wav_emotion)})[0]  # shape (1, freq_bins, time_frames)
    input_tensor = spec_tensor.unsqueeze(0).to(device)  # shape (1, 1, F, T)

    # --------------------------------------------------------------
    # GradCAM setup
    # --------------------------------------------------------------
    target_layers = [model.module3.blocks[-1].conv2]
    cam = GradCAM(model=model, target_layers=target_layers)

    pred_idx = model(input_tensor).argmax(dim=1).item()
    target_label = [ClassifierOutputTarget(pred_idx)]

    cam_mask = cam(input_tensor=input_tensor,
                   targets=target_label,
                   aug_smooth=True,
                   eigen_smooth=True)[0]

    # --------------------------------------------------------------
    # Prepare raw mel spectrogram
    # --------------------------------------------------------------
    raw_spec = audio_to_mel_spectrogram(
        file_path=wav_path,
        max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
        normalization_fn=lambda x: x  # keep real dB
    ).astype("float32")

    raw_norm = (raw_spec - raw_spec.min()) / (raw_spec.ptp() + 1e-6)
    rgb_base = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)

    overlay = show_cam_on_image(rgb_base, cam_mask, use_rgb=True, image_weight=0)

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"Actor {actor_index} - {wav_emotion}", fontsize=16, y=0.95)

    # Load waveform
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    time_waveform = np.linspace(0, len(y) / SAMPLE_RATE, num=len(y))
    axes[0].plot(time_waveform, y)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")

    # Plot mel spectrogram + F0
    im2 = axes[1].imshow(raw_spec,
                         origin="lower",
                         aspect="auto",
                         extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2])
    axes[1].set_title("Mel Spectrogram (dB) + F0 Overlay")
    axes[1].set_ylabel("Hz")
    axes[1].set_xlabel("Time")
    fig.colorbar(im2, ax=axes[1])

    # Extract and plot F0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    valid_idx = ~np.isnan(f0)
    axes[1].plot(times[valid_idx], f0[valid_idx], color='orange', linewidth=2, label="F0 (Hz)")
    axes[1].legend(loc='upper right')

    # GradCAM heatmap
    axes[2].imshow(overlay,
                   origin="lower",
                   aspect="auto",
                   extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2])

    emotion_classified = label_emotion_mapping[pred_idx]
    axes[2].set_title(f"Grad-CAM (Predicted: {emotion_classified})")

    # Save figure
    save_folder = Path("Benchmark_Results") / f"Actor_{actor_index}"
    save_folder.mkdir(parents=True, exist_ok=True)
    save_name = wav_path.stem + "_subplot.png"
    save_path = save_folder / save_name

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_name}")