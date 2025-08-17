# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import correlate2d
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
from Models.SentimentAnalysis.Visualizations import plot_mel_spectrogram
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from audio_dataset import EmotionSpecDataset
from correlation_kernel_playground import K_vert_5x3

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
correlation_kernel = K_vert_5x3

RECORDINGS_TO_PROCESS_HANDPICKED = []

# For automated picking!
RECORDINGS_TO_PROCESS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path("ResNetWithAttention.pt"), map_location=device, weights_only=False)
model.eval()

PANDAS_FLAG: bool | None = True
RECORDINGS_TO_PROCESS = []
if PANDAS_FLAG is True:
    df = pd.read_csv(Path("attributes/99_attributes.csv"), usecols=['path'], quoting=csv.QUOTE_NONE,
                     encoding='utf-8', engine='python', dtype=str)
    for wav in df['path']:
        wav_path = Path(wav.replace("\\", "/").strip())
        if is_valid_ravdess_file(wav_path):
            RECORDINGS_TO_PROCESS.append(wav_path)
else:
    RECORDINGS_TO_PROCESS = [
        wav for wav in RavdessPaths.AUDIO_ORIGINAL_DATA.rglob("*.wav")
        if is_valid_ravdess_file(wav)
    ]

# --------------------------------------------------------------
# 3.  Group by emotion, actor, statement+repetition
# --------------------------------------------------------------
emotion_to_actor_sentence_repetition = defaultdict(lambda: defaultdict(dict))

for wav_path in RECORDINGS_TO_PROCESS:
    parts = wav_path.stem.split("-")
    if len(parts) != 7:
        continue
    emotion_idx, statement, repetition, actor = parts[2], parts[4], parts[5], parts[6]
    emotion = index_emotion_mapping[emotion_idx]
    key = (statement, repetition)

    if actor in ACTORS_TO_INCLUDE and emotion in index_emotion_mapping.values():
        emotion_to_actor_sentence_repetition[emotion][actor][key] = wav_path

# --------------------------------------------------------------
# 4.  Create and save composite plots per emotion-actor
# --------------------------------------------------------------
for emotion, actor_dict in emotion_to_actor_sentence_repetition.items():
    print(f"\n===> Processing emotion: {emotion}")

    for actor, combo_dict in actor_dict.items():
        print(f"  Actor {actor}")
        fig, axes = plt.subplots(3, 4, figsize=(30, 12), sharex=True)
        fig.suptitle(f"{emotion.capitalize()} – Actor {actor}", fontsize=18, y=0.98)

        sorted_keys = sorted(combo_dict.keys(), key=lambda x: (x[0], x[1]))  # (statement, repetition)

        img2 = None
        for col_idx, (statement, repetition) in enumerate(sorted_keys):
            wav_path = combo_dict[(statement, repetition)]
            print(f"    Statement {statement}, Repetition {repetition} → {wav_path.name}")
            spec_tensor, _ = EmotionSpecDataset({(wav_path, emotion)})[0]
            input_tensor = spec_tensor.unsqueeze(0).to(device)

            target_layers = [model.module3.blocks[-1].conv2]
            cam = GradCAM(model=model, target_layers=target_layers)
            pred_idx = model(input_tensor).argmax(dim=1).item()
            cam_mask = cam(input_tensor=input_tensor,
                           targets=[ClassifierOutputTarget(pred_idx)],
                           aug_smooth=True,
                           eigen_smooth=True)[0]

            # high_activation_regions = extract_important_time_regions(
            #     cam_mask, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, threshold_quantile=0.75)

            # y, sr = librosa.load(wav_path, mono=True, sr=SAMPLE_RATE)
            # if len(y) < MAX_SAMPLES:
            #     y = np.pad(y, (0, MAX_SAMPLES - len(y)))
            # else:
            #     y = y[:MAX_SAMPLES]
            # rms = librosa.feature.rms(y=y, frame_length=WINDOW_LENGTH, hop_length=HOP_LENGTH)[0]
            # rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
            #
            # f0, _, _ = librosa.pyin(
            #     y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            #     sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            # f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)

            raw_spec = audio_to_mel_spectrogram(
                file_path=wav_path, max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
                normalization_fn=lambda x: x).astype("float32")
            raw_norm = (raw_spec - raw_spec.min()) / (raw_spec.ptp() + 1e-6)
            rgb_base = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)
            overlay = show_cam_on_image(rgb_base, cam_mask, use_rgb=True, image_weight=0)

            # axes[0][col_idx].set_title(f"Stmt {statement} Rep {repetition}")
            # plot_segmented_line(axes[0][col_idx], rms_times, rms, high_activation_regions,
            #                     base_color='lightgray', highlight_color='crimson', linewidth=2)
            # axes[0][col_idx].set_ylabel("Volume")
            # axes[0][col_idx].set_ylim(0, 0.5)
            # axes[0][col_idx].grid(True)
            #
            # plot_segmented_line(axes[1][col_idx], f0_times, f0, high_activation_regions,
            #                     base_color='lightgray', highlight_color='black', linewidth=2)
            # axes[1][col_idx].set_ylabel("F0 (Hz)")
            # axes[1][col_idx].set_ylim(0, 500)
            # axes[1][col_idx].grid(True)

            axes[0, col_idx].imshow(
                overlay, origin="lower", aspect="auto",
                extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2],
            )
            emotion_classified = label_emotion_mapping[pred_idx]
            axes[0][col_idx].set_title(f"Statement: {statement} Repetition: {repetition}")
            axes[0][col_idx].set_xlabel("Time (s)")
            axes[0][col_idx].set_ylabel("Freq (Hz)")

            # NEW: Plot CAM-filtered spectrogram
            alpha_mask = np.clip(cam_mask, 0, 1)

            # Keep only yellow-red regions (~ top 30% of activation)
            threshold = np.quantile(alpha_mask, 0.88) #84%, 86%, 88%, 90%(?),
            strong_activation_mask = (alpha_mask >= threshold).astype(np.float32)

            # Apply the mask to the original spectrogram (not RGB)
            masked_spec = raw_spec * strong_activation_mask

            # Plot using librosa with a colormap (e.g., magma, viridis)
            img2 = librosa.display.specshow(
                masked_spec,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                x_axis='time',
                y_axis='linear',
                ax=axes[1, col_idx],
                cmap='magma',  # or 'viridis', 'inferno', etc.
            )

            axes[1, col_idx].set_title("Filtered by Attention (Top 30%)")
            axes[1, col_idx].set_xlabel("Time (s)")
            axes[1, col_idx].set_ylabel("Freq (Hz)")

            corr = correlate2d(masked_spec, correlation_kernel, mode='full', boundary='symm')

            # Normalize correlation to [-1,1] for visualization
            corr /= np.max(np.abs(corr)) + 1e-12

            # --- 5. Plot cross-correlation map ---
            im3 = axes[2, col_idx].imshow(corr, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2])
            axes[2, col_idx].set_title("Cross-Correlation with Smooth Flat Kernel")
            if col_idx == axes.shape[1] - 1:  # last column
                cbar = plt.colorbar(im3, ax=axes[2, :], orientation='horizontal', label='Correlation')

        cbar_spec = fig.colorbar(img2, ax=axes[1, :], orientation='horizontal')
        cbar_spec.set_label("Spectrogram Magnitude (dB)")
        save_folder = Path("Benchmark_Results") / "Summary_By_Actor" / actor
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"Emotion_{emotion}_overview.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

# --------------------------------------------------------------
# 5. PCA Summary for All Recordings – RMS and F0 by Emotion
# --------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused but required for 3D plot
#
# rms_vectors = []
# f0_vectors = []
# emotion_labels = []

# print("\n===> Gathering data for PCA summary...")
#
# for emotion, actor_dict in emotion_to_actor_sentence_repetition.items():
#     for actor, combo_dict in actor_dict.items():
#         for (statement, repetition), wav_path in combo_dict.items():
#             y, sr = librosa.load(wav_path, mono = True, sr=SAMPLE_RATE)
#
#             # RMS feature
#             rms = librosa.feature.rms(y=y, frame_length=WINDOW_LENGTH, hop_length=HOP_LENGTH)[0]
#             rms = np.interp(np.linspace(0, len(rms) - 1, 100), np.arange(len(rms)), rms)
#             rms_vectors.append(rms)
#
#             # F0 feature (handle NaNs)
#             f0, _, _ = librosa.pyin(
#                 y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
#                 sr=sr, hop_length=HOP_LENGTH)
#             f0 = pd.Series(f0).interpolate(limit_direction="both").bfill().ffill().to_numpy()
#             f0 = np.interp(np.linspace(0, len(f0) - 1, 100), np.arange(len(f0)), f0)
#             f0_vectors.append(f0)
#
#             emotion_labels.append(emotion)
#
# rms_matrix = np.vstack(rms_vectors)
# f0_matrix = np.vstack(f0_vectors)
#
# def pca_and_plot(data_matrix, labels, feature_name, axes_2d, axes_3d):
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data_matrix)
#
#     pca = PCA(n_components=3)
#     data_pca = pca.fit_transform(data_scaled)
#
#     unique_labels = sorted(set(labels))
#     colors = plt.cm.tab10.colors
#     label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
#
#     for label in unique_labels:
#         mask = np.array(labels) == label
#         axes_2d.scatter(data_pca[mask, 0], data_pca[mask, 1],
#                         label=label, color=label_to_color[label], alpha=0.7)
#     axes_2d.set_title(f"{feature_name} – PCA 2D")
#     axes_2d.set_xlabel("PC1")
#     axes_2d.set_ylabel("PC2")
#     axes_2d.legend()
#
#     for label in unique_labels:
#         mask = np.array(labels) == label
#         axes_3d.scatter(data_pca[mask, 0], data_pca[mask, 1], data_pca[mask, 2],
#                         label=label, color=label_to_color[label], alpha=0.7)
#     axes_3d.set_title(f"{feature_name} – PCA 3D")
#     axes_3d.set_xlabel("PC1")
#     axes_3d.set_ylabel("PC2")
#     axes_3d.set_zlabel("PC3")
#
# fig = plt.figure(figsize=(16, 10))
# ax_rms_2d = fig.add_subplot(2, 2, 1)
# ax_f0_2d = fig.add_subplot(2, 2, 2)
# ax_rms_3d = fig.add_subplot(2, 2, 3, projection='3d')
# ax_f0_3d = fig.add_subplot(2, 2, 4, projection='3d')
#
# pca_and_plot(rms_matrix, emotion_labels, "RMS", ax_rms_2d, ax_rms_3d)
# pca_and_plot(f0_matrix, emotion_labels, "F0", ax_f0_2d, ax_f0_3d)
#
# plt.tight_layout()
# summary_dir = Path("Benchmark_Results") / "Summary_By_Actor"
# summary_dir.mkdir(parents=True, exist_ok=True)
# save_pca_path = summary_dir / "PCA_Overview.png"
# plt.savefig(save_pca_path)
# #plt.show()
# print(f"Saved PCA plot: {save_pca_path}")
