# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
from typing import Tuple

import librosa
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
from Visualizations import plot_mel_spectrogram, save_mel_spectrogram
from models import SentimentModelHandler, ResNetWithAttention
import torch
from pathlib import Path
from audio_dataset import RavdessRawData, EmotionSpecDataset
from Preprocess import audio_to_mel_spectrogram, standardization
from PreprocessParams import HOP_LENGTH, MAX_SPECTOGRAM_DURATION_IN_SECONDS, SAMPLE_RATE
from imageio import imwrite
import numpy as np
import re

EMOTIONS_TO_INCLUDE = ['01', '02', '03', '04', '05', '06', '07', '08']
ACTORS_TO_INCLUDE = ['01', '02', '03', '04']
STATEMENTS_TO_INCLUDE = ['01']
INTENSITY_TO_INCLUDE = ['01']
REPETITION_TO_INCLUDE = ['01']

def wav_indexer(file_name: Path) -> Tuple[str, str]:
    """
    For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
    readable label.
    :param file_name: File path from RAVDESS dataset.
    :return: Label of the file according to the index.
    """
    numbers = re.findall(r'\d+', file_name.name.__str__())

    index_emotion_mapping = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    emotion_index = numbers[2]
    actor_index = numbers[-1]
    emotion = index_emotion_mapping[emotion_index]
    return emotion, actor_index

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
# 2.  Prepare a single-spectrogram batch [1, 1, F, T]
# --------------------------------------------------------------
for wav_path in RECORDINGS_TO_PROCESS:
    print(f"Running GradCam, spectrogram and waveform on {wav_path.stem}")

    wav_emotion, actor_index = wav_indexer(wav_path)

    spec_tensor, _ = EmotionSpecDataset({(wav_path, wav_emotion)})[0]    # shape (1, freq_bins, time_frames)
    input_tensor = spec_tensor.unsqueeze(0).to(device)               # shape (1, 1, freq_bins, time_frames)

    # --------------------------------------------------------------
    # 3.  Pick the layer you want to “look” at
    #     • last conv in module3 ≈ highest-level features
    #     • you can swap to module2 / module1 for lower-level detail
    # --------------------------------------------------------------
    target_layers = [model.module3.blocks[-1].conv2]

    # --------------------------------------------------------------
    # 4.  Build + run Grad-CAM
    # --------------------------------------------------------------
    print("Running GradCam")
    cam = GradCAM(model=model, target_layers=target_layers)

    pred_idx = model(input_tensor).argmax(dim=1).item()              # predicted class id
    target_label = [ClassifierOutputTarget(pred_idx)]                     # focus heat-map on that class

    cam_mask = cam(input_tensor=input_tensor,
                        targets=target_label,
                        aug_smooth=True,
                        eigen_smooth=True)[0]        # (F, T)   values 0-1

    print("Finished GradCam\n")

    # --------------------------------------------------------------
    # 5.  Get *raw* dB mel spectrogram (for prettier colours)
    #     – request **no normalisation** so we can min-max for display only
    # --------------------------------------------------------------

    print("Plotting Spectrogram")

    raw_spec = audio_to_mel_spectrogram(
        file_path=wav_path,
        max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
        normalization_fn=lambda x: x               # keep real dB
    ).astype("float32")                             # (F, T)

    raw_norm = (raw_spec - raw_spec.min()) / (raw_spec.ptp() + 1e-6) # 0-1
    rgb_base  = np.stack([raw_norm]*3, axis=-1).astype(np.float32)   # (F, T, 3)

    overlay = show_cam_on_image(rgb_base,
                                cam_mask,
                                use_rgb=True,
                                image_weight=0)   # 0→only heat-map, 1→only spec

    print("Finished spectrogram\n")

    # Step 4: Plot all components
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    fig.suptitle(f"Actor {actor_index} - {wav_emotion}", fontsize=16, y=0.95)

    # 1. Waveform

    print("Plotting waveform")
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    print("Finished waveform\n")

    axes[0].plot(y)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Samples")

    # 2. Mel-spectrogram
    im2 = axes[1].imshow(raw_spec, origin="lower", aspect="auto")
    axes[1].set_title("Mel Spectrogram (dB)")
    fig.colorbar(im2, ax=axes[1])

    # 3. Grad-CAM heatmap
    axes[2].imshow(overlay, origin="lower")
    axes[2].set_title(f"Grad-CAM (Predicted: {pred_idx})")

    # Save the figure
    # Create subfolder per actor
    save_folder = Path("Benchmark_Results") / f"Actor_{actor_index}"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Save the figure inside that subfolder
    save_name = wav_path.stem + "_subplot.png"
    save_path = save_folder / save_name

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_name}")

    #save_mel_spectrogram(overlay, file_save_path=Path("heatmap_only_heatmap"), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)