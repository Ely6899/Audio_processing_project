# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
from pytorch_grad_cam import GradCAM                   # (already in your file)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt                        #  ← NEW (for quick display)

from models import SentimentModelHandler, ResNetWithAttention
import torch
from pathlib import Path
from audio_dataset import RavdessRawData, EmotionSpecDataset
from Preprocess import audio_to_mel_spectrogram, standardization
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from imageio import imwrite
import numpy as np

# --------------------------------------------------------------
# 1.  Load model exactly as you do now
# --------------------------------------------------------------
model = ResNetWithAttention(num_classes=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r"C:\Users\User\Documents\ARIEL\4th_Year\Audio_processing_project\ResNetWithAttention.pt",
                   map_location=device)
model.eval()

# --------------------------------------------------------------
# 2.  Prepare a single-spectrogram batch [1, 1, F, T]
# --------------------------------------------------------------
wav_path = Path(r"C:\Users\User\Documents\ARIEL\4th_Year\Audio_processing_project\RAVDESS\original_data\Audio_Speech_Actors_01-24\Actor_01\03-01-03-01-01-01-01.wav")
spec_tensor, _ = EmotionSpecDataset({(wav_path, "happy")})[0]    # shape (1, freq_bins, time_frames)
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
cam = GradCAM(model=model, target_layers=target_layers)

pred_idx = model(input_tensor).argmax(dim=1).item()              # predicted class id
target_label = [ClassifierOutputTarget(pred_idx)]                     # focus heat-map on that class

cam_mask = cam(input_tensor=input_tensor,
                    targets=target_label,
                    aug_smooth=True,
                    eigen_smooth=True)[0]        # (F, T)   values 0-1

# --------------------------------------------------------------
# 5.  Get *raw* dB mel spectrogram (for prettier colours)
#     – request **no normalisation** so we can min-max for display only
# --------------------------------------------------------------
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
                            image_weight=0.60)   # 0→only heat-map, 1→only spec

# --------------------------------------------------------------
# 6.  Save **and** show
# --------------------------------------------------------------
out_path = Path("cam_overlay_last_layer.png")
imwrite(out_path, (overlay).astype("uint8"))
print(f"🔥 Saved Grad-CAM overlay → {out_path.resolve()}")

plt.figure(figsize=(10, 4))
plt.imshow(overlay)
plt.axis("off")
plt.tight_layout()
plt.show()
