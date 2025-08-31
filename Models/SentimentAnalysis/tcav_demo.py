import torch
import numpy as np
from pathlib import Path
from captum.concept import TCAV, Concept
#from captum.attr import LayerActivation
from functorch.dim import Tensor
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from concepts_creation import generate_random_pattern_spectrogram, show_arrays_in_separate_windows
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS, TARGET_FRAMES, FREQUENCY_BIN_COUNT, HOP_LENGTH, SAMPLE_RATE


class PreGeneratedRandomSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that pre-generates all random spectrograms in memory.
    """

    def __init__(self, n_samples: int, freq_count = FREQUENCY_BIN_COUNT, frames = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.freq_count = freq_count
        self.frames = frames
        self.rng = np.random.default_rng(rng_seed)

        # Pre-generate all spectrograms in memory
        self.data = [generate_random_pattern_spectrogram(freq_count, frames, rng=self.rng)
                     for _ in range(n_samples)]
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def get_data(self):
        return self.data

index_emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

label_emotion_mapping = {
    0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised',
}

# Spectrograms in Tensor object.
label_inputs = {
    'angry': Tensor(),
    'calm':  Tensor(),

}

# -----------------------------
# 1️⃣ Load pretrained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path("ResNetWithAttention.pt"), map_location=device, weights_only=False)
model.eval()

# -----------------------------
# 2️⃣ Choose layer for TCAV to work on
# -----------------------------

layer = "module3.blocks.0.conv2"

# -----------------------------
# 3️⃣ Compute TCAV
# -----------------------------
# Captum TCAV expects a dictionary of concept activations, with positive and negative examples.


# Define TCAV object
tcav = TCAV(model, [layer])
tcav_scores_per_label= {}

# Anything not considered 'random' is positive concepts.
concept_list: list[Concept] = [Concept(id=label_id, name=label_name, data_iter=None)
                               for label_id, label_name in label_emotion_mapping.items()]

# This concept is the negatives of concepts.
negative_concept_dataset = PreGeneratedRandomSpectrogramDataset(n_samples=10, freq_count=FREQUENCY_BIN_COUNT, frames=TARGET_FRAMES)
concept_list.append(Concept(id=8, name='random', data_iter=DataLoader(negative_concept_dataset, shuffle=False)))

# Debug call, don't uncomment
# show_arrays_in_separate_windows(negative_concept_dataset.get_data)

print("Reached tcav interpret")
for label_index, label_name in label_emotion_mapping.items():
    tcav_scores_per_label[label_name] = tcav.interpret(
        inputs=label_inputs[label_name],
        experimental_sets=[[c] for c in concept_list if c.name != 'random'],
        target=label_index  # integer index of target class
    )

# -----------------------------
# 4️⃣ Inspect results
# -----------------------------

for label_name, score_dict in tcav_scores_per_label.items():
     print(f"TCAV scores for label {label_name}: {score_dict}")
