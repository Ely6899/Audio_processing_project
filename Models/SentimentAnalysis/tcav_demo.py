import pandas as pd
import torch
import numpy as np
from pathlib import Path
from captum.concept import TCAV, Concept
#from captum.attr import LayerActivation
# from functorch.dim import Tensor #! makes a bug because functorch.dim isn't supported in python 3.12 !!
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from Preprocess import audio_to_mel_spectrogram
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
        self.data = np.array([generate_random_pattern_spectrogram(freq_count, frames, rng=self.rng)
                     for _ in range(n_samples)])
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure shape [1, H, W] per sample
        x = self.data[idx]
        return x.unsqueeze(0)

    @property
    def get_data(self):
        return self.data
    
class PreGeneratedConceptDataset(Dataset):
    """
    PyTorch Dataset that pre-generates the dataset for a spesific concept in memory.
    """

    def __init__(self, n_samples: int, concept_name: str, root_concept_dir: Path = Path("positive concepts dataset") , freq_count = FREQUENCY_BIN_COUNT, frames_count = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.concept_name = concept_name
        self.root_concept_dir = root_concept_dir
        self.freq_count = freq_count
        self.frames = frames_count
        self.rng = np.random.default_rng(rng_seed)

        # load all .npy files from root_concept_dir/concept_name
        self.data = []
        concept_dir = self.root_concept_dir / self.concept_name
        for npy_file in concept_dir.glob("*.npy"):
            self.data.append(np.load(npy_file))
        self.data = np.array(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure shape [1, H, W] per sample
        x = self.data[idx]
        return x.unsqueeze(0)

    @property
    def get_data(self):
        return self.data

concept_unique_names = [
                        "long_constant_thick",
                        "long_dropping_flat_thick",
                        "long_dropping_steep_thick",
                        "long_dropping_steep_thin",
                        "long_rising_flat_thick",
                        "long_rising_steep_thick",
                        "long_rising_steep_thin",
                        "short_constant_thick",
                        "short_dropping_steep_thick",
                        "short_dropping_steep_thin",
                        "short_rising_steep_thick",
                        "short_rising_steep_thin"
                        ]

index_emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

label_emotion_mapping = {
    0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised',
}

def get_emotion_tensor(emotion_label: str, drop_false_positive: bool) -> torch.Tensor:
    '''
    return emotion tensor containing all the spectrograms that the model predicted as "emotion_label"

    Args:
        emotion_label (str): The emotion label to filter by.
        drop_false_positive (bool): e.g. if emotion_label='angry', then audio classified as 'angry' but not actually 'angry' will be dropped.

    Returns:
        torch.Tensor: A tensor containing the spectrograms for the specified emotion label. in shape: [Batch, 1, Height, Width]
    '''
    df = pd.read_csv("attributes/all_attributes.csv")

    df_emotion = df[(df["predicted_label"] == emotion_label) & (df["true_label"] == emotion_label)]["path"] if drop_false_positive else df[(df["predicted_label"] == emotion_label)]["path"]
    
    # audio_to_mel_spectrogram to all audio samples
    mel_specs = [audio_to_mel_spectrogram(Path(path)) for path in df_emotion]

    # stack all the mel spectrograms to one big tensor
    emotion_tensor = torch.stack([torch.tensor(spec) for spec in mel_specs])
    
    # if shape=[B,H,W] change it to [B,1,H,W]
    if emotion_tensor.dim() == 3:
        emotion_tensor = emotion_tensor.unsqueeze(1)  # [B, 1, H, W]
        
    return emotion_tensor

# store spectrograms of each emotion in Tensor object.
label_inputs = {
    label_name: get_emotion_tensor(label_name, drop_false_positive=True)
    for label_name in label_emotion_mapping.values()
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


positive_concepts: list[Concept] = [Concept(id=concept_idx, name=concept_name, data_iter=DataLoader(PreGeneratedConceptDataset(n_samples=60, concept_name=concept_name), shuffle=False))
                               for concept_idx, concept_name in enumerate(concept_unique_names)]

# This concept is the negative of concepts.
negative_concept_dataset = PreGeneratedRandomSpectrogramDataset(n_samples=10, freq_count=FREQUENCY_BIN_COUNT, frames=TARGET_FRAMES)
random_concept = Concept(id=len(positive_concepts), name='random', data_iter=DataLoader(negative_concept_dataset, shuffle=False))

# Debug call, don't uncomment
# show_arrays_in_separate_windows(negative_concept_dataset.get_data)

print("Reached tcav interpret")
for label_index, label_name in label_emotion_mapping.items():
    tcav_scores_per_label[label_name] = tcav.interpret(
        inputs=label_inputs[label_name],
        experimental_sets=[[c, random_concept] for c in positive_concepts],
        target=label_index  # integer index of target class
    )

# -----------------------------
# 4️⃣ Inspect results
# -----------------------------

for label_name, score_dict in tcav_scores_per_label.items():
     print(f"TCAV scores for label {label_name}: {score_dict}")
