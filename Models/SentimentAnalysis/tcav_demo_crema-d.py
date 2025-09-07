
from IPython.display import display
from captum.concept import TCAV, Concept
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from pathlib import Path
import sys

from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT, MAX_SPECTOGRAM_DURATION_IN_SECONDS
from concepts_creation import generate_random_pattern_spectrogram
from ConstPaths import CremaPaths

# os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
here = Path(__file__).resolve()
DEFAULT_CREMAD_ROOT = here.parents[0]              # …/ (two levels up)
# print(DEFAULT_CREMAD_ROOT)
sys.path.append(DEFAULT_CREMAD_ROOT)

CREMAD_ROOT = CremaPaths.WAV_DATA  # <-- set your path
device="cpu"
N_SAMPLES_PER_LABEL = 100  # <-- how many spectrograms per class you want

class PreGeneratedRandomSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that pre-generates all random spectrogram in memory.
    """

    def __init__(self, n_samples: int, freq_count = FREQUENCY_BIN_COUNT, frames = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.freq_count = freq_count
        self.frames = frames
        self.rng = np.random.default_rng(rng_seed)

        # Pre-generate all spectrograms in memory
        self.data = np.array([generate_random_pattern_spectrogram(freq_count, frames, rng=self.rng)
                     for _ in range(n_samples)])
        self.data = torch.tensor(self.data, dtype=torch.float32).to(device=device)

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
    PyTorch Dataset that pre-generates the dataset for a specific concept in memory.
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
        concept_dir.mkdir(exist_ok=True, parents=True)
        for npy_file in concept_dir.glob("*.npy"):
            self.data.append(np.load(npy_file))
        self.data = np.array(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32).to(device=device)

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


label_emotion_mapping = {
    0: 'angry',
    1: 'disgust',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
}

# optional short-code map from filename to label name
cremad_code_to_name = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

allowed_emotions = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

CREMAD_PATTERN = re.compile(
    r'^(?P<actor>\d{4})_(?P<utt>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<intensity>[A-Z]{2})\.wav$'
)

def parse_cremad_filename(path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse a CREMA-D filename and return (actor, utterance, emotion_code, intensity).
    Returns None if it doesn't match the expected pattern.
    """
    m = CREMAD_PATTERN.match(path.name)
    if not m:
        return None
    return (
        m.group('actor'),
        m.group('utt'),
        m.group('emo'),
        m.group('intensity'),
    )

def list_cremad_files(root: Path, allowed_emotions: Optional[set] = None) -> List[Path]:
    """
    Recursively list all CREMA-D wavs under `root`. Optionally filter by allowed_emotions (codes like ANG/HAP/...).
    """
    wavs = []
    for p in root.rglob('*.wav'):
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if allowed_emotions is None or emo_code in allowed_emotions:
            wavs.append(p)
    return wavs

def group_by_emotion_cremad(paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Group paths by normalized label name using `cremad_code_to_name`.
    """
    buckets: Dict[str, List[Path]] = {name: [] for name in cremad_code_to_name.values()}
    for p in paths:
        parsed = parse_cremad_filename(p)
        if not parsed:
            continue
        _, _, emo_code, _ = parsed
        if emo_code in cremad_code_to_name:
            label_name = cremad_code_to_name[emo_code]
            buckets[label_name].append(p)
    # Remove empties to avoid surprises
    return {k: v for k, v in buckets.items() if len(v) > 0}

import random
import torch
import numpy as np
from typing import Callable, Dict, List

# You already have these:
# from Preprocess import audio_to_mel_spectrogram
# from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS


def get_emotion_tensor_cremad(
    root: Path,
    label_name: str,
    max_seconds: float,
    normalization_fn=lambda x: x,
    n_samples: int | None = None,
):
    # 1. List all matching WAV paths for this label
    all_paths = list_cremad_files(root, allowed_emotions=allowed_emotions)
    by_label = group_by_emotion_cremad(all_paths)

    if label_name not in by_label or len(by_label[label_name]) == 0:
        raise ValueError(
            f"No CREMA-D files found for label '{label_name}' under {root}"
        )

    chosen = by_label[label_name]

    # 2. Optionally sample n files
    if n_samples is not None and n_samples < len(chosen):
        chosen = random.sample(chosen, n_samples)  # without replacement

    # 3. Process each file → spectrogram tensor
    tensors = []
    for wav_path in chosen:
        mel = audio_to_mel_spectrogram(
            file_path=wav_path,
            max_length_in_seconds=max_seconds,
            normalization_fn=normalization_fn,
        ).astype("float32")
        tensors.append(torch.from_numpy(mel))

    # 4. Stack and adjust shape for CNN input
    batch = torch.stack(tensors)
    if batch.dim() == 3:  # (N, H, W)
        batch = batch.unsqueeze(1)  # → (N, 1, H, W)

    return batch.to(device=device)

def tcav_scores_to_df(scores_by_label: dict, concept_names: list[str]) -> pd.DataFrame:
    """
    Flatten Captum TCAV results into a DataFrame with:
    columns = ["label_name", "concept_name", "layer_name", "positive_sign_count", "positive_magnitude"]
    """
    rows = []
    for label_name, exp_sets in scores_by_label.items():
        # exp_key looks like "0-12" where 0 is the positive concept index, 12 is random/baseline
        for exp_key, layer_dict in exp_sets.items():
            try:
                pos_idx = int(str(exp_key).split("-")[0])
            except Exception:
                continue  # skip malformed keys
            if not (0 <= pos_idx < len(concept_names)):
                continue
            concept_name = concept_names[pos_idx]

            # Usually there's a single chosen layer, but handle multiple layers just in case
            for layer_name, metrics in layer_dict.items():
                sc = metrics.get("sign_count")
                mg = metrics.get("magnitude")
                if sc is None or mg is None:
                    continue

                # Convert torch tensors to Python floats
                if isinstance(sc, torch.Tensor):
                    sc = sc.detach().cpu().tolist()
                if isinstance(mg, torch.Tensor):
                    mg = mg.detach().cpu().tolist()

                # Positive direction = index 0
                rows.append({
                    "label_name": label_name,
                    "concept_name": concept_name,
                    "layer_name": layer_name,
                    "positive_sign_count": float(sc[0]),
                    "positive_magnitude": float(mg[0]),
                })

    return pd.DataFrame(rows, columns=[
        "label_name", "concept_name", "layer_name", "positive_sign_count", "positive_magnitude"
    ])

label_inputs = {
    label_name: get_emotion_tensor_cremad(
            root=CREMAD_ROOT,
            label_name=label_name,
            max_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
            normalization_fn=lambda x: x,
            n_samples=N_SAMPLES_PER_LABEL,
    )  # -> [B, 1, H, W]; remove if your mel already has channel dim
    for label_name in label_emotion_mapping.values()
}

# -----------------------------
# 1️⃣ Load pretrained model
# -----------------------------
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

# -----------------------------
# 3️⃣b Compute TCAV per-sample
# -----------------------------
tcav_scores_per_sample = {}

for label_index, label_name in label_emotion_mapping.items():
    tcav_scores_per_label[label_name] = tcav.interpret(
        inputs=label_inputs[label_name],
        experimental_sets=[[c, random_concept] for c in positive_concepts],
        target=label_index  # integer index of target class
    )



# Now tcav_scores_per_sample[label_name][i] contains the TCAV results for the i-th sample.

# -----------------------------
# 4️⃣ Inspect results
# -----------------------------
df_tcav = tcav_scores_to_df(tcav_scores_per_label, concept_unique_names)
df_tcav.to_csv("crema_d_tcav_results.csv", index=False, encoding="utf-8")
# for label_name, score_dict in tcav_scores_per_label.items():
#      print(f"TCAV scores for label {label_name}: {score_dict}")

# pprint(f"TCAV scores for label {'angry'}: {tcav_scores_per_label['angry']}", depth=1)

# df_tcav = tcav_scores_to_df(tcav_scores_per_label, concept_unique_names)

display(df_tcav)
