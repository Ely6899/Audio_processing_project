import random
from gradcam_initilaization import *
from typing import Dict, List, Optional

import os
import sys

current_directory = os.getcwd()
print(f"Current Working Directory (os.getcwd()): {current_directory}")
# Get the path object for the current file
current_file_path = Path(__file__).resolve()

# Get the path object for the parent directory
parent_dir_path = current_file_path.parents[2]

# Add the parent directory to sys.path
sys.path.append(str(parent_dir_path))
print(sys.path)

PANDAS_FLAG: bool | None = True
RECORDINGS_TO_PROCESS = []
if PANDAS_FLAG is True:
    df = pd.read_csv(Path("attributes/all_attributes.csv"), usecols=['path'], quoting=csv.QUOTE_NONE,
                     encoding='utf-8', engine='python', dtype=str)
    # Filter only rows where prediction matches true label
    # df_match = df[df["predicted_label"] == df["true_label"]]
    for wav in df['path']:
        wav_path = Path(wav.replace("\\", "/").strip())
        wav_path = Path(os.path.join(parent_dir_path, wav_path))
        if is_valid_ravdess_file(wav_path):
            RECORDINGS_TO_PROCESS.append(wav_path)
else:
    RECORDINGS_TO_PROCESS = [
        wav for wav in RavdessPaths.AUDIO_ORIGINAL_DATA.rglob("*.wav")
        if is_valid_ravdess_file(wav)
    ]
    
    # take spectograms that have predicted == true 
    # same preprocess as in gradcam utils
    # sort paths according to label 
    # save each label paths in a tuple of tensors
    
def group_paths_by_label(paths: List[Path]) -> Dict[str, List[Path]]:
    by_label = {}
    for p in paths:
        try:
            emotion, _ = wav_indexer(Path(p))
            by_label.setdefault(emotion, []).append(Path(p))
        except Exception as e:
            print(
                f"[WARN] Could not extract emotion label from file '{p.name}'. "
                f"Skipping this file. (Error: {type(e).__name__}: {e})"
            )
    return by_label

def sample_equal_counts(by_label: Dict[str, List[Path]],
                        samples_per_label: int,
                        seed: Optional[int] = 42,
                        strict: bool = True) -> Dict[str, List[Path]]:
    """
    Sample exactly 'samples_per_label' per label.
    - strict=True: raise if any label has fewer than requested.
    - strict=False: take as many as available (<= requested).
    """
    rng = random.Random(seed)
    sampled: Dict[str, List[Path]] = {}
    for lbl, paths in by_label.items():
        paths = paths.copy()
        rng.shuffle(paths)
        if len(paths) < samples_per_label:
            if strict:
                raise ValueError(f"Label '{lbl}' has only {len(paths)} recordings "
                                 f"(requested {samples_per_label}).")
            sampled[lbl] = paths  # undersample
        else:
            sampled[lbl] = paths[:samples_per_label]
    return sampled

def build_spectrogram_tensor(file_path: Path) -> torch.Tensor:
    """
    Uses your existing preprocessing:
      audio_to_mel_spectrogram(file_path=..., max_length_in_seconds=MAX_..., normalization_fn=lambda x: x)
      .astype('float32')
    Returns a torch.Tensor [n_mels, time] (or whatever your function outputs).
    """
    mel: np.ndarray = audio_to_mel_spectrogram(
        file_path=file_path,
        max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
        normalization_fn=lambda x: x,  # you can swap in your real normalization here
    ).astype("float32")
    return torch.from_numpy(mel)  # shape should already be consistent across files

def make_label_spectrograms(
    recordings: List[Path],
    samples_per_label: int,
    seed: Optional[int] = 42,
    strict: bool = True,
    save_dir: Optional[Path] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Main entry:
      - recordings: list of wav paths (RECORDINGS_TO_PROCESS)
      - samples_per_label: equal count to sample per label
      - strict: if True, require enough files per label; if False, allow fewer
      - save_dir: if provided, saves one .pt file per label with the list of tensors
    Returns: dict {label: [tensor, tensor, ...]}
    """
    # 1) group by label
    by_label = group_paths_by_label([Path(p) for p in recordings])

    # 2) sample equal counts per label
    sampled_paths = sample_equal_counts(by_label, samples_per_label, seed=seed, strict=strict)

    # 3) build spectrogram tensors per label
    label_tensors: Dict[str, List[torch.Tensor]] = {}
    for lbl, paths in sampled_paths.items():
        tensors = []
        for wav_path in paths:
            try:
                spec = build_spectrogram_tensor(wav_path)
                tensors.append(spec)
            except Exception as e:
                # Skip problematic files but continue others
                print(f"[WARN] Failed on {wav_path}: {e}")
        if len(tensors) == 0:
            # drop labels that ended up empty (e.g., all failures)
            continue
        label_tensors[lbl] = tensors
        
    # 4) optional: persist each label’s list as a single .pt
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for lbl, tensors in label_tensors.items():
            torch.save(tensors, save_dir / f"{lbl}_spectrograms.pt")

    return label_tensors
        


## FOR TESTS:
if __name__ == "__main__":
    # Assume you already filled RECORDINGS_TO_PROCESS (list[Path or str])
    SAMPLES_PER_LABEL = 50  # set as you like

    label_to_specs = make_label_spectrograms(
        recordings=RECORDINGS_TO_PROCESS,
        samples_per_label=SAMPLES_PER_LABEL,
        seed=123,
        strict=True,                 # flip to False to allow undersampling
        save_dir=Path("tcav/input_specs")  # or None to skip saving
    )
    
    # print(label_to_specs)