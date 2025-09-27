from xml.parsers.expat import model
from IPython.display import display

from pathlib import Path
# from captum.attr import LayerActivation
# from functorch.dim import Tensor #! makes a bug because functorch.dim isn't supported in python 3.12 !!
from pprint import pprint
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from captum.concept import TCAV, Concept
from torch.utils.data import DataLoader, Dataset
import tqdm

from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import LABEL_STRINGS, TARGET_FRAMES, FREQUENCY_BIN_COUNT
from concepts_creation import generate_random_pattern_spectrogram
from tqdm import tqdm

from ConstPaths import TessPaths, conceptPaths


CONCEPT_UNIQUE_NAMES = [
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

INDEX_EMOTION_MAPPING = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

LABEL_EMOTION_MAPPING = {
    0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised',
}


# PyTorch Datasets for TCAV

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
    PyTorch Dataset that pre-generates the dataset for a specific concept in memory.
    """

    def __init__(self, n_samples: int, concept_name: str, root_concept_dir: Path = conceptPaths.ALL_CONCEPTS, freq_count = FREQUENCY_BIN_COUNT, frames_count = TARGET_FRAMES, rng_seed: Optional[int] = None):
        self.n_samples = n_samples
        self.concept_name = concept_name
        self.root_concept_dir = root_concept_dir
        self.freq_count = freq_count
        self.frames = frames_count
        self.rng = np.random.default_rng(rng_seed)

        # load all .npy files from root_concept_dir/concept_name
        self.data = []
        concept_dir = self.root_concept_dir / self.concept_name
        concept_dir.mkdir(exist_ok=True)
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

# Functions

def init_tcav_with_pamalia_dict(model_path: Optional[Path] = Path("ResNetWithAttention.pt")):
    # -----------------------------
    # 1️⃣ Load pretrained model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
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
    tcav = TCAV(model, [layer], test_split_ratio=0.33)


    positive_concepts: list[Concept] = [Concept(id=concept_idx, name=concept_name, data_iter=DataLoader(PreGeneratedConceptDataset(n_samples=100, concept_name=concept_name), shuffle=False))
                                for concept_idx, concept_name in enumerate(CONCEPT_UNIQUE_NAMES)]

    # This concept is the negative of concepts.
    negative_concept_dataset = PreGeneratedRandomSpectrogramDataset(n_samples=100, freq_count=FREQUENCY_BIN_COUNT, frames=TARGET_FRAMES)
    random_concept = Concept(id=len(positive_concepts), name='random', data_iter=DataLoader(negative_concept_dataset, shuffle=False))
    
    return {'tcav': tcav, 'positive_concepts': positive_concepts, 'random_concept': random_concept, 'layer': layer}


def _compute_cav_accuracy_df(tcav: TCAV,
                             positive_concepts: List[Concept],
                             random_concept: Concept) -> pd.DataFrame:
    """
    Trains / loads CAVs once and extracts the linear concept-classifier accuracy
    per (concept, layer). Returns a DataFrame with columns:
    [concept_name, layer_name, cav_acc]
    """
    # One experimental set per concept: [concept, random]
    experimental_sets = [[c, random_concept] for c in positive_concepts]

    # Train / load CAVs for all concepts & layers in one shot
    cavs_dict = tcav.compute_cavs(experimental_sets, force_train=False)

    rows = []
    # cavs_dict maps "<id>-<id>-..." -> {layer_name: CAV}
    for concepts_key, layer_map in cavs_dict.items():
        try:
            pos_id = int(str(concepts_key).split("-")[0])  # first id is the positive concept id
        except Exception:
            continue
        if not (0 <= pos_id < len(positive_concepts)):
            continue
        concept_name = positive_concepts[pos_id].name

        for layer_name, cav_obj in layer_map.items():
            if cav_obj is None or cav_obj.stats is None:
                continue
            acc = cav_obj.stats.get("accs", None)  # DefaultClassifier returns {"accs": <tensor/float>}
            if isinstance(acc, torch.Tensor):
                acc = acc.detach().cpu().item()
            rows.append({
                "concept_name": concept_name,
                "layer_name": layer_name,
                "cav_acc": float(acc) if acc is not None else np.nan,
            })

    return pd.DataFrame(rows, columns=["concept_name", "layer_name", "cav_acc"])


def _tcav_dict_per_sample_to_df(scores_by_sample: dict, concept_names: list[str]) -> pd.DataFrame:
    # """
    # Flatten Captum TCAV results into a DataFrame with:
    # columns = ["label_name", "concept_name", "layer_name", "positive_percentage", "magnitude"]
    # """
    rows = []
    for path, exp_sets in scores_by_sample.items():
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
                    "path": path,
                    "concept_name": concept_name,
                    "layer_name": layer_name,
                    "positive_percentage": float(sc[0]),
                    "magnitude": float(mg[0]),
                })
    per_sample_df = pd.DataFrame(rows, columns=[
        "path", "concept_name", "layer_name", "positive_percentage", "magnitude"
    ])
    
    tcav_dict = init_tcav_with_pamalia_dict()
    acc_df = _compute_cav_accuracy_df(tcav=tcav_dict['tcav'], positive_concepts=tcav_dict['positive_concepts'], random_concept=tcav_dict['random_concept'])
    # acc_df has columns: ["concept_name", "layer_name", "cav_acc"]
    # merge each row of acc_df with every row in per_sample_df that has the same concept_name and layer_name
    per_sample_acc_df = per_sample_df.merge(acc_df, on=["concept_name", "layer_name"], how="left")
    return per_sample_acc_df


# all_filtered_data is for droping men samples and/or false positive samples 
def _get_tcav_dict_per_sample(all_filtered_data: pd.DataFrame, model_path: Optional[Path] = Path("ResNetWithAttention.pt")) -> dict: 
    tcav_dict = init_tcav_with_pamalia_dict(model_path=model_path)
    tcav = tcav_dict['tcav']
    positive_concepts = tcav_dict['positive_concepts']
    random_concept = tcav_dict['random_concept']
    
    # Debug call, don't uncomment
    # show_arrays_in_separate_windows(negative_concept_dataset.get_data)

    print("Reached tcav interpret")
    
    tcav_dict_per_sample = {}
    
    # for row(pandas series) in df:
    for i, row in tqdm(all_filtered_data.iterrows(), total=len(all_filtered_data), desc="Processing samples"):
        label_name = row['predicted_label']
        path = row['path']
        sample = torch.tensor(audio_to_mel_spectrogram(Path(path)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
        label_2_index = {LABEL_STRINGS.ANGRY: 0,
                         LABEL_STRINGS.DISGUSTED: 1,
                         LABEL_STRINGS.FEARFUL: 2,
                         LABEL_STRINGS.HAPPY: 3,
                         LABEL_STRINGS.NEUTRAL: 4,
                         LABEL_STRINGS.SAD: 5,
                         LABEL_STRINGS.SURPRISED: 6}
        label_index = label_2_index.get(label_name)
        tcav_dict_per_sample[path] = {}
        
        score_for_label = tcav.interpret(
                inputs=sample,
                experimental_sets=[[c, random_concept] for c in positive_concepts],
                target=label_index
            )
        
        tcav_dict_per_sample[path] = score_for_label
        
    
    return tcav_dict_per_sample

def get_tcav_per_sample(attribute_csv_path: Path, model_path: Optional[Path]) -> pd.DataFrame:
    df_attributes = pd.read_csv(attribute_csv_path)

    # ## !debug:
    # df_attributes = df_attributes.head(10)
    # ## !debug
    

    # drop unnecessary columns
    df_attributes.drop(columns=df_attributes.filter(regex=r'^prob ').columns, inplace=True)

    dic = _get_tcav_dict_per_sample(df_attributes, model_path=model_path)
    
    df_tcav = _tcav_dict_per_sample_to_df(dic, CONCEPT_UNIQUE_NAMES)
    
    # create a new df, which is df_tcav but added attributes from df_attributes based on the 'path' column
    df_merged = df_tcav.merge(df_attributes, on='path', how='left')

    # # rearrange columns in a custom order
    # desired_order = ['path', 'true_label', 'predicted_label', 'predicted_probability', 'concept_name', 'layer_name', 'positive_percentage', 'magnitude']  # Specify the desired order
    # df_merged = df_merged[desired_order]
    
    return df_merged


if __name__ == "__main__":
    df_merged = get_tcav_per_sample(attribute_csv_path=TessPaths.PROB_VECTOR_SHUFFLED, model_path=Path(r"TESS\models\2025-09-25_11-23-34\ResNetWithAttention_Tess_spk_shuffeled.pt"))
    df_merged.to_csv("Tcav_Tess_spk_shuffeled.csv", index=False)
