import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AnyStr, Iterable, Optional
from typing import Tuple
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ConstPaths import RavdessPaths, CremaPaths, TessPaths
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS, LABEL_STRINGS
from Preprocess import audio_to_mel_spectrogram


class AudioRawData(ABC):
    """
    Wrapper abstract class to handle Dataset saving in run-time.
    """
    def __init__(self, data_root: Path, supported_formats: set[str], file_pattern: AnyStr = r".*"):
        self._data_root: Path = data_root
        self._supported_formats: set[str] = supported_formats
        self._data: set[Tuple[Path, Any]] = self._scan_supported_files(file_pattern=file_pattern) # File/s path/s, Label.  used as (Path, str) and (Dict[Path, Path], str).

        try:
            self._file_paths, self._file_labels = zip(*list(self._data))
        except ValueError:
            raise ValueError(f"Error: No data found in {data_root} with the given supported formats and file pattern.")

    def _scan_supported_files(self, file_pattern: AnyStr) -> set[Tuple[Path, Any]]:
        base_name_pattern = re.compile(file_pattern)

        files = {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file()
               and any(file.name.endswith(suffix) for suffix in self._supported_formats)
               and base_name_pattern.match(file.stem)
        }

        result = set(map(lambda x: (x, self.emotion_indexer(x)), files))
        return result

    @abstractmethod
    def emotion_indexer(self, file_path: Path) -> str:
        pass

    @property
    def all_data(self) -> set[Tuple[Path, Any]]:
        return self._data

    def print_all_label_counts(self) -> None:
        counter: dict = dict(Counter(self._file_labels))
        print(f"Total label counts: {counter}")

class RavdessRawData(AudioRawData):
    def __init__(self, include_calm: bool = False, include_aug: bool = False):
        self._include_calm = include_calm
        self._include_aug = include_aug
        super().__init__(RavdessPaths.AUDIO_ORIGINAL_DATA, {".wav"}, r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

    def emotion_indexer(self, file_path: Path) -> str:
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', file_path.name.__str__())

        index_emotion_mapping = {
            '01': LABEL_STRINGS.NEUTRAL,
            '02': LABEL_STRINGS.NEUTRAL if not self._include_calm else LABEL_STRINGS.CALM,
            '03': LABEL_STRINGS.HAPPY,
            '04': LABEL_STRINGS.SAD,
            '05': LABEL_STRINGS.ANGRY,
            '06': LABEL_STRINGS.FEARFUL,
            '07': LABEL_STRINGS.DISGUSTED,
            '08': LABEL_STRINGS.SURPRISED
        }

        emotion_index = numbers[2]
        emotion = index_emotion_mapping[emotion_index]
        return emotion

class CREMARawData(AudioRawData):
    def __init__(self):
        super().__init__(CremaPaths.WAV_DATA, {".wav"}, r"^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}$")

    def emotion_indexer(self, file_path: Path) -> str:
        """
        The sentences were presented using different emotion (in parentheses is the three-letter code used in the third part of the filename):

        Anger (ANG)
        Disgust (DIS)
        Fear (FEA)
        Happy/Joy (HAP)
        Neutral (NEU)
        Sad (SAD)

        :param file_name: Name of the file to process.
        :return: String of emotion indexing.
        """
        index_emotion_mapping = {
            'ANG': LABEL_STRINGS.ANGRY,
            'DIS': LABEL_STRINGS.DISGUSTED,
            'FEA': LABEL_STRINGS.FEARFUL,
            'HAP': LABEL_STRINGS.HAPPY,
            'NEU': LABEL_STRINGS.NEUTRAL,
            'SAD': LABEL_STRINGS.SAD
        }

        pattern = re.compile(r'^\d{4}_[A-Z]{3}_([A-Z]{3})_[A-Z]{2}\.wav$', re.IGNORECASE)
        match = pattern.match(file_path.name)

        if not match:
            raise ValueError(f"Filename format not recognized: {file_path}")

        emotion_code = match.group(1).upper()
        emotion = index_emotion_mapping.get(emotion_code)

        if emotion is None:
            raise ValueError(f"Unknown emotion code '{emotion_code}' in file: {file_path}")

        return emotion

class TESSRawData(AudioRawData):
    def __init__(self, data_root: Path = TessPaths.ALL_DATA):
        super().__init__(data_root, {".wav"})

    def emotion_indexer(self, file_path: Path) -> str:
        
        file_emotion_mapping = {
            'angry': LABEL_STRINGS.ANGRY,
            'disgust': LABEL_STRINGS.DISGUSTED,
            'fear': LABEL_STRINGS.FEARFUL,
            'happy': LABEL_STRINGS.HAPPY,
            'neutral': LABEL_STRINGS.NEUTRAL,
            'ps': LABEL_STRINGS.SURPRISED,  # 'ps' stands for 'pleasant surprised'
            'sad': LABEL_STRINGS.SAD
        }
        
        # Try to find a known emotion token in the filename stem (robust to different naming conventions)
        stem = file_path.stem
        parts = stem.split('_')
        emotion_index = 2
        emotion_ds_id = parts[emotion_index]
        return file_emotion_mapping[emotion_ds_id]


class AllRawData:
    def __init__(self, raw_datasets: tuple[AudioRawData, ...], val_ratio: float = 0.3):
        sets_of_data = (raw_dataset.all_data for raw_dataset in raw_datasets)
        self._all_raw_data = set().union(*sets_of_data)
        self._train_data, self._val_data = self.train_val_test_split(val_ratio=val_ratio)

    def train_val_test_split(self, val_ratio: float = 0.3):
        self._train_data, self._val_data = train_test_split(list(self._all_raw_data), test_size=val_ratio, stratify=[label for _, label in self._all_raw_data], random_state=42)
        return self._train_data, self._val_data

    @property
    def all_data(self) -> set[tuple[Path, Any]]:
        return self._all_raw_data

    @property
    def train_data(self) -> set[tuple[Path, Any]]:
        return self._train_data

    @property
    def val_data(self) -> set[tuple[Path, Any]]:
        return self._val_data

    def print_all_label_counts(self) -> None:
        label_counts_all =  Counter([label for _, label in self._all_raw_data])
        label_counts_train = Counter([label for _, label in self.train_data])
        label_counts_val = Counter([label for _, label in self.val_data])
        print(f"Labels all: {label_counts_all}\n"
              f"Labels train: {label_counts_train}\n"
              f"Labels validation: {label_counts_val}")

    def remove_labels(self, label_list: tuple[str] = ('surprised', 'calm')) -> None:
        self._all_raw_data = set(((file_name, label) for file_name, label in self._all_raw_data if label not in label_list))


class SplitttedAudioRawData:
    def __init__(self, train_raw_data: AudioRawData, val_raw_data: AudioRawData, test_raw_data: Optional[AudioRawData], supported_formats: set[str], file_pattern: AnyStr = r".*"):
        self._train_raw_data: AudioRawData = train_raw_data
        self._val_raw_data: AudioRawData = val_raw_data
        self._test_raw_data: Optional[AudioRawData] = test_raw_data
        self._supported_formats: set[str] = supported_formats
        self._file_pattern: AnyStr = file_pattern
    
    @property
    def train_data(self) -> set[Tuple[Path, Any]]:
        return self._train_raw_data.all_data

    @property
    def val_data(self) -> set[Tuple[Path, Any]]:
        return self._val_raw_data.all_data

    @property
    def test_data(self) -> Optional[set[Tuple[Path, Any]]]:
        return self._test_raw_data.all_data if self._test_raw_data is not None else set()

    @property
    def all_data(self) -> set[Tuple[Path, Any]]:
        return self.train_data | self.val_data | self.test_data
    

class TessSplitttedRawData(SplitttedAudioRawData):
    def __init__(self):
        tess_train = TESSRawData(TessPaths.TRAIN_DATA)
        tess_val = TESSRawData(TessPaths.TEST_DATA)
        tess_test = None # no test set in TESS
        super().__init__(tess_train, tess_val, tess_test, {".wav"})
    
    
class EmotionSpecDataset(Dataset):
    def __init__(self, file_paths: Iterable, max_length_in_seconds: float = MAX_SPECTOGRAM_DURATION_IN_SECONDS):
        self._data = list(file_paths)
        self._paths , self._labels = zip(*self._data)

        self.__label_encoder = LabelEncoder()
        self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))
        self._class_names = list(self.__label_encoder.classes_)

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

        self.max_length_in_seconds = max_length_in_seconds

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_path = self._paths[idx]
        label = self._labels[idx]

        # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
        mel_spectrogram = audio_to_mel_spectrogram(file_path=file_path, max_length_in_seconds=self.max_length_in_seconds)

        # Convert to torch.Tensor
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        
        # Now expand to shape (1, freq_bins, time_frames)
        mel_spectrogram = mel_spectrogram.unsqueeze(dim=0)
        
        label = label.long()

        return mel_spectrogram, label

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_counts(self):
        return self._class_counts

    def decode_label(self, encoded_label):
        return self.__label_encoder.inverse_transform([encoded_label])[0]

    def __compute_class_weights(self) -> torch.Tensor:
        """
        Computes class weights based on the frequency of each class in the dataset.

        Returns:
            torch.Tensor: Tensor of class weights (inverse frequency).
        """
        class_counts = torch.bincount(self._labels, minlength=self.num_classes)
        self._class_counts = class_counts
        total_samples = len(self._labels)
        class_weights = total_samples / (class_counts + 1e-6)  # Avoid division by zero
        return class_weights.float()

class EmotionSpecDataset2d(Dataset):
    def __init__(self, data: set):
        self._data = data
        self._paths , self._labels = zip(*self._data)


        self.__label_encoder = LabelEncoder()
        self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))
        self._class_names = list(self.__label_encoder.classes_)

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_pair = self._paths[idx]
        label = self._labels[idx]

        # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
        original_mel_spectrogram = audio_to_mel_spectrogram(file_path=file_pair[0]) # original audio
        synthesized_mel_spectrogram = audio_to_mel_spectrogram(file_path=file_pair[1]) # synthesized audio
        
        # Convert to torch.Tensor
        original_mel_spectrogram = torch.from_numpy(original_mel_spectrogram).float()
        synthesized_mel_spectrogram = torch.from_numpy(synthesized_mel_spectrogram).float()
        
        # now add the two spectrograms to create a 2D tensor with shape (2, freq_bins, time_frames)
        mel_spectrogram = torch.stack((original_mel_spectrogram, synthesized_mel_spectrogram), dim=0)

        label = label.long()

        return mel_spectrogram, label

    @property
    def class_names(self):
        return self._class_names

    def decode_label(self, encoded_label):
        return self.__label_encoder.inverse_transform([encoded_label])[0]

    def __compute_class_weights(self) -> torch.Tensor:
        """
        Computes class weights based on the frequency of each class in the dataset.

        Returns:
            torch.Tensor: Tensor of class weights (inverse frequency).
        """
        class_counts = torch.bincount(self._labels, minlength=self.num_classes)
        total_samples = len(self._labels)
        class_weights = total_samples / (class_counts + 1e-6)  # Avoid division by zero
        return class_weights.float()

