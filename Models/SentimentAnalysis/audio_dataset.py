from abc import ABC, abstractmethod
from typing import Tuple
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
from Models.SentimentAnalysis.Preprocess import audio_to_mel_spectogram

class AudioRawData(ABC):
    def __init__(self, data_root: Path, supported_files: set):
        self._data_root = data_root
        self._supported_files = supported_files
        self._file_paths: set = self._scan_supported_files()

        self._train_data, self._val_data, self._test_data = self._train_val_test_split()

    @abstractmethod
    def _scan_supported_files(self) -> set:
        pass

    @abstractmethod
    def _train_val_test_split(self, test_size=0.2, val_size=0.1, random_state=None)-> Tuple[set, set, set]:
        pass

    @property
    def all_data(self) -> set:
        return self._file_paths

    @property
    def train_data(self) -> set:
        return self._train_data

    @property
    def val_data(self) -> set:
        return self._val_data

    @property
    def test_data(self) -> set:
        return self._test_data

class RavdessRawData(AudioRawData):

    def __init__(self):
        super().__init__(RavdessPaths.AUDIO_FILES_DATA, {".wav"})

    def _scan_supported_files(self) -> set:
        return {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_files)
        }

    def _train_val_test_split(self, test_size=0.2, val_size=0.1, random_state=None) -> Tuple[set, set, set]:
        train_files, temp_files = train_test_split(list(self._file_paths), test_size=test_size + val_size,
                                                   random_state=random_state)

        val_size_adj = val_size / (test_size + val_size)
        val_files, test_files = train_test_split(temp_files, test_size=1 - val_size_adj, random_state=random_state)

        return set(train_files), set(val_files), set(test_files)

def get_emotion_from_index(filename) -> str:
    numbers = re.findall(r'\d+', filename.name.__str__())

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
    emotion = index_emotion_mapping[emotion_index]
    return emotion


class EmotionDataset(Dataset):
    def __init__(self, file_paths: set):
        self._data = list(file_paths)
        self._labels = list(filter(None, map(get_emotion_from_index, self._data)))

        self.__label_encoder = LabelEncoder()
        self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_path = self._data[idx]
        label = self._labels[idx]

        mel_spectrogram = audio_to_mel_spectogram(file_path=file_path)
        label = label.long()

        return mel_spectrogram, label

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
