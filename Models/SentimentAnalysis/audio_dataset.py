from collections import Counter as LabelCounter
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Any
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from ConstPaths import RavdessPaths, TessPaths
from Preprocess import audio_to_mel_spectrogram, audio_to_waveform

class AudioRawData(ABC):
    """
    Wrapper abstract class to handle Dataset saving in run-time.
    """
    def __init__(self, data_root: Path, supported_formats: set[str]):
        self._data_root: Path = data_root
        self._supported_formats: set[str] = supported_formats
        self._data: set[Tuple[Path, Any]] = self._scan_supported_files() # Path, Label

        self._file_paths, self._file_labels = zip(*list(self._data))
        self._train_data, self._val_data, self._test_data = self._train_val_test_split()

    def _train_val_test_split(self, test_size: float=0.2, val_size: float=0.1, random_state=42) -> Tuple[set, set, set]:
        """
        Applies stratified train_val_test split.
        @param test_size: Percentage of test size.
        @param val_size: Percentage of val size.
        @param random_state: Put a specific number to ensure determinism.
        @return: Three sets of Train, Val, Test.
        """
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self._file_paths, self._file_labels, test_size=0.2, stratify=self._file_labels, random_state=random_state
        )

        # Validation + Test split (50% val, 50% test from temp, making each 10% of total)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=random_state
        )

        # Convert back to sets
        train_set = set(zip(train_paths, train_labels))
        val_set = set(zip(val_paths, val_labels))
        test_set = set(zip(test_paths, test_labels))

        return train_set, val_set, test_set

    @abstractmethod
    def _scan_supported_files(self) -> set:
        pass

    @property
    def all_data(self) -> set:
        return self._data

    @property
    def train_data(self) -> set:
        return self._train_data

    @property
    def val_data(self) -> set:
        return self._val_data

    @property
    def test_data(self) -> set:
        return self._test_data

    def labels_count_total(self) -> dict:
        counter: dict = dict(LabelCounter(self._file_labels))
        return counter

    def labels_count_train(self) -> dict:
        _, train_labels = zip(*list(self._train_data))
        counter: dict = dict(LabelCounter(train_labels))
        return counter

    def labels_count_val(self) -> dict:
        _, val_labels = zip(*list(self._val_data))
        counter: dict = dict(LabelCounter(val_labels))
        return counter

    def labels_count_test(self) -> dict:
        _, test_labels = zip(*list(self._test_data))
        counter: dict = dict(LabelCounter(test_labels))
        return counter

    def print_all_label_counts(self) -> None:
        print(f"Total label counts: {self.labels_count_total()}")
        print(f"Train label counts: {self.labels_count_train()}")
        print(f"Val label counts: {self.labels_count_val()}")
        print(f"Test label counts: {self.labels_count_test()}")


class RavdessRawData(AudioRawData):
    def __init__(self):
        super().__init__(RavdessPaths.AUDIO_FILES_DATA, {".wav"})

    def _scan_supported_files(self) -> set[Tuple[Path, str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """
        files = {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = set(map(lambda x: (x, RavdessRawData.__get_emotion_from_index(x)), files))
        return result

    @staticmethod
    def __get_emotion_from_index(filename):
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
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

class TessRawData(AudioRawData):
    def __init__(self):
        super().__init__(TessPaths.AUDIO_FILES_DATA, {".wav"})

    def _scan_supported_files(self) -> set[Tuple[Path, str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the audio file name.
        @return: A set of tuples, each tuple holds (file path, label).
        """
        files = {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = set(map(lambda x: (x, TessRawData.__get_emotion_from_filename(x)), files))
        return result

    @staticmethod
    def __get_emotion_from_filename(filename):
        """
        For TESS, label is indicated in the last word in the name(before the file extention). This function handles mapping it to a
        readable label.
        @param filename: File path from TESS dataset.
        @return: Label of the file according to the filename.
        """
        # Extract filename without extension
        stem = filename.stem  # Removes .wav or other extensions

        # Split by underscores or spaces (TESS filenames typically use underscores)
        words = stem.split("_")
        
        # The last word is the emotion label
        emotion = words[-1]
        
        # if the label is ps - return pleasant surprise
        if words[-1] == "ps":
            emotion = "pleasant surprise"
        
        return emotion

class EmotionSpecDataset(Dataset):
    def __init__(self, file_paths: set):
        self._data = list(file_paths)
        self._paths , self._labels = zip(*self._data)

        self.__label_encoder = LabelEncoder()
        self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_path = self._paths[idx]
        label = self._labels[idx]

        # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
        mel_spectrogram = audio_to_mel_spectrogram(file_path=file_path)

        # Convert to torch.Tensor
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        
        # Now expand to shape (1, freq_bins, time_frames)
        mel_spectrogram = mel_spectrogram.unsqueeze(dim=0)
        
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

class EmotionWaveDataset(Dataset):
    def __init__(self, file_paths: set):
        self._data = list(file_paths)
        self._paths , self._labels = zip(*self._data)

        self.__label_encoder = LabelEncoder()
        self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))

        # Get the number of classes
        self.num_classes = len(self.__label_encoder.classes_)

        # Compute class weights
        self.class_weights = self.__compute_class_weights()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_path = self._paths[idx]
        label = self._labels[idx]

        waveform = audio_to_waveform(file_path=file_path)
        label = label.long()

        return waveform, label

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
