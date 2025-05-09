from collections import Counter as LabelCounter
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Any, Dict
import re
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from ConstPaths import RavdessPaths, TessPaths
from Models.SentimentAnalysis.PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from Preprocess import audio_to_mel_spectrogram, audio_to_waveform
import os
"""
MICHAL ADDED - 4.5.2025
"""
from sklearn.model_selection import train_test_split
import random
from typing import Tuple, Set
# from split_by_speaker import make_actor_split, ORIGINAL_RE   # import from the helper
from split_by_speaker import (
    list_actor_dirs,
    collect_files,          # includes/excludes augmentations
)


class AudioRawData(ABC):
    """
    Wrapper abstract class to handle Dataset saving in run-time.
    """
    def __init__(self, data_root: Path, supported_formats: set[str]):
        self._data_root: Path = data_root
        self._supported_formats: set[str] = supported_formats
        self._data: set[Tuple[Path | Dict[Path, Path], Any]] = self._scan_supported_files() # File/s path/s, Label.  used as (Path, str) and (Dict[Path, Path], str).

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
    def __init__(self, raw_data_root = RavdessPaths.AUDIO_ORIGINAL_DATA,include_calm: bool = False, include_aug: bool = True):
        self._include_calm = include_calm
        self._include_aug = include_aug
        super().__init__(raw_data_root, {".wav"})
        
    
    """
    MICHAL -  ADD IN 5.5
    """
    def _train_val_test_split(
        self,
        test_size: float = 0.2,     # overall test share (relative to *all* data)
        val_size:  float = 0.1,     # overall val  share
        random_state: int = 42,
    ) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """
        Speaker-clean split that mimics the original logic:
            1) Split actors into TRAIN  vs  TEMP (VAL+TEST) by the combined fraction
            (val_size + test_size).
            2) Split TEMP 50/50 into VAL and TEST.
        • TRAIN  : originals + augmentations
        • VAL    : originals only
        • TEST   : originals only  (change `include_aug` below if you prefer)
        """

        # ------------------------------------------------------------------ #
        # 1)  Build actor lists                                              #
        # ------------------------------------------------------------------ #
        data_root = Path(self._data_root)                # .../Audio_Speech_Actors_01-24
        actor_dirs    = list_actor_dirs(data_root)           # 24 actor folders
        """
        - in order to keep track of the results or noise of a specific speaker , no shuffle 
        """
        # random.seed(random_state)
        # random.shuffle(actor_dirs)

        # How many actors go to TEMP (val + test)?
        temp_fraction = test_size + val_size             # e.g. 0.3  (20 % + 10 %)
        n_temp = max(1, round(len(actor_dirs) * temp_fraction))
        temp_actor_dirs  = actor_dirs[:n_temp]
        train_actor_dirs = actor_dirs[n_temp:]

        # ------------------------------------------------------------------ #
        # 2)  Split TEMP by val_size and test_size  →  VAL / TEST                                #
        # ------------------------------------------------------------------ #
        desired_ratio = val_size / (val_size + test_size)   # 0.1 / 0.3 ≈ 0.333
        n_val  = max(1, round(len(temp_actor_dirs) * desired_ratio))
        n_test = len(temp_actor_dirs) - n_val               # remainder

        val_actor_dirs  = temp_actor_dirs[:n_val]
        test_actor_dirs = temp_actor_dirs[n_val:]

        print(train_actor_dirs)
        print(val_actor_dirs)
        print(test_actor_dirs)

        # ------------------------------------------------------------------ #
        # 3)  Collect wav paths                                              #
        # ------------------------------------------------------------------ #
        train_paths = collect_files(train_actor_dirs, include_aug=self._include_aug)
        val_paths   = collect_files(val_actor_dirs,   include_aug=False)
        test_paths  = collect_files(test_actor_dirs,  include_aug=False)  # set True if you want augments

        # ------------------------------------------------------------------ #
        # 4)  Turn them into the expected { (Path, label) } sets             #
        # ------------------------------------------------------------------ #
        def label(p: Path) -> str:
            return self.__get_emotion_from_index(p)

        train_set = {(p, label(p)) for p in train_paths}
        val_set   = {(p, label(p)) for p in val_paths}
        test_set  = {(p, label(p)) for p in test_paths}

        return train_set, val_set, test_set


    def _scan_supported_files(self) -> set[Tuple[Path, str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """
        files = {
            file for file in Path(self._data_root).rglob('*')
            if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = set(map(lambda x: (x, self.__get_emotion_from_index(x)), files))
        return result

    def __get_emotion_from_index(self, filename):
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', filename.name.__str__())

        index_emotion_mapping = {
            '01': 'neutral',
            '02': 'neutral' if not self._include_calm else 'calm',
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
    def __init__(self, file_paths: set, max_length_in_seconds: float = MAX_SPECTOGRAM_DURATION_IN_SECONDS):
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

# add RavdessRawData in which every sample is two audio-file-paths: file2classify and originalneutral
class RavdessRawDataWithNeutral(AudioRawData):
    def __init__(self, include_calm = True, include_aug=False):
        self._include_aug = include_aug
        self._include_calm = include_calm
        super().__init__(RavdessPaths.ALL_AUDIO_DATA, {".wav"})

    def _scan_supported_files(self) -> set[Tuple[Tuple[Path, Path], str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """
        self.neutral_data = Path(os.path.join(self._data_root, RavdessPaths.NEUTRAL_RELATIVE_PATH))
        self.original_data = Path(os.path.join(self._data_root, RavdessPaths.ORIGINAL_RELATIVE_PATH))
        
        files = {
            (file, self.get_assosiated_neutral_file(file))
            for file in self.original_data.rglob('*')
            if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = {
            (tuple, self.get_attribute_from_filename(tuple[0], "emotion")[1]) 
            for tuple in files
        }
        
        return result

    def __get_emotion_from_index(self, filename):
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', filename.name.__str__())

        index_emotion_mapping = {
            '01': 'neutral',
            '02': 'neutral' if not self._include_calm else 'calm',
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

    def _train_val_test_split(self, test_size: float=0.2, val_size: float=0.1, random_state=42) -> Tuple[set, set, set]:
        # ------------------------------------------------------------------ #
        # 1)  Build actor lists                                              #
        # ------------------------------------------------------------------ #
        data_root = Path(self.original_data)  # .../Audio_Speech_Actors_01-24
        actor_dirs = list_actor_dirs(data_root)  # 24 actor folders
        """
        - in order to keep track of the results or noise of a specific speaker , no shuffle 
        """
        # random.seed(random_state)
        # random.shuffle(actor_dirs)

        # How many actors go to TEMP (val + test)?
        temp_fraction = test_size + val_size  # e.g. 0.3  (20 % + 10 %)
        n_temp = max(1, round(len(actor_dirs) * temp_fraction))
        temp_actor_dirs = actor_dirs[:n_temp]
        train_actor_dirs = actor_dirs[n_temp:]

        # ------------------------------------------------------------------ #
        # 2)  Split TEMP by val_size and test_size  →  VAL / TEST            #
        # ------------------------------------------------------------------ #
        desired_ratio = val_size / (val_size + test_size)  # 0.1 / 0.3 ≈ 0.333
        n_val = max(1, round(len(temp_actor_dirs) * desired_ratio))
        n_test = len(temp_actor_dirs) - n_val  # remainder

        val_actor_dirs = temp_actor_dirs[:n_val]
        test_actor_dirs = temp_actor_dirs[n_val:]

        print(train_actor_dirs)
        print(val_actor_dirs)
        print(test_actor_dirs)

        # ------------------------------------------------------------------ #
        # 3)  Collect wav paths                                              #
        # ------------------------------------------------------------------ #
        train_paths = collect_files(train_actor_dirs, include_aug=self._include_aug)
        val_paths = collect_files(val_actor_dirs, include_aug=False)
        test_paths = collect_files(test_actor_dirs, include_aug=False)  # set True if you want augments

        # ------------------------------------------------------------------ #
        # 4)  Turn them into the expected { (Path, Path, label) } sets             #
        # ------------------------------------------------------------------ #
        def label(p: Path) -> str:
            return self.__get_emotion_from_index(p)

        train_set = {((p, self.get_assosiated_neutral_file(p)), label(p)) for p in train_paths}
        val_set = {((p, self.get_assosiated_neutral_file(p)), label(p)) for p in val_paths}
        test_set = {((p, self.get_assosiated_neutral_file(p)), label(p)) for p in test_paths}

        return train_set, val_set, test_set

    def get_assosiated_neutral_file(self, file):
        # get audio required attributes
        statement_num, statment = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "statement")
        
        repetition_num_str, repetition_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "repetition")
        
        actor_num_str, actor_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "actor")
        
        # create a path to the neutral file
        neutral_file_path = Path(os.path.join(self.neutral_data, f'Actor_{actor_num_int:02d}', f"{statment}_rep{repetition_num_int}_act{actor_num_int}.wav"))
        
        return neutral_file_path
    
    @staticmethod
    def get_attribute_from_filename(filename, attribute: str):
        attribute2index = {
            "modality": 0,
            "vocal_channel": 1,
            "emotion": 2,
            "emotional_intensity": 3,
            "statement": 4,
            "repetition": 5,
            "actor": 6
        }
        num2attrvalue = {
            "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
            "vocal_channel": {"01": "speech", "02": "song"},
            "emotion": {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
                        "07": "disgust", "08": "surprised"},
            "emotional_intensity": {"01": "normal", "02": "strong"},
            "statement": {"01": "kids", "02": "dogs"},
            "repetition": {"01": 1, "02": 2},
            "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
        }
        ########################################################
        # e.g. filename: "03-01-02-01-02-01-13.wav"
                # desired attribute: "actor"
        ########################################################
        
        # get the filename numbers | 
        file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
        # get the desired attribute index
        attribute_index = attribute2index[attribute] # e.g. 6 (actor)
        # get the desired attribute number
        attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
        # get the desired attribute value
        attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13
        
        return attribute_number, attribute_value
        
        
# TODO : add RavdessRawData in which every sample is two audio-file-paths: file2classify and originalneutral
# MICHAL 
# TODO : CHECK IF IT IS RIGHT AND LOGICAL
class AudioRawDataWithOriginalNeutral(AudioRawData):
    def __init__(self):
        super().__init__(RavdessPaths.AUDIO_ORIGINAL_DATA, {".wav"}) # the path that will be the data root 

    def _scan_supported_files(self) -> set[Tuple[Path, str]]:
        """
        Scans and saves the file paths of the model and a relevant label based on the index in the name.
        @return: A set of tuples, each tuple holds (file path, label).
        """

        files = {
        (file, self.get_path_of_neutral(file))
        for file in self._data_root.rglob('*')
        if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
        }

        result = set(map(lambda x: (x, AudioRawDataWithOriginalNeutral.__get_emotion_from_index(x[0])), files))
        return result

    # MICHAL 
    # TODO : CHECK IF IT IS RIGHT AND LOGICAL
    def get_path_of_neutral(self , original_file : Path):
        
        filename = original_file.name  # e.g. "03-01-05-01-02-02-16.wav"
            
        # Extract numbers from filename
        numbers = re.findall(r'\d+', filename)
        
        # the case where the original audio is neutral - search for the same audio with different repetition value 
        if self.get_attribute_from_filename(original_file,"emotion")[1] == "neutral" :
            """
            Given a neutral original RAVDESS audio file, returns the neutral version
            with different repetition value. 
            """  
            if self.get_attribute_from_filename(original_file,"repetition")[0] == '01' : 
                # Set repetition (index 5) to "02"
                numbers[5] = '02'
            else : 
                # Set repetition (index 5) to "01"
                numbers[5] = '01'
                
                # Reconstruct the new filename
            new_filename = '-'.join(numbers) + ".wav"
            
            # The new file should be in the same actor folder as the original
            actor_folder = original_file.parent  # This is .../Actor_##/
            
            neutral_file_path = actor_folder / new_filename
            return neutral_file_path   
        
        # the case where the original audio is not neutral - search for the same audio in neutral and repetition =1 
        else : 
            """
            Given an original RAVDESS audio file, returns the corresponding neutral version
            with repetition set to 01.
            """
            # Replace emotion (index 2) with "01" for neutral
            numbers[2] = "01"
            
            # Set repetition (index 5) to "01"
            numbers[5] = "01"
            
            # Reconstruct the new filename
            new_filename = '-'.join(numbers) + ".wav"
            
            # The new file should be in the same actor folder as the original
            actor_folder = original_file.parent  # This is .../Actor_##/
            
            neutral_file_path = actor_folder / new_filename
            return neutral_file_path               
        
        
    @staticmethod
    def get_attribute_from_filename(filename, attribute: str):
        attribute2index = {
            "modality": 0,
            "vocal_channel": 1,
            "emotion": 2,
            "emotional_intensity": 3,
            "statement": 4,
            "repetition": 5,
            "actor": 6
        }
        num2attrvalue = {
            "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
            "vocal_channel": {"01": "speech", "02": "song"},
            "emotion": {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
                        "07": "disgust", "08": "surprised"},
            "emotional_intensity": {"01": "normal", "02": "strong"},
            "statement": {"01": "kids", "02": "dogs"},
            "repetition": {"01": 1, "02": 2},
            "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
        }
        ########################################################
        # e.g. filename: "03-01-02-01-02-01-13.wav"
                # desired attribute: "actor"
        ########################################################
        
        # get the filename numbers | 
        file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
        # get the desired attribute index
        attribute_index = attribute2index[attribute] # e.g. 6 (actor)
        # get the desired attribute number
        attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
        # get the desired attribute value
        attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13
        
        return attribute_number, attribute_value
        
        
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
    
