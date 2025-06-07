import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AnyStr
from typing import Tuple
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ConstPaths import RavdessPaths, CremaPaths
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from Preprocess import audio_to_mel_spectrogram


class AudioRawData(ABC):
    """
    Wrapper abstract class to handle Dataset saving in run-time.
    """
    def __init__(self, data_root: Path, supported_formats: set[str], file_pattern: AnyStr):
        self._data_root: Path = data_root
        self._supported_formats: set[str] = supported_formats
        self._data: set[Tuple[Path, Any]] = self._scan_supported_files(file_pattern=file_pattern) # File/s path/s, Label.  used as (Path, str) and (Dict[Path, Path], str).

        self._file_paths, self._file_labels = zip(*list(self._data))

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
    def emotion_indexer(self, file_name: Path) -> str:
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

    def emotion_indexer(self, file_name: Path) -> str:
        """
        For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
        readable label.
        @param filename: File path from RAVDESS dataset.
        @return: Label of the file according to the index.
        """
        numbers = re.findall(r'\d+', file_name.name.__str__())

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


class CREMARawData(AudioRawData):
    def __init__(self):
        super().__init__(CremaPaths.WAV_DATA, {".wav"}, r"^\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}$")

    def emotion_indexer(self, file_name: Path) -> str:
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
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }

        pattern = re.compile(r'^\d{4}_[A-Z]{3}_([A-Z]{3})_[A-Z]{2}\.wav$', re.IGNORECASE)
        match = pattern.match(file_name.name)

        if not match:
            raise ValueError(f"Filename format not recognized: {file_name}")

        emotion_code = match.group(1).upper()
        emotion = index_emotion_mapping.get(emotion_code)

        if emotion is None:
            raise ValueError(f"Unknown emotion code '{emotion_code}' in file: {file_name}")

        return emotion

class AllRawData:
    def __init__(self, raw_datasets: tuple[AudioRawData, ...]):
        sets_of_data = (raw_dataset.all_data for raw_dataset in raw_datasets)
        self._all_raw_data = set().union(*sets_of_data)
        self._train_data, self._val_data = None, None

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

# add RavdessRawData in which every sample is two audio-file-paths: file2classify and originalneutral
# class RavdessRawDataWithNeutral(AudioRawData):
#     def __init__(self, include_calm = True, include_aug=False):
#         self._include_aug = include_aug
#         self._include_calm = include_calm
#         super().__init__(RavdessPaths.ALL_AUDIO_DATA, {".wav"})
#
#     def _scan_supported_files(self) -> set[Tuple[Tuple[Path, Path], str]]:
#         """
#         Scans and saves the file paths of the model and a relevant label based on the index in the name.
#         @return: A set of tuples, each tuple holds (file path, label).
#         """
#         self.neutral_data = Path(os.path.join(self._data_root, RavdessPaths.NEUTRAL_RELATIVE_PATH))
#         self.original_data = Path(os.path.join(self._data_root, RavdessPaths.ORIGINAL_RELATIVE_PATH))
#
#         base_name_pattern = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
#
#
#         files = {
#             (file, self.get_assosiated_neutral_file(file))
#             for file in self.original_data.rglob('*')
#             if file.is_file()
#             and any(file.name.endswith(suffix) for suffix in self._supported_formats)
#             and base_name_pattern.match(file.stem)
#         }
#
#         result = {
#             (tuple, self.get_attribute_from_filename(tuple[0], "emotion")[1])
#             for tuple in files
#         }
#
#         return result
#
#     def __get_emotion_from_index(self, filename):
#         """
#         For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
#         readable label.
#         @param filename: File path from RAVDESS dataset.
#         @return: Label of the file according to the index.
#         """
#         numbers = re.findall(r'\d+', filename.name.__str__())
#
#         index_emotion_mapping = {
#             '01': 'neutral',
#             '02': 'neutral' if not self._include_calm else 'calm',
#             '03': 'happy',
#             '04': 'sad',
#             '05': 'angry',
#             '06': 'fearful',
#             '07': 'disgust',
#             '08': 'surprised'
#         }
#
#         emotion_index = numbers[2]
#         emotion = index_emotion_mapping[emotion_index]
#         return emotion
#
#     def get_assosiated_neutral_file(self, file):
#         # get audio required attributes
#         statement_num, statment = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "statement")
#
#         repetition_num_str, repetition_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "repetition")
#
#         actor_num_str, actor_num_int = RavdessRawDataWithNeutral.get_attribute_from_filename(file, "actor")
#
#         # create a path to the neutral file
#         neutral_file_path = Path(os.path.join(self.neutral_data, f'Actor_{actor_num_int:02d}', f"{statment}_rep{repetition_num_int}_act{actor_num_int}.wav"))
#
#         return neutral_file_path
#
#     @staticmethod
#     def get_attribute_from_filename(filename, attribute: str):
#         attribute2index = {
#             "modality": 0,
#             "vocal_channel": 1,
#             "emotion": 2,
#             "emotional_intensity": 3,
#             "statement": 4,
#             "repetition": 5,
#             "actor": 6
#         }
#         num2attrvalue = {
#             "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
#             "vocal_channel": {"01": "speech", "02": "song"},
#             "emotion": {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
#                         "07": "disgust", "08": "surprised"},
#             "emotional_intensity": {"01": "normal", "02": "strong"},
#             "statement": {"01": "kids", "02": "dogs"},
#             "repetition": {"01": 1, "02": 2},
#             "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
#         }
#         ########################################################
#         # e.g. filename: "03-01-02-01-02-01-13.wav"
#                 # desired attribute: "actor"
#         ########################################################
#
#         # get the filename numbers |
#         file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
#         # get the desired attribute index
#         attribute_index = attribute2index[attribute] # e.g. 6 (actor)
#         # get the desired attribute number
#         attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
#         # get the desired attribute value
#         attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13
#
#         return attribute_number, attribute_value
#
#
# # class EmotionWavDataset(Dataset):
# #     def __init__(self, file_paths: set):
# #         self._data = list(file_paths)
# #         self._paths, self._labels = zip(*self._data)
# #
# #         self.__label_encoder = LabelEncoder()
# #         self._labels = torch.tensor(self.__label_encoder.fit_transform(self._labels))
# #         self._class_names = list(self.__label_encoder.classes_)
# #
# #         # Get the number of classes
# #         self.num_classes = len(self.__label_encoder.classes_)
# #
# #         # Compute class weights
# #         self.class_weights = self.__compute_class_weights()
# #
# #
# #     def __len__(self):
# #         return len(self._data)
# #
# #     def __getitem__(self, idx):
# #         file_path = self._paths[idx]
# #         label = self._labels[idx]
# #
# #         # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
# #         waveform, sr = audio_to_waveform(file_path)
# #
# #         inputs = processor(waveform, sampling_rate=sr, return_tensors="pt").input_values
# #
# #         with torch.no_grad():
# #             outputs = model(inputs.to("cuda"))
# #
# #         # outputs.last_hidden_state shape: (1, sequence_length, hidden_size)
# #         hidden_states = outputs.last_hidden_state.squeeze(0)
# #
# #         # Option: mean-pool over time
# #         embedding = hidden_states.mean(dim=0)  # Shape: (hidden_size,)
# #         return embedding, label
#
#
#
#     @property
#     def class_names(self):
#         return self._class_names
#
#     @property
#     def class_counts(self):
#         return self._class_counts
#
#     def decode_label(self, encoded_label):
#         return self.__label_encoder.inverse_transform([encoded_label])[0]
#
#     def __compute_class_weights(self) -> torch.Tensor:
#         """
#         Computes class weights based on the frequency of each class in the dataset.
#
#         Returns:
#             torch.Tensor: Tensor of class weights (inverse frequency).
#         """
#         class_counts = torch.bincount(self._labels, minlength=self.num_classes)
#         self._class_counts = class_counts
#         total_samples = len(self._labels)
#         class_weights = total_samples / (class_counts + 1e-6)  # Avoid division by zero
#         return class_weights.float()
#
# # TODO : add RavdessRawData in which every sample is two audio-file-paths: file2classify and originalneutral
# # MICHAL
# # TODO : CHECK IF IT IS RIGHT AND LOGICAL
# class AudioRawDataWithOriginalNeutral(AudioRawData):
#     def __init__(self):
#         super().__init__(RavdessPaths.AUDIO_ORIGINAL_DATA, {".wav"}) # the path that will be the data root
#
#     def _scan_supported_files(self) -> set[Tuple[Path, str]]:
#         """
#         Scans and saves the file paths of the model and a relevant label based on the index in the name.
#         @return: A set of tuples, each tuple holds (file path, label).
#         """
#
#         files = {
#         (file, self.get_path_of_neutral(file))
#         for file in self._data_root.rglob('*')
#         if file.is_file() and any(file.name.endswith(suffix) for suffix in self._supported_formats)
#         }
#
#         result = set(map(lambda x: (x, AudioRawDataWithOriginalNeutral.__get_emotion_from_index(x[0])), files))
#         return result
#
#     # MICHAL
#     # TODO : CHECK IF IT IS RIGHT AND LOGICAL
#     def get_path_of_neutral(self , original_file : Path):
#
#         filename = original_file.name  # e.g. "03-01-05-01-02-02-16.wav"
#
#         # Extract numbers from filename
#         numbers = re.findall(r'\d+', filename)
#
#         # the case where the original audio is neutral - search for the same audio with different repetition value
#         if self.get_attribute_from_filename(original_file,"emotion")[1] == "neutral" :
#             """
#             Given a neutral original RAVDESS audio file, returns the neutral version
#             with different repetition value.
#             """
#             if self.get_attribute_from_filename(original_file,"repetition")[0] == '01' :
#                 # Set repetition (index 5) to "02"
#                 numbers[5] = '02'
#             else :
#                 # Set repetition (index 5) to "01"
#                 numbers[5] = '01'
#
#                 # Reconstruct the new filename
#             new_filename = '-'.join(numbers) + ".wav"
#
#             # The new file should be in the same actor folder as the original
#             actor_folder = original_file.parent  # This is .../Actor_##/
#
#             neutral_file_path = actor_folder / new_filename
#             return neutral_file_path
#
#         # the case where the original audio is not neutral - search for the same audio in neutral and repetition =1
#         else :
#             """
#             Given an original RAVDESS audio file, returns the corresponding neutral version
#             with repetition set to 01.
#             """
#             # Replace emotion (index 2) with "01" for neutral
#             numbers[2] = "01"
#
#             # Set repetition (index 5) to "01"
#             numbers[5] = "01"
#
#             # Reconstruct the new filename
#             new_filename = '-'.join(numbers) + ".wav"
#
#             # The new file should be in the same actor folder as the original
#             actor_folder = original_file.parent  # This is .../Actor_##/
#
#             neutral_file_path = actor_folder / new_filename
#             return neutral_file_path
#
#
#     @staticmethod
#     def get_attribute_from_filename(filename, attribute: str):
#         attribute2index = {
#             "modality": 0,
#             "vocal_channel": 1,
#             "emotion": 2,
#             "emotional_intensity": 3,
#             "statement": 4,
#             "repetition": 5,
#             "actor": 6
#         }
#         num2attrvalue = {
#             "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"},
#             "vocal_channel": {"01": "speech", "02": "song"},
#             "emotion": {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
#                         "07": "disgust", "08": "surprised"},
#             "emotional_intensity": {"01": "normal", "02": "strong"},
#             "statement": {"01": "kids", "02": "dogs"},
#             "repetition": {"01": 1, "02": 2},
#             "actor": {f"{i:02d}": i for i in range(1, 25)} # maps from string number to int number
#         }
#         ########################################################
#         # e.g. filename: "03-01-02-01-02-01-13.wav"
#                 # desired attribute: "actor"
#         ########################################################
#
#         # get the filename numbers |
#         file_numbers = re.findall(r'\d+', filename.name.__str__()) # e.g. ['03', '01', '02', '01', '02', '01', '13']
#         # get the desired attribute index
#         attribute_index = attribute2index[attribute] # e.g. 6 (actor)
#         # get the desired attribute number
#         attribute_number = file_numbers[attribute_index] # e.g. "13"(index 6 at the filename numbers)
#         # get the desired attribute value
#         attribute_value = num2attrvalue[attribute][attribute_number] # e.g. "13" -> 13
#
#         return attribute_number, attribute_value
#
#
#     @staticmethod
#     def __get_emotion_from_index(filename):
#         """
#         For RAVDESS, label is indicated in the third number in the name. This function handles mapping it to a
#         readable label.
#         @param filename: File path from RAVDESS dataset.
#         @return: Label of the file according to the index.
#         """
#         numbers = re.findall(r'\d+', filename.name.__str__())
#
#         index_emotion_mapping = {
#             '01': 'neutral',
#             '02': 'calm',
#             '03': 'happy',
#             '04': 'sad',
#             '05': 'angry',
#             '06': 'fearful',
#             '07': 'disgust',
#             '08': 'surprised'
#         }
#
#         emotion_index = numbers[2]
#         emotion = index_emotion_mapping[emotion_index]
#         return emotion
    
