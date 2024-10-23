from pathlib import Path
from typing import Dict, Any

import numpy as np

from numpy import ndarray
import json

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER, LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER
from GeneralUtils.Exceptions import FileNotSupportedException
from GeneralUtils.FileUtils import retrieve_full_audio_file_path, retrieve_transcript_of_audio_file
from Preprocessing.AudioPreprocess import read_audio_file_as_waveform


class LibriDataObject:
    format: str
    transcript: str
    audio_waveform_data: ndarray
    sample_rate: int | float
    full_path: Path
    filename: str
    parent_directory: Path

    def __init__(self, file_path: str, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER):
        if file_path.endswith(".json"):
            self._init_from_json(file_path)
        else:
            self._init_from_file(file_path, root_folder)


    def _init_from_json(self, file_path: str):
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)

        self.full_path = Path(json_data['full_path'])
        self.filename = json_data['file_name']
        self.format = json_data['file_format']
        self.parent_directory = Path(json_data['parent_directory'])
        self.audio_waveform_data, self.sample_rate = np.array(json_data['audio_waveform_data']), json_data['sample_rate']
        self.transcript = json_data['transcript']

    def _init_from_file(self, file_path: str, root_folder: Path):
        self.full_path = retrieve_full_audio_file_path(file_path, root_folder)
        self.filename = self.full_path.stem
        self.format = self.full_path.suffix
        self.parent_directory = self.full_path.parent
        self.audio_waveform_data, self.sample_rate = read_audio_file_as_waveform(self.full_path, root_folder)
        self.transcript = retrieve_transcript_of_audio_file(self.full_path, root_folder)

    def __str__(self):
        return (f"File name: {self.filename}\n"
                f"File full path: {self.full_path}\n"
                f"File format: {self.format}\n"
                f"Audio waveform data: {self.audio_waveform_data}\n"
                f"Audio sampling rate: {self.sample_rate}Hz\n"
                f"Audio transcript: {self.transcript}\n")

    def to_dict(self) -> Dict[str, Any]:
        """
        Save object attributes to a dictionary supporting the JSON serialization format.
        @return: JSON formatted dictionary
        """
        return {
            'file_name': self.filename,
            'file_format': self.format,
            'full_path': str(self.full_path),  # Convert Path to string
            'parent_directory': str(self.parent_directory),  # Convert Path to string
            'audio_waveform_data': self.audio_waveform_data.tolist() if isinstance(self.audio_waveform_data,
                                                                                   ndarray) else self.audio_waveform_data,
            'sample_rate': self.sample_rate,
            'transcript': self.transcript,
        }

    def construct_json_from_object(self) -> None:
        """
        Serialize the object as a JSON formatted file.
        """
        save_path: Path = LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER / (self.filename + '.json')
        save_path.write_text(json.dumps(self.to_dict(), indent=4), encoding='utf-8')

    @staticmethod
    def construct_object_from_json(json_file_path: str) -> 'LibriDataObject':
        file_suffix = Path(json_file_path).suffix
        if not json_file_path.endswith(".json"):
            raise FileNotSupportedException(file_type=file_suffix, message = "File type not supported for JSON serialization")

        return LibriDataObject(json_file_path)