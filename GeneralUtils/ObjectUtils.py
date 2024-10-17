from pathlib import Path
from typing import Dict, Any

from IPython.testing.tools import full_path
from numpy import ndarray
import json

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER, LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER
from GeneralUtils.FileUtils import retrieve_full_audio_file_path, retrieve_transcript_of_audio_file
from Preprocessing.AudioPreprocess import read_audio_file_as_waveform


class LibriDataObject:
    format: str
    transcript: str
    audio_waveform_data: ndarray
    sample_rate: int | float
    full_path: Path
    file_name: str
    parent_directory: Path


    def __init__(self, file_path: str, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER):
        self.full_path = retrieve_full_audio_file_path(file_path, root_folder)
        self.file_name = self.full_path.stem
        self.format = self.full_path.suffix
        self.parent_directory = self.full_path.parent
        self.audio_waveform_data, self.sample_rate = read_audio_file_as_waveform(self.full_path, root_folder)
        self.transcript = retrieve_transcript_of_audio_file(self.full_path, root_folder)


    def __str__(self):
        return (f"File name: {self.file_name}\n"
                f"File full path: {self.full_path}\n"
                f"File format: {self.format}\n"
                f"Audio waveform data: {self.audio_waveform_data}\n"
                f"Audio sampling rate: {self.sample_rate}Hz\n"
                f"Audio transcript: {self.transcript}\n")


    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        return {
            'file_name': self.file_name,
            'file_format': self.format,
            'full_path': str(self.full_path),  # Convert Path to string
            'parent_directory': str(self.parent_directory),  # Convert Path to string
            'audio_waveform_data': self.audio_waveform_data.tolist() if isinstance(self.audio_waveform_data,
                                                                                   ndarray) else self.audio_waveform_data,
            'sample_rate': self.sample_rate,
            'transcript': self.transcript,
        }



def construct_json_from_audio_file(data_object: LibriDataObject) -> None:
    """Dump the LibriDataObject to a JSON file."""
    save_path: Path = LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER / (data_object.file_name + '.json')
    with open(save_path, 'w') as json_file:
        json.dump(data_object.to_dict(), json_file, indent=4)