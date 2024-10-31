import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

from numpy import ndarray
import orjson

from tqdm import tqdm
import multiprocessing as mp

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER, LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER
from GeneralUtils.Exceptions import FileNotSupportedException
from GeneralUtils.FileUtils import retrieve_full_audio_file_path, retrieve_transcript_of_audio_file, \
    get_intermediate_folders, retrieve_dataset_file_paths
from Preprocessing.AudioPreprocess import read_audio_file_as_waveform


class LibriDataObject:
    format: str
    transcript: str
    audio_waveform_data: ndarray
    sample_rate: int | float
    file_name: str
    full_path: Path
    parent_directory: Path

    def __init__(self, file_path: Path):
        file_format = file_path.suffix
        if file_format == ".json":
            self._init_from_json(file_path)
        else:
            self._init_from_file(file_path)

    def _init_from_json(self, file_path: Path):
        with open(file_path, 'r') as json_file:
            json_data = orjson.loads(json_file.read())

        self.full_path = Path(json_data['full_path'])
        self.file_name = json_data['file_name']
        self.format = json_data['file_format']
        self.parent_directory = Path(json_data['parent_directory'])
        self.audio_waveform_data, self.sample_rate = np.array(json_data['audio_waveform_data']), json_data['sample_rate']
        self.transcript = json_data['transcript']

    def _init_from_file(self, file_path: Path):
        self.full_path = retrieve_full_audio_file_path(file_path, root_folder=LIBRISPEECH_TRAIN_ROOT_FOLDER)
        self.file_name = self.full_path.stem
        self.format = self.full_path.suffix
        self.parent_directory = self.full_path.parent
        self.audio_waveform_data, self.sample_rate = read_audio_file_as_waveform(self.full_path, data_root_path=LIBRISPEECH_TRAIN_ROOT_FOLDER)
        self.transcript = retrieve_transcript_of_audio_file(self.full_path, root_folder=LIBRISPEECH_TRAIN_ROOT_FOLDER)

    def __str__(self):
        return (f"File name: {self.file_name}\n"
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
            'file_name': self.file_name,
            'file_format': self.format,
            'full_path': self.full_path.__str__(),
            'parent_directory': self.parent_directory.__str__(),
            'audio_waveform_data': self.audio_waveform_data,
            'sample_rate': self.sample_rate,
            'transcript': self.transcript,
        }

    def construct_json_from_object(self) -> None:
        """
        Serialize the object as a JSON formatted file.
        """
        sub_folders: Path = get_intermediate_folders(self.full_path, LIBRISPEECH_TRAIN_ROOT_FOLDER)
        new_path: Path = LIBRISPEECH_SERIALIZED_OBJECT_ROOT_FOLDER / sub_folders
        new_path.mkdir(parents=True, exist_ok=True)
        save_path: Path = new_path / (self.file_name + '.json')
        save_path.write_bytes(orjson.dumps(self.to_dict(), option=orjson.OPT_SERIALIZE_NUMPY))

    @staticmethod
    def construct_object_from_json(json_file_path: Path) -> 'LibriDataObject':
        file_suffix = json_file_path.suffix
        if file_suffix != ".json":
            raise FileNotSupportedException(file_type=file_suffix,
                                            message="File type not supported for JSON deserialization")

        return LibriDataObject(json_file_path)

    @staticmethod
    def save_object(libri_path: Path):
        """Helper function to save a single LibriDataObject as JSON."""
        libri_obj = LibriDataObject(libri_path)
        libri_obj.construct_json_from_object()

    @staticmethod
    def save_all_libri_as_json():
        libri_paths = retrieve_dataset_file_paths(LIBRISPEECH_TRAIN_ROOT_FOLDER)

        num_processes = mp.cpu_count() - 6
        with mp.Pool(processes=num_processes) as pool:
            # Use imap_unordered to allow tqdm to work properly
            for _ in tqdm(pool.imap_unordered(LibriDataObject.save_object, libri_paths), total=len(libri_paths),
                          desc="Saving librispeech data as JSON formats"):
                pass  # Each item processed will automatically update the progress bar

        # for libri_path in tqdm(libri_paths, "Saving librispeech data as JSON formats...", ncols=100):
        #     LibriDataObject(libri_path).construct_json_from_object()