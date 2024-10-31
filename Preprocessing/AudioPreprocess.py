import librosa
import soundfile as sf
import numpy as np

from GeneralUtils.DirPaths import *
from GeneralUtils.Exceptions import FileNotSupportedException
from GeneralUtils.FileUtils import retrieve_full_audio_file_path


def read_audio_file_as_waveform(file_name: Path,
                                data_root_path: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER,
                                sr: float | None = None) -> tuple[np.ndarray, int | float]:
    """
    Reads the supported audio file as an array of amplitude values
    @rtype: Tuple
    @param file_name: Name of a supported audio file to read, or full Path object representation
    @param data_root_path: The root folder of the file. Defaults to the LibriSpeech dataset root.
    @param sr: Sample rate in which to read the audio. Defaults to None for native sample rate of the file.
    @return: File data as a numpy array and sample rate value of the file.
    """

    full_file_path = retrieve_full_audio_file_path(file_name, data_root_path)
    file_data, sample_rate = librosa.load(full_file_path, sr=sr)
    return file_data, sample_rate

