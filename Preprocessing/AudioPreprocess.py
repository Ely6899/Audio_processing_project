import librosa
import soundfile as sf
import numpy as np

from GeneralUtils.DirPaths import *
from GeneralUtils.FileUtils import retrieve_full_audio_file_path


def read_audio_file_as_waveform(file_name: Path,
                                root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER,
                                sr: float | None = None) -> tuple[np.ndarray, int | float]:
    """
    Reads the supported audio file as an array of amplitude values
    @rtype: Tuple
    @param file_name: Name of a supported audio file to read, or full Path object representation
    @param root_folder: The root folder of the file. Defaults to the LibriSpeech dataset root.
    @param sr: Sample rate in which to read the audio. Defaults to None for native sample rate of the file.
    @return: File data as a numpy array and sample rate value of the file.
    """

    full_file_path = retrieve_full_audio_file_path(file_name, root_folder)
    file_data, sample_rate = librosa.load(full_file_path, sr=sr)
    return file_data, sample_rate

