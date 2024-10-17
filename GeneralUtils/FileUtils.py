from pathlib import Path
from typing import Optional
from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.Exceptions import FileNotSupportedException

AUDIO_FILE_SUPPORTED_FORMATS = tuple([".flac", ".wav"])
TEXT_FILE_SUPPORTED_FORMATS = None


def retrieve_full_audio_file_path(filename: str, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> Optional[Path]:
    """
    Given a filename, inside a root directory, returns the full path of the file if found.
    @rtype: Optional[Path]
    @param filename: Filename of supported audio file format
    @param root_folder: Root folder to search the file from. Defaults to the LibriSpeech dataset root.
    @return: Return the full file path if the file is found. Otherwise, throws FileNotSupportedException
    """
    if not filename.endswith(AUDIO_FILE_SUPPORTED_FORMATS):
        extension_not_supported: str = filename.split(".")[-1]
        raise FileNotSupportedException(extension_not_supported)

    # Use rglob to recursively search for the file
    for file_path in root_folder.rglob(filename):
        if file_path.is_file():  # Check if it's a file (not a directory)
            return Path(file_path)  # Return the full path as a string

    raise FileNotFoundError(f"File '{filename}' not found in '{root_folder}'")


def retrieve_transcript_of_audio_file(filename: str, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER):
    raise NotImplementedError(f"{retrieve_transcript_of_audio_file.__name__} not implemented yet")


def retrieve_dataset_file_paths(root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> list[Path]:
    pass
