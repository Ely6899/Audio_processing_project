from pathlib import Path
from typing import Optional, Tuple

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.Exceptions import FileNotSupportedException

AUDIO_FILE_SUPPORTED_FORMATS : Tuple[str, ...] = ('.flac', '.wav', '.mp3')
TEXT_FILE_SUPPORTED_FORMATS = None #TODO: Add supported text formats


def retrieve_full_audio_file_path(filename: str, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> Path:
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


def retrieve_transcript_of_audio_file(filename: str | Path, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> str:
    """
    Retrieves relevant transcript from given file in the data. For now, supports only LibriSpeech,
    where the transcript file is a single .txt file with transcript line per file in the same directory.
    @param filename: Name of the file we want to fetch the transcript for.
    @param root_folder: The root data folder in which the data is located.
    @return: String of the transcript.
    """
    # Handle different instances as input
    if isinstance(filename, str):
        try:
            full_file_path = retrieve_full_audio_file_path(filename, root_folder)
        except FileNotSupportedException as e:
            raise e
    elif isinstance(filename, Path):
        full_file_path = filename
    else:
        raise AttributeError(f"Parameter 'filename' needs to be of type str or Path. Got {type(filename)} instead!")

    filename_without_extension = full_file_path.stem
    parent_directory = full_file_path.parent

    #Obtain text file path
    text_files = list(parent_directory.glob("*.trans.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt file found in {parent_directory}")

    text_file_path = text_files[0]
    transcript = ""

    with open(text_file_path, 'r') as transcript_file:
        for line in transcript_file:
            if line.startswith(filename_without_extension):
                line = line.strip()
                transcript = line.removeprefix(filename_without_extension).strip()
                break

    return transcript


def retrieve_dataset_file_paths(root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER,
                                supported_formats: Tuple[str, ...] = AUDIO_FILE_SUPPORTED_FORMATS) -> list[Path]:
    """
    Retrieve supported file paths in a list.
    @param root_folder: The root from which to search for files downstream.
    @param supported_formats: Supported audio file formats. Defaults to module definition.
    @return: List of all supported files full paths.
    """
    supported_files = []

    # Search recursively for files with extensions matching the supported formats
    for file_path in root_folder.rglob("*"):
        if file_path.suffix.lower() in supported_formats:
            supported_files.append(file_path)

    return supported_files
