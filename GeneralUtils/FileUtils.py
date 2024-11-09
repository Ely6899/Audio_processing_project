from pathlib import Path
from typing import Tuple, Any
from tqdm import tqdm

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.Exceptions import FileNotSupportedException

AUDIO_FILE_SUPPORTED_FORMATS: Tuple[str, ...] = ('.flac', '.wav', '.mp3')
TEXT_FILE_SUPPORTED_FORMATS = None  #TODO: Add supported text formats


def retrieve_full_audio_file_path(file_name: Path, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> Path:
    """
    Given a file_name, inside a root directory, returns the full path of the file if found.
    @rtype: Path
    @param file_name: Filename of supported audio file format
    @param root_folder: Root folder to search the file from. Defaults to the LibriSpeech dataset root.
    @return: Return the full file path if the file is found.
    @raise FileNotSupportedException: If given file is in an unsupported format.
    @raise FileNotFoundError: If file is not found after searching root directory.
    @raise AttributeError: If file_name attribute is not of type Path.
    """

    try:
        if not file_name.exists():
            raise FileNotFoundError(f"File '{file_name}' not found in '{root_folder}'")

        file_format: str = file_name.suffix
        if file_format not in AUDIO_FILE_SUPPORTED_FORMATS:
            raise FileNotSupportedException(file_format)

        # Resolve both paths to their absolute form
        file_name_full = file_name.resolve()
        root_folder_full = root_folder.resolve()

        # Check if file_name is within root_folder
        if file_name_full.is_relative_to(root_folder_full):
            return file_name

        # Use rglob to recursively search for the file
        for file_path in root_folder.rglob(file_name.name):
            if file_path.is_file():
                return file_path

    except AttributeError as attribute_error:
        raise attribute_error


def retrieve_transcript_of_audio_file(file_name: Path, root_folder: Path = LIBRISPEECH_TRAIN_ROOT_FOLDER) -> str | None:
    """
    Retrieves relevant transcript from given file in the data. For now, supports only LibriSpeech,
    where the transcript file is a single .txt file with transcript line per file in the same directory.
    @param file_name: Name of the file we want to fetch the transcript for.
    @param root_folder: The root data folder in which the data is located.
    @return: String of the transcript. Returns None if no transcript found.
    @raise: FileNotFoundError: Given file wasn't found.
    """

    full_file_path: Path = retrieve_full_audio_file_path(file_name=file_name, root_folder=root_folder)

    filename_without_extension: str = full_file_path.stem
    parent_directory: Path = full_file_path.parent

    #Obtain text file path
    text_files: list[Any] = list(parent_directory.glob("*.trans.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt file found in {parent_directory}")

    text_file_path = text_files[0]  #For Librispeech use case, should be only 1 file.
    transcript: str | None = None

    with open(text_file_path, 'r') as transcript_file:
        for line in transcript_file:
            if line.startswith(filename_without_extension):
                line = line.strip()
                transcript = line.removeprefix(filename_without_extension).strip()
                break

    return transcript


def get_intermediate_folders(file_name: Path, root_folder: Path) -> Path:
    """
    Gets the folders between the root and the file itself.
    @param file_name: Name of the file.
    @param root_folder: Root folder we start the search from.
    @return: Folders between root_folder and file_name.
    @raise: FileNotFoundError: Given file wasn't found.
    """

    full_file_path: Path = retrieve_full_audio_file_path(file_name=file_name, root_folder=root_folder)
    try:
        # Ensure both paths are absolute
        file_name : Path = Path(full_file_path).resolve()
        root_folder: Path= Path(root_folder).resolve()

        # Check if file is within base_directory
        if root_folder in file_name.parents:
            # Get the relative path from base_directory to file_path
            relative_path: Path = file_name.relative_to(root_folder)
            sub_folders_tuple: tuple[str, ...] = relative_path.parts[:-1] # Exclude the last part which is the file name
            return Path(*sub_folders_tuple)
        else:
            raise FileNotFoundError(f"'{file_name}' not under root dir '{root_folder}'")
    except AttributeError as attribute_error:
        raise attribute_error


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
    for file_path in tqdm(root_folder.rglob("*"), desc="Retrieving dataset paths...", ncols=100):
        if file_path.suffix.lower() in supported_formats:
            supported_files.append(file_path)

    return supported_files
