from pathlib import Path

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.FileUtils import get_intermediate_folders

if __name__ == '__main__':
    print(get_intermediate_folders(Path("LibriSpeech/19/198/19-198-0000.flac"), LIBRISPEECH_TRAIN_ROOT_FOLDER))

