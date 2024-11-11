from pathlib import Path

from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.FileUtils import get_intermediate_folders

import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())

