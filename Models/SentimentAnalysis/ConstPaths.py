import os.path
from enum import EnumType
from pathlib import Path

class MeldPaths(EnumType):
    AUDIO_FILES_DATA = Path(os.path.join("wav_splits"))
    TRAIN_DATA_CSV = Path(os.path.join("train.csv"))
    DEV_DATA_CSV = Path(os.path.join("dev.csv"))
    TEST_DATA_CSV = Path(os.path.join("test.csv"))

class RavdessPaths(EnumType):
    AUDIO_FILES_DATA = Path(os.path.join("RAVDESS"))

