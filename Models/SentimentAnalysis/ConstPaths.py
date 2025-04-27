import os.path
from enum import EnumType
from pathlib import Path

class ProjectPaths(EnumType):
    MODEL_RESULTS = Path(os.path.join("Model Results"))

class MeldPaths(EnumType):
    AUDIO_FILES_DATA = Path(os.path.join("wav_splits"))
    TRAIN_DATA_CSV = Path(os.path.join("train.csv"))
    DEV_DATA_CSV = Path(os.path.join("dev.csv"))
    TEST_DATA_CSV = Path(os.path.join("test.csv"))

class RavdessPaths(EnumType):
    ORIGINAL_RELATIVE_PATH = 'original_data'
    NEUTRAL_RELATIVE_PATH = 'neutral_synthesized'
    ALL_AUDIO_DATA = Path(os.path.join("RAVDESS"))
    AUDIO_ORIGINAL_DATA = Path(os.path.join(ALL_AUDIO_DATA, ORIGINAL_RELATIVE_PATH))
    AUDIO_NEUTRAL_SYNTHESIZED_DATA = Path(os.path.join(ALL_AUDIO_DATA, NEUTRAL_RELATIVE_PATH))
    TXT_FOR_DEEPFAKE_PATH = Path(os.path.join(ALL_AUDIO_DATA, "txt_for_deepfake"))
    WORD_CHUNKED_AUDIO_DATA = Path(os.path.join(ALL_AUDIO_DATA, "chunked_word_audio"))
class TessPaths(EnumType):
    AUDIO_FILES_DATA = Path(os.path.join("TESS"))