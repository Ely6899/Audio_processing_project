from pathlib import Path

import torch.nn.functional

from PreprocessParams import *
import pandas as pd
import numpy as np
import os
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import logging

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (optional, default is WARNING)
logger.setLevel(logging.DEBUG)

# Create a console handler (you can add more handlers for file or other destinations)
console_handler = logging.StreamHandler()

# Set the log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
#logger.addHandler(console_handler)


def audio_to_waveform(file_path: Path, target_sample_rate: int = SAMPLE_RATE):
    waveform, loaded_sample_rate = torchaudio.load(uri=file_path)
    logger.debug(f"Loaded audio file with shape: {waveform.shape}")

    if loaded_sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=loaded_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        logger.debug(f"Waveform shape after Resampling: {waveform.shape}")

    return waveform, target_sample_rate


def audio_to_mel_spectogram(file_path: Path,
                            sample_rate: int = SAMPLE_RATE,
                            n_mels: int = FREQUENCY_BIN_COUNT,
                            max_length_in_frames: int = MAX_AUDIO_LENGTH_IN_FRAMES,
                            pad_or_truncate: bool = True):
    """
    Generates a Mel-Spectrogram of the given audio file.
    @param file_path: Path of audio file.
    @param sample_rate: Sample rate to load the audio with. Write None to preserve native SR.
    @param n_mels: Number of frequency bins to be displayed on the Spectrogram (The Y axis).
    @param max_length_in_frames: Maximum length (in frames) of the spectrogram to process.
    @param pad_or_truncate: True if you want to keep the length of the spectrogram at max_length.
    @return: Mel spectrogram of the audio file.
    """

    # audio, sr = librosa.load(file_path, sr=sample_rate)
    # mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    #
    # if pad_or_truncate:
    #     if mel_spec.shape[1] < max_length:  # Padding
    #         pad_width = max_length - mel_spec.shape[1]
    #         mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    #     else:  # Truncating
    #         mel_spec = mel_spec[:, :max_length]
    #
    # # Normalize
    # mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
    # return mel_spec, sr

    waveform, sample_rate = audio_to_waveform(file_path, sample_rate)

    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform)
    logger.debug(f"Spectogram shape: {mel_spectrogram.shape}\n"
                 f"Num of channels: {mel_spectrogram.shape[0]}\n"
                 f"Num of frequency bins: {mel_spectrogram.shape[1]}\n"
                 f"Num of time frames: {mel_spectrogram.shape[2]}\n")

    spectrogram_numb_of_frames = mel_spectrogram.shape[-1]
    if spectrogram_numb_of_frames < max_length_in_frames:
        padding = max_length_in_frames - spectrogram_numb_of_frames
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
    else:
        mel_spectrogram = mel_spectrogram[:, :, :max_length_in_frames]

    logger.debug(f"Spectrogram shape at return: {mel_spectrogram.shape}")

    return mel_spectrogram




def save_datasets_from_csv(csv_path, output_dir, split_name):
    """
    Preprocess audio recordings into mel spectrogram datasets from a given CSV file.
    """
    # Load CSV
    data = pd.read_csv(csv_path)
    audio_paths = data['path'].tolist()
    labels = data['label'].tolist()

    # Prepare directory
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Save spectrograms
    for i, (path, label) in enumerate(zip(audio_paths, labels)):
        try:
            mel_spec = preprocess_audio_to_mel(path)
            np.save(os.path.join(split_dir.__str__(), f"{i}_spec.npy"), mel_spec)
            with open(os.path.join(split_dir.__str__(), f"{i}_label.txt"), 'w') as f:
                f.write(str(label))
        except Exception as e:
            print(f"Error processing {path}: {e}")
