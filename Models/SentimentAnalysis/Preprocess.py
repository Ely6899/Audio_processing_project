from pathlib import Path

import torch.nn.functional
from PreprocessParams import *

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
logger.addHandler(console_handler)


def audio_to_waveform(file_path: Path, target_sample_rate: int = SAMPLE_RATE):
    waveform, loaded_sample_rate = torchaudio.load(uri=file_path)
    logger.debug(f"Loaded audio file with shape: {waveform.shape}")

    if waveform.shape[0] > 1:  # Check if there are multiple channels
        waveform = waveform.mean(dim=0, keepdim=True)

    if loaded_sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=loaded_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        logger.debug(f"Waveform shape after Resampling: {waveform.shape}")

    return waveform, target_sample_rate


def audio_to_mel_spectogram(file_path: Path,
                            sample_rate: int = SAMPLE_RATE,
                            n_fft = N_FFT,
                            window_length = WINDOW_LENGTH,
                            hop_length = HOP_LENGTH,
                            n_mels: int = FREQUENCY_BIN_COUNT,
                            max_length_in_seconds: float = MAX_SPECTOGRAM_DURATION_IN_SECONDS):


    waveform, sample_rate = audio_to_waveform(file_path, sample_rate)

    mel_transform = MelSpectrogram(sample_rate=sample_rate,
                                   n_mels=n_mels,
                                   n_fft = n_fft,
                                   win_length=window_length,
                                   hop_length=hop_length
                                   ,normalized=True,
                                   center=False)
    mel_spectrogram = mel_transform(waveform)
    logger.debug(f"Spectogram shape: {mel_spectrogram.shape}\n"
                 f"Num of channels: {mel_spectrogram.shape[0]}\n"
                 f"Num of frequency bins: {mel_spectrogram.shape[1]}\n"
                 f"Num of time frames: {mel_spectrogram.shape[2]}\n")

    mel_spectrogram_padded = pad_spectrogram_to_max_duration(spectrogram=mel_spectrogram,
                                                             max_duration_seconds=max_length_in_seconds,
                                                             sample_rate=sample_rate,
                                                             win_length=window_length,
                                                             hop_length=hop_length)

    logger.debug(f"Spectrogram shape at return: {mel_spectrogram_padded.shape}")

    return mel_spectrogram_padded


def pad_spectrogram_to_max_duration(spectrogram, max_duration_seconds, sample_rate, win_length, hop_length):
    """
    Pads a spectrogram to match the maximum duration in seconds.

    Args:
        spectrogram (torch.Tensor): Input spectrogram (shape: [channels, n_mels, n_frames]).
        max_duration_seconds (float): Target maximum duration in seconds.
        sample_rate (int): Sampling rate of the audio signal.
        win_length (int): Window size used in the STFT.
        hop_length (int): Hop size used in the STFT.

    Returns:
        torch.Tensor: Padded spectrogram with consistent frame count.
    """

    # Calculate the max number of samples for the target duration
    max_samples = int(max_duration_seconds * sample_rate)

    # Calculate the target number of frames for the spectrogram
    target_frames = (max_samples - win_length) // hop_length + 1

    # Get current frame count in the spectrogram
    current_frames = spectrogram.shape[-1]

    logger.debug(f"Given the input, target amount of frames is {target_frames}\n"
                 f"and current amount of frames is {current_frames}")

    if current_frames > target_frames:
        spectrogram = spectrogram[:, :, :target_frames]
        logger.debug(f"Truncated spectrogram to {target_frames} frames")
    else:
        padding_needed = target_frames - current_frames
        spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_needed), mode='constant', value=0)
        logger.debug(f"Padded spectrogram to {target_frames} frames")

    return spectrogram


# def save_datasets_from_csv(csv_path, output_dir, split_name):
#     """
#     Preprocess audio recordings into mel spectrogram datasets from a given CSV file.
#     """
#     # Load CSV
#     data = pd.read_csv(csv_path)
#     audio_paths = data['path'].tolist()
#     labels = data['label'].tolist()
#
#     # Prepare directory
#     split_dir = os.path.join(output_dir, split_name)
#     os.makedirs(split_dir, exist_ok=True)
#
#     # Save spectrograms
#     for i, (path, label) in enumerate(zip(audio_paths, labels)):
#         try:
#             mel_spec = preprocess_audio_to_mel(path)
#             np.save(os.path.join(split_dir.__str__(), f"{i}_spec.npy"), mel_spec)
#             with open(os.path.join(split_dir.__str__(), f"{i}_label.txt"), 'w') as f:
#                 f.write(str(label))
#         except Exception as e:
#             print(f"Error processing {path}: {e}")
