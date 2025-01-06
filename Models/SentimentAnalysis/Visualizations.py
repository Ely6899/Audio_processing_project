import librosa.display
from matplotlib import pyplot as plt
import torch

from Models.SentimentAnalysis.PreprocessParams import FREQUENCY_BIN_COUNT

def plot_waveform(waveform, sample_rate):
    # Plot each channel separately
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(time_axis, waveform[i].numpy(), label=f'Channel {i + 1}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Waveform for Channel {i + 1}')

    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(mel_spec, sr, n_mels=FREQUENCY_BIN_COUNT):
    """
    Plot a mel spectrogram.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()