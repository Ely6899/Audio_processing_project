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

def plot_mel_spectrogram(mel_spec, sr):
    """
    Plot a mel spectrogram.
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.squeeze().numpy()  # Remove extra dimensions and convert to NumPy

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, x_axis='off', y_axis='mel', cmap='viridis', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    #plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

def plot_loss_per_epoch(file_save_name: str, **kwargs):
    training_loss = kwargs.get("training_loss", None)
    validation_loss = kwargs.get("validation_loss", None)

    if training_loss is None and validation_loss is None:
        print("No data given for plotting")
    else:
        # Plot
        plt.figure(figsize=(10, 6))

        if training_loss is not None:
            plt.plot(training_loss, marker='o', linestyle='-', color='b', label='Training Loss')

        if validation_loss is not None:
            plt.plot(validation_loss, marker='o', linestyle='--', color='r', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{file_save_name}.png")

def plot_accuracy_per_epoch(file_save_name: str, **kwargs):
    training_accuracy = kwargs.get("training_accuracy", None)
    validation_accuracy = kwargs.get("validation_accuracy", None)

    if training_accuracy is None and validation_accuracy is None:
        print("No data given for plotting")
    else:
        # Plot
        plt.figure(figsize=(10, 6))

        if training_accuracy is not None:
            plt.plot(training_accuracy, marker='o', linestyle='-', color='b', label='Training Accuracy')

        if validation_accuracy is not None:
            plt.plot(validation_accuracy, marker='o', linestyle='--', color='r', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{file_save_name}.png")

