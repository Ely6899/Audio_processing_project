import os.path
from typing import Tuple, Optional

import librosa.display
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ConstPaths import ProjectPaths


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
    librosa.display.specshow(mel_spec, sr=sr, x_axis='off', y_axis='mel', cmap='inferno', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
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
        os.makedirs(ProjectPaths.MODEL_RESULTS, exist_ok=True)
        plt.savefig(os.path.join(ProjectPaths.MODEL_RESULTS, f"{file_save_name}.png"))

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
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        os.makedirs(ProjectPaths.MODEL_RESULTS, exist_ok=True)
        plt.savefig(os.path.join(ProjectPaths.MODEL_RESULTS, f"{file_save_name}.png"))

def plot_confusion_matrix(file_save_name: str, **kwargs):
    train_values_data: Optional[Tuple[list, list, list]]= kwargs.get("train_label_data")
    val_values_data: Optional[Tuple[list, list, list]] = kwargs.get("val_label_data")

    if not train_values_data and not val_values_data:
        print("No data provided for confusion matrices")

    conf_matrices = []
    titles = []
    data_classes = []

    if train_values_data:
        train_truth_labels, train_pred_labels, train_classes = train_values_data
        conf_matrices.append(confusion_matrix(train_truth_labels, train_pred_labels))
        titles.append("Train Confusion Matrix")
        data_classes.append(train_classes)

    if val_values_data:
        val_truth_labels, val_pred_labels, val_classes = val_values_data
        conf_matrices.append(confusion_matrix(val_truth_labels, val_pred_labels))
        titles.append("Validation Confusion Matrix")
        data_classes.append(val_classes)

    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(6 * len(conf_matrices), 5))

    # If only one confusion matrix, turn variable to iterable.
    if len(conf_matrices) == 1:
        axes = [axes]

    for ax, conf_matrix, title, class_names in zip(axes, conf_matrices, titles, data_classes):
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues" if "Train" in title else "Oranges",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax)
        ax.set_title(titles)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.savefig(file_save_name)