from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt

from Preprocess import audio_to_mel_spectrogram, audio_to_waveform
from PreprocessParams import SAMPLE_RATE, HOP_LENGTH
from Visualizations import plot_mel_spectrogram, plot_waveform
from models import ResNetWithAttention, RavdessPaperModel
from audio_dataset import EmotionSpecDataset, EmotionWaveDataset, RavdessRawData
from models import SentimentModelHandler
from pprint import pprint 


if __name__ == '__main__':
    ravdess_raw_data = RavdessRawData()
    # pprint(ravdess_raw_data.all_data)

    # create the dataset with the preprocessing logic:
    train = EmotionSpecDataset(ravdess_raw_data.train_data)
    val = EmotionSpecDataset(ravdess_raw_data.val_data)
    
    # create the model:
    model = ResNetWithAttention()
    
    # create the handler:
    handler = SentimentModelHandler(model, train, val)
    
    # train the model:
    handler.train_model()
    
    # save the results in a plot:
    handler.plot_accuracies("ResNetWithAttention after 2 db - ACC")
    handler.plot_losses("ResNetWithAttention after 2 db - LOSS")
    
    # S_dB_np = mel_spectogram.cpu().numpy()
    # # Invert mel spectrogram to get the magnitude spectrogram
    # S_inv = librosa.db_to_power(S_dB_np)  # Convert back to power spectrogram

    # Reconstruct the audio using Griffin-Lim algorithm
    # y_reconstructed = librosa.feature.inverse.mel_to_audio(S_inv, sr=16000, n_iter=32, hop_length=512)
    #
    # print(f"Audio data type: {y_reconstructed.dtype}")
    # print(f"Audio waveform shape: {y_reconstructed.shape}")
    # print(f"Min value: {np.min(y_reconstructed)}, Max value: {np.max(y_reconstructed)}")
    #
    # y_reconstructed = y_reconstructed.flatten()
    # # Save the reconstructed audio to a file
    # sf.write('reconstructed_audio_old_algo.wav', y_reconstructed, 16000)

    # plot_mel_spectrogram(mel_spectogram, sample_rate)
    #plot_waveform(waveform, SAMPLE_RATE)
    
    # Load datasets
    # train_dataset = EmotionSpecDataset(ravdess_raw_data.train_data)
    # val_dataset = EmotionSpecDataset(ravdess_raw_data.val_data)



    # y, sr = librosa.load(file_path, sr=16000)  # y is the audio signal, sr is the sampling rate
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    #
    # # Convert to decibels (log scale) for better visualization
    # S_dB = librosa.power_to_db(S, ref=np.max)
    #
    # # Plot the Mel spectrogram
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.show()
    #
    # # Invert mel spectrogram to get the magnitude spectrogram
    # S_inv = librosa.db_to_power(S_dB)  # Convert back to power spectrogram
    #
    # y_reconstructed = librosa.feature.inverse.mel_to_audio(S_inv, sr=16000, n_iter=32, hop_length=512)
    #
    # print(f"Audio data type: {y_reconstructed.dtype}")
    # print(f"Audio waveform shape: {y_reconstructed.shape}")
    # print(f"Min value: {np.min(y_reconstructed)}, Max value: {np.max(y_reconstructed)}")
    #
    # y_reconstructed = y_reconstructed.flatten()
    # # Save the reconstructed audio to a file
    # sf.write('reconstructed_audio_new_algo.wav', y_reconstructed, 16000)


