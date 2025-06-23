import os
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf

from ConstPaths import RavdessPaths
from audio_dataset import CREMARawData, AllRawData
from PreprocessParams import SAMPLE_RATE
from audio_dataset import RavdessRawData, EmotionSpecDataset
from models import SentimentModelHandler, ResNetWithAttention


def train_1channel():
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)
    #crema_raw_data = CREMARawData()

    ravdess_raw_data.print_all_label_counts()
    #crema_raw_data.print_all_label_counts()

    all_data_raw = tuple([ravdess_raw_data])
    all_data = AllRawData(all_data_raw)

    train_set, val_set = all_data.train_val_test_split(0.3)

    # print(len(all_data.all_data))
    # print(len(train_set))
    # print(len(val_set))
    #
    # print(all_data.print_all_label_counts())


    # pprint(ravdess_raw_data.all_data)

    # create the dataset with the preprocessing logic:
    train_ds = EmotionSpecDataset(train_set)
    val_ds = EmotionSpecDataset(val_set)

    # print(train.class_counts)
    # print(val.class_counts)

    #For testing pre-processing spectrogram
    # mel_spectrogram = audio_to_mel_spectrogram(Path("RAVDESS/Actor_01/03-01-02-02-02-02-01.wav"), normalization_fn=standardization)
    # plot_mel_spectrogram(mel_spectrogram, SAMPLE_RATE)


    # # create the model:
    model_paper = ResNetWithAttention(num_classes=8)
    #
    # #
    # # # create the handler:
    handler_paper = SentimentModelHandler(model_paper, train_ds, val_ds, batch_size=32, learning_rate=0.001, raw_data_class_name="ravdess_raw_data, crema_raw_data")
    # #
    # # # train the model:
    try:
        handler_paper.train_model(epochs = 50, verbose=True, save_model=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    handler_paper.plot_accuracies("SINGLE-SENTENCE-SGD-70-30-no-split-ACC")
    handler_paper.plot_losses("SINGLE-SENTENCE-SGD-70-30-no-split-LOSS")
    handler_paper.plot_confusion_matrix("SINGLE-SENTENCE-SGD-70-30-no-split-Matrix")

# def train_2channel():
#     ravdess_raw_data = RavdessRawDataWithNeutral()
#     #pprint(ravdess_raw_data.all_data)
#
#     # # create the dataset with the preprocessing logic:
#     train = EmotionSpecDataset2d(ravdess_raw_data.train_data)
#     val = EmotionSpecDataset2d(ravdess_raw_data.val_data)
#
#     #print(train.__getitem__(2)[0].shape)
#
#     # create the model:
#     model_paper = ResNetWithAttention2d()
#     model_name = model_paper.__class__.__name__
#
#     # create the handler:
#     handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=16, learning_rate=0.001)
#
#     # train the model:
#     try:
#         handler_paper.train_model(epochs=50, verbose=True)
#     except KeyboardInterrupt:
#         print("Training was interrupted by the user.")
#
#     # # save the results in a plot:
#     handler_paper.plot_accuracies(f"DEPTH-MODEL-SGD-70-30-no-split-ACC")
#     handler_paper.plot_losses(f"DEPTH-MODEL-SGD-70-30-no-split-LOSS")
#     handler_paper.plot_confusion_matrix(f"DEPTH-MODEL-SGD-70-30--no-split-Confusion-Matrix")


if __name__ == '__main__':
    train_1channel()

