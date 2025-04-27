import os
from typing import Union

import librosa
import numpy as np

from ConstPaths import RavdessPaths
import soundfile as sf
import torch.cuda
from whisper import Whisper

from Models.SentimentAnalysis.Visualizations import plot_waveform
from audio_dataset import EmotionSpecDataset, RavdessRawData, RavdessRawDataWithNeutral, EmotionSpecDataset2d
from models import ResNetWithAttentionDropOut, ResNetWithAttention2d, ResNetWithAttentionDropOut2d, ResNetWithAttention
from models import SentimentModelHandler
from pprint import pprint
from Preprocess import audio_to_mel_spectrogram, standardization
from PreprocessParams import SAMPLE_RATE
from Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch, plot_confusion_matrix, plot_mel_spectrogram
from pathlib import Path

import whisper

def train_1channel():
    ravdess_raw_data = RavdessRawData()
    # pprint(ravdess_raw_data.all_data)

    # create the dataset with the preprocessing logic:
    train = EmotionSpecDataset(ravdess_raw_data.train_data)
    val = EmotionSpecDataset(ravdess_raw_data.val_data)


    #For testing pre-processing spectrogram
    # mel_spectrogram = audio_to_mel_spectrogram(Path("RAVDESS/Actor_01/03-01-02-02-02-02-01.wav"), normalization_fn=standardization)
    # plot_mel_spectrogram(mel_spectrogram, SAMPLE_RATE)


    # # create the model:
    model_paper = ResNetWithAttention(num_classes=7)

    #
    # # create the handler:
    handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=32, learning_rate=0.001)
    #
    # # train the model:
    try:
        handler_paper.train_model(epochs = 50, verbose=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    handler_paper.plot_accuracies("PaperModelSeven-ACC-fixed")
    handler_paper.plot_losses("PaperModelSeven-LOSS-fixed")
    handler_paper.plot_confusion_matrix("PaperModel-Confusion-Matrix-Seven")

def train_2channel():
    ravdess_raw_data = RavdessRawDataWithNeutral()
    pprint(ravdess_raw_data.all_data)

    # # create the dataset with the preprocessing logic:
    # train = EmotionSpecDataset2d(ravdess_raw_data.train_data)
    # val = EmotionSpecDataset2d(ravdess_raw_data.val_data)
    
    # # create the model:
    # model_paper = ResNetWithAttentionDropOut2d()
    # model_name = model_paper.__class__.__name__
    
    # # create the handler:
    # handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=32, learning_rate=0.001)
    
    # # train the model:
    # try:
    #     handler_paper.train_model(epochs=100, verbose=True)
    # except KeyboardInterrupt:
    #     print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    # handler_paper.plot_accuracies(f"{model_name}-ACC-fixed")
    # handler_paper.plot_losses(f"{model_name}-LOSS-fixed")
    # handler_paper.plot_confusion_matrix(f"{model_name}-Confusion-Matrix")

def segment_words_with_timestamps(model: Whisper, file_path: Union[str, Path]) -> set[tuple[str, float, float]]:
    #segmentation_model = whisper.load_model("small.en", device="cuda" if torch.cuda.is_available() else "cpu")
    results = whisper.transcribe(model,
                                 file_path.__str__(),
                                 word_timestamps=True)

    print(results)
    segmented_words = results['segments'][0]['words']
    organized_results: set = set()
    for segmented_word_data_dict in segmented_words:
        segmented_word: str = segmented_word_data_dict['word']
        timestamp_start: float = segmented_word_data_dict['start']
        timestamp_end: float = segmented_word_data_dict['end']
        organized_results.add((segmented_word, timestamp_start, timestamp_end))

    return organized_results

def create_new_audio_from_segments(audio_path: Union[str, Path], segments: set[tuple[str, float, float]]) -> None:
    audio_path = Path(audio_path)
    actor_folder, audio_file_name = audio_path.parts[-2:]
    audio_data, sr = librosa.load(audio_path, mono=True, sr=SAMPLE_RATE)
    new_audio_data: list = []

    for _, start, end in segments:
        start_index = int(start*sr)
        end_index = int(end*sr)
        audio_segment = audio_data[start_index:end_index]

        padded_audio_segment = np.pad(audio_segment, pad_width=(5, 5), mode='constant', constant_values=0)
        new_audio_data.extend(padded_audio_segment)

    new_audio_data = np.array(new_audio_data).reshape(-1,1)

    folder_to_create = RavdessPaths.WORD_CHUNKED_AUDIO_DATA / actor_folder
    os.makedirs(folder_to_create, exist_ok=True)
    output_path = folder_to_create / audio_file_name
    print(f"Saving new file to {output_path}")

    sf.write(output_path, new_audio_data, samplerate=SAMPLE_RATE)


if __name__ == '__main__':
    segmentation_model = whisper.load_model("small.en", device="cuda" if torch.cuda.is_available() else "cpu")
    # results = whisper.transcribe(segmentation_model,
    #                              "RAVDESS/original_data/Actor_01/03-01-01-01-01-01-01.wav",
    #                              word_timestamps=True)

    file_to_test: str = "RAVDESS/original_data/Actor_02/03-01-01-01-01-01-02.wav"

    segmentation_results = segment_words_with_timestamps(model=segmentation_model,
                                                         file_path=file_to_test)

    create_new_audio_from_segments(file_to_test, segmentation_results)

    segment_words_with_timestamps(segmentation_model, "RAVDESS/chunked_word_audio/Actor_02/03-01-01-01-01-01-02.wav")


    #print(segmentation_results)

    # segmented_words = results['segments'][0]['words']
    # for segmented_word_data_dict in segmented_words:
    #     segmented_word: str = segmented_word_data_dict['word']
    #     timestamp_start: float = segmented_word_data_dict['start']
    #     timestamp_end: float = segmented_word_data_dict['end']
    #     print(f"{segmented_word} starts at {timestamp_start:.2f} and ends at {timestamp_end:.2f}")



    # plot_mel_spectrogram(audio_to_mel_spectrogram(Path(r"RAVDESS\original_data\Actor_01\03-01-01-01-01-01-01.wav"), top_db=20), SAMPLE_RATE)
    #train_1channel()
