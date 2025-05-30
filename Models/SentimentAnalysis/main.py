import os
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch.cuda
import whisper
from whisper import Whisper
from transformers import pipeline

from ConstPaths import RavdessPaths
from Models.SentimentAnalysis.audio_dataset import CREMARawData, AllRawData
from Models.SentimentAnalysis.models import MlpModel
from PreprocessParams import SAMPLE_RATE
from audio_dataset import EmotionSpecDataset, RavdessRawData, EmotionSpecDataset2d
from models import ResNetWithAttentionDropOut2d, ResNetWithAttention, ResNetWithAttentionDropOut
from models import SentimentModelHandler, SentimentModelAttentionDropOut, ResNetWithAttention2d


def train_1channel():
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)
    crema_raw_data = CREMARawData()

    ravdess_raw_data.print_all_label_counts()
    crema_raw_data.print_all_label_counts()

    all_data = AllRawData((ravdess_raw_data, crema_raw_data))
    print(len(all_data.all_data))
    print(len(all_data.train_data))
    print(len(all_data.val_data))

    print(all_data.print_all_label_counts())


    # pprint(ravdess_raw_data.all_data)

    # create the dataset with the preprocessing logic:
    # train = EmotionWavDataset(ravdess_raw_data.train_data)
    # val = EmotionWavDataset(ravdess_raw_data.val_data)

    # print(train.class_counts)
    # print(val.class_counts)

    #For testing pre-processing spectrogram
    # mel_spectrogram = audio_to_mel_spectrogram(Path("RAVDESS/Actor_01/03-01-02-02-02-02-01.wav"), normalization_fn=standardization)
    # plot_mel_spectrogram(mel_spectrogram, SAMPLE_RATE)


    # # create the model:
    # model_paper = MlpModel(num_classes=8)
    #
    # #
    # # # create the handler:
    # handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=16, learning_rate=0.001)
    # #
    # # # train the model:
    # try:
    #     handler_paper.train_model(epochs = 50, verbose=True)
    # except KeyboardInterrupt:
    #     print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    #handler_paper.plot_accuracies("SINGLE-SENTENCE-MLP-SGD-70-30-no-split-dropout-ACC")
    #handler_paper.plot_losses("SINGLE-SENTENCE-MLP-SGD-70-30-no-split-dropout-LOSS")
    #handler_paper.plot_confusion_matrix("SINGLE-SENTENCE-MLP-SGD-70-30-no-split-dropout-Matrix")

def train_2channel():
    ravdess_raw_data = RavdessRawDataWithNeutral()
    #pprint(ravdess_raw_data.all_data)

    # # create the dataset with the preprocessing logic:
    train = EmotionSpecDataset2d(ravdess_raw_data.train_data)
    val = EmotionSpecDataset2d(ravdess_raw_data.val_data)

    #print(train.__getitem__(2)[0].shape)
    
    # create the model:
    model_paper = ResNetWithAttention2d()
    model_name = model_paper.__class__.__name__
    
    # create the handler:
    handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=16, learning_rate=0.001)
    
    # train the model:
    try:
        handler_paper.train_model(epochs=50, verbose=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    handler_paper.plot_accuracies(f"DEPTH-MODEL-SGD-70-30-no-split-ACC")
    handler_paper.plot_losses(f"DEPTH-MODEL-SGD-70-30-no-split-LOSS")
    handler_paper.plot_confusion_matrix(f"DEPTH-MODEL-SGD-70-30--no-split-Confusion-Matrix")

def segment_words_with_timestamps(model: Whisper, file_path_original: Union[str, Path]) -> list[tuple[str, float, float]]:
    #segmentation_model = whisper.load_model("small.en", device="cuda" if torch.cuda.is_available() else "cpu")
    results = whisper.transcribe(model,
                                 file_path_original.__str__(),
                                 word_timestamps=True)

    print(results)
    segmented_words = results['segments'][0]['words']
    organized_results: list = []
    for segmented_word_data_dict in segmented_words:
        segmented_word: str = segmented_word_data_dict['word']
        timestamp_start: float = segmented_word_data_dict['start']
        timestamp_end: float = segmented_word_data_dict['end']

        print(f"({segmented_word}, {timestamp_start:.4f}, {timestamp_end:.4f})")
        organized_results.append((segmented_word, timestamp_start, timestamp_end))

    return organized_results

def create_new_audio_from_segments(audio_path_original: Union[str, Path],
                                   segments_original: list[tuple[str, float, float]],
                                   audio_path_neutral: [Union[str, Path]],
                                   segments_neutral: [list[tuple[str, float, float]]]) -> None:

    assert len(segments_original) == len(segments_neutral), "Word segments don't match in length!"

    audio_path_original = Path(audio_path_original)
    actor_folder, audio_file_name = audio_path_original.parts[-2:]

    audio_path_neutral = Path(audio_path_neutral)

    pause_duration = 0.3  # in seconds
    pause = np.zeros(int(SAMPLE_RATE * pause_duration))

    audio_data_original, sr = librosa.load(audio_path_original, mono=True, sr=SAMPLE_RATE)
    audio_data_neutral, sr = librosa.load(audio_path_neutral, mono=True, sr=SAMPLE_RATE)

    new_audio_data: list = []

    for original_values, neutral_values in zip(segments_original, segments_neutral):
        word_original, start_time_original, end_time_original = original_values
        word_neutral, start_time_neutral, end_time_neutral = neutral_values

        assert word_original == word_neutral, "Non matching words!"

        start_index_original, start_index_neutral = int(start_time_original*sr), int(start_time_neutral*sr)
        end_index_original, end_index_neutral = int(end_time_original*sr), int(end_time_neutral*sr)

        audio_segment_original = audio_data_original[start_index_original:end_index_original]
        audio_segment_neutral = audio_data_neutral[start_index_neutral:end_index_neutral]

        new_audio_data.append(audio_segment_original)
        new_audio_data.append(pause)
        new_audio_data.append(audio_segment_neutral)
        new_audio_data.append(pause)

    final_audio = np.concatenate(new_audio_data)

    folder_to_create = RavdessPaths.WORD_CHUNKED_AUDIO_DATA / actor_folder
    os.makedirs(folder_to_create, exist_ok=True)
    output_path = folder_to_create / audio_file_name
    print(f"Saving new file to {output_path}")

    sf.write(output_path, final_audio, samplerate=SAMPLE_RATE)


if __name__ == '__main__':
    #segmentation_model = whisper.load_model("medium.en", device="cuda" if torch.cuda.is_available() else "cpu")
    # results = whisper.transcribe(segmentation_model,
    #                              "RAVDESS/original_data/Actor_01/03-01-01-01-01-01-01.wav",
    #                              word_timestamps=True)

    #Kids, Rep1
    #file_to_test: str = "RAVDESS/original_data/Actor_02/03-01-01-01-01-01-02.wav"

    #matching_neutral_file: str = "RAVDESS/neutral_synthesized/Actor_02/kids_rep1_act2.wav"

    #combine_audios(file_path_original=file_to_test, file_path_neutral=matching_neutral_file)

    # segmentation_results_original = segment_words_with_timestamps(model=segmentation_model,
    #                                                      file_path_original=file_to_test)
    #
    # segmentation_results_neutral = segment_words_with_timestamps(model = segmentation_model,
    #                                                              file_path_original=matching_neutral_file)
    #
    # create_new_audio_from_segments(file_to_test, segmentation_results_original, matching_neutral_file, segmentation_results_neutral)
    #
    # segment_words_with_timestamps(segmentation_model, "RAVDESS/chunked_word_audio/Actor_02/03-01-01-01-01-01-02.wav")


    # print(segmentation_results)
    #
    # segmented_words = results['segments'][0]['words']
    # for segmented_word_data_dict in segmented_words:
    #     segmented_word: str = segmented_word_data_dict['word']
    #     timestamp_start: float = segmented_word_data_dict['start']
    #     timestamp_end: float = segmented_word_data_dict['end']
    #     print(f"{segmented_word} starts at {timestamp_start:.2f} and ends at {timestamp_end:.2f}")



    # plot_mel_spectrogram(audio_to_mel_spectrogram(Path(r"RAVDESS\original_data\Actor_01\03-01-01-01-01-01-01.wav"), top_db=20), SAMPLE_RATE)

    """
    MICHAL - ADD IN 5.5
    """
    # rav_data = RavdessRawData()
    # rav_data.print_all_label_counts()
    # #train_2channel()
    #
    # train_ds = EmotionSpecDataset(rav_data.train_data)
    # val_ds   = EmotionSpecDataset(rav_data.val_data)
    # test_ds  = EmotionSpecDataset(rav_data.test_data)
    train_1channel()

