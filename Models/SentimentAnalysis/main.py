from audio_dataset import EmotionSpecDataset, RavdessRawData, RavdessRawDataWithNeutral, EmotionSpecDataset2d, AudioRawDataWithOriginalNeutral
from models import ResNetWithAttentionDropOut, ResNetWithAttention2d, ResNetWithAttentionDropOut2d, ResNetWithAttention
from models import SentimentModelHandler
from pprint import pprint
from Preprocess import audio_to_mel_spectrogram, standardization
from PreprocessParams import SAMPLE_RATE
from Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch, plot_confusion_matrix, plot_mel_spectrogram
from pathlib import Path

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
    ravdess_neutral_original_data = AudioRawDataWithOriginalNeutral()
    pprint(ravdess_neutral_original_data.all_data)

    # create the dataset with the preprocessing logic:
    train = EmotionSpecDataset2d(ravdess_neutral_original_data.train_data)
    val = EmotionSpecDataset2d(ravdess_neutral_original_data.val_data)
    
    # create the model:
    model_paper = ResNetWithAttentionDropOut2d()
    model_name = model_paper.__class__.__name__
    
    # create the handler:
    handler_paper = SentimentModelHandler(model_paper, train, val, batch_size=32, learning_rate=0.001)
    
    # train the model:
    try:
        handler_paper.train_model(epochs=100, verbose=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        
    # save the results in a plot:
    handler_paper.plot_accuracies(f"{model_name}-ACC-fixed")
    handler_paper.plot_losses(f"{model_name}-LOSS-fixed")
    handler_paper.plot_confusion_matrix(f"{model_name}-Confusion-Matrix")
    
    


if __name__ == '__main__':
    plot_mel_spectrogram(audio_to_mel_spectrogram(Path(r"RAVDESS\original_data\Actor_01\03-01-01-01-01-01-01.wav"), top_db=20), SAMPLE_RATE)
    
    # """
    # MICHAL - ADD IN 5.5
    # """
    # rav_data = RavdessRawData()
    # rav_data.print_all_label_counts()

    # train_ds = EmotionSpecDataset(rav_data.train_data)
    # val_ds   = EmotionSpecDataset(rav_data.val_data)
    # test_ds  = EmotionSpecDataset(rav_data.test_data)
    
  