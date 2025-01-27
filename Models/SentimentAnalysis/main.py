from pathlib import Path

#from Models.SentimentAnalysis.ConstPaths import TRAIN_DATA_CSV, DEV_DATA_CSV, MeldPaths
from Models.SentimentAnalysis.Preprocess import audio_to_mel_spectogram
from Models.SentimentAnalysis.PreprocessParams import SAMPLE_RATE
from Models.SentimentAnalysis.Visualizations import plot_mel_spectrogram
from Models.SentimentAnalysis.audio_dataset import EmotionDataset, RavdessRawData
from Models.SentimentAnalysis.models import EmotionClassifier2, SentimentModelHandler, EmotionClassifier1
from visualizations import Visualizations

if __name__ == '__main__':
    ravdess_raw_data = RavdessRawData()

    print(ravdess_raw_data.train_data)
    print(ravdess_raw_data.val_data)
    print(ravdess_raw_data.test_data)
    #plot_mel_spectrogram(spectogram, SAMPLE_RATE)

    # Load datasets
    # train_dataset = EmotionDataset(csv_file=TRAIN_DATA_CSV)
    # val_dataset = EmotionDataset(csv_file=DEV_DATA_CSV)
    #
    # model_handler_base = SentimentModelHandler(EmotionClassifier1(), train_dataset=train_dataset, val_dataset=val_dataset)
    # model_handler_big = SentimentModelHandler(EmotionClassifier2(), train_dataset=train_dataset, val_dataset=val_dataset)
    #
    # model_handler_big.train_model(verbose=True)
    # model_handler_big.plot_losses(file_name="big_losses")
    # model_handler_big.plot_accuracies(file_name="big_accuracies")
    #
    # model_handler_base.train_model(verbose=True)
    # model_handler_base.plot_losses(file_name="base_losses")
    # model_handler_base.plot_accuracies(file_name="base_accuracies")
    #model_handler.train_model(10)

    # Data loaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    #
    # # Model, loss, optimizer
    # model = EmotionClassifier1()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # # Train the model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
