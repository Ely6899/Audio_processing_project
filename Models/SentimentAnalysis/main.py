from Models.SentimentAnalysis.ConstPaths import TRAIN_DATA_CSV, DEV_DATA_CSV
from Models.SentimentAnalysis.audio_dataset import EmotionDataset
from Models.SentimentAnalysis.models import EmotionClassifier1, SentimentModelHandler

if __name__ == '__main__':
    #audio_to_mel_spectogram(Path("wav_splits/dev_dia0_utt0.wav"))

    # Load datasets
    train_dataset = EmotionDataset(csv_file=TRAIN_DATA_CSV)
    val_dataset = EmotionDataset(csv_file=DEV_DATA_CSV)

    model_handler = SentimentModelHandler(EmotionClassifier1(), train_dataset=train_dataset, val_dataset=val_dataset)
    model_handler.train_model(verbose=True)
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
