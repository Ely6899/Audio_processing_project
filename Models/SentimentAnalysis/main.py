from models import ResNetWithAttention, RavdessPaperModel
from audio_dataset import EmotionDataset, RavdessRawData
from models import SentimentModelHandler

if __name__ == '__main__':
    ravdess_raw_data = RavdessRawData()

    # Load datasets
    train_dataset = EmotionDataset(ravdess_raw_data.train_data)
    val_dataset = EmotionDataset(ravdess_raw_data.val_data)

    # model_handler_base = SentimentModelHandler(EmotionClassifier0(), train_dataset=train_dataset, val_dataset=val_dataset)
    # model_handler_big = SentimentModelHandler(EmotionClassifier2(), train_dataset=train_dataset, val_dataset=val_dataset)
    #
    # model_handler_big.train_model(verbose=True)
    # model_handler_big.plot_losses(file_name="big_losses")
    # model_handler_big.plot_accuracies(file_name="big_accuracies")
    #
    # model_handler_base.train_model(verbose=True)
    # model_handler_base.plot_losses(file_name="base_losses")
    # model_handler_base.plot_accuracies(file_name="base_accuracies")

    model_handler_with_attention = SentimentModelHandler(ResNetWithAttention(), train_dataset=train_dataset, val_dataset=val_dataset)
    model_handler_with_attention.train_model(verbose=True)
    model_handler_with_attention.plot_losses(file_name="emo_net_losses")
    model_handler_with_attention.plot_accuracies(file_name="emo_net_accuracies")
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
