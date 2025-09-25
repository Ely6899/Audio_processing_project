from audio_dataset import AllRawData, RavdessRawDataWithNeutral, EmotionSpecDataset2d
from audio_dataset import RavdessRawData, EmotionSpecDataset
from models import SentimentModelHandler, ResNetWithAttention, ResNetWithAttention2d


def train_1channel():
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)
    ravdess_raw_data.print_all_label_counts()

    all_data_raw = tuple([ravdess_raw_data])
    all_data = AllRawData(all_data_raw)

    train_set, val_set, test_set = all_data.train_val_test_split(0.1, 0.2)

    # create the dataset with the preprocessing logic:
    train_ds = EmotionSpecDataset(train_set)
    val_ds = EmotionSpecDataset(val_set)
    test_ds = EmotionSpecDataset(test_set)


    # # create the model:
    model_paper = ResNetWithAttention(num_classes=8)
    #
    # #
    # # # create the handler:
    handler_paper = SentimentModelHandler(model_paper,
                                          train_ds,
                                          val_ds,
                                          test_ds,
                                          batch_size=32,
                                          learning_rate=0.001,
                                          raw_data_class_name="ravdess_raw_data")
    # #
    # # # train the model:
    try:
        handler_paper.train_model(epochs = 40, verbose=True, save_model=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        
    # # save the results in a plot:
    handler_paper.plot_accuracies("SINGLE-SENTENCE-SGD-70-10-20-no-split-ACC")
    handler_paper.plot_losses("SINGLE-SENTENCE-SGD-70-10-20-no-split-LOSS")
    handler_paper.plot_confusion_matrix("SINGLE-SENTENCE-SGD-70-10-20-no-split-Matrix")

def train_2channel():
    ravdess_raw_data = RavdessRawDataWithNeutral()
    all_data = AllRawData(tuple([ravdess_raw_data]))

    train, val, test = all_data.train_val_test_split(0.1, 0.2)

    # # create the dataset with the preprocessing logic:
    train_ds = EmotionSpecDataset2d(train)
    val_ds = EmotionSpecDataset2d(val)
    test_ds = EmotionSpecDataset2d(test)

    # create the model:
    model_paper_2d = ResNetWithAttention2d()

    # create the handler:
    handler_2d = SentimentModelHandler(model_paper_2d,
                                          train_ds,
                                          val_ds,
                                          test_ds,
                                          batch_size=32,
                                          learning_rate=0.001,
                                          raw_data_class_name="ravdess_raw_data_2d")

    # train the model:
    try:
        handler_2d.train_model(epochs=40, verbose=True)
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")

    # # save the results in a plot:
    handler_2d.plot_accuracies(f"DEPTH-MODEL-SGD-70-10-20-no-split-ACC")
    handler_2d.plot_losses(f"DEPTH-MODEL-SGD-70-10-20-no-split-LOSS")
    handler_2d.plot_confusion_matrix(f"DEPTH-MODEL-SGD-70-10-20--no-split-Confusion-Matrix")


if __name__ == '__main__':
    #train_1channel()
    train_2channel()

