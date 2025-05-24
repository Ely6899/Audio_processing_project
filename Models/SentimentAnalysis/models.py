from typing import Callable

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT
from Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch, plot_confusion_matrix
from audio_dataset import EmotionSpecDataset, EmotionSpecDataset2d

"""
for extracting meta-data for organized plot savings
"""
import inspect
from collections import OrderedDict
from typing import Any


class SentimentModelHandler:
    """
    Wrapper class for general model hyperparameters.
    """
    def __init__(self, model: nn.Module, train_dataset: EmotionSpecDataset | EmotionSpecDataset2d, val_dataset: EmotionSpecDataset | EmotionSpecDataset2d, **kwargs):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: nn.Module = model
        self._train_dataset: EmotionSpecDataset = train_dataset
        self._val_dataset: EmotionSpecDataset = val_dataset

        self._train_class_names = self._train_dataset.class_names
        self._val_class_names = self._val_dataset.class_names

        self._batch_size: int = kwargs.get('batch_size', 16)
        self._lr: float = kwargs.get("learning_rate", 0.1)

        self._train_loader: DataLoader = DataLoader(self._train_dataset, self._batch_size, shuffle=True)
        self._val_loader: DataLoader = DataLoader(self._val_dataset, self._batch_size, shuffle=False)

        self._class_weights = train_dataset.class_weights
        #print(f"Training set class Weights: {self._class_weights}")

        self._criterion = kwargs.get("criterion", nn.CrossEntropyLoss)(weight=self._class_weights.to(self._device))
        self._optimizer = kwargs.get("optimizer", optim.SGD)(self._model.parameters(), momentum = 0.9, lr=self._lr, weight_decay=1e-6)
        #self._scheduler = kwargs.get("scheduler", None)()

        self._training_logs: dict = dict()

        # Tuple structure is (Loss score, Accuracy score)
        self._train_scores: list[tuple[float, float]] = []
        self._val_scores: list[tuple[float, float]] = []

        # New lists to store the true labels and predicted labels for confusion matrix(Of both train and validation).
        self._true_labels_train, self._true_labels_val = [], []
        self._pred_labels_train, self._pred_labels_val = [], []

        """
        for extracting meta-data for organized plot savings
        """
        # 1) Keep the hyper-parameters exactly as passed
        self.hparams: OrderedDict[str, Any] = OrderedDict(kwargs)
         # 2) Optional – auto-collect the *model* constructor kwargs
        sig = inspect.signature(self._model.__class__.__init__)
        ctor_args = {
            k: v.default
            for k, v in sig.parameters.items()
            if k != "self" and v.default is not inspect._empty
        }
        self.hparams.update({f"model_{k}": v for k, v in ctor_args.items()})
        


    def __train_one_epoch(self):
        """
        Trains a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
        self._true_labels_train.clear()
        self._pred_labels_train.clear()

        self._model.train()
        running_loss = 0.0
        correct = 0

        for mel_spec, label in tqdm(self._train_loader, desc="Model training"):
            mel_spec, label = mel_spec.to(self._device), label.to(self._device)

            self._optimizer.zero_grad()
            output = self._model(mel_spec)
            #print("Output sample:", output[0].detach().cpu().numpy())
            #print("Label sample:", label[0].item())
            loss = self._criterion(output, label)
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item() * mel_spec.size(0)

            predictions = output.argmax(1)
            correct += (predictions == label).sum().item()

            self._true_labels_train.extend(label.cpu().numpy())
            self._pred_labels_train.extend(predictions.cpu().numpy())

        total_samples = len(self._train_loader.dataset)

        return running_loss / total_samples, correct, total_samples

    def __validate(self):
        """
        Validates a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
        self._true_labels_val.clear()
        self._pred_labels_val.clear()

        self._model.eval()
        running_loss = 0.0
        correct = 0


        with torch.no_grad():
            for mel_spec, label in tqdm(self._val_loader, desc="Model validation"):
                mel_spec, label = mel_spec.to(self._device), label.to(self._device)
                output = self._model(mel_spec)
                loss = self._criterion(output, label)

                running_loss += loss.item() * mel_spec.size(0)

                predictions = output.argmax(1)
                correct += (predictions == label).sum().item()

                self._true_labels_val.extend(label.cpu().numpy())
                self._pred_labels_val.extend(predictions.cpu().numpy())

        total_samples = len(self._val_loader.dataset)
        #self._scheduler.step(running_loss / total_samples)
        return running_loss / total_samples, correct, total_samples

    def train_model(self, epochs: int = 10, verbose: bool = False):
        """
        Applies the entire training logic and saves the results.
        @param epochs: Number of epochs to train the model. Defaults to 10.
        @param verbose: Verbosity of results. Defaults to False.
        """
        self._model.to(self._device)

        print(f"Training model with device: {self._device}")
        for epoch in range(epochs):
            train_loss, train_correct, train_total = self.__train_one_epoch() # noam question eli: does train loss is the average train loss per batch?
            val_loss, val_correct, val_total = self.__validate()

            train_accuracy = (train_correct / train_total) * 100.0
            val_accuracy = (val_correct / val_total) * 100.0
            current_lr = self._optimizer.param_groups[0]['lr']

            epoch_string: str = f"Epoch {epoch + 1}"
            results_string: str = f'Train Loss: {train_loss:.4f},\n' \
                                  f'Train Accuracy: {train_accuracy:.2f}%\n' \
                                  f'[{train_correct}/{train_total}]\n' \
                                  f'Val Loss: {val_loss:.4f},\n' \
                                  f'Val Accuracy: {val_accuracy:.2f}%\n' \
                                  f'[{val_correct}/{val_total}]\n' \
                                  f'LR: {current_lr}'

            self._training_logs[epoch_string] = results_string
            self._train_scores.append((train_loss, train_accuracy))
            self._val_scores.append((val_loss, val_accuracy))

            if verbose:
                print(f"Epoch {epoch + 1}")
                print(results_string)
                print("--------------------------------\n")

    def __str__(self):
        return (f"Model name: {self._model.__class__.__name__}\n"
                f"Device: {self._device}\n"
                f"Batch size: {self._batch_size}\n"
                f"Criterion: {self._criterion.__class__.__name__}\n"
                f"Optimizer: {self._optimizer.__class__.__name__}")

    """Properties"""

    @property
    def model_name(self) -> str:
        return self._model.__class__.__name__.__str__()

    @property
    def batch_size(self) -> str:
        return self._batch_size.__str__()

    @property
    def starting_lr(self) -> str:
        return self._lr.__str__()

    @property
    def criterion(self) -> str:
        return self._criterion.__class__.__name__.__str__()

    @property
    def optimizer(self) -> str:
        return self._optimizer.__class__.__name__.__str__()

    def __generate_plot_title(self) -> str:
        return (f"{self.model_name}_"
                f"")

    def plot_losses(self, file_name: str | None = None):
        file_name = f"{self._model.__class__.__name__}_losses" if None else file_name
        train_losses = [scores[0] for scores in self._train_scores]
        val_losses = [scores[0] for scores in self._val_scores]
        plot_loss_per_epoch(file_save_name= file_name,
                            hparams=self.hparams,              # ← lives in the handler
                            training_loss=train_losses,
                            validation_loss=val_losses)

    def plot_accuracies(self, file_name: str | None = None):
        file_name = f"{self._model.__class__.__name__}_accuracies" if None else file_name
        train_accuracies = [scores[1] for scores in self._train_scores]
        val_accuracies = [scores[1] for scores in self._val_scores]
        plot_accuracy_per_epoch(file_save_name=file_name,
                                hparams=self.hparams,              # ← lives in the handler
                            training_accuracy=train_accuracies,
                            validation_accuracy=val_accuracies)

    def plot_confusion_matrix(self, file_name: str | None = None):
        file_name = f"{self._model.__class__.__name__}_confusion_matrix" if None else file_name
        train_confusion_data = (self._true_labels_train, self._pred_labels_train, self._train_class_names)
        val_confusion_data = (self._true_labels_val, self._pred_labels_val, self._val_class_names)
        plot_confusion_matrix(file_save_name=file_name,
                              hparams=self.hparams,              # ← lives in the handler
                              train_label_data=train_confusion_data,
                              val_label_data=val_confusion_data)

    def save_plots(self):
        #TODO: Add robust title for plots for easy identification
        self.plot_accuracies()
        self.plot_losses()
        self.plot_confusion_matrix()


"""Emo-Net Logic"""

class ResidualBlockNew(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=False):
        super(ResidualBlockNew, self).__init__()

        self.stride = stride
        self.shortcut = shortcut

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if self.shortcut:
            self.shortcut_pool = nn.AvgPool2d(2, stride=2, ceil_mode=True)
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
        identity = x

        # print(f"Identity before: {identity.shape}")
        # print(f"x before: {x.shape}")

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #print(f"{self.shortcut}")
        if self.shortcut:
            identity = self.shortcut_pool(identity)
            identity = self.shortcut_conv(identity)

        # print(f"Identity after: {identity.shape}")
        # print(f"x after(output): {out.shape}")
        # print()

        out += identity
        out = self.relu(out)
        return out

class ResNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super(ResNetModule, self).__init__()

        self.blocks = []
        for i in range(num_blocks):
            if i == 0:  # First block requires a shortcut connection
                self.blocks.append(ResidualBlockNew(in_channels, out_channels, stride=stride, shortcut=True))
            else:
                self.blocks.append(ResidualBlockNew(out_channels, out_channels, stride=1, shortcut=False))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttention, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # First submodule with 64 filters
        self.module1 = ResNetModule(8, 16, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(16, 32, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        #self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        self.fc1 = nn.Linear(64 * (TARGET_FRAMES // 8) * (FREQUENCY_BIN_COUNT // 8), 64)  # Assuming input size (32x32)
        self.bn_fc1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        #x, _ = self.attention(x, x, x)
        #x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))

        # Output layer (classification)
        x = self.fc_out(x)

        return x

class ResNetWithAttentionDropOut(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttentionDropOut, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # First submodule with 64 filters
        self.module1 = ResNetModule(8, 16, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(16, 32, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        self.fc1 = nn.Linear(64 * (TARGET_FRAMES // 8) * (FREQUENCY_BIN_COUNT // 8), 256)  # Assuming input size (32x32)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.6)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        # Output layer (classification)
        x = self.fc_out(x)

        return x


"""ResNet 2 channels"""
class ResNetWithAttention2d(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttention2d, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # First submodule with 64 filters
        self.module1 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(64, 128, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(128, 256, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        self.fc1 = nn.Linear(256 * (TARGET_FRAMES // 8) * (FREQUENCY_BIN_COUNT // 8), 1024)  # Assuming input size (32x32)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))

        # Output layer (classification)
        x = self.fc_out(x)

        return x

class ResNetWithAttentionDropOut2d(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttentionDropOut2d, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # First submodule with 64 filters
        self.module1 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(64, 128, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(128, 256, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        self.fc1 = nn.Linear(256 * (TARGET_FRAMES // 8) * (FREQUENCY_BIN_COUNT // 8), 1024)  # Assuming input size (32x32)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.6)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        # Output layer (classification)
        x = self.fc_out(x)

        return x


"""New Simplified models"""

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.Tanh()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BasicFcBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: Callable, include_dropout: bool, dropout_rate: float = 0.3):
        super(BasicFcBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.bn_fc = nn.BatchNorm1d(output_size)
        self.activation_function = activation_function()
        self._include_dropout = include_dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.activation_function(self.bn_fc(self.fc(x)))
        return self.dropout(x) if self._include_dropout else x


class SentimentModelAttentionDropOut(nn.Module):
    def __init__(self, dual_channel = False, num_classes=8, include_dropout = True, include_attention = False):
        super(SentimentModelAttentionDropOut, self).__init__()
        self._include_dropout = include_dropout
        self._include_attention = include_attention

        self.conv1 = BasicConvBlock(1 if not dual_channel else 2, 32)
        #self.conv2 = BasicConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        #self.conv3 = BasicConvBlock(64, 128)


        # Attention layer (Self-Attention)
        if self._include_attention:
            self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        # Final batch normalization and ReLU
        #self.bn2 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        self.fc1 = BasicFcBlock(32 * 8 * 8, 256, nn.Tanh, include_dropout=self._include_dropout, dropout_rate=0.6)
        self.fc2 = BasicFcBlock(256, 128, nn.Tanh, include_dropout=self._include_dropout, dropout_rate=0.6)
        self.fc3 = BasicFcBlock(128, 64, nn.Tanh, include_dropout=self._include_dropout, dropout_rate=0.6)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        #x = self.conv2(x)

        #x = self.conv3(x)

        x = self.pool(x)

        # Apply attention
        if self._include_attention:
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
            x, _ = self.attention(x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        x = x.reshape(x.size(0), -1)   # Flatten

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.fc_out(x)

        return x
