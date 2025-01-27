import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Models.SentimentAnalysis.PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT
from Models.SentimentAnalysis.Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch


class SentimentModelHandler:
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, **kwargs):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: nn.Module = model
        self._train_dataset: Dataset = train_dataset
        self._val_dataset: Dataset = val_dataset

        self._batch_size: int = kwargs.get('batch_size', 16)
        self._lr: float = kwargs.get("learning_rate", 0.001)

        self._train_loader: DataLoader = DataLoader(self._train_dataset, self._batch_size, shuffle=True)
        self._val_loader: DataLoader = DataLoader(self._val_dataset, self._batch_size, shuffle=False)

        self._class_weights = train_dataset.class_weights
        print(f"Class Weights: {self._class_weights}")

        self._criterion = kwargs.get("criterion", nn.CrossEntropyLoss)(weight=self._class_weights.to(self._device))
        self._optimizer = kwargs.get("optimizer", optim.Adam)(self._model.parameters(), lr=self._lr)

        self._training_logs: dict = dict()

        # Tuple structure is (Loss score, Accuracy score)
        self._train_scores: list[tuple[float, float]] = []
        self._val_scores: list[tuple[float, float]] = []


    def __train_one_epoch(self):
        self._model.train()
        running_loss = 0.0
        correct = 0

        for mel_spec, label in tqdm(self._train_loader, desc="Model training"):
            mel_spec, label = mel_spec.to(self._device), label.to(self._device)

            self._optimizer.zero_grad()
            output = self._model(mel_spec)
            loss = self._criterion(output, label)
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()

        total_samples = len(self._train_loader.dataset)

        return running_loss / len(self._train_loader), correct, total_samples

    def __validate(self):
        self._model.eval()
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for mel_spec, label in tqdm(self._val_loader, desc="Model validation"):
                mel_spec, label = mel_spec.to(self._device), label.to(self._device)
                output = self._model(mel_spec)
                loss = self._criterion(output, label)

                running_loss += loss.item()
                correct += (output.argmax(1) == label).sum().item()

        total_samples = len(self._val_loader.dataset)
        return running_loss / len(self._val_loader), correct, total_samples

    def train_model(self, epochs: int = 10, verbose: bool = False):
        self._model.to(self._device)

        print(f"Training model with device: {self._device}")
        for epoch in range(epochs):
            train_loss, train_correct, train_total = self.__train_one_epoch()
            val_loss, val_correct, val_total = self.__validate()

            train_accuracy = (train_correct / train_total) * 100.0
            val_accuracy = (val_correct / val_total) * 100.0

            epoch_string: str = f"Epoch {epoch + 1}"
            results_string: str = f'Train Loss: {train_loss:.4f},\n' \
                                  f'Train Accuracy: {train_accuracy:.2f}%\n' \
                                  f'[{train_correct}/{train_total}]\n' \
                                  f'Val Loss: {val_loss:.4f},\n' \
                                  f'Val Accuracy: {val_accuracy:.2f}%\n' \
                                  f'[{val_correct}/{val_total}]'

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

    def plot_losses(self, file_name: str | None = None):
        file_name = f"{self._model.__class__.__name__}_losses" if None else file_name
        train_losses = [scores[0] for scores in self._train_scores]
        val_losses = [scores[0] for scores in self._val_scores]
        plot_loss_per_epoch(file_save_name= file_name,
                            training_loss=train_losses,
                            validation_loss=val_losses)

    def plot_accuracies(self, file_name: str | None = None):
        file_name = f"{self._model.__class__.__name__}_accuracies" if None else file_name
        train_accuracies = [scores[1] for scores in self._train_scores]
        val_accuracies = [scores[1] for scores in self._val_scores]
        plot_accuracy_per_epoch(file_save_name=file_name,
                            training_accuracy=train_accuracies,
                            validation_accuracy=val_accuracies)


class EmotionClassifier0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * (TARGET_FRAMES // 2) * (FREQUENCY_BIN_COUNT // 2), 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EmotionClassifier1(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Second convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Third convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(128 * (TARGET_FRAMES // 2) * (FREQUENCY_BIN_COUNT // 2), 8)  # Update the input size based on pooling

    def forward(self, x):
        # Apply first convolution layer
        x = self.conv1(x)
        x = self.relu1(x)

        # Apply second convolution layer
        x = self.conv2(x)
        x = self.relu2(x)

        # Apply third convolution layer
        x = self.conv3(x)
        x = self.relu3(x)

        # Apply pooling layer
        x = self.pool(x)

        # Flatten the output before passing to the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

class EmotionClassifier2(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolution layer with BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Residual stack with 10 filters
        self.residual_stack = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(32 * (TARGET_FRAMES // 2) * (FREQUENCY_BIN_COUNT // 2), 8)

    def forward(self, x):
        # Apply first convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply residual stack
        residual = x  # Save input for residual connection
        x = self.residual_stack(x)
        x += residual  # Add residual connection
        x = torch.relu(x)  # Apply ReLU after residual addition

        # Apply pooling layer
        x = self.pool(x)

        # Flatten the output before passing to the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

class EmotionClassifier3(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolution layer with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Residual stack
        self.residual_stack = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        residual = x  # Save input for residual connection
        x = self.residual_stack(x)
        x += residual  # Add residual connection
        x = torch.relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ModelWithAttention(nn.Module):
    def __init__(self):
        super(ModelWithAttention, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # Adjusted in_channels to 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.residual_stack = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Attention mechanism
        self.attention = nn.Conv2d(64, 1, kernel_size=1)  # Reduce to 1 channel for attention weights

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        self.fc = nn.Linear(64, 8)  # Fully connected for 10 classes

    def forward(self, x):
        # Apply first convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply residual stack
        residual = x  # Save input for residual connection
        x = self.residual_stack(x)
        x += residual  # Add residual connection
        x = torch.relu(x)  # Apply ReLU after residual addition

        # Apply attention head
        attention_weights = torch.sigmoid(self.attention(x))  # Compute attention weights
        x = x * attention_weights  # Apply attention to the feature map

        # Apply pooling layer
        x = self.pool(x)

        # Flatten the output before passing to the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

