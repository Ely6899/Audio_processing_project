import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

from PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT
from Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch
from audio_dataset import EmotionSpecDataset


class SentimentModelHandler:
    """
    Wrapper class for general model hyper-parameters.
    """
    def __init__(self, model: nn.Module, train_dataset: EmotionSpecDataset, val_dataset: EmotionSpecDataset, **kwargs):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: nn.Module = model
        self._train_dataset: Dataset = train_dataset
        self._val_dataset: Dataset = val_dataset

        self._batch_size: int = kwargs.get('batch_size', 16)
        self._lr: float = kwargs.get("learning_rate", 0.1)

        self._train_loader: DataLoader = DataLoader(self._train_dataset, self._batch_size, shuffle=True)
        self._val_loader: DataLoader = DataLoader(self._val_dataset, self._batch_size, shuffle=False)

        self._class_weights = train_dataset.class_weights
        print(f"Class Weights: {self._class_weights}")

        self._criterion = kwargs.get("criterion", nn.CrossEntropyLoss)(weight=self._class_weights.to(self._device))
        self._optimizer = kwargs.get("optimizer", optim.SGD)(self._model.parameters(), lr=self._lr, momentum=0.9, weight_decay=1e-6)
        self._scheduler = kwargs.get("scheduler", optim.lr_scheduler.MultiStepLR)(self._optimizer, milestones=[int(0.33 * 100), int(0.66 * 100)], gamma=0.1)

        self._training_logs: dict = dict()

        # Tuple structure is (Loss score, Accuracy score)
        self._train_scores: list[tuple[float, float]] = []
        self._val_scores: list[tuple[float, float]] = []

        # New lists to store the true labels and predicted labels for confusion matrix
        self._true_labels = []
        self._pred_labels = []


    def __train_one_epoch(self):
        """
        Trains a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
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

            running_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()

        self._scheduler.step()

        total_samples = len(self._train_loader.dataset)

        return running_loss / total_samples, correct, total_samples

    def __validate(self):
        """
        Validates a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
        self._model.eval()
        running_loss = 0.0
        correct = 0

        # Clear true and predicted labels at the start of validation
        self._true_labels.clear()
        self._pred_labels.clear()

        with torch.no_grad():
            for mel_spec, label in tqdm(self._val_loader, desc="Model validation"):
                mel_spec, label = mel_spec.to(self._device), label.to(self._device)
                output = self._model(mel_spec)
                loss = self._criterion(output, label)

                running_loss += loss.item()
                correct += (output.argmax(1) == label).sum().item()

        total_samples = len(self._val_loader.dataset)
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
        self.fc = nn.Linear(32 * TARGET_FRAMES * FREQUENCY_BIN_COUNT, 8)

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
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Residual stack
        self.residual_stack = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = nn.Linear(64, 8)

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

class RavdessPaperModel(nn.Module):
    def __init__(self):
        super(RavdessPaperModel, self).__init__()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding='same')
        self.dropout = nn.Dropout(0.2)
        self.fc = None  # Will define the fully connected layer dynamically

    def forward(self, x):
        # x.shape is (batch_size, num_channels, sequence_length)
        x = self.conv2(x)  # After Conv1D: (batch_size, out_channels, sequence_length)
        x = F.relu(x)
        x = self.dropout(x)

        # Flatten the output for the fully connected layer (batch_size, -1)
        x = x.view(x.size(0), -1)

        # Define the fully connected layer dynamically based on output shape after conv1
        if self.fc is None:
            # Calculate the shape dynamically
            conv_out_shape = x.shape[1]
            self.fc = nn.Linear(conv_out_shape, 8).to("cuda")  # Define the fc layer for dynamic shape

        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)


#NOTE: Too slow for unknown reason.
class ResidualModel(nn.Module):
    def __init__(self):
        super(ResidualModel, self).__init__()
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(32)

        self.residual_stack1 = ResidualBlock(32, 64)
        self.residual_stack2 = ResidualBlock(64, 64)
        self.residual_stack3 = ResidualBlock(64, 64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * TARGET_FRAMES * FREQUENCY_BIN_COUNT, 128)  # Assuming 8x8 feature maps after Conv layers
        self.fc2 = nn.Linear(128, 64)

        self.output_layer = nn.Linear(64, 8)

    def forward(self, x):
        x = torch.relu(self.initial_bn(self.initial_conv(x)))
        x = self.residual_stack1(x)
        x = self.residual_stack2(x)
        x = self.residual_stack3(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output_layer(x)



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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
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

        # Pass through the modules
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






