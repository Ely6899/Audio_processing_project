import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from Models.SentimentAnalysis.Preprocess import audio_to_mel_spectogram


class EmotionDataset(Dataset):
    def __init__(self, csv_file: Path):
        self.data = pd.read_csv(csv_file)
        self._file_paths = self.data['file']
        self.labels = self.data['label']

        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(self.label_encoder.fit_transform(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self._file_paths[idx]
        label = self.labels[idx]

        mel_spectrogram = audio_to_mel_spectogram(file_path=file_path)
        label = label.long()

        return mel_spectrogram, label

    def decode_label(self, encoded_label):
        return self.label_encoder.inverse_transform([encoded_label])[0]