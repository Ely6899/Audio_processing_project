import os
import sys
from pathlib import Path
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Models.SentimentAnalysis.audio_dataset import RavdessRawData

index_emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

@pytest.mark.parametrize("file_path, expected_emotion", RavdessRawData().data)
def test_fetch_labels(file_path, expected_emotion):
    filename = Path(file_path).name  # Extract filename
    emotion_index = filename.split("-")[2]  # Get third number
    emotion = index_emotion_mapping.get(emotion_index, "unknown")

    assert emotion == expected_emotion



