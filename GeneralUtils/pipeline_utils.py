from pathlib import Path

import numpy as np
import torch.cuda

from audio import preprocess_wav
from inference import load_model, embed_utterance




def encoder_inference(audio_file_path: str | Path) -> np.ndarray:
    encoder_model_path = Path("Models/Encoder/saved_models/train1.pt")
    load_model(encoder_model_path, device = None)
    embedding = embed_utterance(preprocess_wav(audio_file_path, None)).reshape(1, -1)

    return embedding