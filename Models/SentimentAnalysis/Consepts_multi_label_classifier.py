"""
    define: 
        dataset; train, evaluate and predict methods
    
    execute:
        load model and dataset, train the model, evaluate it and save the best model.

    ALL FOR MULTI-LABEL CONCEPT CLASSIFICATION OF SPECTROGRAMS
"""


# =========================
# Multi-label spectrogram tagger (ResNet-18)
# =========================

# Imports
import os
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
from collections import defaultdict

from Preprocess import audio_to_mel_spectrogram
from preproccess_for_concept_classification import prepare_spectrogram_for_backbone
from tcav_demo import CONCEPT_UNIQUE_NAMES

# ---- your function is assumed to be defined/imported beforehand ----
# from your_module import prepare_spectrogram_for_backbone

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 40
NUM_WORKERS = 0 # no multiprocessing
FREEZE_TO_LAST_BLOCK = True         # freeze all except layer4 + fc (few-shot friendly)
CALIBRATE_THRESHOLDS = True         # find per-label thresholds on val set
CKPT_PATH = "best_resnet18_multilabel.pt"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- Your label space ----------
# List every possible tag (order defines the output indices)
ALL_TAGS: List[str] = CONCEPT_UNIQUE_NAMES
TAG2IDX = {t: i for i, t in enumerate(ALL_TAGS)}
NUM_TAGS = len(ALL_TAGS)

# ---------- Your data ----------
# Provide lists of samples: (image_path, [list_of_present_tags])
# Example:
# train_items = [
#   ("path/to/img1.png", ["long-thick-constant"]),
#   ("path/to/img2.png", ["long-steep-thick-rising","long-thick-constant"]),
# ]
train_items: List[Tuple[str, List[str]]] = [(r"RAVDESS\original_data\Actor_06\03-01-01-01-02-01-06.wav", ["short_constant_thick", "long_constant_thick", "short_dropping_steep_thick"])]  #TODO <-- FILL WITH REAL VALUES!
val_items:   List[Tuple[str, List[str]]] = [(r'RAVDESS\original_data\Actor_08\03-01-03-01-01-01-08.wav', ["short_dropping_steep_thick", "short_constant_thick"])]  #TODO <-- FILL WITH REAL VALUES!



# ---------- Dataset ----------
class MultiLabelSpectrogramDataset(Dataset):
    """
    Multi-label dataset for spectrogram images.

    Each item is a tuple (image_path, tags), where:
      - image_path: str path to a spectrogram image (PNG/JPG).
      - tags: List[str] of concept names present in the sample.

    Produces for each index:
      - model_img: torch.FloatTensor (3, 224, 224), normalized and resized for the backbone
                   using prepare_spectrogram_for_backbone.
      - y: torch.FloatTensor (NUM_TAGS,), a multi-hot vector corresponding to ALL_TAGS.
      - path: str, the original image path (useful for debugging).
    """

    def __init__(self, items: List[Tuple[str, List[str]]], letterbox: bool = False):
        """
        Initialize the dataset.

        Parameters:
          - items: List of (path, tags) pairs. Tags must be in ALL_TAGS to be set in the label vector.
          - letterbox: Placeholder flag for future letterbox handling (not used in current pipeline).
        """
        self.items = items
        self.letterbox = letterbox

    def __len__(self):
        """
        Returns:
          int: Number of samples in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        Build a multi-hot label vector and return the backbone-ready image.

        Returns:
          Tuple[torch.FloatTensor, torch.FloatTensor, str]:
            - model_img: (3, 224, 224) float tensor, normalized as required by the backbone.
            - y: (NUM_TAGS,) float multi-hot vector aligned with ALL_TAGS.
            - path: Original image path.

        Notes:
          - Uses prepare_spectrogram_for_backbone with out_size=(224, 224), to_rgb=True, normalize=True.
        """
        path, tags = self.items[idx]

        path = Path(path)
        # build multi-hot vector
        y = torch.zeros(NUM_TAGS, dtype=torch.float32)
        for t in tags:
            if t in TAG2IDX:
                y[TAG2IDX[t]] = 1.0

        # load spectrogram image:
        img = audio_to_mel_spectrogram(path)
        img = torch.from_numpy(img)
        # your function handles resizing + normalization for ResNet
        # we only need the model_img return
        _, model_img = prepare_spectrogram_for_backbone(
            img,
            out_size=(224, 224),
            letterbox=False,
            to_rgb=True,
            normalize=True,
            return_both=True,
        )
        return model_img, y, str(path)  # keep path for debugging, sending string as path will make dataloader create an error.


# ---------- DataLoaders ----------
train_ds = MultiLabelSpectrogramDataset(train_items, letterbox=False)
val_ds   = MultiLabelSpectrogramDataset(val_items,   letterbox=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ---------- Model ----------
base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_dim = base.fc.in_features
base.fc = nn.Linear(in_dim, NUM_TAGS)
model = base.to(DEVICE)

if FREEZE_TO_LAST_BLOCK:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


#? ---------- Metrics ----------
def f1_micro_macro(y_true: torch.Tensor, y_prob: torch.Tensor, thresholds: Optional[torch.Tensor] = None):
    """
    y_true: (N, C) float {0,1}
    y_prob: (N, C) sigmoid probabilities
    thresholds: (C,) per-label thresholds in [0,1] or None->0.5
    """
    eps = 1e-9
    if thresholds is None:
        thresholds = torch.full((y_true.shape[1],), 0.5, device=y_prob.device)
    y_pred = (y_prob >= thresholds.view(1, -1)).float()

    # per-class stats
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)

    precision_c = tp / (tp + fp + eps)
    recall_c    = tp / (tp + fn + eps)
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

    # macro
    f1_macro = f1_c.mean().item()

    # micro (sum over classes first)
    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    precision_micro = TP / (TP + FP + eps)
    recall_micro    = TP / (TP + FN + eps)
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)).item()

    return f1_micro, f1_macro, f1_c.detach().cpu().numpy()

def calibrate_thresholds(y_true: torch.Tensor, y_logit: torch.Tensor) -> torch.Tensor:
    """
    For each class, sweep thresholds and pick the one that maximizes F1 on the validation set.
    Returns a (C,) tensor in [0,1].
    """
    y_prob = torch.sigmoid(y_logit)
    C = y_true.shape[1]
    thresholds = torch.zeros(C, device=y_true.device)
    grid = torch.linspace(0.05, 0.95, steps=19, device=y_true.device)
    eps = 1e-9
    for c in range(C):
        best_t, best_f1 = 0.5, -1.0
        p = y_prob[:, c]
        t = y_true[:, c]
        for th in grid:
            pred = (p >= th).float()
            tp = (pred * t).sum()
            fp = (pred * (1 - t)).sum()
            fn = ((1 - pred) * t).sum()
            prec = tp / (tp + fp + eps)
            rec  = tp / (tp + fn + eps)
            f1 = (2 * prec * rec / (prec + rec + eps)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, th.item()
        thresholds[c] = best_t
    return thresholds

# ---------- Train / Validate ----------
def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train for a single epoch over a multi-label spectrogram dataset.

    Args:
        model (torch.nn.Module): Model that maps a batch of images to logits of shape (N, C).
        loader (torch.utils.data.DataLoader): Yields tuples (imgs, targets, path) where:
            - imgs: Float tensor (N, 3, 224, 224) to be moved to DEVICE.
            - targets: Float multi-hot tensor (N, C) aligned with ALL_TAGS.
            - path: Original sample path (unused here).
        optimizer (torch.optim.Optimizer): Optimizer for updating trainable parameters.
        criterion (Callable): Loss function taking (logits, targets) -> scalar loss
            (e.g., nn.BCEWithLogitsLoss for multi-label classification).

    Returns:
        float: Average loss over the entire dataset (sum of batch losses normalized by dataset size).

    Notes:
        - Sets model.train().
        - Respects requires_grad filtering (useful when freezing layers).
        - Uses optimizer.zero_grad(set_to_none=True) for performance.
    """
    model.train()
    running_loss = 0.0
    for imgs, targets, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    Evaluate the model on a validation/test split without gradient tracking.

    Args:
        model (torch.nn.Module): Model that produces logits of shape (N, C).
        loader (torch.utils.data.DataLoader): Yields (imgs, targets, path) batches.
        criterion (Callable): Loss function for reporting average loss.

    Returns:
        Tuple[
            float,               # avg_loss: Mean loss over the dataset
            float,               # f1_micro: Micro-averaged F1 at threshold 0.5
            float,               # f1_macro: Macro-averaged F1 at threshold 0.5
            torch.Tensor,        # logits: Concatenated raw logits of shape (N, C)
            torch.Tensor         # targets: Concatenated multi-hot targets of shape (N, C)
        ]

    Notes:
        - Sets model.eval() and runs under torch.no_grad().
        - F1 metrics are computed with f1_micro_macro using default per-class threshold 0.5
          (thresholds=None). For calibrated thresholds, pass them explicitly or adapt this call.
        - Moves batches to DEVICE with non_blocking=True.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)

        all_logits.append(logits)
        all_targets.append(targets)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits)
    f1_micro, f1_macro, f1_per_class = f1_micro_macro(targets, probs, thresholds=None)
    return total_loss / len(loader.dataset), f1_micro, f1_macro, logits, targets


# ---------- Fit ----------
best_val_score = -1.0
best_thresholds = torch.full((NUM_TAGS,), 0.5)

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, f1_micro, f1_macro, val_logits, val_targets = evaluate(model, val_loader, criterion)

    # optional per-label threshold calibration on validation set
    if CALIBRATE_THRESHOLDS:
        best_thresholds = calibrate_thresholds(val_targets, val_logits)
        val_probs = torch.sigmoid(val_logits)
        f1_micro_cal, f1_macro_cal, _ = f1_micro_macro(val_targets, val_probs, thresholds=best_thresholds)
        display_f1 = f1_micro_cal
        display_macro = f1_macro_cal
    else:
        display_f1 = f1_micro
        display_macro = f1_macro

    print(f"[{epoch:02d}/{EPOCHS}] "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"F1_micro={display_f1:.4f}  F1_macro={display_macro:.4f}")

    # scheduler on micro-F1
    scheduler.step(display_f1)

    if display_f1 > best_val_score:
        best_val_score = display_f1
        torch.save({
            "model_state": model.state_dict(),
            "thresholds": best_thresholds.cpu(),
            "all_tags": ALL_TAGS,
        }, CKPT_PATH)
        print(f"  ↪ saved best checkpoint to {CKPT_PATH} (F1_micro={best_val_score:.4f})")

print("Training done.")


# ---------- Inference helper ----------
@torch.no_grad()
def predict_paths(paths: List[str], ckpt_path: str = CKPT_PATH, letterbox: bool=False):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    thresholds = ckpt.get("thresholds", torch.full((NUM_TAGS,), 0.5)).to(DEVICE)

    model.eval()
    preds = []
    for p in paths:
        # load + prepare
        img = torch.from_numpy(audio_to_mel_spectrogram(Path(p)))
        _, x_m = prepare_spectrogram_for_backbone(img, out_size=(224,224), letterbox=letterbox,
                                                  to_rgb=True, normalize=True, return_both=True)
        x_m = x_m.unsqueeze(0).to(DEVICE)

        logit = model(x_m)
        prob = torch.sigmoid(logit).squeeze(0)          # (C,)
        pred = (prob >= thresholds).nonzero(as_tuple=False).view(-1).tolist()
        tags = [ALL_TAGS[i] for i in pred]
        preds.append({"path": p, "probs": prob.cpu().numpy(), "tags": tags})
    return preds


if __name__ == "__main__":
  
  #? assert dataset works

  your_items_list = [("RAVDESS/original_data/Actor_06/03-01-01-01-01-01-06.wav", ["long_constant_thick"])]

  dataset = MultiLabelSpectrogramDataset(items=your_items_list)

  assert dataset[0][0].shape == (3, 224, 224)

  #? assert train step and an eval works
  print("Sanity check: running one train step and one eval pass...")
  try:
      train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
      print(f"[sanity] train_loss={train_loss:.4f}")

      val_loss, f1_micro, f1_macro, _, _ = evaluate(model, val_loader, criterion)
      print(f"[sanity] val_loss={val_loss:.4f} | f1_micro={f1_micro:.3f} | f1_macro={f1_macro:.3f}")

      # Optional: step LR scheduler with the validation metric
      try:
          scheduler.step(f1_micro)
      except Exception:
          pass
  except Exception as e:
      print(f"[sanity] Skipped (dataset/files may be missing): {e}")  
      
  
  #? assert predict_paths works
  preds = predict_paths(["RAVDESS/original_data/Actor_06/03-01-01-01-01-01-06.wav", "RAVDESS/original_data/Actor_08/03-01-03-01-01-01-08.wav"])
  print("hi")
