import os, json, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tokenizer import build_vocab
from dataset import HateDataset
from model import TextCNN

DATA_DIR = "data/processed"
ART_DIR = "backend/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    probs_all = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            y = b["y"].cpu().numpy().astype(int)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs >= 0.5).astype(int)

            ys.extend(y.tolist())
            preds.extend(pred.tolist())
            probs_all.extend(probs.tolist())

    acc = accuracy_score(ys, preds)
    p, r, f1, _ = precision_recall_fscore_support(ys, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
