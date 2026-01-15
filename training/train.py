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

def main():
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv   = os.path.join(DATA_DIR, "val.csv")
    test_csv  = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_csv, encoding="latin1")
    vocab = build_vocab(train_df["text"].astype(str).tolist(), min_freq=1, max_size=20000)

    max_len = 40
    batch_size = 16

    train_ds = HateDataset(train_csv, vocab, max_len=max_len)
    val_ds   = HateDataset(val_csv, vocab, max_len=max_len)
    test_ds  = HateDataset(test_csv, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TextCNN(vocab_size=len(vocab.itos), pad_idx=vocab.pad_idx).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = -1
    best_path = os.path.join(ART_DIR, "textcnn_best.pth")

    for epoch in range(1, 21):
        model.train()
        losses = []
        for b in train_loader:
            x = b["x"].to(DEVICE)
            y = b["y"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch} | loss={np.mean(losses):.4f} | val={val_metrics}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    # Test best
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader)
    print("TEST:", test_metrics)

    # Save vocab + config for backend
    with open(os.path.join(ART_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(ART_DIR, "config.json"), "w") as f:
        json.dump({"max_len": max_len, "threshold": 0.5}, f, indent=2)

    print("✅ Saved artifacts to:", ART_DIR)

if __name__ == "__main__":
    main()
