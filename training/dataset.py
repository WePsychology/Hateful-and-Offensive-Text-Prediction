import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizer import Vocab, encode, pad_or_truncate


class HateDataset(Dataset):
    def __init__(self, csv_path: str, vocab: Vocab, max_len: int = 40):
        df = pd.read_csv(csv_path, encoding="latin1")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(subset=["text", "label"])

        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab)
        ids = pad_or_truncate(ids, self.max_len, self.vocab.pad_idx)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"x": x, "y": y}
