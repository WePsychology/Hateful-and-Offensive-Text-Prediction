import json
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ART_DIR = "backend/artifacts"

# -------- Tokenizer (same logic as training, but self-contained) --------
PAD = "<PAD>"
UNK = "<UNK>"

def basic_tokenize(text: str) -> List[str]:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.findall(r"[a-z0-9']+", text)

def pad_or_truncate(ids: List[int], max_len: int, pad_idx: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_idx: int
    unk_idx: int

# -------- Model (same architecture as training) --------
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, embed_dim=64, num_filters=64, kernel_sizes=(3,4,5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x)          # (B, L, E)
        emb = emb.transpose(1, 2)        # (B, E, L)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))        # (B, F, L')
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # (B, F)
            pooled.append(h)

        feats = torch.cat(pooled, dim=1)
        feats = self.dropout(feats)
        logits = self.fc(feats).squeeze(1)  # (B,)
        return logits

def _load_vocab():
    with open(f"{ART_DIR}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab

def _load_config():
    with open(f"{ART_DIR}/config.json", "r", encoding="utf-8") as f:
        return json.load(f)
        
class HateSpeechPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = _load_vocab()
        self.cfg = _load_config()

        # Must match train.py defaults
        self.model = TextCNN(
            vocab_size=len(self.vocab.itos),
            pad_idx=self.vocab.pad_idx,
            embed_dim=64,
            num_filters=64,
            kernel_sizes=(3, 4, 5),
            dropout=0.3
        ).to(self.device)

        state = torch.load(f"{ART_DIR}/textcnn_best.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.max_len = int(self.cfg.get("max_len", 40))
        self.threshold = float(self.cfg.get("threshold", 0.5))

    def _encode(self, text: str) -> List[int]:
        toks = basic_tokenize(text)
        ids = [self.vocab.stoi.get(t, self.vocab.unk_idx) for t in toks]
        return pad_or_truncate(ids, self.max_len, self.vocab.pad_idx)

    def predict(self, text: str) -> Dict[str, Any]:
        ids = self._encode(text)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()

        label = 1 if prob >= self.threshold else 0
        confidence = prob if label == 1 else (1 - prob)

        flagged_words = self._simple_flag_words(text, label)

        return {
            "label": int(label),
            "confidence": float(round(confidence, 4)),
            "flagged_words": flagged_words
        }

    def _simple_flag_words(self, text: str, label: int) -> List[str]:
        if label == 0:
            return []
        tokens = basic_tokenize(text)
        bad_seed = {"hate","stupid","idiot","trash","moron","whore","bitch","slut","loser","worthless","disgusting","shut","dumb","kill","hurt","annoying","pathetic","failure","useless","incompetent","garbage","ridiculous","nasty","horrible","toxic","despise","worst"}
        flagged = []
        for t in tokens:
            if t in bad_seed and t not in flagged:
                flagged.append(t)
        return flagged


def censor_text(text: str, flagged_words: List[str]) -> str:
    if not flagged_words:
        return text
    pattern = r"\b(" + "|".join(map(re.escape, flagged_words)) + r")\b"
    return re.sub(pattern, "****", text, flags=re.IGNORECASE)


