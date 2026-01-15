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



def censor_text(text: str, flagged_words: List[str]) -> str:
    if not flagged_words:
        return text
    pattern = r"\b(" + "|".join(map(re.escape, flagged_words)) + r")\b"
    return re.sub(pattern, "****", text, flags=re.IGNORECASE)

