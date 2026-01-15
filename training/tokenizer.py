import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

PAD = "<PAD>"
UNK = "<UNK>"

def basic_tokenize(text: str) -> List[str]:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.findall(r"[a-z0-9']+", text)

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_idx: int
    unk_idx: int

def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = 20000) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    itos = [PAD, UNK]
    for w, f in counter.most_common():
        if f < min_freq:
            continue
        itos.append(w)
        if len(itos) >= max_size:
            break
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_idx=stoi[PAD], unk_idx=stoi[UNK])

def encode(text: str, vocab: Vocab) -> List[int]:
    toks = basic_tokenize(text)
    return [vocab.stoi.get(t, vocab.unk_idx) for t in toks]

def pad_or_truncate(ids: List[int], max_len: int, pad_idx: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))
