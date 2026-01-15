import re
from dataclasses import dataclass
from typing import Dict, List

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
