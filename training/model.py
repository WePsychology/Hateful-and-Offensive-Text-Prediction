import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, embed_dim=64, num_filters=64, kernel_sizes=(3,4,5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
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
    
