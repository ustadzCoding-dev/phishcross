import torch
from torch import nn


class UrlCharCNN(nn.Module):
    def __init__(self, vocab_size, out_dim=128, emb_dim=48, channels=128, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, channels, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, channels, kernel_size=5, padding=2),
                nn.Conv1d(emb_dim, channels, kernel_size=7, padding=3),
            ]
        )
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, url_ids):
        x = self.embedding(url_ids).transpose(1, 2)
        pooled = [torch.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        return self.proj(torch.cat(pooled, dim=1))

