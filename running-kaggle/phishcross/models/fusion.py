import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings, names=None):
        return self.classifier(torch.cat(embeddings, dim=1)), None


class AttentionFusion(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings, names=None):
        stack = torch.stack(embeddings, dim=1)
        scores = self.score(stack).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused), weights


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion: HTML sebagai Query, modalitas LAIN sebagai Key/Value.

    HTML dikeluarkan dari stack K/V sehingga mekanisme ini murni cross-modal
    (bukan self-attention terselubung).
    Ketika HTML tidak tersedia, fallback ke weighted mean pooling.
    """

    def __init__(self, emb_dim=128, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings, names=None):
        # --- Fallback: jika HTML tidak tersedia, gunakan mean pooling ---
        if names is None or "html" not in names:
            stack = torch.stack(embeddings, dim=1)
            fused = stack.mean(dim=1)
            fused_concat = torch.cat([fused, fused], dim=1)
            return self.classifier(fused_concat), None

        # --- Pure Cross-Attention: HTML → attend ke modalitas lain ---
        html_idx = names.index("html")
        query = embeddings[html_idx].unsqueeze(1)  # (B, 1, D)

        # Bangun K/V hanya dari modalitas SELAIN HTML
        other_embeddings = [emb for i, emb in enumerate(embeddings) if i != html_idx]
        other_names = [n for i, n in enumerate(names) if i != html_idx]

        if len(other_embeddings) == 0:
            # Hanya HTML yang tersedia — tidak ada modalitas lain untuk cross-attend
            query_squeeze = query.squeeze(1)
            fused_concat = torch.cat([query_squeeze, query_squeeze], dim=1)
            return self.classifier(fused_concat), None

        kv = torch.stack(other_embeddings, dim=1)  # (B, N-1, D)

        attn_output, attn_weights = self.cross_attn(query, kv, kv)
        attn_output = attn_output.squeeze(1)  # (B, D)
        if attn_weights is not None:
            # Rekonstruksi attention weights lengkap (termasuk HTML=0)
            # agar kompatibel dengan logging downstream
            full_weights = torch.zeros(
                attn_weights.shape[0], len(names), device=attn_weights.device
            )
            attn_weights = attn_weights.squeeze(1)  # (B, N-1)
            j = 0
            for i, name in enumerate(names):
                if i == html_idx:
                    full_weights[:, i] = 0.0  # HTML tidak attend ke dirinya sendiri
                else:
                    full_weights[:, i] = attn_weights[:, j]
                    j += 1
            attn_weights = full_weights

        query_squeeze = query.squeeze(1)
        fused = self.layer_norm(query_squeeze + attn_output)

        fused_concat = torch.cat([query_squeeze, fused], dim=1)

        return self.classifier(fused_concat), attn_weights
