import torch
from torch import nn

from phishcross.data.preprocess import UrlVocab
from phishcross.models.fusion import AttentionFusion, ConcatFusion, CrossAttentionFusion
from phishcross.models.url_encoder import UrlCharCNN


class PhishCrossNet(nn.Module):
    def __init__(
        self,
        mode="attention",
        use_url=True,
        use_html=True,
        use_image=True,
        html_dim=384,
        image_dim=576,
        proj_dim=128,
        dropout=0.25,
        modality_dropout_prob=0.15,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.use_url = use_url
        self.use_html = use_html
        self.use_image = use_image
        self.mode = mode

        # ---- Instansiasi submodule secara kondisional ----
        # Hanya membuat encoder/projector untuk modalitas yang aktif,
        # sehingga count_parameters() melaporkan jumlah yang akurat.
        if use_url:
            vocab = UrlVocab()
            self.url_encoder = UrlCharCNN(vocab.size, out_dim=proj_dim, dropout=dropout)
        if use_html:
            self.html_proj = nn.Sequential(nn.Linear(html_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        if use_image:
            self.image_proj = nn.Sequential(nn.Linear(image_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))

        n_modalities = sum([use_url, use_html, use_image])
        if mode == "concat":
            self.fusion = ConcatFusion(proj_dim * n_modalities, dropout=dropout)
        elif mode == "attention":
            self.fusion = AttentionFusion(proj_dim, dropout=dropout)
        elif mode == "cross":
            self.fusion = CrossAttentionFusion(proj_dim, dropout=dropout)
        elif mode == "single":
            self.fusion = ConcatFusion(proj_dim * n_modalities, dropout=dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, batch):
        embeddings = []
        names = []
        if self.use_url:
            embeddings.append(self.url_encoder(batch["url_ids"]))
            names.append("url")
        if self.use_html:
            embeddings.append(self.html_proj(batch["html_feat"]))
            names.append("html")
        if self.use_image:
            embeddings.append(self.image_proj(batch["image_feat"]))
            names.append("image")

        # Modality dropout: zero-out random modalities during training
        # untuk meningkatkan robustness terhadap missing modalities
        if self.training and self.modality_dropout_prob > 0.0:
            for i in range(len(embeddings)):
                if torch.rand(1).item() < self.modality_dropout_prob:
                    embeddings[i] = torch.zeros_like(embeddings[i])

        logits, weights = self.fusion(embeddings, names)
        return logits, weights, names
