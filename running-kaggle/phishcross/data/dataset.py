import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phishcross.data.preprocess import UrlVocab, read_text


class PhishFeatureDataset(Dataset):
    def __init__(self, manifest_csv, feature_dir, split, max_url_len=256):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feature_dir = feature_dir
        self.max_url_len = max_url_len
        self.vocab = UrlVocab()

        self.html = np.load(f"{feature_dir}/{split}_html.npy", mmap_mode="r")
        self.image = np.load(f"{feature_dir}/{split}_image.npy", mmap_mode="r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = read_text(row["url_path"]).strip()
        url_ids = torch.tensor(self.vocab.encode(url, self.max_url_len), dtype=torch.long)
        html_feat = torch.tensor(self.html[idx], dtype=torch.float32)
        image_feat = torch.tensor(self.image[idx], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {
            "url_ids": url_ids,
            "html_feat": html_feat,
            "image_feat": image_feat,
            "label": label,
            "sample_id": row["sample_id"],
        }

