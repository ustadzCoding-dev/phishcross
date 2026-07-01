import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from phishcross.config import Config
from phishcross.data.preprocess import clean_html_text, read_text
from phishcross.utils import ensure_dirs, get_device, set_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="white")
        return self.transform(image)


class ImageDescriptorDataset(Dataset):
    def __init__(self, paths, image_size=24):
        self.paths = list(paths)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("L").resize((self.image_size, self.image_size))
        except Exception:
            image = Image.new("L", (self.image_size, self.image_size), color=255)
        arr = np.asarray(image, dtype="float32") / 255.0
        return arr.reshape(-1)


def extract_image_descriptors(df, split):
    split_df = df[df["split"] == split].reset_index(drop=True)
    dataset = ImageDescriptorDataset(split_df["image_path"], image_size=24)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    features = []
    for batch in tqdm(loader, desc=f"Image descriptors {split}"):
        features.append(batch.numpy())
    return np.concatenate(features).astype("float32")


def extract_html_features(df, split, cfg):
    split_df = df[df["split"] == split].reset_index(drop=True)
    texts = [
        clean_html_text(read_text(path), max_chars=cfg.html_max_chars)
        for path in tqdm(split_df["html_path"], desc=f"HTML text {split}")
    ]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.html_model_name, device=str(get_device()))
        emb = model.encode(
            texts,
            batch_size=cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        return emb
    except Exception as exc:
        print(f"WARNING: MiniLM extraction failed: {exc}")
        print("Falling back to HashingVectorizer HTML features.")
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            n_features=cfg.html_dim,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        return vectorizer.transform(texts).toarray().astype("float32")


def extract_image_features(df, split, cfg):
    if getattr(cfg, "offline_features", False):
        return extract_image_descriptors(df, split)

    from torchvision import models, transforms

    device = get_device()
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        mean = weights.transforms().mean
        std = weights.transforms().std
    except Exception as exc:
        print(f"WARNING: Pretrained MobileNetV3 weights failed: {exc}")
        print("Falling back to deterministic 24x24 grayscale visual descriptors.")
        return extract_image_descriptors(df, split)
    feature_extractor = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(device)
    feature_extractor.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    split_df = df[df["split"] == split].reset_index(drop=True)
    loader = DataLoader(
        ImagePathDataset(split_df["image_path"], transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    with torch.no_grad():
        for images in tqdm(loader, desc=f"Image features {split}"):
            images = images.to(device)
            features.append(feature_extractor(images).cpu().numpy())
    return np.concatenate(features).astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Config.manifest_path)
    parser.add_argument("--feature-dir", default=Config.feature_dir)
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--offline-features", action="store_true", help="Use no-download HTML/image features.")
    args = parser.parse_args()

    cfg = Config()
    cfg.manifest_path = args.manifest
    cfg.feature_dir = args.feature_dir
    cfg.data_root = args.data_root
    cfg.batch_size = args.batch_size
    cfg.offline_features = args.offline_features

    set_seed(cfg.seed)
    ensure_dirs(cfg.feature_dir)
    df = pd.read_csv(cfg.manifest_path)

    for split in ["trainval", "test"]:
        html_out = Path(cfg.feature_dir) / f"{split}_html.npy"
        image_out = Path(cfg.feature_dir) / f"{split}_image.npy"

        if not (args.skip_existing and html_out.exists()):
            np.save(html_out, extract_html_features(df, split, cfg))
            print(f"Saved {html_out}")

        if not (args.skip_existing and image_out.exists()):
            np.save(image_out, extract_image_features(df, split, cfg))
            print(f"Saved {image_out}")


if __name__ == "__main__":
    main()
