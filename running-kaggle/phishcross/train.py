import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from phishcross.config import Config
from phishcross.data.dataset import PhishFeatureDataset
from phishcross.models.phishcross_net import PhishCrossNet
from phishcross.utils import count_parameters, ensure_dirs, get_device, save_json, set_seed


def build_model(experiment, cfg):
    mapping = {
        "url_only": dict(mode="single", use_url=True, use_html=False, use_image=False),
        "html_only": dict(mode="single", use_url=False, use_html=True, use_image=False),
        "image_only": dict(mode="single", use_url=False, use_html=False, use_image=True),
        "concat": dict(mode="concat", use_url=True, use_html=True, use_image=True),
        "attention": dict(mode="attention", use_url=True, use_html=True, use_image=True),
        "attention_no_url": dict(mode="attention", use_url=False, use_html=True, use_image=True),
        "attention_no_html": dict(mode="attention", use_url=True, use_html=False, use_image=True),
        "attention_no_image": dict(mode="attention", use_url=True, use_html=True, use_image=False),
        "attention_no_moddrop": dict(mode="attention", use_url=True, use_html=True, use_image=True),
        "phishcross": dict(mode="cross", use_url=True, use_html=True, use_image=True),
        "phishcross_no_url": dict(mode="cross", use_url=False, use_html=True, use_image=True),
        "phishcross_no_html": dict(mode="cross", use_url=True, use_html=False, use_image=True),
        "phishcross_no_image": dict(mode="cross", use_url=True, use_html=True, use_image=False),
        "phishcross_no_moddrop": dict(mode="cross", use_url=True, use_html=True, use_image=True),
    }
    if experiment not in mapping:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    modality_dropout_prob = 0.0 if "no_moddrop" in experiment else cfg.modality_dropout_prob
    
    return PhishCrossNet(
        **mapping[experiment],
        html_dim=cfg.html_dim,
        image_dim=cfg.image_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
        modality_dropout_prob=modality_dropout_prob,
    )


def make_loader(dataset, batch_size, shuffle=False, weighted=False, num_workers=2):
    """Buat DataLoader dengan opsi WeightedRandomSampler untuk menangani class imbalance."""
    sampler = None
    if weighted:
        # Ambil label: jika dataset adalah Subset, akses underlying df via indices
        if isinstance(dataset, Subset):
            labels = dataset.dataset.df.iloc[dataset.indices]["label"].to_numpy()
        else:
            labels = dataset.df["label"].to_numpy()
        class_counts = np.bincount(labels, minlength=2)
        weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch, device):
    return {
        "url_ids": batch["url_ids"].to(device),
        "html_feat": batch["html_feat"].to(device),
        "image_feat": batch["image_feat"].to(device),
        "label": batch["label"].to(device),
    }


def evaluate(model, loader, device, measure_time=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    attn_sum = None
    attn_count = 0
    elapsed = 0.0
    seen = 0

    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            if measure_time:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
            logits, weights, names = model(batch_dev)
            if measure_time:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                seen += batch_dev["label"].shape[0]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            y_true.extend(batch_dev["label"].cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            if weights is not None:
                batch_sum = weights.sum(dim=0).detach().cpu().numpy()
                attn_sum = batch_sum if attn_sum is None else attn_sum + batch_sum
                attn_count += weights.shape[0]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None
    if measure_time and seen:
        metrics["inference_time_per_sample_sec"] = elapsed / seen
    if attn_sum is not None and attn_count:
        metrics["attention_weights"] = {name: float(value) for name, value in zip(names, attn_sum / attn_count)}
    return metrics


def train_one(experiment, cfg):
    device = get_device()

    # ----------------------------------------------------------------
    # PERBAIKAN S1: Pemisahan Validation Set yang benar
    # ----------------------------------------------------------------
    # Muat seluruh dataset trainval, lalu bagi menjadi train (85%) + val (15%)
    # secara stratified berdasarkan label untuk menjaga rasio kelas.
    full_trainval_ds = PhishFeatureDataset(cfg.manifest_path, cfg.feature_dir, "trainval", cfg.max_url_len)
    test_ds = PhishFeatureDataset(cfg.manifest_path, cfg.feature_dir, "test", cfg.max_url_len)

    trainval_labels = full_trainval_ds.df["label"].to_numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(full_trainval_ds)),
        test_size=0.15,
        random_state=cfg.seed,
        stratify=trainval_labels,
    )

    train_subset = Subset(full_trainval_ds, train_idx)
    val_subset = Subset(full_trainval_ds, val_idx)

    train_loader = make_loader(train_subset, cfg.batch_size, weighted=True, num_workers=cfg.num_workers)
    val_loader = make_loader(val_subset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"[DATA SPLIT] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_ds)}")

    model = build_model(experiment, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # PERBAIKAN S2: CosineAnnealingLR Scheduler
    # ----------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_path = Path(cfg.checkpoint_dir) / f"{experiment}.pt"
    history = []
    patience = 7
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{experiment} epoch {epoch}/{cfg.epochs}"):
            batch_dev = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(batch_dev)
            loss = criterion(logits, batch_dev["label"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Step scheduler setelah setiap epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------------
        # Evaluasi pada VALIDATION SET (bukan test set!)
        # ----------------------------------------------------------
        val_metrics = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = float(np.mean(losses))
        val_metrics["learning_rate"] = current_lr
        history.append(val_metrics)

        print(
            f"{experiment} epoch={epoch} lr={current_lr:.2e} "
            f"loss={val_metrics['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        # Model selection berdasarkan validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            print(f"  -> Checkpoint tersimpan (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        # ----------------------------------------------------------
        # PERBAIKAN S2: Early Stopping
        # ----------------------------------------------------------
        if patience_counter >= patience:
            print(f"  -> Early stopping! Val F1 tidak membaik selama {patience} epoch.")
            break

    # ------------------------------------------------------------------
    # Evaluasi FINAL pada TEST SET (murni terisolasi, hanya sekali)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Evaluasi final pada TEST SET (terisolasi) untuk: {experiment}")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_metrics = evaluate(model, test_loader, device, measure_time=True)
    final_metrics["experiment"] = experiment
    final_metrics["trainable_parameters"] = count_parameters(model)
    final_metrics["checkpoint"] = str(best_path)
    final_metrics["best_val_f1"] = best_val_f1
    final_metrics["stopped_epoch"] = len(history)
    final_metrics["history"] = history
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Config.manifest_path)
    parser.add_argument("--feature-dir", default=Config.feature_dir)
    parser.add_argument("--result-dir", default=Config.result_dir)
    parser.add_argument("--checkpoint-dir", default=Config.checkpoint_dir)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--modality-dropout-prob", type=float, default=Config.modality_dropout_prob)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    cfg = Config()
    cfg.manifest_path = args.manifest
    cfg.feature_dir = args.feature_dir
    cfg.result_dir = args.result_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers
    cfg.dropout = args.dropout
    cfg.modality_dropout_prob = args.modality_dropout_prob
    cfg.weight_decay = args.weight_decay
    cfg.seed = args.seed

    set_seed(cfg.seed)
    ensure_dirs(cfg.result_dir, cfg.checkpoint_dir)
    metrics = train_one(args.experiment, cfg)
    out_path = os.path.join(cfg.result_dir, f"{args.experiment}.json")
    save_json(metrics, out_path)
    print(metrics)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
