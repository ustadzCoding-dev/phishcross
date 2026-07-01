import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def chi_square_continuity(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 0.0
    return ((abs(b - c) - 1) ** 2) / n


def refresh_manifest_paths(old_manifest: Path, data_root: Path, out_manifest: Path, build_manifest):
    old_df = pd.read_csv(old_manifest)
    current_df = build_manifest(str(data_root), str(out_manifest.with_suffix(".current.csv")))

    key_cols = ["split", "sample_id"]
    path_cols = ["url_path", "html_path", "image_path"]
    merged = old_df.drop(columns=path_cols).merge(
        current_df[key_cols + path_cols],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if merged[path_cols].isna().any().any():
        missing = merged[merged[path_cols].isna().any(axis=1)][key_cols].head(10)
        raise RuntimeError(f"Some samples from the old manifest were not found in data_root:\n{missing}")

    old_labels = old_df[key_cols + ["label"]].sort_values(key_cols).reset_index(drop=True)
    new_labels = current_df[key_cols + ["label"]].sort_values(key_cols).reset_index(drop=True)
    if not old_labels.equals(new_labels):
        raise RuntimeError("Label mismatch between old manifest and current Phish360 manifest.")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_manifest, index=False)
    return out_manifest


def resolve_phish360_root(data_root: Path) -> Path:
    candidates = [
        data_root,
        data_root / "Phish360",
        data_root / "phish360",
    ]
    for candidate in candidates:
        if (candidate / "trainval").is_dir() and (candidate / "test").is_dir():
            return candidate

    nested = []
    if data_root.exists():
        for child in data_root.rglob("*"):
            if child.is_dir() and (child / "trainval").is_dir() and (child / "test").is_dir():
                nested.append(child)
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        options = "\n".join(str(p) for p in nested[:10])
        raise RuntimeError(f"Multiple possible Phish360 roots found. Please pass one explicitly:\n{options}")

    raise FileNotFoundError(
        "Could not find Phish360 root. The --data-root folder must contain trainval/ and test/, "
        "or contain a Phish360/ subfolder with trainval/ and test/."
    )


def resolve_artifact_root(artifact_root: Path) -> Path:
    candidates = [
        artifact_root,
        artifact_root / "phishcross_runs",
        artifact_root / "phishcross-runs",
    ]
    for candidate in candidates:
        if (candidate / "manifest.csv").is_file() and (candidate / "features").is_dir() and (candidate / "checkpoints").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find artifact root. The --artifact-root folder must contain manifest.csv, features/, and checkpoints/."
    )


def move_batch(batch, device):
    return {
        "url_ids": batch["url_ids"].to(device),
        "html_feat": batch["html_feat"].to(device),
        "image_feat": batch["image_feat"].to(device),
        "label": batch["label"].to(device),
    }


def predict(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            logits, _, _ = model(batch_dev)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            label = batch["label"].numpy()
            for sample_id, y_true, y_pred, y_prob in zip(batch["sample_id"], label, pred, prob):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": int(y_true),
                        "y_pred": int(y_pred),
                        "y_prob": float(y_prob),
                    }
                )
    return pd.DataFrame(rows)


def load_model(experiment, checkpoint_path, cfg, build_model, device):
    model = build_model(experiment, cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run(args):
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))

    from phishcross.config import Config
    from phishcross.data.dataset import PhishFeatureDataset
    from phishcross.data.manifest import build_manifest
    from phishcross.train import build_model

    artifact_root = resolve_artifact_root(Path(args.artifact_root).resolve())
    data_root = resolve_phish360_root(Path(args.data_root).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using artifact_root: {artifact_root}")
    print(f"Using data_root: {data_root}")

    old_manifest = artifact_root / "manifest.csv"
    feature_dir = artifact_root / "features"
    checkpoint_root = artifact_root / "checkpoints"
    eval_manifest = refresh_manifest_paths(
        old_manifest=old_manifest,
        data_root=data_root,
        out_manifest=out_dir / "manifest_eval_paths.csv",
        build_manifest=build_manifest,
    )

    cfg = Config()
    cfg.manifest_path = str(eval_manifest)
    cfg.feature_dir = str(feature_dir)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_ds = PhishFeatureDataset(str(eval_manifest), str(feature_dir), "test", cfg.max_url_len)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    result_rows = []

    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        seed_dir = checkpoint_root / f"seed_{seed}"
        target_ckpt = seed_dir / f"{args.target}.pt"
        target_model = load_model(args.target, target_ckpt, cfg, build_model, device)
        target_pred = predict(target_model, loader, device).rename(
            columns={"y_pred": f"y_pred_{args.target}", "y_prob": f"y_prob_{args.target}"}
        )

        seed_pred = target_pred
        for baseline in baselines:
            baseline_ckpt = seed_dir / f"{baseline}.pt"
            baseline_model = load_model(baseline, baseline_ckpt, cfg, build_model, device)
            baseline_pred = predict(baseline_model, loader, device).rename(
                columns={"y_pred": f"y_pred_{baseline}", "y_prob": f"y_prob_{baseline}"}
            )
            seed_pred = seed_pred.merge(
                baseline_pred[["sample_id", f"y_pred_{baseline}", f"y_prob_{baseline}"]],
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )

            target_correct = seed_pred[f"y_pred_{args.target}"] == seed_pred["y_true"]
            baseline_correct = seed_pred[f"y_pred_{baseline}"] == seed_pred["y_true"]
            b = int((target_correct & ~baseline_correct).sum())
            c = int((~target_correct & baseline_correct).sum())
            result_rows.append(
                {
                    "seed": seed,
                    "comparison": f"{args.target}_vs_{baseline}",
                    "b_target_correct_baseline_wrong": b,
                    "c_target_wrong_baseline_correct": c,
                    "mcnemar_chi2_continuity": chi_square_continuity(b, c),
                    "mcnemar_exact_p": exact_mcnemar_pvalue(b, c),
                    "target_accuracy": float(target_correct.mean()),
                    "baseline_accuracy": float(baseline_correct.mean()),
                    "n_test": len(seed_pred),
                }
            )

        pred_path = out_dir / f"predictions_seed_{seed}.csv"
        seed_pred.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

    results = pd.DataFrame(result_rows)
    results_path = out_dir / "mcnemar_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved McNemar results: {results_path}")
    print(results.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", default="running-kaggle", help="Folder containing the phishcross package.")
    parser.add_argument("--artifact-root", required=True, help="Folder containing manifest.csv, features/, checkpoints/.")
    parser.add_argument("--data-root", required=True, help="Current Phish360 root containing trainval/ and test/.")
    parser.add_argument("--out-dir", default="/kaggle/working/mcnemar_eval")
    parser.add_argument("--target", default="phishcross")
    parser.add_argument("--baselines", default="attention,concat")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
