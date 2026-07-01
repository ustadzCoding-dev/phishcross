import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_EXPERIMENTS = ["url_only", "html_only", "image_only", "concat", "attention", "phishcross"]
EXTRA_EXPERIMENTS = [
    "attention_no_url",
    "attention_no_html",
    "attention_no_image",
    "phishcross_no_url",
    "phishcross_no_html",
    "phishcross_no_image",
    "attention_no_moddrop",
    "phishcross_no_moddrop",
]


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize(result_dir, seeds):
    rows = []
    for seed in seeds:
        seed_dir = Path(result_dir) / f"seed_{seed}"
        if not seed_dir.exists():
            seed_dir = Path(result_dir)
        for path in sorted(seed_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "experiment": data.get("experiment"),
                        "seed": seed,
                        "accuracy": data.get("accuracy"),
                        "precision": data.get("precision"),
                        "recall": data.get("recall"),
                        "f1": data.get("f1"),
                        "roc_auc": data.get("roc_auc"),
                        "params": data.get("trainable_parameters"),
                        "inference_time_sec": data.get("inference_time_per_sample_sec"),
                    }
                )
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if not rows:
        print("No results found to summarize.")
        return

    df_raw = pd.DataFrame(rows)
    raw_csv = Path(result_dir) / "all_runs.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"Saved raw runs: {raw_csv}")

    if len(seeds) > 1:
        agg_funcs = {
            "accuracy": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "params": "first",
            "inference_time_sec": "mean",
        }
        df_agg = df_raw.groupby("experiment").agg(agg_funcs)
        df_agg.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != "first" else col[0]
            for col in df_agg.columns
        ]
        df_agg = df_agg.reset_index().sort_values("f1_mean", ascending=False)
        out_csv = Path(result_dir) / "summary.csv"
        df_agg.to_csv(out_csv, index=False)
        print("\n--- Aggregated Results (Mean ± Std) ---")
        print(df_agg.to_string(index=False))
        print(f"Saved aggregated summary: {out_csv}")
    else:
        df_simple = df_raw.drop(columns=["seed"]).sort_values("f1", ascending=False)
        out_csv = Path(result_dir) / "summary.csv"
        df_simple.to_csv(out_csv, index=False)
        print("\n--- Summary ---")
        print(df_simple.to_string(index=False))
        print(f"Saved summary: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/kaggle/input/phish360/Phish360")
    parser.add_argument("--work-dir", default="/kaggle/working/phishcross_runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extra", action="store_true", help="Run missing-modality and ablation experiments too.")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds to run (e.g., 42,43,44)")
    args = parser.parse_args()

    work = Path(args.work_dir)
    manifest = work / "manifest.csv"
    feature_dir = work / "features"
    result_dir = work / "results"
    checkpoint_dir = work / "checkpoints"
    work.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "phishcross.data.manifest",
            "--data-root",
            args.data_root,
            "--out-csv",
            str(manifest),
        ]
    )

    if not args.skip_features:
        run(
            [
                sys.executable,
                "-m",
                "phishcross.extract_features",
                "--manifest",
                str(manifest),
                "--feature-dir",
                str(feature_dir),
                "--batch-size",
                str(args.batch_size),
                "--skip-existing",
            ]
        )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    experiments = DEFAULT_EXPERIMENTS + (EXTRA_EXPERIMENTS if args.extra else [])

    for seed in seeds:
        print(f"\n==================================================")
        print(f"RUNNING EXPERIMENTS WITH SEED {seed}")
        print(f"==================================================")
        seed_result_dir = result_dir / f"seed_{seed}"
        seed_checkpoint_dir = checkpoint_dir / f"seed_{seed}"
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for exp in experiments:
            run(
                [
                    sys.executable,
                    "-m",
                    "phishcross.train",
                    "--manifest",
                    str(manifest),
                    "--feature-dir",
                    str(feature_dir),
                    "--result-dir",
                    str(seed_result_dir),
                    "--checkpoint-dir",
                    str(seed_checkpoint_dir),
                    "--experiment",
                    exp,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(seed),
                ]
            )

    summarize(result_dir, seeds)


if __name__ == "__main__":
    main()


