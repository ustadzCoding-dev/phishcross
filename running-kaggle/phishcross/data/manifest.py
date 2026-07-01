from pathlib import Path

import pandas as pd


def infer_label(sample_dir: Path) -> int:
    label_file = sample_dir / "Label" / "label.txt"
    if label_file.exists():
        text = label_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
        if "phish" in text or text == "1":
            return 1
        if "legit" in text or text == "0":
            return 0
    return 1 if sample_dir.name.startswith("P") else 0


def build_manifest(data_root: str, out_csv: str) -> pd.DataFrame:
    root = Path(data_root)
    rows = []
    for split in ["trainval", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
        for sample_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            rows.append(
                {
                    "sample_id": sample_dir.name,
                    "split": split,
                    "label": infer_label(sample_dir),
                    "url_path": str(sample_dir / "URL" / "url.txt"),
                    "html_path": str(sample_dir / "RAW-HTML" / "index.html"),
                    "image_path": str(sample_dir / "SCREEN-SHOT" / "screen_shoot.png"),
                }
            )
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root, args.out_csv)
    print(manifest.groupby(["split", "label"]).size())
    print(f"Saved: {args.out_csv}")

