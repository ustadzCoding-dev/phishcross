from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "/kaggle/input/phish360/Phish360"
    work_dir: str = "/kaggle/working/phishcross_runs"
    manifest_path: str = "/kaggle/working/phishcross_runs/manifest.csv"
    feature_dir: str = "/kaggle/working/phishcross_runs/features"
    result_dir: str = "/kaggle/working/phishcross_runs/results"
    checkpoint_dir: str = "/kaggle/working/phishcross_runs/checkpoints"

    max_url_len: int = 256
    html_max_chars: int = 4000
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 42
    num_workers: int = 2

    url_dim: int = 128
    html_dim: int = 384
    image_dim: int = 576
    proj_dim: int = 128
    dropout: float = 0.35
    modality_dropout_prob: float = 0.15

    html_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_size: int = 224

