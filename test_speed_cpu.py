"""
Uji Kecepatan Inferensi CPU — PhishCross (Arsitektur Asli)

Script ini mengukur rata-rata waktu inferensi model PhishCross yang sebenarnya
di CPU, menggunakan arsitektur dan dimensi yang identik dengan model pelatihan.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Pastikan package phishcross bisa di-import dari lokasi relatif
ROOT_DIR = Path(__file__).resolve().parent / "running-kaggle"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phishcross.config import Config
from phishcross.models.phishcross_net import PhishCrossNet

from phishcross.data.preprocess import UrlVocab

cfg = Config()
url_vocab = UrlVocab()

# ---- Inisialisasi Model Asli (mode="cross") ----
model = PhishCrossNet(
    mode="cross",
    use_url=True,
    use_html=True,
    use_image=True,
    html_dim=cfg.html_dim,       # 384
    image_dim=cfg.image_dim,     # 576
    proj_dim=cfg.proj_dim,       # 128
    dropout=cfg.dropout,         # 0.35
)

checkpoint_path = Path(__file__).resolve().parent / "phishcross_result" / "phishcross_runs" / "checkpoints" / "seed_44" / "phishcross.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

print(f"Memuat checkpoint: {checkpoint_path}")
model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
model.eval()

# Hitung parameter
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameter PhishCross: {total_params:,}")
print(f"Parameter Trainable      : {trainable_params:,}")

# ---- Buat Dummy Input Sesuai Dimensi Asli ----
dummy_batch = {
    "url_ids": torch.randint(0, url_vocab.size, (1, cfg.max_url_len), dtype=torch.long),
    "html_feat": torch.randn(1, cfg.html_dim),
    "image_feat": torch.randn(1, cfg.image_dim),
}

# ---- Warmup CPU (20 iterasi) ----
print("\nMelakukan warmup CPU...")
with torch.no_grad():
    for _ in range(20):
        _ = model(dummy_batch)

# ---- Ukur Kecepatan Inferensi ----
iterations = 1000
print(f"Mengukur kecepatan inferensi ({iterations} iterasi)...")
times = []
with torch.no_grad():
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = model(dummy_batch)
        times.append(time.perf_counter() - start_time)

avg_time = float(np.mean(times))
std_time = float(np.std(times))
p50_time = float(np.percentile(times, 50))
p95_time = float(np.percentile(times, 95))

print(f"\n{'='*60}")
print(f"  HASIL UJI KECEPATAN INFERENSI CPU (ARSITEKTUR ASLI)")
print(f"{'='*60}")
print(f"  Mode Fusi       : Cross-Attention (PhishCross)")
print(f"  Input URL       : max_len={cfg.max_url_len}")
print(f"  Input HTML      : dim={cfg.html_dim}")
print(f"  Input Image     : dim={cfg.image_dim}")
print(f"  Projection Dim  : {cfg.proj_dim}")
print(f"  Total Parameter : {total_params:,}")
print(f"  Iterasi Uji     : {iterations}")
print(f"{'='*60}")
print(f"  Rata-rata Waktu Inferensi per Website: {avg_time:.6f} detik")
print(f"  Standar Deviasi                      : {std_time:.6f} detik")
print(f"  Median (P50)                         : {p50_time:.6f} detik")
print(f"  P95                                  : {p95_time:.6f} detik")
print(f"  Throughput                           : {1/avg_time:.0f} website/detik")
print(f"{'='*60}")
print()
print("CATATAN: Waktu ini hanya mengukur forward pass model fusi PhishCross.")
print("Waktu end-to-end (termasuk feature extraction oleh MiniLM & MobileNetV3)")
print("akan lebih tinggi dan bergantung pada hardware yang digunakan.")
