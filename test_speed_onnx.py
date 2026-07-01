"""
Uji Kecepatan Inferensi CPU - PhishCross ONNX Runtime

Script ini mengukur waktu inferensi ONNX Runtime untuk forward pass classifier.
Fitur HTML dan image diasumsikan sudah tersedia sebagai vektor embedding.
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent / "running-kaggle"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phishcross.config import Config
from phishcross.data.preprocess import UrlVocab


cfg = Config()
url_vocab = UrlVocab()
onnx_path = Path(__file__).resolve().parent / "phishcross_optimized.onnx"

if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX model tidak ditemukan: {onnx_path}")

model_proto = onnx.load(str(onnx_path))
onnx.checker.check_model(model_proto)

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

rng = np.random.default_rng(42)
dummy_inputs = {
    "url_input": rng.integers(0, url_vocab.size, size=(1, cfg.max_url_len), dtype=np.int64),
    "html_input": rng.normal(size=(1, cfg.html_dim)).astype(np.float32),
    "image_input": rng.normal(size=(1, cfg.image_dim)).astype(np.float32),
}

print(f"ONNX model        : {onnx_path}")
print(f"ONNX opset        : {model_proto.opset_import[0].version}")
print(f"Execution provider: {session.get_providers()[0]}")

print("\nMelakukan warmup ONNX Runtime...")
for _ in range(50):
    session.run(None, dummy_inputs)

iterations = 1000
print(f"Mengukur kecepatan inferensi ({iterations} iterasi)...")

times = []
for _ in range(iterations):
    start_time = time.perf_counter()
    session.run(None, dummy_inputs)
    times.append(time.perf_counter() - start_time)

times = np.array(times)
avg_time = float(np.mean(times))
std_time = float(np.std(times))
p50_time = float(np.percentile(times, 50))
p95_time = float(np.percentile(times, 95))

print(f"\n{'='*60}")
print("  HASIL UJI KECEPATAN INFERENSI CPU (ONNX RUNTIME)")
print(f"{'='*60}")
print("  Mode Fusi       : Cross-Attention (PhishCross)")
print(f"  Input URL       : max_len={cfg.max_url_len}")
print(f"  Input HTML      : dim={cfg.html_dim}")
print(f"  Input Image     : dim={cfg.image_dim}")
print(f"  Iterasi Uji     : {iterations}")
print(f"{'='*60}")
print(f"  Rata-rata Waktu Inferensi per Website: {avg_time:.6f} detik")
print(f"  Standar Deviasi                      : {std_time:.6f} detik")
print(f"  Median (P50)                         : {p50_time:.6f} detik")
print(f"  P95                                  : {p95_time:.6f} detik")
print(f"  Throughput                           : {1/avg_time:.0f} website/detik")
print(f"{'='*60}")
print()
print("CATATAN: Waktu ini hanya mengukur forward pass classifier ONNX.")
print("Waktu end-to-end termasuk ekstraksi fitur MiniLM dan MobileNetV3")
print("akan lebih tinggi dan bergantung pada hardware yang digunakan.")
