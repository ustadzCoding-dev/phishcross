import torch
import torch.onnx
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load arsitektur dari folder lokal
ROOT_DIR = Path(__file__).resolve().parent / "running-kaggle"
sys.path.insert(0, str(ROOT_DIR))

from phishcross.config import Config
from phishcross.models.phishcross_net import PhishCrossNet
from phishcross.data.preprocess import UrlVocab

print("1. Inisialisasi arsitektur PhishCross...")
cfg = Config()
model = PhishCrossNet(
    mode="cross",
    use_url=True,
    use_html=True,
    use_image=True,
    html_dim=cfg.html_dim,
    image_dim=cfg.image_dim,
    proj_dim=cfg.proj_dim,
    dropout=cfg.dropout,
)

print("2. Memuat bobot .pt hasil pelatihan Kaggle...")
# Sesuaikan path ini ke lokasi checkpoint hasil running Kaggle
CHECKPOINT_PATH = Path(__file__).resolve().parent / "phishcross_result" / "phishcross_runs" / "checkpoints" / "seed_44" / "phishcross.pt"
print(f"   Memuat checkpoint dari: {CHECKPOINT_PATH}")
model.load_state_dict(torch.load(str(CHECKPOINT_PATH), map_location="cpu"))
model.eval()

print("3. Membuat tensor input tiruan (Dummy Inputs)...")
# Dimensi disesuaikan dengan output dari konfigurasi
url_vocab = UrlVocab()
dummy_url = torch.randint(0, url_vocab.size, (1, cfg.max_url_len), dtype=torch.long)
dummy_html = torch.randn(1, cfg.html_dim)
dummy_img = torch.randn(1, cfg.image_dim)

print("4. Mengekspor model ke format ONNX...")
ONNX_PATH = "phishcross_optimized.onnx"

# Buat Wrapper agar ONNX tidak kesulitan memparsing input berupa Dictionary
class ONNXWrapper(torch.nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.model = core_model

    def forward(self, url_ids, html_feat, image_feat):
        batch = {
            "url_ids": url_ids,
            "html_feat": html_feat,
            "image_feat": image_feat
        }
        logits, weights, names = self.model(batch)
        return logits, weights

wrapped_model = ONNXWrapper(model)
wrapped_model.eval()

# Proses ekspor ke ONNX
torch.onnx.export(
    wrapped_model,                               # Model yang dibungkus
    (dummy_url, dummy_html, dummy_img),          # Input tensor langsung
    ONNX_PATH,                                   # Nama file output
    export_params=True,                          # Simpan bobot latih di dalam file ONNX
    opset_version=18,                            # Sesuai exporter PyTorch terbaru
    do_constant_folding=True,                    # Optimasi melipat konstanta
    input_names=["url_input", "html_input", "image_input"],   # Nama input untuk mempermudah pemanggilan
    output_names=["classification_output", "attention_weights"] # Nama output (tanpa string names)
)

print(f"SUKSES! Model berhasil dikonversi dan disimpan sebagai: {ONNX_PATH}")
print("Sekarang model ini bisa dijalankan murni menggunakan CPU melalui 'onnxruntime' tanpa perlu menginstall PyTorch yang berat.")
