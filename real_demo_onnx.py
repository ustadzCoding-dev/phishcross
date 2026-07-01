import time
import sys
import hashlib
from pathlib import Path
import torch

import gradio as gr
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

# Tambahkan path root agar bisa membaca package `phishcross`
ROOT_DIR = Path(__file__).resolve().parent / "running-kaggle"
sys.path.insert(0, str(ROOT_DIR))

from phishcross.config import Config
from phishcross.data.preprocess import UrlVocab, clean_html_text

# ==========================================
# 1. INISIALISASI MODEL & PRE-PROCESSING
# ==========================================
cfg = Config()

print("Memuat arsitektur ONNX Runtime PhishCross...")
ONNX_PATH = Path(__file__).resolve().parent / "phishcross_optimized.onnx"
if not ONNX_PATH.exists():
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {ONNX_PATH}")

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
onnx_sha = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()[:12]
onnx_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
onnx_opset = onnx_model.opset_import[0].version

# Inisialisasi ONNX Runtime Inference Session (Sangat Ringan, pure CPU)
ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
onnx_inputs_expected = [i.name for i in onnx_model.graph.input]
onnx_outputs_expected = [o.name for o in onnx_model.graph.output]

MODEL_INFO = {
    "path": str(ONNX_PATH),
    "opset": onnx_opset,
    "sha12": onnx_sha,
    "size_mb": onnx_size_mb,
    "provider": ort_session.get_providers()[0],
    "inputs": onnx_inputs_expected,
    "outputs": onnx_outputs_expected,
}

print(f"ONNX siap: {MODEL_INFO}")

print("Memuat Tokenizer dan Pengekstrak Fitur Visual...")
# --- A. URL Preprocessor ---
url_vocab = UrlVocab()

# --- B. HTML Preprocessor (MiniLM) ---
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer(cfg.html_model_name, device="cpu")

# --- C. Image Preprocessor (MobileNetV3) ---
from torchvision import models, transforms
mobilenet_weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
mobilenet_model = models.mobilenet_v3_small(weights=mobilenet_weights)
mobilenet_extractor = torch.nn.Sequential(mobilenet_model.features, mobilenet_model.avgpool, torch.nn.Flatten()).to("cpu")
mobilenet_extractor.eval()

img_mean = mobilenet_weights.transforms().mean
img_std = mobilenet_weights.transforms().std
img_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std),
])

print("Semua modul siap! Membuka UI...")

# ==========================================
# 2. FUNGSI INFERENSI NYATA (REAL ONNX)
# ==========================================
def predict_phishing_onnx(url_text, html_text, image):
    start_time = time.time()
    
    # --- Ekstraksi URL ---
    url_ids = url_vocab.encode(url_text, max_len=cfg.max_url_len)
    url_array = np.array([url_ids], dtype=np.int64) # ONNX butuh NumPy array (INT64)
    
    # --- Ekstraksi HTML ---
    clean_html = clean_html_text(html_text, max_chars=cfg.html_max_chars)
    html_array = sentence_model.encode([clean_html], convert_to_numpy=True).astype(np.float32) # FLOAT32
    
    # --- Ekstraksi Image ---
    if image is None:
        image = Image.new("RGB", (224, 224), color="white")
    else:
        image = image.convert("RGB")
        
    img_transformed = img_transform(image).unsqueeze(0)
    with torch.no_grad():
        image_tensor = mobilenet_extractor(img_transformed)
        image_array = image_tensor.numpy().astype(np.float32) # ONNX butuh NumPy array (FLOAT32)
        
    # --- Inferensi PhishCross menggunakan ONNX Runtime ---
    # Memasukkan NumPy array ke ONNX Runtime tanpa PyTorch Graph Overhead
    onnx_inputs = {
        "url_input": url_array,
        "html_input": html_array,
        "image_input": image_array
    }
    
    onnx_outputs = ort_session.run(None, onnx_inputs)
    logits = onnx_outputs[0] # output pertama adalah classification_output
    
    # Softmax secara manual menggunakan NumPy
    exp_logits = np.exp(logits[0] - np.max(logits[0]))
    probs = exp_logits / exp_logits.sum()
    
    phishing_score = float(probs[1])
    legit_score = float(probs[0])
    
    end_time = time.time()
    latency = end_time - start_time
    
    result_dict = {"Legitimate (Aman)": legit_score, "Phishing (Berbahaya)": phishing_score}
    status_msg = (
        "[INFO] Eksekusi ONNX Runtime selesai\n"
        f"[MODEL] {ONNX_PATH.name} | opset={onnx_opset} | sha12={onnx_sha}\n"
        f"[PROVIDER] {MODEL_INFO['provider']}\n"
        f"[WAKTU] Latensi pipeline lokal aktual: {latency:.6f} detik\n"
        "[CATATAN] Latensi ini mencakup ekstraksi MiniLM + MobileNetV3 + classifier ONNX"
    )
    
    return result_dict, status_msg

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
theme = gr.themes.Soft(
    primary_hue="emerald", 
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
)

with gr.Blocks(title="Demo ONNX PhishCross") as demo:
    gr.HTML("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="font-weight: 800; font-size: 2.5rem; margin-bottom: 0px; color: #1e293b;">PhishCross (ONNX Runtime)</h1>
        <h3 style="font-weight: 400; color: #64748b; margin-top: 5px;">Prototype Lokal Berbasis Model ONNX</h3>
    </div>
    """)
    gr.Markdown(
        f"""
**Model aktif:** `{ONNX_PATH.name}`  
**ONNX opset:** `{onnx_opset}` | **Provider:** `{MODEL_INFO['provider']}` | **SHA-256 pendek:** `{onnx_sha}` | **Ukuran:** `{onnx_size_mb:.2f} MB`  
**Input graph:** `{', '.join(onnx_inputs_expected)}` | **Output graph:** `{', '.join(onnx_outputs_expected)}`
"""
    )
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Masukkan Parameter Web Target")
            url_input = gr.Textbox(label="String Alamat URL", placeholder="Contoh: http://paypal-verify-secure.com")
            html_input = gr.Textbox(label="Teks Kode HTML", lines=3, placeholder="<form action='...'><input type='password'></form>")
            img_input = gr.Image(label="Tata Letak Visual (Screenshot)", type="pil", height=200)
            predict_btn = gr.Button("Eksekusi Inferensi ONNX", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    [
                        "http://paypal-verify-secure.example/login",
                        "<html><body><form action='http://evil.example'><input type='password'></form></body></html>",
                        None,
                    ],
                    [
                        "https://www.wikipedia.org/",
                        "<html><body><h1>Wikipedia</h1><p>The Free Encyclopedia</p></body></html>",
                        None,
                    ],
                ],
                inputs=[url_input, html_input, img_input],
                label="Contoh input demo",
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Hasil Arsitektur (ONNX)")
            label_output = gr.Label(label="Probabilitas Klasifikasi", num_top_classes=2, color="red")
            time_output = gr.Textbox(label="Log Kecepatan (Metrik ONNX CPU Aktual)", interactive=False, lines=3)
            
    predict_btn.click(
        fn=predict_phishing_onnx, 
        inputs=[url_input, html_input, img_input], 
        outputs=[label_output, time_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme)
