import os
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# Başlık/etiketler
APP_TITLE = "CIFAR-10 Image Classifier"
APP_DESC = "ResNet18 (PyTorch) ile eğitilmiş model. Görsel yükle → sınıf tahmini + güven skoru (Top-3)."

def load_model(model_path="models/model.pth"):
    """
    Eğitilmiş modeli diskten yükler ve tahmin için hazırlar.
    - model.pth içinden model ağırlıklarını ve sınıf isimlerini alır
    - modeli eval moduna alır
    - giriş görselleri için dönüşüm (resize+normalize) pipeline'ını döndürür
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, classes, tf, device

def predict(img: Image.Image):
    """
    Kullanıcının yüklediği görsel üzerinde sınıflandırma yapar.
    Çıktı olarak:
    - Tahmin edilen sınıf adı
    - Güven skoru (0-1 arası)
    - Sınıfların olasılık dağılımı (Top-3 arayüzde gösterilir) döndürür.
    """
    if not os.path.exists("models/model.pth"):
        return "Model yok. Önce python train.py çalıştır.", 0.0, {}

    model, classes, tf, device = load_model("models/model.pth")
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (10,)

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    # Gradio Label için: {class_name: prob}
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

    pred_text = f"Tahmin: {classes[top_idx]}"
    return pred_text, top_conf, prob_dict

# Basit CSS (Gradio içinde çalışır)
CUSTOM_CSS = """
:root {
  --bg: #0b1220;
  --card: #111b2e;
  --text: #e9eefc;
  --muted: #a9b6d3;
  --accent: #7c5cff;
  --accent2: #22c55e;
}

.gradio-container {
  background: radial-gradient(1000px 600px at 15% 10%, rgba(124,92,255,.25), transparent 55%),
              radial-gradient(900px 500px at 90% 25%, rgba(34,197,94,.18), transparent 55%),
              linear-gradient(180deg, var(--bg), #070b14);
  color: var(--text) !important;
}

#app_card {
  background: rgba(17,27,46,.78);
  border: 1px solid rgba(124,92,255,.25);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

#small_note {
  color: var(--muted);
  font-size: 0.95rem;
}

#badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(124,92,255,.18);
  border: 1px solid rgba(124,92,255,.35);
  color: var(--text);
  font-size: 0.9rem;
}
"""

# Theme
theme = gr.themes.Soft()

with gr.Blocks(theme=theme, css=CUSTOM_CSS) as demo:
    gr.Markdown(f"# {APP_TITLE}\n\n<span id='badge'>Confidence + Top-3</span>\n\n{APP_DESC}", elem_id="app_card")

    with gr.Row():
        img_in = gr.Image(type="pil", label="Görsel Yükle", height=280)
        with gr.Column():
            pred_out = gr.Textbox(label="Sonuç", lines=1)
            conf_out = gr.Slider(label="Güven (0–1)", minimum=0, maximum=1, step=0.001, interactive=False)
            top3_out = gr.Label(label="Sınıf Olasılıkları (Top-3 gösterir)", num_top_classes=3)

    gr.Markdown(
        "<div id='small_note'>Not: Model sadece CIFAR-10 sınıflarıyla eğitildiği için, tamamen alakasız görsellerde en yakın sınıfa zorunlu tahmin yapabilir.</div>",
        elem_id="app_card"
    )

    btn = gr.Button("Tahmin Et", variant="primary")
    btn.click(fn=predict, inputs=img_in, outputs=[pred_out, conf_out, top3_out])

if __name__ == "__main__":
    demo.launch()
