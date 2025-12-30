import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 224

ITALIAN_TO_TR = {
    "cane": "köpek",
    "cavallo": "at",
    "elefante": "fil",
    "farfalla": "kelebek",
    "gallina": "tavuk",
    "gatto": "kedi",
    "mucca": "inek",
    "pecora": "koyun",
    "ragno": "örümcek",
    "scoiattolo": "sincap",
}

def load_css():
    css_path = Path("app/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)

    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("Model checkpoint içinde 'classes' bulunamadı. best_model.pth bozuk olabilir.")

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def predict(image: Image.Image, model, classes, device):
    tf = get_transform()
    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]

    probs = softmax(logits)
    top_idx = probs.argsort()[::-1][:3]
    top = [(classes[i], float(probs[i])) for i in top_idx]
    return top

def main():
    st.set_page_config(page_title="Animals-10 Classifier", page_icon="🧠", layout="wide")
    load_css()

    # Header
    st.markdown(
        """
        <div class="header">
          <h1>🧠 Animals-10 Sınıflandırma</h1>
          <p>MobileNetV2 (Transfer Learning) ile 10 hayvan sınıfı için görsel tahminleme arayüzü.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Safety checks
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model bulunamadı: {MODEL_PATH}\nÖnce eğitim tamamlanmalı ve models/best_model.pth oluşmalı.")
        st.stop()

    model, classes, device = load_model()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Görsel yükleyiniz</div>', unsafe_allow_html=True)
        st.markdown('<span class="badge">Desteklenen formatlar: JPG / PNG</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Bir hayvan görseli seç:", type=["jpg", "jpeg", "png"])
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    HERO_IMAGE_PATH = os.path.join("app", "assets", "hero.png")
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tahmin sonucu</div>', unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Yükleme yoksa vitrin görselini göster
        if uploaded is None:
            if os.path.exists(HERO_IMAGE_PATH):
                hero = Image.open(HERO_IMAGE_PATH).convert("RGB")
                st.image(hero, use_container_width=True)
                st.info("Tahmin yapmak için görsel yükleyiniz.")
            else:
                st.warning("Vitrin görseli bulunamadı: app/assets/hero.png")

            st.markdown("</div>", unsafe_allow_html=True)

        # Yükleme varsa: normal tahmin akışı
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image,  use_container_width=True)

            top3 = predict(image, model, classes, device)

            best_label, best_prob = top3[0]
            tr_label = ITALIAN_TO_TR.get(best_label, best_label)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">✅ Tahmin: {tr_label}</div>
                    <div class="result-sub">Güven skoru: {best_prob*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Top-3 Olasılıklar")

            for label, prob in top3:
                tr = ITALIAN_TO_TR.get(label, label)
                st.progress(min(max(prob, 0.0), 1.0), text=f"{tr} — {prob*100:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
