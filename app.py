import os
import json
from datetime import datetime
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
from streamlit_paste_button import paste_image_button
from transformers import pipeline

# ---------------------------------------------------------
# 1. Page Configuration & Custom Cyber Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="MonoVision | AI Image Forensics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: #090d16;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(9, 13, 22, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 2.2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(56, 189, 248, 0.08);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .verdict-fake {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #f43f5e;
        color: #fda4af;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 0 30px rgba(244, 63, 94, 0.2);
    }

    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #10b981;
        color: #6ee7b7;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">MONOVISION</div>
    <div class="hero-subtitle">Deepfake & Synthetic Image Forensics</div>
    <div>
        <span class="status-badge">ENGINE ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Active Pipeline")
    with st.container(border=True):
        st.markdown("**Model:** `Smogy/SMOGY-Ai-images-detector`")
        st.caption("Fine-tuned on SDXL, DALL-E 3, FLUX, and artwork domain datasets.")

# ---------------------------------------------------------
# 4. Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    """Loads the model pipeline cleanly."""
    return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")

detector = load_detector()

def parse_predictions(results):
    """
    Parses output probabilities directly from the neural network
    without any manual heuristic overrides.
    """
    fake_score = 0.0
    real_score = 0.0
    
    for res in results:
        label = str(res['label']).lower()
        score = res['score'] * 100.0
        
        if any(k in label for k in ['fake', 'ai', 'generated', 'synthetic', 'label_1']):
            fake_score = score
        elif any(k in label for k in ['real', 'human', 'authentic', 'photography', 'label_0']):
            real_score = score

    if fake_score == 0.0 and real_score > 0.0:
        fake_score = 100.0 - real_score
    elif real_score == 0.0 and fake_score > 0.0:
        real_score = 100.0 - fake_score
        
    return fake_score, real_score

# ---------------------------------------------------------
# 5. Visual Forensic Visualizers (ELA & FFT)
# ---------------------------------------------------------
def generate_ela(image, quality=90):
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    compressed_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, compressed_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if max([ex[1] for ex in extrema]) != 0 else 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return ela_image

def generate_fft(image_pil):
    img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(magnitude_spectrum)

# ---------------------------------------------------------
# 6. Input Interface
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📁 File Upload", "📋 Clipboard Import"])

image = None

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard",
        background_color="#0284c7",
        hover_background_color="#0369a1",
    )
    if paste_result.image_data is not None:
        image = paste_result.image_data.convert('RGB')

# ---------------------------------------------------------
# 7. Analysis & Output Dashboard
# ---------------------------------------------------------
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("#### 🖼️ Source Media")
            st.image(image, use_container_width=True)
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem; text-align:center;'>Resolution: {image.width} × {image.height}px</p>", unsafe_allow_html=True)
        
    with col_right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Forensic Control")
            st.write("Evaluating image features using fine-tuned neural classification.")
            analyze_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)

        if analyze_btn and detector is not None:
            with st.spinner("Analyzing image features..."):
                raw_results = detector(image)
                ai_score, real_score = parse_predictions(raw_results)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if ai_score >= 50.0:
                st.markdown(f'<div class="verdict-fake">⚠️ Verdict: Synthetically Generated ({ai_score:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "AI-Generated"
            else:
                st.markdown(f'<div class="verdict-real">✅ Verdict: Authentic Photograph ({real_score:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "Authentic Photo"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Neural Breakdown")
            
            with st.container(border=True):
                st.progress(int(ai_score), text=f"AI/Deepfake Probability: {ai_score:.1f}%")
                st.progress(int(real_score), text=f"Authentic Photography Probability: {real_score:.1f}%")

            # --- Visual Forensics ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🕵️ Visual Diagnostics")
            
            tab_ela, tab_fft = st.tabs(["Error Level Analysis (ELA)", "Frequency Spectrum (FFT)"])
            
            with tab_ela:
                st.image(generate_ela(image), use_container_width=True)
            
            with tab_fft:
                st.image(generate_fft(image), use_container_width=True)

            # --- Export Audit Log ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_data = {
                "platform": "MonoVision Forensics Studio",
                "model": "Smogy/SMOGY-Ai-images-detector",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "verdict": verdict_str,
                "probabilities": {
                    "ai_probability": f"{ai_score:.2f}%",
                    "real_probability": f"{real_score:.2f}%"
                }
            }
            
            st.download_button(
                label="📄 Download Forensic Audit Log (JSON)",
                data=json.dumps(report_data, indent=4),
                file_name=f"monovision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
    )
