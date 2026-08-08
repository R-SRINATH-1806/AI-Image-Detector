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
    page_title="MonoVision | Deepfake Forensics Studio",
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

    .pulse-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
        margin-right: 6px;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
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

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #0f172a;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
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

    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #f59e0b;
        color: #fde68a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Branding Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">MONOVISION</div>
    <div class="hero-subtitle">Deepfake & Synthetic Image Forensics Platform</div>
    <div>
        <span class="status-badge"><span class="pulse-online"></span> ENSEMBLE ENGINE ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Dual-Model Pipeline")
    
    with st.container(border=True):
        st.markdown("**Engine 1:** `umm-maybe/AI-image-detector`")
        st.markdown("**Engine 2:** `Falconsai/intent_image_classifier`")
        st.caption("Focus: Midjourney, Stable Diffusion & DALL-E Detection")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛡️ System Telemetry")
    st.caption("Analysis Mode: Weighted Ensemble Averaging")

# ---------------------------------------------------------
# 4. Hugging Face Dual-Model Ensemble Loader
# ---------------------------------------------------------
@st.cache_resource
def load_ensemble_detectors():
    """Loads two models specialized in Midjourney and general synthetic art."""
    pipe1 = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    pipe2 = pipeline("image-classification", model="Falconsai/intent_image_classifier")
    return pipe1, pipe2

pipe1, pipe2 = load_ensemble_detectors()

def parse_model_scores(results):
    """Cleanly extracts fake/real percentages regardless of output label syntax."""
    fake_score, real_score = 0.0, 0.0
    for res in results:
        label = str(res['label']).lower()
        score = res['score'] * 100.0
        if any(k in label for k in ['fake', 'ai', 'generated', 'synthetic', 'artificial', 'label_1']):
            fake_score = score
        elif any(k in label for k in ['real', 'authentic', 'human', 'label_0']):
            real_score = score
            
    if fake_score == 0.0 and real_score > 0:
        fake_score = 100.0 - real_score
    elif real_score == 0.0 and fake_score > 0:
        real_score = 100.0 - fake_score
        
    return fake_score, real_score

# ---------------------------------------------------------
# 5. Visual Forensic Generators (ELA & FFT)
# ---------------------------------------------------------
def generate_ela(image, quality=90):
    """Generates Error Level Analysis (ELA) map highlighting compression artifacts."""
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
    """Generates Frequency Spectrum (FFT) map highlighting grid artifacts."""
    img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(magnitude_spectrum)

# ---------------------------------------------------------
# 6. Input Interface Tabs
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
# 7. Diagnostic Dashboard & Analysis Execution
# ---------------------------------------------------------
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if image.width < 500 or image.height < 500:
        st.warning(
            f"⚠️ **Low Resolution Media ({image.width} × {image.height}px):** "
            "Small images have heavy JPEG compression, which can impact detection precision."
        )

    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("#### 🖼️ Source Media")
            st.image(image, use_container_width=True)
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem; text-align:center;'>Resolution: {image.width} × {image.height}px</p>", unsafe_allow_html=True)
        
    with col_right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Forensic Control")
            st.write("Evaluating against dual Midjourney and synthetic image classification models.")
            analyze_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner("Analyzing neural texture patterns..."):
                res1 = pipe1(image)
                fake1, real1 = parse_model_scores(res1)
                
                res2 = pipe2(image)
                fake2, real2 = parse_model_scores(res2)
                
                # Weighted ensemble (Engine 1: 50%, Engine 2: 50%)
                avg_fake = (fake1 + fake2) / 2.0
                avg_real = (real1 + real2) / 2.0

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- BALANCED VERDICT LOGIC ---
            if avg_fake >= 50.0:
                st.markdown(f'<div class="verdict-fake">⚠️ Verdict: Synthetically Generated ({avg_fake:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "AI-Generated"
            elif avg_real > 50.0:
                st.markdown(f'<div class="verdict-real">✅ Verdict: Authentic Photograph ({avg_real:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "Authentic Photo"
            else:
                st.markdown(f'<div class="verdict-uncertain">🤔 Verdict: Inconclusive Signal ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)</div>', unsafe_allow_html=True)
                verdict_str = "Inconclusive"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Ensemble Transformer Breakdown")
            
            with st.container(border=True):
                st.progress(int(avg_fake), text=f"Combined AI/Deepfake Signature: {avg_fake:.1f}%")
                st.progress(int(avg_real), text=f"Combined Authentic Signature: {avg_real:.1f}%")
                st.caption(f"Engine 1 (General AI): {fake1:.1f}% AI | Engine 2 (Generator Classifier): {fake2:.1f}% AI")

            # --- Advanced Visual Forensics (ELA & FFT) ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🕵️‍♂️ Advanced Visual Forensics")
            
            tab_ela, tab_fft = st.tabs(["Error Level Analysis (ELA)", "Frequency Spectrum (FFT)"])
            
            with tab_ela:
                st.write("Highlights areas with different JPEG compression ratios. Digital splices and synthetic regions often glow brighter than untouched areas.")
                st.image(generate_ela(image), use_container_width=True)
            
            with tab_fft:
                st.write("Visualizes high-frequency patterns. Real photos have continuous radial glows, while AI generators often leave unnatural geometric grids.")
                st.image(generate_fft(image), use_container_width=True)

            # --- Forensic Report Download ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_data = {
                "platform": "MonoVision Forensics Studio",
                "engine": "Dual-Ensemble (umm-maybe + Falconsai)",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "verdict": verdict_str,
                "engine_breakdown": {
                    "engine_1_ai_score": f"{fake1:.2f}%",
                    "engine_2_ai_score": f"{fake2:.2f}%",
                    "ensemble_average_fake": f"{avg_fake:.2f}%",
                    "ensemble_average_real": f"{avg_real:.2f}%"
                }
            }
            
            st.download_button(
                label="📄 Download Forensic Audit Log (JSON)",
                data=json.dumps(report_data, indent=4),
                file_name=f"monovision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
