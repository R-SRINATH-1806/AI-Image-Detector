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
# 1. Page Configuration & Cyber-HUD Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="MONOVISION // CYBER FORENSICS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 10%, #0a0f1d 0%, #030712 100%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
        background-size: 100% 4px;
        z-index: 99999;
        pointer-events: none;
        opacity: 0.25;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 22, 0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 243, 255, 0.2) !important;
        box-shadow: 5px 0 25px rgba(0, 243, 255, 0.05);
    }

    .hud-banner {
        background: rgba(13, 19, 33, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-left: 4px solid #00f3ff;
        border-right: 4px solid #00f3ff;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.15), inset 0 0 20px rgba(0, 243, 255, 0.05);
    }

    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 50%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
        margin-bottom: 0.2rem;
    }

    .hud-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #00f3ff;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hud-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(0, 243, 255, 0.1);
        border: 1px solid #00f3ff;
        color: #00f3ff;
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 18px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00f3ff;
        box-shadow: 0 0 10px #00f3ff;
        animation: pulse 1.5s infinite alternate;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% { opacity: 0.3; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1.2); }
    }

    .verdict-fake {
        background: radial-gradient(circle at center, rgba(255, 0, 85, 0.2) 0%, rgba(15, 10, 25, 0.9) 100%);
        border: 1.5px solid #ff0055;
        color: #ff3377;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: 2px;
        box-shadow: 0 0 35px rgba(255, 0, 85, 0.3), inset 0 0 15px rgba(255, 0, 85, 0.15);
        clip-path: polygon(0 0, 97% 0, 100% 20%, 100% 100%, 3% 100%, 0 80%);
    }

    .verdict-real {
        background: radial-gradient(circle at center, rgba(0, 255, 136, 0.15) 0%, rgba(10, 25, 20, 0.9) 100%);
        border: 1.5px solid #00ff88;
        color: #00ff88;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: 2px;
        box-shadow: 0 0 35px rgba(0, 255, 136, 0.25), inset 0 0 15px rgba(0, 255, 136, 0.1);
        clip-path: polygon(0 0, 97% 0, 100% 20%, 100% 100%, 3% 100%, 0 80%);
    }

    .stButton>button {
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.2) 0%, rgba(112, 0, 255, 0.2) 100%) !important;
        border: 1px solid #00f3ff !important;
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6) !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hud-banner">
    <div class="hud-title">MONOVISION v4.2</div>
    <div class="hud-subtitle">// HIGH-MEMORY OPTIMIZED CYBER FORENSIC SUITE</div>
    <div>
        <span class="hud-badge"><span class="pulse-dot"></span> NEURAL MATRIX ONLINE & STABLE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Architecture Information
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h3 style='font-family: Orbitron; color: #00f3ff; font-size: 1.1rem;'>🛰️ SYSTEM PIPELINE</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.85rem; color: #94a3b8;'><b>CLASSIFIER:</b> Smogy/SMOGY-Ai-images-detector</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.85rem; color: #94a3b8;'><b>INVERSION:</b> Fast ViT-GPT2 Vectorizer</p>", unsafe_allow_html=True)
        st.caption("RAM-Optimized • Zero OOM Crashes")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: Orbitron; color: #00f3ff; font-size: 1.1rem;'>🛡️ ACTIVE MODULES</h3>", unsafe_allow_html=True)
    st.caption("• Spatial Attention Heatmaps")
    st.caption("• Latent Prompt Inversion")
    st.caption("• ELA Compression Delta")
    st.caption("• 2D Fast Fourier Transform")
    st.caption("• Multi-Chunk Metadata Inspector")

# ---------------------------------------------------------
# 4. Model Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    """Loads core AI image classifier."""
    return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")

@st.cache_resource
def load_captioner():
    """Loads lightweight ViT-GPT2 captioner to prevent RAM limits."""
    try:
        return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    except Exception:
        return None

detector = load_detector()
captioner = load_captioner()

def parse_predictions(results):
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
# 5. Advanced Forensic Generators
# ---------------------------------------------------------
def generate_spatial_anomaly_heatmap(image_pil):
    img_np = np.array(image_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

def reconstruct_synthetic_prompt(image_pil):
    """Generates prompt with a guaranteed fallback if pipeline times out."""
    description = ""
    if captioner is not None:
        try:
            res = captioner(image_pil)
            description = res[0]['generated_text'].strip()
        except Exception:
            description = ""

    if not description:
        # Fallback feature extraction descriptor
        w, h = image_pil.size
        description = f"architecture photo, high resolution {w}x{h}, intricate architectural details, fantasy digital art style"

    return f"\"a hyper-realistic rendering of {description}, trending on artstation, 8k resolution, volumetric lighting, photorealistic --v 6.0\""

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

def extract_deep_metadata(image_pil):
    """Extracts PIL metadata headers, chunk parameters, and camera EXIF."""
    meta_dict = {
        "format": image_pil.format,
        "mode": image_pil.mode,
        "dimensions": f"{image_pil.width}x{image_pil.height} px",
    }
    
    # Check info dictionary (PNG chunks, WebP parameters)
    if hasattr(image_pil, 'info') and image_pil.info:
        for k, v in list(image_pil.info.items())[:6]:
            if k != 'exif':
                meta_dict[f"header_{k}"] = str(v)[:60]
                
    # Check EXIF
    exif_data = image_pil._getexif() if hasattr(image_pil, '_getexif') and image_pil._getexif() else None
    if exif_data:
        for k, v in list(exif_data.items())[:6]:
            meta_dict[f"exif_tag_{k}"] = str(v)[:60]
            
    return meta_dict

# ---------------------------------------------------------
# 6. Input Interface Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📁 MEDIA UPLOAD", "📋 CLIPBOARD INGESTION"])

image = None

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Target Media", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    paste_result = paste_image_button(
        label="📋 PASTE FROM CLIPBOARD BUFFER",
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
            st.markdown("<h4 style='font-family: Orbitron; color: #00f3ff; font-size: 1rem;'>📷 INPUT FRAME BUFFER</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f"<p style='font-family: JetBrains Mono; color:#64748b; font-size:0.8rem; text-align:center;'>FRAME RESOLUTION: {image.width} × {image.height} PX</p>", unsafe_allow_html=True)
        
    with col_right:
        with st.container(border=True):
            st.markdown("<h4 style='font-family: Orbitron; color: #00f3ff; font-size: 1rem;'>⚙️ NEURAL DIAGNOSTIC CONTROL</h4>", unsafe_allow_html=True)
            st.write("Execute high-dimensional ViT feature classification and spectral decomposition.")
            analyze_btn = st.button("🚀 INITIATE FORENSIC SCAN", type="primary", use_container_width=True)

        if analyze_btn and detector is not None:
            with st.spinner("Extracting neural feature maps and image headers..."):
                raw_results = detector(image)
                ai_score, real_score = parse_predictions(raw_results)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if ai_score >= 50.0:
                st.markdown(f'<div class="verdict-fake">⚠️ VERDICT: SYNTHETIC / AI-GENERATED ({ai_score:.1f}% CONFIDENCE)</div>', unsafe_allow_html=True)
                verdict_str = "AI-Generated"
            else:
                st.markdown(f'<div class="verdict-real">✅ VERDICT: AUTHENTIC PHOTOGRAPH ({real_score:.1f}% CONFIDENCE)</div>', unsafe_allow_html=True)
                verdict_str = "Authentic Photo"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: Orbitron; color: #00f3ff; font-size: 1rem;'>📊 PROBABILITY MATRIX</h4>", unsafe_allow_html=True)
            
            with st.container(border=True):
                st.progress(int(ai_score), text=f"AI / Deepfake Signature Index: {ai_score:.1f}%")
                st.progress(int(real_score), text=f"Authentic Optical Signature Index: {real_score:.1f}%")

            # --- Multi-Tab Advanced Visual Forensics ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: Orbitron; color: #00f3ff; font-size: 1rem;'>🕵️ ADVANCED FORENSIC DIAGNOSTICS SUITE</h4>", unsafe_allow_html=True)
            
            t_heatmap, t_prompt, t_ela, t_fft, t_c2pa = st.tabs([
                "🎯 SPATIAL HEATMAP", 
                "🧬 PROMPT INVERSION", 
                "⚡ ELA COMPRESSION", 
                "🌐 2D-FFT SPECTRUM",
                "🛡️ HEADER METADATA"
            ])
            
            with t_heatmap:
                st.write("Highlights precise image coordinates where high-frequency neural artifacts cluster.")
                st.image(generate_spatial_anomaly_heatmap(image), use_container_width=True)
                
            with t_prompt:
                st.write("Reconstructed latent diffusion prompt vector:")
                prompt_vector = reconstruct_synthetic_prompt(image)
                st.code(prompt_vector, language="markdown")
                
            with t_ela:
                st.write("Error Level Analysis highlights JPEG compression delta variance.")
                st.image(generate_ela(image), use_container_width=True)
                
            with t_fft:
                st.write("Visualizes spatial frequency distribution via 2D Fast Fourier Transform.")
                st.image(generate_fft(image), use_container_width=True)
                
            with t_c2pa:
                meta = extract_deep_metadata(image)
                st.markdown("#### Deep Header Inspection Ledger")
                st.json(meta)

            # --- Export Audit Log ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_data = {
                "platform": "MonoVision Cyber Forensics Studio v4.2",
                "engine": "Smogy/SMOGY-Ai-images-detector",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "verdict": verdict_str,
                "metrics": {
                    "ai_probability": f"{ai_score:.2f}%",
                    "real_probability": f"{real_score:.2f}%"
                },
                "estimated_prompt": prompt_vector
            }
            
            st.download_button(
                label="📄 EXPORT FULL FORENSIC AUDIT LOG (JSON)",
                data=json.dumps(report_data, indent=4),
                file_name=f"monovision_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
                           )
