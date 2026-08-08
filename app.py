import os
import urllib.request
import gc
import json
from datetime import datetime
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image
from streamlit_paste_button import paste_image_button

# ---------------------------------------------------------
# 1. Page Config & High-End Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="MonoVision | Deepfake Forensics Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Cyber Core Background */
    .stApp {
        background: #090d16;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(9, 13, 22, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 2.2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(56, 189, 248, 0.08);
        position: relative;
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

    /* Pulsating Status Indicator */
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

    /* Tab Custom Styling */
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

    /* Verdict Card Highlight Overrides */
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
        <span class="status-badge"><span class="pulse-online"></span> ENGINES ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Engine Configuration & Telemetry
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    st.markdown("---")
    
    threshold = st.slider(
        "AI Detection Threshold (%)",
        min_value=50,
        max_value=90,
        value=60,
        step=5,
        help="Confidence level required to classify an image as AI-generated."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Active Model Pipeline")
    
    with st.container(border=True):
        st.markdown("**1. ViT-Tiny (Vision Transformer)**")
        st.caption("Weight: 50% | Resolution: 224×224")
        st.markdown("**2. ResNet-18 (Deep ConvNet)**")
        st.caption("Weight: 50% | Resolution: 224×224")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛡️ System Status")
    st.caption("Device: CPU Inference Mode")
    st.caption("Spatial Sampling: 5-Crop Strategy")

# ---------------------------------------------------------
# 4. Model Downloads & Processing Setup
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading weights ({file_path})..."):
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())

device = torch.device("cpu")

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

five_crop = transforms.FiveCrop(224)

@st.cache_resource
def load_models():
    download_file_if_missing("vit_highres_model.pth", VIT_URL)
    download_file_if_missing("resnet_highres_model.pth", RESNET_URL)
    
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load("vit_highres_model.pth", map_location=device, weights_only=True))
    vit.eval()
    
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load("resnet_highres_model.pth", map_location=device, weights_only=True))
    resnet.eval()

    gc.collect()
    return vit, resnet

vit_model, resnet_model = load_models()

# ---------------------------------------------------------
# 5. Input Console
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Clipboard Import", "📁 File Upload", "🔗 Image URL"])

image = None

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard",
        background_color="#0284c7",
        hover_background_color="#0369a1",
    )
    if paste_result.image_data is not None:
        image = paste_result.image_data.convert('RGB')

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    url_input = st.text_input("Direct Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    if url_input:
        try:
            req = urllib.request.Request(url_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                image = Image.open(response).convert('RGB')
        except Exception:
            st.error("⚠️ Unable to load image URL. Please check the link or paste the image directly.")

# ---------------------------------------------------------
# 6. Diagnostic Dashboard & Forensic Analysis
# ---------------------------------------------------------
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("#### 🖼️ Media Sample")
            st.image(image, use_container_width=True)
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem; text-align:center;'>Resolution: {image.width} × {image.height}px</p>", unsafe_allow_html=True)
        
    with col_right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Forensic Execution")
            st.write("Ready to analyze micro-textures across Center + 4-Corner high-res patches.")
            analyze_btn = st.button("🚀 Run MonoVision Analysis", type="primary", use_container_width=True)

        if analyze_btn:
            patches = five_crop(image)
            patch_tensors = torch.stack([base_transform(p) for p in patches])

            with st.spinner("Executing dual-model micro-texture inference..."):
                with torch.no_grad():
                    vit_probs = F.softmax(vit_model(patch_tensors), dim=1).mean(dim=0).tolist()
                    res_probs = F.softmax(resnet_model(patch_tensors), dim=1).mean(dim=0).tolist()

                avg_fake = (vit_probs[0] + res_probs[0]) / 2.0 * 100
                avg_real = (vit_probs[1] + res_probs[1]) / 2.0 * 100

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Dynamic Threshold Verdict Card
            if avg_fake >= threshold:
                st.markdown(f'<div class="verdict-fake">⚠️ Verdict: Synthetically Generated ({avg_fake:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "AI-Generated"
            elif avg_real >= threshold:
                st.markdown(f'<div class="verdict-real">✅ Verdict: Authentic Photograph ({avg_real:.1f}% Confidence)</div>', unsafe_allow_html=True)
                verdict_str = "Authentic Photo"
            else:
                st.markdown(f'<div class="verdict-uncertain">🤔 Verdict: Inconclusive Signal ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)</div>', unsafe_allow_html=True)
                verdict_str = "Inconclusive"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Neural Model Breakdown")
            
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("##### ViT-Tiny (Transformer)")
                    st.progress(int(vit_probs[0] * 100), text=f"AI Artifacts: {vit_probs[0]*100:.1f}%")
                    st.progress(int(vit_probs[1] * 100), text=f"Authentic Signal: {vit_probs[1]*100:.1f}%")
                
            with c2:
                with st.container(border=True):
                    st.markdown("##### ResNet-18 (Deep ConvNet)")
                    st.progress(int(res_probs[0] * 100), text=f"AI Artifacts: {res_probs[0]*100:.1f}%")
                    st.progress(int(res_probs[1] * 100), text=f"Authentic Signal: {res_probs[1]*100:.1f}%")

            # --- Visual 5-Crop Patch Inspector ---
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Inspection Area: Sampled 5-Crop Micro-Patches", expanded=False):
                st.write("These 224x224 crops were passed through both neural networks for high-frequency noise evaluation:")
                crop_cols = st.columns(5)
                crop_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
                for idx, crop in enumerate(patches):
                    with crop_cols[idx]:
                        st.image(crop, caption=crop_names[idx], use_container_width=True)

            # --- Downloadable Forensic Report ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_data = {
                "platform": "MonoVision Forensics Studio",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "verdict": verdict_str,
                "confidence_threshold_used": f"{threshold}%",
                "ensemble_averages": {
                    "ai_probability": f"{avg_fake:.2f}%",
                    "real_probability": f"{avg_real:.2f}%"
                },
                "models": {
                    "vit_tiny": {
                        "ai_prob": f"{vit_probs[0]*100:.2f}%",
                        "real_prob": f"{vit_probs[1]*100:.2f}%"
                    },
                    "resnet18": {
                        "ai_prob": f"{res_probs[0]*100:.2f}%",
                        "real_prob": f"{res_probs[1]*100:.2f}%"
                    }
                }
            }
            
            st.download_button(
                label="📄 Download Forensic Audit Log (JSON)",
                data=json.dumps(report_data, indent=4),
                file_name=f"monovision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
