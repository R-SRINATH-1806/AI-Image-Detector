import os
import urllib.request
import gc
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image
from streamlit_paste_button import paste_image_button

# ---------------------------------------------------------
# 1. Page Configuration & MonoVision Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MonoVision | AI Forensics Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injected CSS for MonoVision Dark Cyber-Forensic Interface
st.markdown("""
<style>
    /* Global Base Theme Override */
    .stApp {
        background-color: #07090e;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* MonoVision Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(7, 9, 14, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 2.2rem 1.8rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.08);
    }
    
    .hero-logo {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-tagline {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto 1.2rem auto;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .m-badge {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
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

    /* Verdict Card Highlights */
    .verdict-fake {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #f43f5e;
        color: #fda4af;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.15);
    }

    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #10b981;
        color: #6ee7b7;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }

    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #f59e0b;
        color: #fde68a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Branding Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-logo">MONOVISION</div>
    <div class="hero-tagline">Deepfake & Synthetic Media Forensics Platform</div>
    <div class="badge-container">
        <span class="m-badge">⚡ Dual-Engine Ensemble</span>
        <span class="m-badge">🔬 5-Crop Spatial Sampling</span>
        <span class="m-badge">🛡️ Real-Time Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model Downloads & Processing Setup
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Initializing Neural Weights ({file_path})..."):
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
    
    # Model 1: ViT Tiny
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load("vit_highres_model.pth", map_location=device, weights_only=True))
    vit.eval()
    
    # Model 2: ResNet-18
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load("resnet_highres_model.pth", map_location=device, weights_only=True))
    resnet.eval()

    gc.collect()
    return vit, resnet

vit_model, resnet_model = load_models()

# ---------------------------------------------------------
# 4. Input Console (Clipboard, Upload, Web Link)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Clipboard Import", "📁 Local File Upload", "🔗 Web Image Link"])

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
            st.error("⚠️ Direct link connection failed. Please verify the link or paste the image directly.")

# ---------------------------------------------------------
# 5. Diagnostic Dashboard & Forensic Analysis
# ---------------------------------------------------------
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        with st.container(border=True):
            st.markdown("#### 🖼️ Media Preview")
            st.image(image, use_container_width=True)
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem; text-align:center;'>Source Image Resolution: {image.width} × {image.height}px</p>", unsafe_allow_html=True)
        
    with col_right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Forensic Engine")
            st.write("Extracting micro-texture spatial distributions using Center + 4-Corner high-res image crops.")
            analyze_btn = st.button("🚀 Run MonoVision Forensics", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner("Analyzing high-frequency pixel artifacts..."):
                patches = five_crop(image)
                patch_tensors = torch.stack([base_transform(p) for p in patches])

                with torch.no_grad():
                    vit_probs = F.softmax(vit_model(patch_tensors), dim=1).mean(dim=0).tolist()
                    res_probs = F.softmax(resnet_model(patch_tensors), dim=1).mean(dim=0).tolist()

                avg_fake = (vit_probs[0] + res_probs[0]) / 2.0 * 100
                avg_real = (vit_probs[1] + res_probs[1]) / 2.0 * 100

            st.markdown("<br>", unsafe_allow_html=True)
            
            # MonoVision Verdict Card
            if avg_fake > 60.0:
                st.markdown(f'<div class="verdict-fake">⚠️ Verdict: Synthetically Generated ({avg_fake:.1f}% Confidence)</div>', unsafe_allow_html=True)
            elif avg_real > 60.0:
                st.markdown(f'<div class="verdict-real">✅ Verdict: Authentic Photograph ({avg_real:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-uncertain">🤔 Verdict: Inconclusive Signal ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Model Diagnostic Breakdown")
            
            # Native Streamlit Containers for metrics
            c1, c2 = st.columns(2)
            
            with c1:
                with st.container(border=True):
                    st.markdown("##### ViT-Tiny (Vision Transformer)")
                    st.progress(int(vit_probs[0] * 100), text=f"AI Artifacts: {vit_probs[0]*100:.1f}%")
                    st.progress(int(vit_probs[1] * 100), text=f"Authentic Signal: {vit_probs[1]*100:.1f}%")
                
            with c2:
                with st.container(border=True):
                    st.markdown("##### ResNet-18 (Deep ConvNet)")
                    st.progress(int(res_probs[0] * 100), text=f"AI Artifacts: {res_probs[0]*100:.1f}%")
                    st.progress(int(res_probs[1] * 100), text=f"Authentic Signal: {res_probs[1]*100:.1f}%")
