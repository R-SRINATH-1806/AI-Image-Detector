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
# 1. Modern Page & Theme Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VeriSight AI | Deepfake & Media Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Modern Dark Glassmorphism CSS styling
st.markdown("""
<style>
    /* Global Container Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(13, 17, 23, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        max-width: 650px;
        margin: 0 auto;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0.25rem 0 0.25rem;
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
    }
    
    /* Result Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Verdict Alerts */
    .verdict-fake {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.15) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid #f85149;
        color: #ff7b72;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.15) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid #2ea043;
        color: #3fb950;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
    }
    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.15) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid #d29922;
        color: #e3b341;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
    }
    
    /* Custom Streamlit Tabs Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ VeriSight Neural Forensics</div>
    <div class="hero-subtitle">High-precision AI image detection powered by dual Vision Architectures & 5-crop spatial texture sampling.</div>
    <div>
        <span class="badge">ViT-Tiny Vision Transformer</span>
        <span class="badge">ResNet-18 Deep ConvNet</span>
        <span class="badge">5-Crop Spatial Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model Downloads & Processing Utilities
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
# 4. Professional Input Console Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Clipboard Import", "📁 Direct File Upload", "🔗 Web Image Link"])

image = None

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    paste_result = paste_image_button(
        label="📋 Paste Image directly from Clipboard",
        background_color="#238636",
        hover_background_color="#2ea043",
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
    url_input = st.text_input("Source Image Link", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    if url_input:
        try:
            req = urllib.request.Request(url_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                image = Image.open(response).convert('RGB')
        except Exception:
            st.error("⚠️ Direct link access failed. Please verify the URL or try pasting the image directly.")

# ---------------------------------------------------------
# 5. Diagnostic Dashboard & Ensemble Analysis
# ---------------------------------------------------------
if image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Media Sample")
        st.image(image, use_container_width=True)
        st.markdown(f"<p style='color:#8b949e; font-size:0.85rem; text-align:center;'>Dimensions: {image.width} × {image.height}px</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Forensic Engine")
        st.write("Ready to analyze micro-texture spatial patterns across center and 4 corner high-res crops.")
        
        analyze_btn = st.button("🚀 Run Deep Forensic Inspection", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_btn:
            with st.spinner("Extracting spatial micro-features across Neural Ensemble..."):
                patches = five_crop(image)
                patch_tensors = torch.stack([base_transform(p) for p in patches])

                with torch.no_grad():
                    vit_probs = F.softmax(vit_model(patch_tensors), dim=1).mean(dim=0).tolist()
                    res_probs = F.softmax(resnet_model(patch_tensors), dim=1).mean(dim=0).tolist()

                avg_fake = (vit_probs[0] + res_probs[0]) / 2.0 * 100
                avg_real = (vit_probs[1] + res_probs[1]) / 2.0 * 100

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive Verdict Card
            if avg_fake > 60.0:
                st.markdown(f'<div class="verdict-fake">⚠️ Verdict: Synthetically Generated ({avg_fake:.1f}% Confidence)</div>', unsafe_allow_html=True)
            elif avg_real > 60.0:
                st.markdown(f'<div class="verdict-real">✅ Verdict: Authentic Photograph ({avg_real:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-uncertain">🤔 Verdict: Uncertain Analysis ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Neural Breakdown")
            
            # Neural Network Metric Breakdown
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### ViT-Tiny (Transformer)")
                st.progress(int(vit_probs[0] * 100), text=f"AI Artifacts: {vit_probs[0]*100:.1f}%")
                st.progress(int(vit_probs[1] * 100), text=f"Authentic: {vit_probs[1]*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("##### ResNet-18 (ConvNet)")
                st.progress(int(res_probs[0] * 100), text=f"AI Artifacts: {res_probs[0]*100:.1f}%")
                st.progress(int(res_probs[1] * 100), text=f"Authentic: {res_probs[1]*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
