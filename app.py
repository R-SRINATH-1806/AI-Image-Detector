import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (UI Styling)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 AI vs Real Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze image artifacts and micro-textures with ViT Tiny & ResNet18 models.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. GitHub Release Asset Links
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            urllib.request.urlretrieve(url, file_path)

# ---------------------------------------------------------
# 3. Model Loading & Preprocessing
# ---------------------------------------------------------
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models():
    download_file_if_missing("vit_highres_model.pth", VIT_URL)
    download_file_if_missing("resnet_highres_model.pth", RESNET_URL)
    
    # Load ViT
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load("vit_highres_model.pth", map_location=device, weights_only=True))
    vit.eval()
    
    # Load ResNet18
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load("resnet_highres_model.pth", map_location=device, weights_only=True))
    resnet.eval()
    
    return vit, resnet

vit_model, resnet_model = load_models()

# ---------------------------------------------------------
# 4. Upload & Paste Interface
# ---------------------------------------------------------
st.info("💡 **How to input an image:** Click the area below to browse files **OR** press `Ctrl+V` (`Cmd+V` on Mac) while focusing on the box to paste a copied image directly from your clipboard!")

uploaded_file = st.file_uploader(
    "Upload or Paste Image", 
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Display preview image
    image = Image.open(uploaded_file).convert('RGB')
    
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Input Preview", use_container_width=True)
    with col_info:
        st.write("### Image Details")
        st.write(f"**Format:** {image.format if image.format else 'Clipboard Image'}")
        st.write(f"**Dimensions:** {image.width} x {image.height} px")
        analyze_btn = st.button("🚀 Analyze Image", type="primary")

    if analyze_btn:
        with st.spinner("Running AI detection models..."):
            img_tensor = transform(image).unsqueeze(0)

            # ViT Prediction
            with torch.no_grad():
                vit_out = vit_model(img_tensor)
                vit_probs = F.softmax(vit_out, dim=1).squeeze().tolist()
                
            # ResNet Prediction
            with torch.no_grad():
                res_out = resnet_model(img_tensor)
                res_probs = F.softmax(res_out, dim=1).squeeze().tolist()

            # Ensemble Prediction (Average of both)
            avg_fake = (vit_probs[0] + res_probs[0]) / 2.0 * 100
            avg_real = (vit_probs[1] + res_probs[1]) / 2.0 * 100

        st.divider()
        st.subheader("📊 Model Predictions")

        # Top Metric Banner
        if avg_fake > avg_real:
            st.error(f"⚠️ **Verdict:** Likely AI-Generated ({avg_fake:.1f}% Confidence)")
        else:
            st.success(f"✅ **Verdict:** Likely Real Photo ({avg_real:.1f}% Confidence)")

        # Detailed Model Breakdowns
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Vision Transformer (ViT)")
            st.progress(int(vit_probs[0] * 100), text=f"AI/Fake: {vit_probs[0]*100:.1f}%")
            st.progress(int(vit_probs[1] * 100), text=f"Real: {vit_probs[1]*100:.1f}%")
            
        with c2:
            st.markdown("#### ResNet-18 Architecture")
            st.progress(int(res_probs[0] * 100), text=f"AI/Fake: {res_probs[0]*100:.1f}%")
            st.progress(int(res_probs[1] * 100), text=f"Real: {res_probs[1]*100:.1f}%")
