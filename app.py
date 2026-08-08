import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image
from streamlit_paste_button import paste_image_button

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI vs Real Image Detector")
st.write("Detect if an image is real or AI-generated using ViT Tiny & ResNet18 models.")

# ---------------------------------------------------------
# 2. Model Setup
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            urllib.request.urlretrieve(url, file_path)

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
    
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load("vit_highres_model.pth", map_location=device, weights_only=True))
    vit.eval()
    
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load("resnet_highres_model.pth", map_location=device, weights_only=True))
    resnet.eval()
    
    return vit, resnet

vit_model, resnet_model = load_models()

# ---------------------------------------------------------
# 3. Input Options (Clipboard Paste vs File Upload)
# ---------------------------------------------------------
st.markdown("### 📥 Choose Image Input")

col_paste, col_upload = st.columns(2)

image = None

with col_paste:
    st.write("**Option A: Clipboard**")
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard",
        background_color="#FF4B4B",
        hover_background_color="#D33636",
    )
    if paste_result.image_data is not None:
        image = paste_result.image_data.convert('RGB')

with col_upload:
    st.write("**Option B: Upload File**")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

# ---------------------------------------------------------
# 4. Analysis & Results
# ---------------------------------------------------------
if image is not None:
    st.divider()
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(image, caption="Selected Image", use_container_width=True)
        
    with col_info:
        st.write("### Ready to Analyze")
        analyze_btn = st.button("🚀 Analyze Image Now", type="primary")

    if analyze_btn:
        with st.spinner("AI is analyzing image details..."):
            img_tensor = transform(image).unsqueeze(0)

            # ViT Prediction
            with torch.no_grad():
                vit_out = vit_model(img_tensor)
                vit_probs = F.softmax(vit_out, dim=1).squeeze().tolist()
                
            # ResNet Prediction
            with torch.no_grad():
                res_out = resnet_model(img_tensor)
                res_probs = F.softmax(res_out, dim=1).squeeze().tolist()

            avg_fake = (vit_probs[0] + res_probs[0]) / 2.0 * 100
            avg_real = (vit_probs[1] + res_probs[1]) / 2.0 * 100

        st.divider()
        st.subheader("📊 Detection Results")

        if avg_fake > avg_real:
            st.error(f"⚠️ **Verdict:** Likely AI-Generated ({avg_fake:.1f}% Confidence)")
        else:
            st.success(f"✅ **Verdict:** Likely Real Photo ({avg_real:.1f}% Confidence)")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Vision Transformer (ViT)")
            st.progress(int(vit_probs[0] * 100), text=f"AI/Fake: {vit_probs[0]*100:.1f}%")
            st.progress(int(vit_probs[1] * 100), text=f"Real: {vit_probs[1]*100:.1f}%")
            
        with c2:
            st.markdown("#### ResNet-18 Architecture")
            st.progress(int(res_probs[0] * 100), text=f"AI/Fake: {res_probs[0]*100:.1f}%")
            st.progress(int(res_probs[1] * 100), text=f"Real: {res_probs[1]*100:.1f}%")
