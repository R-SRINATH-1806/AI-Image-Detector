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
st.write("Native micro-texture analysis using Vision Transformer & ResNet18 models.")

# ---------------------------------------------------------
# 2. Model Setup & Transforms
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            urllib.request.urlretrieve(url, file_path)

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
    
    return vit, resnet

vit_model, resnet_model = load_models()

# ---------------------------------------------------------
# 3. Input Options (Clipboard vs File Upload vs URL)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Paste from Clipboard", "📁 File Upload", "🔗 Image Web Link (URL)"])

image = None

with tab1:
    st.write("**Click the button below to grab an image copied to your clipboard:**")
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard",
        background_color="#FF4B4B",
        hover_background_color="#D33636",
    )
    if paste_result.image_data is not None:
        image = paste_result.image_data.convert('RGB')

with tab2:
    st.write("**Browse an image file from your computer:**")
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

with tab3:
    st.write("**Paste a direct web link ending in .jpg, .png, or .webp:**")
    url_input = st.text_input("Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    
    if url_input:
        try:
            req = urllib.request.Request(url_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                image = Image.open(response).convert('RGB')
        except Exception:
            st.error("⚠️ Unable to load image from this URL. Please verify the link is a direct image URL.")

# ---------------------------------------------------------
# 4. Multi-Patch Analysis & Results
# ---------------------------------------------------------
if image is not None:
    st.divider()
    
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(image, caption="Loaded Image", use_container_width=True)
        
    with col_info:
        st.write("### Ready to Analyze")
        analyze_btn = st.button("🚀 Analyze Native Micro-Textures", type="primary")

    if analyze_btn:
        with st.spinner("Analyzing 5 high-resolution patches across the image..."):
            
            patches = five_crop(image)
            patch_tensors = torch.stack([base_transform(p) for p in patches])

            with torch.no_grad():
                vit_outs = vit_model(patch_tensors)
                vit_probs_batch = F.softmax(vit_outs, dim=1)
                vit_avg_probs = vit_probs_batch.mean(dim=0).tolist()
                
                res_outs = resnet_model(patch_tensors)
                res_probs_batch = F.softmax(res_outs, dim=1)
                res_avg_probs = res_probs_batch.mean(dim=0).tolist()

            avg_fake = (vit_avg_probs[0] + res_avg_probs[0]) / 2.0 * 100
            avg_real = (vit_avg_probs[1] + res_avg_probs[1]) / 2.0 * 100

        st.divider()
        st.subheader("📊 Multi-Patch Detection Results")

        if avg_fake > 60.0:
            st.error(f"⚠️ **Verdict:** Likely AI-Generated ({avg_fake:.1f}% Confidence)")
        elif avg_real > 60.0:
            st.success(f"✅ **Verdict:** Likely Real Photo ({avg_real:.1f}% Confidence)")
        else:
            st.warning(f"🤔 **Verdict:** Uncertain / Natural Photo ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Vision Transformer (ViT)")
            st.progress(int(vit_avg_probs[0] * 100), text=f"AI/Fake: {vit_avg_probs[0]*100:.1f}%")
            st.progress(int(vit_avg_probs[1] * 100), text=f"Real: {vit_avg_probs[1]*100:.1f}%")
            
        with c2:
            st.markdown("#### ResNet-18 Architecture")
            st.progress(int(res_avg_probs[0] * 100), text=f"AI/Fake: {res_avg_probs[0]*100:.1f}%")
            st.progress(int(res_avg_probs[1] * 100), text=f"Real: {res_avg_probs[1]*100:.1f}%")
