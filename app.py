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
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI vs Real Image Detector")
st.write("3-Model Ensemble (ViT Base + ResNet18 + ViT Landmark) with 5-crop micro-texture analysis.")

# ---------------------------------------------------------
# 2. Exact Model URLs & Download Utility
# ---------------------------------------------------------
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"
LANDMARK_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_landmark_model.1.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())

device = torch.device("cpu")

# Transform pipelines for 5-crop analysis
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

five_crop = transforms.FiveCrop(224)

# ---------------------------------------------------------
# 3. Load Models (With Memory Cleanup & Architecture Fix)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    download_file_if_missing("vit_highres_model.pth", VIT_URL)
    download_file_if_missing("resnet_highres_model.pth", RESNET_URL)
    download_file_if_missing("vit_landmark_model.1.pth", LANDMARK_URL)
    
    # Model 1: ViT-Tiny (192-dim)
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load("vit_highres_model.pth", map_location=device, weights_only=True))
    vit.eval()
    
    # Model 2: ResNet-18
    resnet = models.resnet18(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load("resnet_highres_model.pth", map_location=device, weights_only=True))
    resnet.eval()

    # Model 3: ViT-Base (768-dim) for the Landmark Model
    vit_landmark = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    
    raw_checkpoint = torch.load("vit_landmark_model.1.pth", map_location=device)
    
    # Extract inner state dict if wrapped in a dictionary
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        state_dict = raw_checkpoint["state_dict"]
    elif isinstance(raw_checkpoint, dict) and "model" in raw_checkpoint:
        state_dict = raw_checkpoint["model"]
    else:
        state_dict = raw_checkpoint

    # Strip 'module.' prefix if trained using PyTorch DataParallel
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    vit_landmark.load_state_dict(clean_state_dict, strict=False)
    vit_landmark.eval()
    
    # Free up unused RAM so Streamlit Cloud does not crash
    del raw_checkpoint, state_dict, clean_state_dict
    gc.collect()
    
    return vit, resnet, vit_landmark

vit_model, resnet_model, landmark_model = load_models()

# ---------------------------------------------------------
# 4. Input Interface (Clipboard, File, URL)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Paste from Clipboard", "📁 File Upload", "🔗 Image Web Link (URL)"])

image = None

with tab1:
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard",
        background_color="#FF4B4B",
        hover_background_color="#D33636",
    )
    if paste_result.image_data is not None:
        image = paste_result.image_data.convert('RGB')

with tab2:
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

with tab3:
    url_input = st.text_input("Image URL", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    if url_input:
        try:
            req = urllib.request.Request(url_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                image = Image.open(response).convert('RGB')
        except Exception:
            st.error("⚠️ Unable to load image from this URL. Please ensure it's a direct link.")

# ---------------------------------------------------------
# 5. Multi-Model Inference & Results Analysis
# ---------------------------------------------------------
if image is not None:
    st.divider()
    
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        # Fixed the use_container_width warning here by using width="stretch"
        st.image(image, caption="Loaded Image", width="stretch")
        
    with col_info:
        st.write("### Ready to Analyze")
        analyze_btn = st.button("🚀 Analyze Micro-Textures", type="primary")

    if analyze_btn:
        with st.spinner("Analyzing 5 high-resolution crops across 3 Neural Networks..."):
            
            # Extract Center + 4 Corners patches
            patches = five_crop(image)
            patch_tensors = torch.stack([base_transform(p) for p in patches])

            with torch.no_grad():
                # Model 1: ViT Base
                vit_probs = F.softmax(vit_model(patch_tensors), dim=1).mean(dim=0).tolist()
                
                # Model 2: ResNet-18
                res_probs = F.softmax(resnet_model(patch_tensors), dim=1).mean(dim=0).tolist()
                
                # Model 3: ViT Landmark
                lmark_probs = F.softmax(landmark_model(patch_tensors), dim=1).mean(dim=0).tolist()

            # Combined Ensemble Average across all 3 models
            avg_fake = (vit_probs[0] + res_probs[0] + lmark_probs[0]) / 3.0 * 100
            avg_real = (vit_probs[1] + res_probs[1] + lmark_probs[1]) / 3.0 * 100

        st.divider()
        st.subheader("📊 Ensemble Detection Results")

        # Threshold Decision Check
        if avg_fake > 60.0:
            st.error(f"⚠️ **Verdict:** Likely AI-Generated ({avg_fake:.1f}% Confidence)")
        elif avg_real > 60.0:
            st.success(f"✅ **Verdict:** Likely Real Photo ({avg_real:.1f}% Confidence)")
        else:
            st.warning(f"🤔 **Verdict:** Uncertain / Natural Photo ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)")

        # Individual Breakdown per Neural Network
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### ViT Tiny")
            st.progress(int(vit_probs[0] * 100), text=f"AI: {vit_probs[0]*100:.1f}%")
            st.progress(int(vit_probs[1] * 100), text=f"Real: {vit_probs[1]*100:.1f}%")
            
        with c2:
            st.markdown("#### ResNet-18")
            st.progress(int(res_probs[0] * 100), text=f"AI: {res_probs[0]*100:.1f}%")
            st.progress(int(res_probs[1] * 100), text=f"Real: {res_probs[1]*100:.1f}%")

        with c3:
            st.markdown("#### ViT Landmark")
            st.progress(int(lmark_probs[0] * 100), text=f"AI: {lmark_probs[0]*100:.1f}%")
            st.progress(int(lmark_probs[1] * 100), text=f"Real: {lmark_probs[1]*100:.1f}%")
