import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image

st.set_page_config(page_title="AI vs Real Image Detector", page_icon="🔍", layout="centered")
st.title("🔍 AI vs Real Image Detector")
st.write("Native micro-texture analysis using Vision Transformer & ResNet18.")

# Model Setup
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            urllib.request.urlretrieve(url, file_path)

device = torch.device("cpu")

# 1. Base transform for individual 224x224 crops
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Extract 5 patches at native resolution (Center + 4 Corners)
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

# Input UI
uploaded_file = st.file_uploader("Upload or Paste Image (Ctrl+V)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Ensure image is at least 224x224 before cropping
    if image.width < 224 or image.height < 224:
        image = image.resize((max(224, image.width), max(224, image.height)))
        
    st.image(image, caption="Loaded Image", use_container_width=True)
    
    if st.button("🚀 Analyze Native Micro-Textures", type="primary"):
        with st.spinner("Analyzing 5 high-resolution patches across the image..."):
            
            # Extract 5 native patches
            patches = five_crop(image)
            # Batch them together into tensor shape [5, 3, 224, 224]
            patch_tensors = torch.stack([base_transform(p) for p in patches])

            with torch.no_grad():
                # Get ViT predictions across all 5 patches and average them
                vit_outs = vit_model(patch_tensors)
                vit_probs_batch = F.softmax(vit_outs, dim=1)
                vit_avg_probs = vit_probs_batch.mean(dim=0).tolist() # [Fake_prob, Real_prob]
                
                # Get ResNet predictions across all 5 patches and average them
                res_outs = resnet_model(patch_tensors)
                res_probs_batch = F.softmax(res_outs, dim=1)
                res_avg_probs = res_probs_batch.mean(dim=0).tolist()

            # Combined Ensemble Score
            avg_fake = (vit_avg_probs[0] + res_avg_probs[0]) / 2.0 * 100
            avg_real = (vit_avg_probs[1] + res_avg_probs[1]) / 2.0 * 100

        st.divider()
        st.subheader("📊 Multi-Patch Detection Results")

        # Set a realistic decision threshold (Requires >60% confidence to call AI)
        if avg_fake > 60.0:
            st.error(f"⚠️ **Verdict:** Likely AI-Generated ({avg_fake:.1f}% Confidence)")
        elif avg_real > 60.0:
            st.success(f"✅ **Verdict:** Likely Real Photo ({avg_real:.1f}% Confidence)")
        else:
            st.warning(f"🤔 **Verdict:** Uncertain / Natural Photo with Smooth Lighting ({avg_real:.1f}% Real / {avg_fake:.1f}% AI)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Vision Transformer (ViT)")
            st.progress(int(vit_avg_probs[0] * 100), text=f"AI/Fake: {vit_avg_probs[0]*100:.1f}%")
            st.progress(int(vit_avg_probs[1] * 100), text=f"Real: {vit_avg_probs[1]*100:.1f}%")
            
        with c2:
            st.markdown("#### ResNet-18 Architecture")
            st.progress(int(res_avg_probs[0] * 100), text=f"AI/Fake: {res_avg_probs[0]*100:.1f}%")
            st.progress(int(res_avg_probs[1] * 100), text=f"Real: {res_avg_probs[1]*100:.1f}%")
