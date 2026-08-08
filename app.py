import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from PIL import Image

# 1. Page Setup
st.set_page_config(page_title="AI vs Real Image Detector", page_icon="🚀")
st.title("Ultimate AI vs. Real Image Detector 🚀")
st.write("Upload a high-resolution image! Our models evaluate native textures without resolution limits.")

# 2. GitHub Release Asset Direct Links (Replace YOUR_USERNAME if needed)
VIT_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/vit_highres_model.pth"
RESNET_URL = "https://github.com/R-SRINATH-1806/AI-Image-Detector/releases/download/v1.0/resnet_highres_model.pth"

# Helper function to download model files if missing
def download_file_if_missing(file_path, url):
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading model weights ({file_path})..."):
            urllib.request.urlretrieve(url, file_path)

# 3. Setup Device and Transforms
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Load Models (Cached so downloading and loading only happen once)
@st.cache_resource
def load_models():
    # Download weights from Release assets if not present
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

# 5. UI and Prediction Logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("AI is analyzing the image..."):
            img_tensor = transform(image).unsqueeze(0)

            # ViT Prediction
            with torch.no_grad():
                vit_out = vit_model(img_tensor)
                vit_probs = F.softmax(vit_out, dim=1).squeeze().tolist()
                
            # ResNet Prediction
            with torch.no_grad():
                res_out = resnet_model(img_tensor)
                res_probs = F.softmax(res_out, dim=1).squeeze().tolist()

            # Results Display
            st.markdown("### 📊 Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**ViT Tiny Verdict**")
                st.write(f"**FAKE:** {vit_probs[0]*100:.2f}%")
                st.write(f"**REAL:** {vit_probs[1]*100:.2f}%")
                
            with col2:
                st.success("**ResNet18 Verdict**")
                st.write(f"**FAKE:** {res_probs[0]*100:.2f}%")
                st.write(f"**REAL:** {res_probs[1]*100:.2f}%")
