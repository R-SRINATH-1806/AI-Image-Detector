import os
import json
import base64
from io import BytesIO
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import plotly.graph_objects as go
from streamlit_paste_button import paste_image_button
from transformers import pipeline

# ---------------------------------------------------------
# 1. Page Configuration & Cyber HUD Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MONOVISION // HOLO-FORENSICS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 10%, #0a0f1d 0%, #030712 100%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }

    .hud-banner {
        background: rgba(13, 19, 33, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-left: 4px solid #00f3ff;
        border-right: 4px solid #00f3ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.15);
    }

    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 50%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .verdict-fake {
        background: rgba(255, 0, 85, 0.15);
        border: 1.5px solid #ff0055;
        color: #ff3377;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.3rem;
    }

    .verdict-real {
        background: rgba(0, 255, 136, 0.15);
        border: 1.5px solid #00ff88;
        color: #00ff88;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.3rem;
    }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)

def render_cyber_header(text):
    st.markdown(f"""
    <h4 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem; margin-top: 15px;">
        {text}
    </h4>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Banner & Sidebar
# ---------------------------------------------------------
st.markdown("""
<div class="hud-banner">
    <div class="hud-title">MONOVISION v5.0</div>
    <div style="font-family: 'JetBrains Mono'; color: #00f3ff; letter-spacing: 2px;">// 3D HOLOGRAM PROJECTION & FORENSIC SUITE</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='font-family: Orbitron; color: #00f3ff;'>🛰️ PIPELINE</h3>", unsafe_allow_html=True)
    st.caption("• Model: Smogy AI Detector")
    st.caption("• 3D Engine: Plotly + WebGL Three.js")

# ---------------------------------------------------------
# 3. Neural Engine Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")

detector = load_detector()

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
# 4. 3D Render Engines (Native Plotly + WebGL Fallback)
# ---------------------------------------------------------
def create_plotly_3d_hologram(image_pil):
    """Generates a guaranteed native Python 3D hologram surface mesh."""
    img_small = image_pil.resize((120, 120))
    gray_np = np.array(img_small.convert('L'))
    color_np = np.array(img_small)
    
    x = np.linspace(0, 10, 120)
    y = np.linspace(0, 10, 120)
    X, Y = np.meshgrid(x, y)
    Z = gray_np / 25.5  # Elevation map
    
    colorscale = [
        [0.0, 'rgb(3,7,18)'],
        [0.5, 'rgb(0,243,255)'],
        [1.0, 'rgb(255,0,85)']
    ]
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale, opacity=0.85)])
    fig.update_layout(
        title="3D VOLUMETRIC METRIC SURFACE MESH",
        autosize=True,
        height=500,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='rgba(3,7,18,0)',
        plot_bgcolor='rgba(3,7,18,0)',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title='Density', backgroundcolor='rgba(3,7,18,0)'),
            aspectratio=dict(x=1, y=1, z=0.4)
        )
    )
    return fig

def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def render_3d_hologram_component(img_b64):
    """WebGL Three.js canvas with auto-resizing canvas listener."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #030712; }}
            #canvas-container {{ width: 100vw; height: 500px; position: relative; }}
            .holo-overlay {{
                position: absolute; top: 10px; left: 10px; color: #00f3ff;
                font-family: monospace; font-size: 12px; border: 1px solid #00f3ff;
                padding: 4px 8px; background: rgba(3,7,18,0.8); z-index: 10;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="holo-overlay">🛸 WebGL 3D HOLOGRAM // CLICK & DRAG TO ROTATE</div>
        </div>
        <script>
            const container = document.getElementById('canvas-container');
            const width = container.clientWidth || window.innerWidth;
            const height = 500;

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(0, 2, 6);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const img = new Image();
            img.src = '{img_b64}';
            img.onload = () => {{
                const texture = new THREE.Texture(img);
                texture.needsUpdate = true;
                
                const aspect = img.width / img.height;
                const geo = new THREE.PlaneGeometry(2.5, 2.5 / aspect, 16, 16);
                const mat = new THREE.MeshBasicMaterial({{
                    map: texture, side: THREE.DoubleSide, transparent: true,
                    opacity: 0.85, blending: THREE.AdditiveBlending
                }});
                
                const mesh = new THREE.Mesh(geo, mat);
                scene.add(mesh);
                
                const wireMat = new THREE.MeshBasicMaterial({{ color: 0x00f3ff, wireframe: true, transparent: true, opacity: 0.2 }});
                const wireMesh = new THREE.Mesh(geo, wireMat);
                wireMesh.position.z = 0.01;
                mesh.add(wireMesh);

                function animate() {{
                    requestAnimationFrame(animate);
                    mesh.rotation.y += 0.005;
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
            }};
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=520)

def generate_spatial_heatmap(image_pil):
    img_np = np.array(image_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    norm_mag = cv2.normalize(np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.4, 0)
    return Image.fromarray(overlay)

# ---------------------------------------------------------
# 5. Media Ingestion
# ---------------------------------------------------------
tab_upload, tab_paste = st.tabs(["📁 MEDIA UPLOAD", "📋 CLIPBOARD INGESTION"])
image = None

with tab_upload:
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

with tab_paste:
    paste_res = paste_image_button(label="📋 PASTE FROM CLIPBOARD", background_color="#0284c7")
    if paste_res.image_data:
        image = paste_res.image_data.convert('RGB')

# ---------------------------------------------------------
# 6. Analysis & Futuristic Visualization
# ---------------------------------------------------------
if image:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        with st.container(border=True):
            render_cyber_header("📷 INPUT FRAME")
            st.image(image, use_container_width=True)
            
    with col2:
        with st.container(border=True):
            render_cyber_header("⚙️ NEURAL DIAGNOSTICS")
            scan_btn = st.button("🚀 INITIATE FORENSIC SCAN", type="primary", use_container_width=True)

        if scan_btn and detector:
            with st.spinner("Extracting High-Dimensional Feature Maps..."):
                raw = detector(image)
                ai_score, real_score = parse_predictions(raw)

            st.markdown("<br>", unsafe_allow_html=True)
            if ai_score >= 50.0:
                st.markdown(f'<div class="verdict-fake">⚠️ VERDICT: AI-GENERATED ({ai_score:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-real">✅ VERDICT: AUTHENTIC PHOTOGRAPH ({real_score:.1f}%)</div>', unsafe_allow_html=True)

            render_cyber_header("📊 PROBABILITY MATRIX")
            st.progress(int(ai_score), text=f"AI / Deepfake Signature: {ai_score:.1f}%")
            st.progress(int(real_score), text=f"Authentic Optical Signature: {real_score:.1f}%")

            # Visual Forensic Modules
            render_cyber_header("🕵️ ADVANCED FUTURISTIC VISUALIZATIONS")
            
            t_plotly3d, t_webgl, t_heatmap = st.tabs([
                "🌋 NATIVE 3D HOLOGRAM MESH",
                "🛸 WebGL THREE.JS PROJECTION",
                "🎯 SPATIAL HEATMAP"
            ])
            
            with t_plotly3d:
                st.write("Native 3D Volumetric Topography (Interactive: Drag to tilt, rotate, and zoom):")
                fig_3d = create_plotly_3d_hologram(image)
                st.plotly_chart(fig_3d, use_container_width=True)

            with t_webgl:
                st.write("WebGL Hologram Projection:")
                img_b64 = image_to_base64(image)
                render_3d_hologram_component(img_b64)

            with t_heatmap:
                st.write("High-Frequency Neural Anomaly Heatmap:")
                st.image(generate_spatial_heatmap(image), use_container_width=True)
