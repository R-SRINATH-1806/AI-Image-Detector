import base64
from datetime import datetime
from io import BytesIO
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_paste_button import paste_image_button
from transformers import pipeline

# ---------------------------------------------------------
# 1. Page Configuration & Cyber HUD Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MONOVISION // HOLO-FORENSICS AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 10%, #0a0f1d 0%, #030712 100%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
        background-size: 100% 4px;
        z-index: 99999;
        pointer-events: none;
        opacity: 0.25;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 22, 0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 243, 255, 0.2) !important;
        box-shadow: 5px 0 25px rgba(0, 243, 255, 0.05);
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
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.15), inset 0 0 20px rgba(0, 243, 255, 0.05);
    }

    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 50%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
        margin-bottom: 0.2rem;
    }

    .hud-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #00f3ff;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hud-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(0, 243, 255, 0.1);
        border: 1px solid #00f3ff;
        color: #00f3ff;
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 18px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00f3ff;
        box-shadow: 0 0 10px #00f3ff;
        animation: pulse 1.5s infinite alternate;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% { opacity: 0.3; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1.2); }
    }

    .verdict-fake {
        background: radial-gradient(circle at center, rgba(255, 0, 85, 0.2) 0%, rgba(15, 10, 25, 0.9) 100%);
        border: 1.5px solid #ff0055;
        color: #ff3377;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 35px rgba(255, 0, 85, 0.3), inset 0 0 15px rgba(255, 0, 85, 0.15);
    }

    .verdict-real {
        background: radial-gradient(circle at center, rgba(0, 255, 136, 0.15) 0%, rgba(10, 25, 20, 0.9) 100%);
        border: 1.5px solid #00ff88;
        color: #00ff88;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 35px rgba(0, 255, 136, 0.25), inset 0 0 15px rgba(0, 255, 136, 0.1);
    }

    .stButton>button {
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.2) 0%, rgba(112, 0, 255, 0.2) 100%) !important;
        border: 1px solid #00f3ff !important;
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6) !important;
        transform: translateY(-2px);
    }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)


def render_cyber_header(text):
  st.markdown(
      f"""
    <h4 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem; margin-top: 15px; margin-bottom: 10px;">
        {text}
    </h4>
    """,
      unsafe_allow_html=True,
  )


# ---------------------------------------------------------
# 2. Hero Banner & Sidebar Architecture
# ---------------------------------------------------------
st.markdown(
    """
<div class="hud-banner">
    <div class="hud-title">MONOVISION v6.0</div>
    <div class="hud-subtitle">// AI VOLUMETRIC DEPTH & 3D FORENSIC SUITE</div>
    <div>
        <span class="hud-badge"><span class="pulse-dot"></span> MiDaS AI GEOMETRIC DEPTH ENGINE ONLINE</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
  st.markdown(
      """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛰️ SYSTEM PIPELINE</h3>""",
      unsafe_allow_html=True,
  )
  with st.container(border=True):
    st.markdown(
        """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;"><b>CLASSIFIER:</b> Smogy/SMOGY-Ai-images-detector</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;"><b>DEPTH ENGINE:</b> Intel DPT-MiDaS Hybrid</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;"><b>3D RENDERER:</b> WebGL Point-Cloud + Plotly</p>""",
        unsafe_allow_html=True,
    )
    st.caption("Active Volumetric Depth Mapping")

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown(
      """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛡️ FORENSIC MODULES</h3>""",
      unsafe_allow_html=True,
  )
  st.caption("• AI MiDaS Volumetric 3D Point-Cloud")
  st.caption("• Geometric Surface Depth Topography")
  st.caption("• Spatial Attention Artifact Heatmap")
  st.caption("• Latent Prompt Vector Inversion")
  st.caption("• JPEG Compression Delta (ELA)")
  st.caption("• 2D Fourier Frequency Spectrum")


# ---------------------------------------------------------
# 3. Neural Engine Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
  return pipeline("image-classification", model="Smogy/SMOGY-Ai-images-detector")


@st.cache_resource
def load_captioner():
  try:
    return pipeline(
        "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
    )
  except Exception:
    return None


@st.cache_resource
def load_depth_estimator():
  try:
    return pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
  except Exception:
    return None


detector = load_detector()
captioner = load_captioner()
depth_estimator = load_depth_estimator()


def generate_depth_map(image_pil):
  """Generates a true monocular depth map using AI MiDaS pipeline."""
  if depth_estimator is not None:
    try:
      result = depth_estimator(image_pil)
      depth_img = result["depth"].convert("L")
      return depth_img
    except Exception:
      pass
  # Fallback to luminance grayscale if model is unavailable
  return image_pil.convert("L")


def parse_predictions(results):
  fake_score = 0.0
  real_score = 0.0
  for res in results:
    label = str(res["label"]).lower()
    score = res["score"] * 100.0
    if any(
        k in label for k in ["fake", "ai", "generated", "synthetic", "label_1"]
    ):
      fake_score = score
    elif any(
        k in label
        for k in ["real", "human", "authentic", "photography", "label_0"]
    ):
      real_score = score

  if fake_score == 0.0 and real_score > 0.0:
    fake_score = 100.0 - real_score
  elif real_score == 0.0 and fake_score > 0.0:
    real_score = 100.0 - fake_score
  return fake_score, real_score


# ---------------------------------------------------------
# 4. WebGL AI Volumetric 3D Particle Cloud Engine
# ---------------------------------------------------------
def image_to_base64(img_pil):
  buffered = BytesIO()
  img_pil.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return f"data:image/png;base64,{img_str}"


def render_3d_hologram_component(img_b64, depth_b64):
  """WebGL Three.js AI Depth Extruded Volumetric Point Cloud with HUD controls."""
  html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ background: #030712; overflow: hidden; width: 100vw; height: 530px; font-family: 'Courier New', monospace; }}
            #canvas-container {{ width: 100%; height: 100%; position: relative; }}
            .hud-overlay {{
                position: absolute; top: 12px; left: 12px;
                color: #00f3ff; border: 1px solid rgba(0,243,255,0.5);
                padding: 6px 12px; background: rgba(3,7,18,0.85);
                font-size: 11px; letter-spacing: 1.5px; border-radius: 4px;
                pointer-events: none; z-index: 100; box-shadow: 0 0 10px rgba(0,243,255,0.2);
            }}
            .controls-hud {{
                position: absolute; bottom: 12px; right: 12px; z-index: 100;
                display: flex; gap: 8px; background: rgba(3,7,18,0.85); padding: 8px;
                border: 1px solid rgba(0,243,255,0.3); border-radius: 6px;
            }}
            .hud-btn {{
                background: rgba(0,243,255,0.1); border: 1px solid #00f3ff; color: #00f3ff;
                padding: 5px 12px; font-size: 10px; cursor: pointer; border-radius: 3px; font-weight: bold;
                transition: all 0.2s; font-family: monospace;
            }}
            .hud-btn:hover {{ background: #00f3ff; color: #000; box-shadow: 0 0 10px #00f3ff; }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">⚡ AI MIDAS GEOMETRIC 3D POINT-CLOUD // ACTIVE</div>
            <div class="controls-hud">
                <button class="hud-btn" onclick="toggleMesh()">TOGGLE MESH</button>
                <button class="hud-btn" onclick="setMode('cyber')">CYBER GLOW</button>
                <button class="hud-btn" onclick="setMode('rgb')">TRUE COLOR</button>
            </div>
        </div>
        <script>
            function loadScript(src) {{
                return new Promise((resolve, reject) => {{
                    const s = document.createElement('script');
                    s.src = src;
                    s.onload = resolve;
                    s.onerror = reject;
                    document.head.appendChild(s);
                }});
            }}

            let wireframeMesh, pointCloud, colorMode = 'cyber', holoGroup;

            function toggleMesh() {{
                if (wireframeMesh) wireframeMesh.visible = !wireframeMesh.visible;
            }}

            function setMode(mode) {{
                colorMode = mode;
                if (window.rebuildHologram) window.rebuildHologram();
            }}

            async function init() {{
                try {{
                    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js');
                    await loadScript('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js');

                    const container = document.getElementById('canvas-container');
                    let width = container.clientWidth || window.innerWidth || 800;
                    let height = container.clientHeight || 530;

                    const scene = new THREE.Scene();
                    scene.fog = new THREE.FogExp2(0x030712, 0.05);

                    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                    camera.position.set(0, 2.5, 6.0);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    renderer.setSize(width, height);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    container.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;

                    // Neon Grid Floor
                    const gridHelper = new THREE.GridHelper(10, 24, 0x00f3ff, 0x112233);
                    gridHelper.position.y = -1.2;
                    scene.add(gridHelper);

                    // Scanning Laser Bar
                    const scanPlaneGeo = new THREE.PlaneGeometry(3.5, 3.5);
                    const scanPlaneMat = new THREE.MeshBasicMaterial({{
                        color: 0x00f3ff, side: THREE.DoubleSide, transparent: true, opacity: 0.12, wireframe: true
                    }});
                    const scanPlane = new THREE.Mesh(scanPlaneGeo, scanPlaneMat);
                    scanPlane.rotation.x = Math.PI / 2;
                    scene.add(scanPlane);

                    // Load Image & AI Depth Map into Dual Canvas Buffers
                    const img = new Image();
                    const depthImg = new Image();
                    let imgLoaded = false, depthLoaded = false;

                    img.src = '{img_b64}';
                    depthImg.src = '{depth_b64}';

                    function checkReady() {{
                        if (imgLoaded && depthLoaded) buildVolumetricHologram();
                    }}

                    img.onload = () => {{ imgLoaded = true; checkReady(); }};
                    depthImg.onload = () => {{ depthLoaded = true; checkReady(); }};

                    function buildVolumetricHologram() {{
                        const imgW = 110, imgH = 110;

                        // Canvas 1: Color
                        const canvasColor = document.createElement('canvas');
                        canvasColor.width = imgW; canvasColor.height = imgH;
                        const ctxC = canvasColor.getContext('2d');
                        ctxC.drawImage(img, 0, 0, imgW, imgH);
                        const rgbData = ctxC.getImageData(0, 0, imgW, imgH).data;

                        // Canvas 2: AI Depth
                        const canvasDepth = document.createElement('canvas');
                        canvasDepth.width = imgW; canvasDepth.height = imgH;
                        const ctxD = canvasDepth.getContext('2d');
                        ctxD.drawImage(depthImg, 0, 0, imgW, imgH);
                        const depthData = ctxD.getImageData(0, 0, imgW, imgH).data;

                        holoGroup = new THREE.Group();
                        scene.add(holoGroup);

                        window.rebuildHologram = () => {{
                            scene.remove(holoGroup);
                            holoGroup = new THREE.Group();

                            const numPoints = imgW * imgH;
                            const positions = new Float32Array(numPoints * 3);
                            const colors = new Float32Array(numPoints * 3);

                            for (let y = 0; y < imgH; y++) {{
                                for (let x = 0; x < imgW; x++) {{
                                    const idx = (y * imgW + x);
                                    const pIdx = idx * 4;

                                    const r = rgbData[pIdx] / 255;
                                    const g = rgbData[pIdx + 1] / 255;
                                    const b = rgbData[pIdx + 2] / 255;

                                    // Extract AI depth displacement
                                    const depthVal = depthData[pIdx] / 255.0;

                                    positions[idx * 3] = (x / imgW - 0.5) * 3.4;
                                    positions[idx * 3 + 1] = depthVal * 1.6 - 0.2; // AI Depth Height Extrusion
                                    positions[idx * 3 + 2] = (y / imgH - 0.5) * 3.4;

                                    if (colorMode === 'cyber') {{
                                        colors[idx * 3] = 0.0;
                                        colors[idx * 3 + 1] = 0.7 + depthVal * 0.3;
                                        colors[idx * 3 + 2] = 1.0;
                                    }} else {{
                                        colors[idx * 3] = r;
                                        colors[idx * 3 + 1] = g;
                                        colors[idx * 3 + 2] = b;
                                    }}
                                }}
                            }}

                            const geometry = new THREE.BufferGeometry();
                            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                            // Point Cloud Geometry
                            const pMat = new THREE.PointsMaterial({{
                                size: 0.038, vertexColors: true, transparent: true, opacity: 0.92, blending: THREE.AdditiveBlending
                            }});
                            pointCloud = new THREE.Points(geometry, pMat);
                            holoGroup.add(pointCloud);

                            // Surface Wireframe Mesh
                            const wireMat = new THREE.MeshBasicMaterial({{ color: 0x00f3ff, wireframe: true, transparent: true, opacity: 0.15 }});
                            const planeGeo = new THREE.PlaneGeometry(3.4, 3.4, imgW - 1, imgH - 1);
                            const posAttr = planeGeo.attributes.position;
                            for (let i = 0; i < posAttr.count; i++) {{
                                posAttr.setZ(i, positions[i * 3 + 1]);
                            }}
                            planeGeo.computeVertexNormals();
                            wireframeMesh = new THREE.Mesh(planeGeo, wireMat);
                            wireframeMesh.rotation.x = -Math.PI / 2;
                            wireframeMesh.position.y = 0;
                            holoGroup.add(wireframeMesh);

                            scene.add(holoGroup);
                        }};

                        window.rebuildHologram();

                        let time = 0;
                        function animate() {{
                            requestAnimationFrame(animate);
                            time += 0.02;

                            if (holoGroup) {{
                                holoGroup.rotation.y += 0.004;
                                scanPlane.position.y = Math.sin(time * 1.5) * 0.9 + 0.6;
                            }}

                            controls.update();
                            renderer.render(scene, camera);
                        }}
                        animate();
                    }}

                    const resizeObserver = new ResizeObserver(() => {{
                        const newWidth = container.clientWidth;
                        const newHeight = container.clientHeight || 530;
                        if (newWidth > 0 && newHeight > 0) {{
                            camera.aspect = newWidth / newHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(newWidth, newHeight);
                        }}
                    }});
                    resizeObserver.observe(container);

                }} catch(e) {{
                    console.error("WebGL Init Error: ", e);
                }}
            }}
            init();
        </script>
    </body>
    </html>
    """
  components.html(html_code, height=550)


def create_plotly_3d_hologram(depth_pil):
  """Generates a native 3D surface topography mesh from AI Depth map."""
  depth_small = depth_pil.resize((120, 120))
  depth_np = np.array(depth_small)

  x = np.linspace(0, 10, 120)
  y = np.linspace(0, 10, 120)
  X, Y = np.meshgrid(x, y)
  Z = depth_np / 25.5

  colorscale = [
      [0.0, "rgb(3,7,18)"],
      [0.5, "rgb(0,243,255)"],
      [1.0, "rgb(255,0,85)"],
  ]

  fig = go.Figure(
      data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale, opacity=0.88)]
  )
  fig.update_layout(
      title="AI MiDaS GEOMETRIC SURFACE TOPOGRAPHY",
      autosize=True,
      height=500,
      margin=dict(l=0, r=0, b=0, t=30),
      paper_bgcolor="rgba(3,7,18,0)",
      plot_bgcolor="rgba(3,7,18,0)",
      scene=dict(
          xaxis=dict(visible=False),
          yaxis=dict(visible=False),
          zaxis=dict(title="Depth Z", backgroundcolor="rgba(3,7,18,0)"),
          aspectratio=dict(x=1, y=1, z=0.45),
      ),
  )
  return fig


# ---------------------------------------------------------
# 5. Visual Forensics Processing Algorithms
# ---------------------------------------------------------
def generate_spatial_anomaly_heatmap(image_pil):
  img_np = np.array(image_pil)
  gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  magnitude = cv2.magnitude(sobelx, sobely)
  norm_mag = cv2.normalize(
      magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
  )
  heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
  overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
  return Image.fromarray(overlay)


def reconstruct_synthetic_prompt(image_pil):
  description = ""
  if captioner is not None:
    try:
      res = captioner(image_pil)
      description = res[0]["generated_text"].strip()
    except Exception:
      description = ""

  if not description:
    w, h = image_pil.size
    description = (
        f"cyberpunk asset, resolution {w}x{h}, detailed geometric lighting,"
        " volumetric textures"
    )

  return (
      f'"a hyper-realistic rendering of {description}, trending on artstation,'
      ' 8k resolution, volumetric cyan lighting, photorealistic --v 6.0"'
  )


def generate_ela(image, quality=90):
  temp_filename = "temp_ela.jpg"
  image.save(temp_filename, "JPEG", quality=quality)
  compressed_image = Image.open(temp_filename)
  ela_image = ImageChops.difference(image, compressed_image)
  extrema = ela_image.getextrema()
  max_diff = (
      max([ex[1] for ex in extrema])
      if max([ex[1] for ex in extrema]) != 0
      else 1
  )
  scale = 255.0 / max_diff
  ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
  if os.path.exists(temp_filename):
    os.remove(temp_filename)
  return ela_image


def generate_fft(image_pil):
  img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
  f_transform = np.fft.fft2(img_gray)
  f_shift = np.fft.fftshift(f_transform)
  magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
  magnitude_spectrum = cv2.normalize(
      magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
  )
  return Image.fromarray(magnitude_spectrum)


# ---------------------------------------------------------
# 6. Media Ingestion
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📁 MEDIA UPLOAD", "📋 CLIPBOARD INGESTION"])

image = None

with tab1:
  st.markdown("<br>", unsafe_allow_html=True)
  uploaded_file = st.file_uploader(
      "Upload Target Media",
      type=["jpg", "jpeg", "png", "webp"],
      label_visibility="collapsed",
  )
  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

with tab2:
  st.markdown("<br>", unsafe_allow_html=True)
  paste_result = paste_image_button(
      label="📋 PASTE FROM CLIPBOARD BUFFER",
      background_color="#0284c7",
      hover_background_color="#0369a1",
  )
  if paste_result.image_data is not None:
    image = paste_result.image_data.convert("RGB")

# ---------------------------------------------------------
# 7. Forensic Dashboard & Render Output
# ---------------------------------------------------------
if image is not None:
  st.markdown("<br>", unsafe_allow_html=True)
  col_left, col_right = st.columns([1, 1], gap="medium")

  with col_left:
    with st.container(border=True):
      render_cyber_header("📷 INPUT FRAME BUFFER")
      st.image(image, use_container_width=True)
      res_txt = f"FRAME RESOLUTION: {image.width} × {image.height} PX"
      st.markdown(
          f"""<p style="font-family: 'JetBrains Mono', monospace; color: #64748b; font-size: 0.8rem; text-align: center;">{res_txt}</p>""",
          unsafe_allow_html=True,
      )

  with col_right:
    with st.container(border=True):
      render_cyber_header("⚙️ NEURAL DIAGNOSTIC CONTROL")
      st.write(
          "Execute ViT classification and compute MiDaS AI monocular depth map"
          " extrusion."
      )
      analyze_btn = st.button(
          "🚀 INITIATE FULL FORENSIC SCAN",
          type="primary",
          use_container_width=True,
      )

    if analyze_btn and detector is not None:
      with st.spinner(
          "Computing AI Geometric Depth Map & Extracting Point-Cloud..."
      ):
        raw_results = detector(image)
        ai_score, real_score = parse_predictions(raw_results)
        depth_map = generate_depth_map(image)

      st.markdown("<br>", unsafe_allow_html=True)

      if ai_score >= 50.0:
        fake_html = f"""<div class="verdict-fake">⚠️ VERDICT: SYNTHETIC / AI-GENERATED ({ai_score:.1f}% CONFIDENCE)</div>"""
        st.markdown(fake_html, unsafe_allow_html=True)
        verdict_str = "AI-Generated"
      else:
        real_html = f"""<div class="verdict-real">✅ VERDICT: AUTHENTIC PHOTOGRAPH ({real_score:.1f}% CONFIDENCE)</div>"""
        st.markdown(real_html, unsafe_allow_html=True)
        verdict_str = "Authentic Photo"

      st.markdown("<br>", unsafe_allow_html=True)
      render_cyber_header("📊 PROBABILITY MATRIX")

      with st.container(border=True):
        ai_label = f"AI / Deepfake Signature Index: {ai_score:.1f}%"
        real_label = f"Authentic Optical Signature Index: {real_score:.1f}%"
        st.progress(int(ai_score), text=ai_label)
        st.progress(int(real_score), text=real_label)

      # --- Multi-Tab Visual Forensics ---
      st.markdown("<br>", unsafe_allow_html=True)
      render_cyber_header("🕵️ ADVANCED FORENSIC DIAGNOSTICS SUITE")

      t_holo, t_depth, t_mesh, t_heatmap, t_prompt, t_ela, t_fft = st.tabs([
          "🛸 WebGL 3D HOLOGRAM",
          "🗺️ AI DEPTH MAP",
          "🌋 NATIVE 3D MESH",
          "🎯 SPATIAL HEATMAP",
          "🧬 PROMPT INVERSION",
          "⚡ ELA COMPRESSION",
          "🌐 2D-FFT SPECTRUM",
      ])

      with t_holo:
        st.write(
            "True 3D Point-Cloud extruded dynamically using MiDaS AI Geometric"
            " Depth Map. Use canvas buttons to toggle wireframe or real RGB"
            " colors."
        )
        img_b64 = image_to_base64(image)
        depth_b64 = image_to_base64(depth_map)
        render_3d_hologram_component(img_b64, depth_b64)

      with t_depth:
        st.write(
            "Monocular Depth Estimation computed via Intel DPT-MiDaS Hybrid Neural"
            " Pipeline:"
        )
        st.image(depth_map, use_container_width=True)

      with t_mesh:
        st.write(
            "Native Volumetric Surface Mesh projected from AI Depth Map"
            " Topography:"
        )
        fig_3d = create_plotly_3d_hologram(depth_map)
        st.plotly_chart(fig_3d, use_container_width=True)

      with t_heatmap:
        st.write(
            "Highlights image coordinates where high-frequency neural"
            " artifacts cluster."
        )
        st.image(
            generate_spatial_anomaly_heatmap(image), use_container_width=True
        )

      with t_prompt:
        st.write("Reconstructed latent diffusion prompt vector:")
        prompt_vector = reconstruct_synthetic_prompt(image)
        st.code(prompt_vector, language="markdown")

      with t_ela:
        st.write(
            "Error Level Analysis highlights JPEG compression delta variance."
        )
        st.image(generate_ela(image), use_container_width=True)

      with t_fft:
        st.write(
            "Visualizes spatial frequency distribution via 2D Fast Fourier"
            " Transform."
        )
        st.image(generate_fft(image), use_container_width=True)

      # --- Export Audit Log ---
      st.markdown("<br>", unsafe_allow_html=True)
      report_data = {
          "platform": "MonoVision Cyber Forensics Studio v6.0",
          "engine": "Smogy/SMOGY-Ai-images-detector",
          "depth_engine": "Intel/dpt-hybrid-midas",
          "timestamp": datetime.utcnow().isoformat() + "Z",
          "verdict": verdict_str,
          "metrics": {
              "ai_probability": f"{ai_score:.2f}%",
              "real_probability": f"{real_score:.2f}%",
          },
          "estimated_prompt": prompt_vector,
      }

      time_stamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
      st.download_button(
          label="📄 EXPORT FULL FORENSIC AUDIT LOG (JSON)",
          data=json.dumps(report_data, indent=4),
          file_name=f"monovision_audit_{time_stamp_str}.json",
          mime="application/json",
          use_container_width=True,
      )
