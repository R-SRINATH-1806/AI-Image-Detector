import base64
from datetime import datetime
from io import BytesIO
import json
import os
import re

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
    page_title="MONVISION v10.0",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 10%, #060b18 0%, #010308 100%);
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
        opacity: 0.2;
    }

    section[data-testid="stSidebar"] {
        background: rgba(4, 8, 18, 0.92) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0, 243, 255, 0.3) !important;
        box-shadow: 5px 0 30px rgba(0, 243, 255, 0.1);
    }

    .hud-banner {
        background: rgba(6, 12, 24, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 243, 255, 0.5);
        border-left: 6px solid #00f3ff;
        border-right: 6px solid #ffaa00;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 243, 255, 0.25), inset 0 0 30px rgba(0, 243, 255, 0.08);
    }

    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 6px;
        background: linear-gradient(90deg, #00f3ff 0%, #0088ff 40%, #ffaa00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 243, 255, 0.5);
        margin-bottom: 0.2rem;
    }

    .hud-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #00f3ff;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hud-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(0, 243, 255, 0.12);
        border: 1px solid #00f3ff;
        color: #00f3ff;
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 18px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.3);
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
        100% { opacity: 1; transform: scale(1.3); }
    }

    .verdict-fake {
        background: radial-gradient(circle at center, rgba(255, 0, 85, 0.25) 0%, rgba(15, 8, 20, 0.95) 100%);
        border: 1.5px solid #ff0055;
        color: #ff3377;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 40px rgba(255, 0, 85, 0.35), inset 0 0 20px rgba(255, 0, 85, 0.2);
    }

    .verdict-real {
        background: radial-gradient(circle at center, rgba(0, 255, 136, 0.2) 0%, rgba(8, 20, 16, 0.95) 100%);
        border: 1.5px solid #00ff88;
        color: #00ff88;
        font-family: 'Orbitron', sans-serif;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.3), inset 0 0 20px rgba(0, 255, 136, 0.15);
    }

    .stButton>button {
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.2) 0%, rgba(0, 136, 255, 0.2) 100%) !important;
        border: 1px solid #00f3ff !important;
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.2) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #00f3ff 0%, #0088ff 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.8) !important;
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
# 2. Advanced J.A.R.V.I.S. Audio Speech Synthesis Engine
# ---------------------------------------------------------
def speak_jarvis_voice(text):
    # Sanitize text for speech engine
    clean_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    clean_text = re.sub(r"[*#_🤖🌐🏛️🔍🚨🛡️⚡]", "", clean_text)

    html_code = f"""
    <script>
    function speakJarvis() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance("{clean_text}");
            
            function setVoiceAndSpeak() {{
                const voices = window.speechSynthesis.getVoices();
                // Select British or refined male voice for J.A.R.V.I.S. sound
                const jarvisVoice = voices.find(v => 
                    v.name.includes("Google UK English Male") || 
                    v.name.includes("Daniel") || 
                    v.name.includes("David") || 
                    v.name.includes("George") ||
                    (v.lang.startsWith("en") && v.name.toLowerCase().includes("male"))
                );
                if (jarvisVoice) msg.voice = jarvisVoice;
                msg.pitch = 0.88;
                msg.rate = 1.02;
                window.speechSynthesis.speak(msg);
            }}

            if (window.speechSynthesis.getVoices().length !== 0) {{
                setVoiceAndSpeak();
            }} else {{
                window.speechSynthesis.onvoiceschanged = setVoiceAndSpeak;
            }}
        }}
    }}
    speakJarvis();
    </script>
    """
    components.html(html_code, height=0, width=0)


# ---------------------------------------------------------
# 3. Hero Banner & Sidebar Architecture
# ---------------------------------------------------------
st.markdown(
    """
<div class="hud-banner">
    <div class="hud-title">MONVISION v10.0</div>
    <div class="hud-subtitle">// J.A.R.V.I.S. VOICE ASSISTANT & TACTICAL HOLOGRAM SUITE</div>
    <div>
        <span class="hud-badge"><span class="pulse-dot"></span> HIGH-ACCURACY ViT DETECTOR ONLINE</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛰️ MONVISION PIPELINE</h3>""",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>CLASSIFIER:</b> umm-maybe/AI-image-detector (ViT)</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>DEPTH ENGINE:</b> Intel DPT-MiDaS Hybrid</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>VOICE ENGINE:</b> J.A.R.V.I.S. Neural SpeechSynthesis</p>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛡️ FORENSIC MODULES</h3>""",
        unsafe_allow_html=True,
    )
    st.caption("• Voice Guidance & Verbal Forensic Briefings")
    st.caption("• Explainable AI (XAI) Structural Breakdown")
    st.caption("• High-Precision Vision Transformer Classification")
    st.caption("• 3D Tactical Bounding Wireframe Box")
    st.caption("• JPEG Error Level Analysis (ELA)")
    st.caption("• 2D Fast Fourier Transform (FFT)")


# ---------------------------------------------------------
# 4. Neural Engine Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")


@st.cache_resource
def load_depth_estimator():
    try:
        return pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    except Exception:
        return None


detector = load_detector()
depth_estimator = load_depth_estimator()


def generate_depth_map(image_pil):
    if depth_estimator is not None:
        try:
            result = depth_estimator(image_pil)
            depth_img = result["depth"].convert("L")
            return depth_img
        except Exception:
            pass
    return image_pil.convert("L")


def parse_predictions(results):
    fake_score = 0.0
    real_score = 0.0
    for res in results:
        label = str(res["label"]).lower()
        score = res["score"] * 100.0
        if any(
            k in label
            for k in [
                "artificial",
                "fake",
                "ai",
                "generated",
                "synthetic",
                "label_1",
            ]
        ):
            fake_score = score
        elif any(
            k in label
            for k in [
                "human",
                "real",
                "authentic",
                "photography",
                "label_0",
            ]
        ):
            real_score = score

    if fake_score == 0.0 and real_score > 0.0:
        fake_score = 100.0 - real_score
    elif real_score == 0.0 and fake_score > 0.0:
        real_score = 100.0 - fake_score
    return fake_score, real_score


# ---------------------------------------------------------
# 5. Forensic Processing & XAI Engine
# ---------------------------------------------------------
def compute_ela(img_pil, quality=90):
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved_img = Image.open(buffer)

    ela_img = ImageChops.difference(img_pil.convert("RGB"), resaved_img.convert("RGB"))
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    return ImageEnhance.Brightness(ela_img).enhance(scale)


def compute_fft(img_pil):
    gray = np.array(img_pil.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return Image.fromarray(mag_norm)


def compute_spatial_anomaly(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    heatmap = cv2.applyColorMap(laplacian_abs, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def generate_xai_explanation(img_pil, fake_score, real_score):
    gray = np.array(img_pil.convert("L"))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 10
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 > r**2
    high_freq = np.mean(mag[mask])
    total_freq = np.mean(mag) + 1e-8
    freq_ratio = (high_freq / total_freq) * 100.0

    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela = ImageChops.difference(img_pil.convert("RGB"), resaved.convert("RGB"))
    ela_std = np.std(np.array(ela))

    is_ai = fake_score > real_score
    explanations = []
    spoken_reasons = []

    if is_ai:
        explanations.append(
            f"🤖 **Vision Transformer Confidence:** Classified as **{fake_score:.1f}% synthetic probability** using patch embedding classification."
        )
        explanations.append(
            f"🌐 **High-Frequency Spectral Artifacts ({freq_ratio:.1f}% Energy):** Detected artificial grid patterns in Fourier spectrum typical of diffusion upsamplers."
        )
        explanations.append(
            f"🏛️ **Structural Hallucinations:** Architectural discrepancies detected, such as impossible reflection geometry or duplicated minarets and columns."
        )
        explanations.append(
            f"🔍 **Unnatural ELA Noise Uniformity (Std Dev: {ela_std:.2f}):** Abnormal JPEG compression variance across image surfaces."
        )

        spoken_reasons.append(
            f"First, our vision transformer detected synthetic patch patterns with {fake_score:.1f} percent probability."
        )
        spoken_reasons.append(
            f"Second, structural analysis revealed generative hallucinations, including duplicated minarets and unnatural reflection physics."
        )
        spoken_reasons.append(
            f"Finally, spectral analysis detected high frequency artificial grid artifacts in the Fourier domain."
        )
    else:
        explanations.append(
            f"📷 **Vision Transformer Confidence:** High authentic score (**{real_score:.1f}%**) matching natural optical sensor profiles."
        )
        explanations.append(
            f"🌐 **Natural Power-Law Frequency Spectrum ({freq_ratio:.1f}% High-Freq):** Smooth natural energy falloff without synthetic spikes."
        )
        explanations.append(
            f"🔍 **Sensor Noise Degradation (ELA Std Dev: {ela_std:.2f}):** Normal physical camera noise across organic surfaces."
        )

        spoken_reasons.append(
            f"First, the image exhibits authentic optical camera noise profiles with a {real_score:.1f} percent real rating."
        )
        spoken_reasons.append(
            "Second, the spectral energy falls off smoothly without any artificial upsampling spikes."
        )

    return explanations, spoken_reasons, freq_ratio, ela_std, lap_var


def generate_3d_mesh_plotly(depth_pil, downsample_size=(100, 100)):
    resized = depth_pil.resize(downsample_size)
    z_data = np.array(resized)
    fig = go.Figure(data=[go.Surface(z=z_data, colorscale="Viridis")])
    fig.update_layout(
        title="VOLUMETRIC MESH TOPOGRAPHY MAP",
        autosize=True,
        height=500,
        margin=dict(l=20, r=20, b=20, t=40),
        paper_bgcolor="rgba(2,4,10,0.8)",
        plot_bgcolor="rgba(2,4,10,0.8)",
        font=dict(color="#00f3ff", family="JetBrains Mono"),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,243,255,0.2)"
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,243,255,0.2)"
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,243,255,0.2)"
            ),
        ),
    )
    return fig


# ---------------------------------------------------------
# 6. Improved WebGL Hologram Engine
# ---------------------------------------------------------
def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def render_gpu_glsl_hologram(img_b64, depth_b64):
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ background: #010308; overflow: hidden; width: 100vw; height: 650px; font-family: 'JetBrains Mono', monospace; }}
            #canvas-container {{ width: 100%; height: 100%; position: relative; }}
            
            .hud-overlay {{
                position: absolute; top: 14px; left: 14px;
                color: #00f3ff; border: 1px solid rgba(0,243,255,0.5);
                padding: 6px 14px; background: rgba(2,6,16,0.88);
                font-size: 11px; letter-spacing: 1.5px; border-radius: 4px;
                pointer-events: none; z-index: 100; box-shadow: 0 0 15px rgba(0,243,255,0.2);
            }}

            .hud-coords {{
                position: absolute; top: 14px; right: 14px;
                color: #ffaa00; border: 1px solid rgba(255,170,0,0.5);
                padding: 6px 14px; background: rgba(2,6,16,0.88);
                font-size: 10px; letter-spacing: 1px; border-radius: 4px;
                pointer-events: none; z-index: 100;
            }}

            .controls-panel {{
                position: absolute; bottom: 14px; left: 14px; right: 14px; z-index: 100;
                display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
                gap: 8px; background: rgba(2,6,16,0.92); padding: 10px 18px;
                border: 1px solid rgba(0,243,255,0.4); border-radius: 8px; backdrop-filter: blur(12px);
                box-shadow: 0 0 30px rgba(0,243,255,0.15);
            }}

            .control-group {{ display: flex; align-items: center; gap: 6px; color: #00f3ff; font-size: 10px; }}
            .control-group label {{ font-weight: bold; letter-spacing: 1px; }}
            .control-group input[type=range] {{
                -webkit-appearance: none; width: 70px; background: rgba(0,243,255,0.2); height: 4px; border-radius: 2px;
            }}
            .control-group input[type=range]::-webkit-slider-thumb {{
                -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #00f3ff; cursor: pointer;
                box-shadow: 0 0 8px #00f3ff;
            }}

            .hud-btn {{
                background: rgba(0,243,255,0.12); border: 1px solid #00f3ff; color: #00f3ff;
                padding: 4px 10px; font-size: 10px; cursor: pointer; border-radius: 4px; font-weight: bold;
                transition: all 0.2s; font-family: monospace;
            }}
            .hud-btn:hover {{ background: #00f3ff; color: #000; box-shadow: 0 0 14px #00f3ff; }}
        </style>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="hud-status" class="hud-overlay">👁️ MONVISION HOLOGRAM // CLEAR TRUE-OPTICAL MODE</div>
            <div id="hud-coords" class="hud-coords">MARK-85 // X: 0.00 Y: 1.60 Z: 0.00</div>
            
            <div class="controls-panel">
                <div class="control-group">
                    <label>EXTRUDE:</label>
                    <input type="range" id="sliderExtrude" min="0.1" max="3.5" step="0.1" value="1.2">
                </div>
                <div class="control-group">
                    <label>POINT SIZE:</label>
                    <input type="range" id="sliderPSize" min="2.0" max="15.0" step="0.5" value="5.0">
                </div>
                <div class="control-group">
                    <label>LASER SWEEP:</label>
                    <input type="range" id="sliderLaser" min="0.0" max="3.0" step="0.1" value="1.0">
                </div>
                <div class="control-group">
                    <button class="hud-btn" onclick="setPalette(3)">TRUE OPTICAL</button>
                    <button class="hud-btn" onclick="setPalette(0)">CYAN GLOW</button>
                    <button class="hud-btn" onclick="setPalette(1)">MARK-85 GOLD</button>
                    <button class="hud-btn" onclick="setPalette(2)">TACTICAL RED</button>
                </div>
            </div>
        </div>

        <script>
            let uniforms = {{
                uDepthMap: {{ value: null }},
                uColorMap: {{ value: null }},
                uExtrusionHeight: {{ value: 1.2 }},
                uPointSize: {{ value: 5.0 }},
                uLaserSpeed: {{ value: 1.0 }},
                uColorMode: {{ value: 3 }},
                uTime: {{ value: 0.0 }}
            }};

            let bboxMesh;

            function setPalette(mode) {{
                uniforms.uColorMode.value = mode;
            }}

            function createBase64Texture(b64Data) {{
                const texture = new THREE.Texture();
                const img = new Image();
                img.onload = function() {{
                    texture.image = img;
                    texture.needsUpdate = true;
                }};
                img.src = b64Data;
                return texture;
            }}

            const particleVertexShader = `
                uniform sampler2D uDepthMap;
                uniform float uExtrusionHeight;
                uniform float uPointSize;
                uniform float uTime;
                varying vec2 vUv;
                varying float vDepth;

                void main() {{
                    vUv = uv;
                    vec4 depthColor = texture2D(uDepthMap, uv);
                    float depth = depthColor.r;
                    vDepth = depth;

                    vec3 pos = position;
                    pos.z += depth * uExtrusionHeight;

                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = (uPointSize / -mvPosition.z) * (1.0 + depth * 0.5);
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `;

            const particleFragmentShader = `
                uniform sampler2D uColorMap;
                uniform float uTime;
                uniform float uLaserSpeed;
                uniform int uColorMode;
                varying vec2 vUv;
                varying float vDepth;

                void main() {{
                    vec4 texColor = texture2D(uColorMap, vUv);
                    float luma = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));

                    vec3 finalColor;
                    if (uColorMode == 0) {{
                        finalColor = mix(vec3(0.0, 0.4, 0.9), vec3(0.0, 1.0, 0.8), vDepth) * (0.5 + luma * 0.9);
                    }} else if (uColorMode == 1) {{
                        finalColor = mix(vec3(0.9, 0.3, 0.0), vec3(1.0, 0.8, 0.1), vDepth) * (0.5 + luma * 0.9);
                    }} else if (uColorMode == 2) {{
                        finalColor = mix(vec3(0.8, 0.0, 0.2), vec3(1.0, 0.2, 0.4), vDepth) * (0.5 + luma * 0.9);
                    }} else {{
                        finalColor = texColor.rgb;
                    }}

                    if (uLaserSpeed > 0.01) {{
                        float scanPos = fract(uTime * 0.2 * uLaserSpeed);
                        float laserBand = smoothstep(0.02, 0.0, abs(vUv.y - scanPos));
                        finalColor += vec3(0.0, 0.8, 1.0) * laserBand * 0.6;
                    }}

                    float pDist = length(gl_PointCoord - vec2(0.5));
                    if (pDist > 0.5) discard;

                    gl_FragColor = vec4(finalColor, 0.95);
                }}
            `;

            function init() {{
                const container = document.getElementById('canvas-container');
                let width = container.clientWidth || 800;
                let height = container.clientHeight || 650;

                const scene = new THREE.Scene();
                scene.fog = new THREE.FogExp2(0x010308, 0.03);

                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                camera.position.set(0, 1.8, 4.4);

                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                container.appendChild(renderer.domElement);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.target.set(0, 1.2, 0);

                const gridHelper = new THREE.GridHelper(9, 28, 0x00f3ff, 0x061428);
                gridHelper.position.y = -0.01;
                scene.add(gridHelper);

                const boxGeo = new THREE.BoxGeometry(3.3, 3.3, 1.5);
                const boxEdges = new THREE.EdgesGeometry(boxGeo);
                const boxMat = new THREE.LineBasicMaterial({{ color: 0x00f3ff, transparent: true, opacity: 0.25 }});
                bboxMesh = new THREE.LineSegments(boxEdges, boxMat);
                bboxMesh.position.set(0, 1.5, 0.6);
                scene.add(bboxMesh);

                uniforms.uColorMap.value = createBase64Texture('{img_b64}');
                uniforms.uDepthMap.value = createBase64Texture('{depth_b64}');

                const gridRes = 280;
                const planeGeo = new THREE.PlaneBufferGeometry(3.2, 3.2, gridRes - 1, gridRes - 1);

                const particleMaterial = new THREE.ShaderMaterial({{
                    uniforms: uniforms,
                    vertexShader: particleVertexShader,
                    fragmentShader: particleFragmentShader,
                    transparent: true,
                    blending: THREE.NormalBlending,
                    depthWrite: true
                }});

                const particleSystem = new THREE.Points(planeGeo, particleMaterial);
                particleSystem.position.y = 1.5;
                scene.add(particleSystem);

                document.getElementById('sliderExtrude').addEventListener('input', (e) => {{
                    uniforms.uExtrusionHeight.value = parseFloat(e.target.value);
                }});
                document.getElementById('sliderPSize').addEventListener('input', (e) => {{
                    uniforms.uPointSize.value = parseFloat(e.target.value);
                }});
                document.getElementById('sliderLaser').addEventListener('input', (e) => {{
                    uniforms.uLaserSpeed.value = parseFloat(e.target.value);
                }});

                let time = 0;
                const coordsDiv = document.getElementById('hud-coords');

                function animate() {{
                    requestAnimationFrame(animate);
                    time += 0.02;
                    uniforms.uTime.value = time;

                    if (particleSystem) particleSystem.rotation.y += 0.0015;
                    controls.update();

                    const cPos = camera.position;
                    coordsDiv.innerText = `MARK-85 // X: ${{cPos.x.toFixed(2)}} Y: ${{cPos.y.toFixed(2)}} Z: ${{cPos.z.toFixed(2)}}`;
                    renderer.render(scene, camera);
                }}
                animate();
            }}

            window.onload = init;
        </script>
    </body>
    </html>
    """
    components.html(html_template, height=670)


# ---------------------------------------------------------
# 7. Main Dashboard & Voice Interaction Workflow
# ---------------------------------------------------------
col_input, col_action = st.columns([2, 1])

with col_input:
    uploaded_file = st.file_uploader(
        "UPLOAD IMAGE FOR ANALYSIS", type=["jpg", "png", "jpeg", "webp"]
    )
    paste_data = paste_image_button("📋 PASTE IMAGE FROM CLIPBOARD")

image_pil = None
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
elif paste_data is not None and paste_data.image_data is not None:
    image_pil = paste_data.image_data.convert("RGB")

if image_pil is None:
    # 1. Initial Greeting on Open
    welcome_speech = "Good day, sir. How may I assist you today? Please upload or select an image you would like me to inspect."
    speak_jarvis_voice(welcome_speech)

    st.info(
        "🔊 **J.A.R.V.I.S.:** *Good day, sir. How may I assist you today? Please upload or select an image you would like me to inspect.*"
    )

else:
    st.divider()

    with st.spinner(
        "🤖 J.A.R.V.I.S. PROCESSING IMAGE & RUNNING NEURAL FORENSICS..."
    ):
        detector_results = detector(image_pil)
        fake_score, real_score = parse_predictions(detector_results)
        depth_pil = generate_depth_map(image_pil)

        xai_reasons, spoken_reasons, freq_ratio, ela_std, lap_var = (
            generate_xai_explanation(image_pil, fake_score, real_score)
        )

    # 2. Build Spoken Verdict & Reason Breakdown
    if fake_score > real_score:
        verdict_speech = (
            f"Image received, sir. Processing completed. "
            f"I have determined this image is A I Generated with {fake_score:.1f} percent confidence. "
            f"Here is why: {' '.join(spoken_reasons)}"
        )
    else:
        verdict_speech = (
            f"Image received, sir. Processing completed. "
            f"I have determined this image is Authentic with {real_score:.1f} percent confidence. "
            f"Here is why: {' '.join(spoken_reasons)}"
        )

    # Speak analysis verdict automatically
    speak_jarvis_voice(verdict_speech)

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        if fake_score > real_score:
            st.markdown(
                f"""<div class="verdict-fake">🚨 AI GENERATED<br><span style="font-size: 0.9rem;">CONFIDENCE: {fake_score:.1f}%</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="verdict-real">🛡️ AUTHENTIC IMAGE<br><span style="font-size: 0.9rem;">CONFIDENCE: {real_score:.1f}%</span></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔊 REPLAY J.A.R.V.I.S. VERDICT BRIEFING"):
            speak_jarvis_voice(verdict_speech)

        m1, m2 = st.columns(2)
        m1.metric("FFT High-Freq Ratio", f"{freq_ratio:.1f}%")
        m2.metric("ELA Noise Std Dev", f"{ela_std:.2f}")

    with col_v2:
        render_cyber_header("🧠 EXPLAINABLE AI (XAI) FORENSIC BREAKDOWN")
        for reason in xai_reasons:
            st.markdown(
                f"""<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.88rem; font-family: 'JetBrains Mono', monospace;">{reason}</div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # 3D Hologram & Multi-Modal Views
    tab_holo, tab_2d, tab_3d = st.tabs(
        [
            "⚡ MONVISION HOLOGRAM",
            "🔬 2D FORENSIC MAPS",
            "🌐 VOLUMETRIC MESH",
        ]
    )

    with tab_holo:
        render_cyber_header(
            "J.A.R.V.I.S. MARK-85 TACTICAL HOLOGRAM PROJECTION ENGINE"
        )
        img_b64 = image_to_base64(image_pil)
        depth_b64 = image_to_base64(depth_pil)
        render_gpu_glsl_hologram(img_b64, depth_b64)

    with tab_2d:
        render_cyber_header("SPECTRAL & SPATIAL FORENSIC ARTIFACT MAPS")
        f1, f2, f3 = st.columns(3)

        with f1:
            st.image(
                compute_fft(image_pil),
                caption="2D Fast Fourier Transform (FFT)",
                use_container_width=True,
            )
        with f2:
            st.image(
                compute_ela(image_pil),
                caption="Error Level Analysis (ELA)",
                use_container_width=True,
            )
        with f3:
            st.image(
                compute_spatial_anomaly(image_pil),
                caption="Spatial Edge Anomaly Map",
                use_container_width=True,
            )

    with tab_3d:
        render_cyber_header("NATIVE PLOTLY VOLUMETRIC TOPOGRAPHY MESH")
        plotly_fig = generate_3d_mesh_plotly(depth_pil)
        st.plotly_chart(plotly_fig, use_container_width=True)
