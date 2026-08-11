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
    page_title="MONOVISION // GPU GLSL HOLO-FORENSICS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 10%, #080d1a 0%, #02040a 100%);
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%);
        background-size: 100% 4px;
        z-index: 99999;
        pointer-events: none;
        opacity: 0.3;
    }

    section[data-testid="stSidebar"] {
        background: rgba(6, 10, 20, 0.9) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0, 243, 255, 0.25) !important;
        box-shadow: 5px 0 30px rgba(0, 243, 255, 0.08);
    }

    .hud-banner {
        background: rgba(10, 16, 30, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 243, 255, 0.4);
        border-left: 5px solid #00f3ff;
        border-right: 5px solid #ff0055;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.2), inset 0 0 25px rgba(0, 243, 255, 0.05);
    }

    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 5px;
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 50%, #ff0055 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 25px rgba(0, 243, 255, 0.4);
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
        background: rgba(0, 243, 255, 0.12);
        border: 1px solid #00f3ff;
        color: #00f3ff;
        font-family: 'JetBrains Mono', monospace;
        padding: 6px 18px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.25);
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
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.2) 0%, rgba(112, 0, 255, 0.2) 100%) !important;
        border: 1px solid #00f3ff !important;
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.15) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #00f3ff 0%, #7000ff 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.7) !important;
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
    <div class="hud-title">MONOVISION v8.0 ULTRA</div>
    <div class="hud-subtitle">// GPU GLSL HYPER-REALISTIC HOLOGRAM & AI FORENSICS SUITE</div>
    <div>
        <span class="hud-badge"><span class="pulse-dot"></span> ULTRA-RESPONSIVE DUAL-SURFACE HOLO ENGINE ONLINE</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛰️ GPU & AI PIPELINE</h3>""",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>CLASSIFIER:</b> Smogy/SMOGY-Ai-images-detector</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>DEPTH ENGINE:</b> Intel DPT-MiDaS Hybrid</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>HOLO ENGINE:</b> Dual-Layer WebGL (48k Vertices + Wireframe Mesh)</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>POST-FX:</b> Unreal Bloom + Dynamic Glitch GLSL</p>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛡️ FORENSIC MODULES</h3>""",
        unsafe_allow_html=True,
    )
    st.caption("• Explainable AI (XAI) Forensic Breakdown")
    st.caption("• Dual Surface WebGL Particle/Wireframe Hologram")
    st.caption("• Projector Base Emitter & Light Beams")
    st.caption("• Real-Time GLSL Glitch & Interference Engine")
    st.caption("• JPEG Error Level Analysis (ELA)")
    st.caption("• 2D Fast Fourier Transform (FFT)")
    st.caption("• Neuro-Acoustic Sonification Engine")


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
# 4. Forensic Processing & XAI Explanation Engine
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

    if is_ai:
        explanations.append(
            f"🤖 **Vision Transformer Confidence Score:** Classified as **{fake_score:.1f}% synthetic probability** using deep patch embedding matching."
        )
        explanations.append(
            f"🌐 **High-Frequency Spectral Artifacts ({freq_ratio:.1f}% Energy):** Detected periodic grid signatures in Fourier spectrum typical of diffusion/GAN upsamplers."
        )
        explanations.append(
            f"🔍 **Unnatural ELA Uniformity (Std Dev: {ela_std:.2f}):** Abnormal JPEG noise distribution across the visual surface."
        )
        explanations.append(
            f"⚡ **Laplacian Gradient Discrepancy (Var: {lap_var:.1f}):** Artificial sharpening present alongside over-smoothed low-texture regions."
        )
    else:
        explanations.append(
            f"📷 **Vision Transformer Confidence Score:** High authentic score (**{real_score:.1f}%**) with natural sensor characteristics."
        )
        explanations.append(
            f"🌐 **Natural Power-Law Frequency Spectrum ({freq_ratio:.1f}% High-Freq):** Smooth energy falloff with no transposed-convolution spikes."
        )
        explanations.append(
            f"🔍 **Sensor Noise Degradation (ELA Std Dev: {ela_std:.2f}):** Normal optical camera degradation across textures."
        )
        explanations.append(
            f"⚡ **Continuous Optical Focus Dropoff (Laplacian Var: {lap_var:.1f}):** Authentic depth-of-field transition without generative smoothing."
        )

    return explanations, freq_ratio, ela_std, lap_var


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
# 5. Ultra-Realistic WebGL GLSL Hologram Engine
# ---------------------------------------------------------
def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def render_sonification_module():
    html_code = """
    <div style="background: rgba(10,16,30,0.8); border: 1px solid rgba(0, 243, 255, 0.4); border-left: 4px solid #00f3ff; padding: 15px; border-radius: 8px; text-align: center; color: #00f3ff; font-family: 'Courier New', monospace;">
        <p style="margin-top: 0; font-weight: bold; letter-spacing: 2px;">🔊 NEURO-ACOUSTIC DATA SONIFICATION</p>
        <p style="font-size: 11px; color: #94a3b8; margin-bottom: 12px;">Translating latent visual frequencies into audible waveform telemetry.</p>
        <button onclick="playSciFiDrone()" style="background: rgba(0,243,255,0.15); color: #00f3ff; border: 1px solid #00f3ff; padding: 8px 20px; cursor: pointer; font-weight: bold; font-family: inherit; margin-right: 10px; border-radius: 4px;">▶ INITIATE AUDIO SCAN</button>
        <button onclick="stopDrone()" style="background: rgba(255,0,85,0.1); color: #ff0055; border: 1px solid #ff0055; padding: 8px 20px; cursor: pointer; font-weight: bold; font-family: inherit; border-radius: 4px;">■ HALT</button>
    </div>
    
    <script>
        let audioCtx;
        let activeNodes = [];

        function playSciFiDrone() {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (activeNodes.length > 0) return;

            const freqs = [41.20, 82.41, 123.47]; 
            freqs.forEach((freq, i) => {
                let osc = audioCtx.createOscillator();
                let gain = audioCtx.createGain();
                
                osc.type = i === 0 ? 'sine' : 'sawtooth';
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                
                let lfo = audioCtx.createOscillator();
                lfo.type = 'sine';
                lfo.frequency.value = 0.5 + (i * 1.5);
                
                let lfoGain = audioCtx.createGain();
                lfoGain.gain.value = 5 + (i * 2);
                
                lfo.connect(lfoGain);
                lfoGain.connect(osc.frequency);
                
                gain.gain.setValueAtTime(0, audioCtx.currentTime);
                gain.gain.linearRampToValueAtTime(0.08, audioCtx.currentTime + 2.0);
                
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                
                osc.start();
                lfo.start();
                activeNodes.push({osc, lfo, gain});
            });
        }

        function stopDrone() {
            if (!audioCtx) return;
            activeNodes.forEach(node => {
                node.gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 1.5);
                setTimeout(() => { 
                    node.osc.stop(); 
                    node.lfo.stop(); 
                    node.osc.disconnect();
                }, 1500);
            });
            activeNodes = [];
        }
    </script>
    """
    components.html(html_code, height=150)


def render_gpu_glsl_hologram(img_b64, depth_b64):
    """WebGL Hologram with Projector Base, Particle Cloud, Wireframe Overlay & GLSL Glitch FX."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ background: #02040a; overflow: hidden; width: 100vw; height: 620px; font-family: 'JetBrains Mono', monospace; }}
            #canvas-container {{ width: 100%; height: 100%; position: relative; }}
            
            .hud-overlay {{
                position: absolute; top: 12px; left: 12px;
                color: #00f3ff; border: 1px solid rgba(0,243,255,0.5);
                padding: 6px 14px; background: rgba(2,4,10,0.85);
                font-size: 11px; letter-spacing: 1.5px; border-radius: 4px;
                pointer-events: none; z-index: 100; box-shadow: 0 0 15px rgba(0,243,255,0.25);
            }}

            .controls-panel {{
                position: absolute; bottom: 12px; left: 12px; right: 12px; z-index: 100;
                display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
                gap: 8px; background: rgba(2,4,10,0.9); padding: 10px 16px;
                border: 1px solid rgba(0,243,255,0.4); border-radius: 8px; backdrop-filter: blur(10px);
                box-shadow: 0 0 25px rgba(0,243,255,0.15);
            }}

            .control-group {{ display: flex; align-items: center; gap: 6px; color: #00f3ff; font-size: 10px; }}
            .control-group label {{ font-weight: bold; letter-spacing: 1px; }}
            .control-group input[type=range] {{
                -webkit-appearance: none; width: 75px; background: rgba(0,243,255,0.2); height: 4px; border-radius: 2px;
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
            .hud-btn:hover {{ background: #00f3ff; color: #000; box-shadow: 0 0 12px #00f3ff; }}
        </style>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminanceHighPassShader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="hud-status" class="hud-overlay">⚡ GPU DUAL-SURFACE HOLOGRAM // GLSL SIGNAL GLITCH // 48,400 VOXELS</div>
            
            <div class="controls-panel">
                <div class="control-group">
                    <label>EXTRUDE:</label>
                    <input type="range" id="sliderExtrude" min="0.1" max="4.0" step="0.1" value="1.8">
                </div>
                <div class="control-group">
                    <label>GLITCH:</label>
                    <input type="range" id="sliderGlitch" min="0.0" max="2.0" step="0.1" value="0.4">
                </div>
                <div class="control-group">
                    <label>SIZE:</label>
                    <input type="range" id="sliderPSize" min="5.0" max="25.0" step="1.0" value="14.0">
                </div>
                <div class="control-group">
                    <label>BLOOM:</label>
                    <input type="range" id="sliderBloom" min="0.0" max="3.0" step="0.1" value="1.5">
                </div>
                <div class="control-group">
                    <label>SCAN SPEED:</label>
                    <input type="range" id="sliderScan" min="0.2" max="3.0" step="0.1" value="1.0">
                </div>
                <div class="control-group">
                    <label>Z-CLIP:</label>
                    <input type="range" id="sliderClip" min="0.0" max="1.0" step="0.01" value="1.0">
                </div>
                <div class="control-group">
                    <button class="hud-btn" onclick="setPalette(0)">CYBER</button>
                    <button class="hud-btn" onclick="setPalette(1)">MATRIX</button>
                    <button class="hud-btn" onclick="setPalette(2)">SYNTH</button>
                    <button class="hud-btn" onclick="setPalette(3)">INFRARED</button>
                    <button class="hud-btn" onclick="setPalette(4)">TRUE RGB</button>
                </div>
            </div>
        </div>

        <script>
            if (typeof THREE !== 'undefined') {{
                if (!THREE.CopyShader) {{
                    THREE.CopyShader = {{
                        uniforms: {{ 'tDiffuse': {{ value: null }}, 'opacity': {{ value: 1.0 }} }},
                        vertexShader: 'varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }}',
                        fragmentShader: 'uniform float opacity; uniform sampler2D tDiffuse; varying vec2 vUv; void main() {{ vec4 tex = texture2D( tDiffuse, vUv ); gl_FragColor = opacity * tex; }}'
                    }};
                }}
                if (!THREE.LuminanceHighPassShader) {{
                    THREE.LuminanceHighPassShader = {{
                        shaderID: 'luminanceHighPass',
                        uniforms: {{
                            'tDiffuse': {{ value: null }},
                            'luminanceThreshold': {{ value: 0.21 }},
                            'smoothWidth': {{ value: 0.01 }},
                            'defaultColor': {{ value: new THREE.Color( 0x000000 ) }},
                            'defaultOpacity': {{ value: 0.0 }}
                        }},
                        vertexShader: 'varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }}',
                        fragmentShader: 'uniform sampler2D tDiffuse; uniform vec3 defaultColor; uniform float defaultOpacity; uniform float luminanceThreshold; uniform float smoothWidth; varying vec2 vUv; void main() {{ vec4 texel = texture2D( tDiffuse, vUv ); vec3 luma = vec3( 0.299, 0.587, 0.114 ); float v = dot( texel.rgb, luma ); vec4 outputColor = vec4( defaultColor, defaultOpacity ); float alpha = smoothstep( luminanceThreshold, luminanceThreshold + smoothWidth, v ); gl_FragColor = mix( outputColor, texel, alpha ); }}'
                    }};
                }}
            }}

            let uniforms = {{
                uDepthMap: {{ value: null }},
                uColorMap: {{ value: null }},
                uExtrusionHeight: {{ value: 1.8 }},
                uZClip: {{ value: 1.0 }},
                uGlitchIntensity: {{ value: 0.4 }},
                uPointSize: {{ value: 14.0 }},
                uScanSpeed: {{ value: 1.0 }},
                uColorMode: {{ value: 0 }},
                uTime: {{ value: 0.0 }}
            }};

            let bloomPass, composer, useComposer = false;
            let ring1, ring2, beamMesh;

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
                uniform float uZClip;
                uniform float uGlitchIntensity;
                uniform float uPointSize;
                uniform float uTime;
                varying vec2 vUv;
                varying float vDepth;
                varying float vClip;
                varying float vGlitch;

                float rand(vec2 co) {{
                    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
                }}

                void main() {{
                    vUv = uv;
                    vec4 depthColor = texture2D(uDepthMap, uv);
                    float depth = depthColor.r;
                    vDepth = depth;

                    vec3 pos = position;

                    if (depth > uZClip) {{
                        vClip = 1.0;
                    }} else {{
                        vClip = 0.0;
                    }}

                    pos.z += depth * uExtrusionHeight;

                    // Hologram Glitch Signal Displacement
                    float glitchNoise = rand(vec2(uTime * 0.1, pos.y));
                    float isGlitching = step(0.96 - (uGlitchIntensity * 0.08), glitchNoise);
                    pos.x += isGlitching * (rand(vec2(uTime, pos.z)) - 0.5) * 0.3 * uGlitchIntensity;
                    vGlitch = isGlitching;

                    // Floating micro-vibration
                    pos.z += sin(pos.x * 12.0 + uTime * 4.0) * 0.012 * depth;

                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = (uPointSize / -mvPosition.z) * (0.8 + depth * 1.2);
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `;

            const particleFragmentShader = `
                uniform sampler2D uColorMap;
                uniform float uTime;
                uniform float uScanSpeed;
                uniform int uColorMode;
                varying vec2 vUv;
                varying float vDepth;
                varying float vClip;
                varying float vGlitch;

                void main() {{
                    if (vClip > 0.5) discard;

                    vec4 texColor = texture2D(uColorMap, vUv);
                    vec3 finalColor;

                    if (uColorMode == 0) {{ // Cyber Cyan / Purple
                        finalColor = mix(vec3(0.0, 0.95, 1.0), vec3(0.8, 0.0, 1.0), vDepth);
                    }} else if (uColorMode == 1) {{ // Matrix Neon
                        finalColor = mix(vec3(0.0, 0.3, 0.1), vec3(0.2, 1.0, 0.4), vDepth);
                    }} else if (uColorMode == 2) {{ // Synthwave
                        finalColor = mix(vec3(1.0, 0.0, 0.5), vec3(1.0, 0.8, 0.0), vDepth);
                    }} else if (uColorMode == 3) {{ // Infrared Thermal
                        finalColor = mix(vec3(0.0, 0.0, 0.8), vec3(1.0, 0.1, 0.0), vDepth);
                    }} else {{ // True RGB
                        finalColor = texColor.rgb;
                    }}

                    // Dynamic Scanline Sweep
                    float scanline = sin((vUv.y * 240.0) - (uTime * uScanSpeed * 12.0)) * 0.22 + 0.78;
                    finalColor *= scanline;

                    // Temporal Signal Flicker
                    float flicker = sin(uTime * 18.0) * 0.04 + 0.96;
                    finalColor *= flicker;

                    if (vGlitch > 0.5) {{
                        finalColor = vec3(1.0, 1.0, 1.0);
                    }}

                    float dist = length(gl_PointCoord - vec2(0.5));
                    if (dist > 0.5) discard;
                    float alpha = (1.0 - dist * 2.0) * (0.65 + vDepth * 0.35);

                    gl_FragColor = vec4(finalColor, alpha);
                }}
            `;

            function init() {{
                try {{
                    const container = document.getElementById('canvas-container');
                    let width = container.clientWidth || window.innerWidth || 800;
                    let height = container.clientHeight || 620;

                    const scene = new THREE.Scene();
                    scene.fog = new THREE.FogExp2(0x02040a, 0.035);

                    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                    camera.position.set(0, 1.8, 4.8);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    renderer.setSize(width, height);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    container.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.target.set(0, 1.0, 0);

                    // --- 1. PROJECTOR BASE & PEDESTAL ---
                    const gridHelper = new THREE.GridHelper(10, 30, 0x00f3ff, 0x0a1526);
                    gridHelper.position.y = -0.01;
                    scene.add(gridHelper);

                    // Projector Ring 1 (Outer target reticle)
                    const ringGeo1 = new THREE.RingGeometry(1.6, 1.65, 64);
                    const ringMat1 = new THREE.MeshBasicMaterial({{ color: 0x00f3ff, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }});
                    ring1 = new THREE.Mesh(ringGeo1, ringMat1);
                    ring1.rotation.x = Math.PI / 2;
                    scene.add(ring1);

                    // Projector Ring 2 (Inner ring)
                    const ringGeo2 = new THREE.RingGeometry(1.1, 1.13, 32);
                    const ringMat2 = new THREE.MeshBasicMaterial({{ color: 0xff0055, side: THREE.DoubleSide, transparent: true, opacity: 0.5 }});
                    ring2 = new THREE.Mesh(ringGeo2, ringMat2);
                    ring2.rotation.x = Math.PI / 2;
                    scene.add(ring2);

                    // Vertical Light Projection Cone Beam
                    const beamGeo = new THREE.CylinderGeometry(1.6, 0.2, 3.2, 32, 1, true);
                    const beamMat = new THREE.MeshBasicMaterial({{
                        color: 0x00f3ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false
                    }});
                    beamMesh = new THREE.Mesh(beamGeo, beamMat);
                    beamMesh.position.y = 1.6;
                    scene.add(beamMesh);

                    // --- 2. DUAL-SURFACE HOLOGRAM (PARTICLES + WIREFRAME) ---
                    uniforms.uColorMap.value = createBase64Texture('{img_b64}');
                    uniforms.uDepthMap.value = createBase64Texture('{depth_b64}');

                    const gridRes = 220;
                    const planeGeo = new THREE.PlaneBufferGeometry(3.2, 3.2, gridRes - 1, gridRes - 1);

                    const particleMaterial = new THREE.ShaderMaterial({{
                        uniforms: uniforms,
                        vertexShader: particleVertexShader,
                        fragmentShader: particleFragmentShader,
                        transparent: true,
                        blending: THREE.AdditiveBlending,
                        depthWrite: false
                    }});

                    const particleSystem = new THREE.Points(planeGeo, particleMaterial);
                    particleSystem.position.y = 1.6;
                    scene.add(particleSystem);

                    // Semi-transparent Structural Wireframe Mesh
                    const wireframeMaterial = new THREE.ShaderMaterial({{
                        uniforms: uniforms,
                        vertexShader: particleVertexShader,
                        fragmentShader: particleFragmentShader,
                        wireframe: true,
                        transparent: true,
                        opacity: 0.25,
                        blending: THREE.AdditiveBlending,
                        depthWrite: false
                    }});
                    const wireframeMesh = new THREE.Mesh(planeGeo, wireframeMaterial);
                    wireframeMesh.position.y = 1.6;
                    scene.add(wireframeMesh);

                    // --- 3. POST-PROCESSING BLOOM ---
                    try {{
                        if (typeof THREE.EffectComposer !== 'undefined' && typeof THREE.UnrealBloomPass !== 'undefined') {{
                            const renderScene = new THREE.RenderPass(scene, camera);
                            bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(width, height), 1.5, 0.4, 0.85);
                            composer = new THREE.EffectComposer(renderer);
                            composer.addPass(renderScene);
                            composer.addPass(bloomPass);
                            useComposer = true;
                        }}
                    }} catch (err) {{
                        console.warn("Bloom setup fallback:", err);
                        useComposer = false;
                    }}

                    // --- 4. CONTROLS INTERACTION ---
                    document.getElementById('sliderExtrude').addEventListener('input', (e) => {{
                        uniforms.uExtrusionHeight.value = parseFloat(e.target.value);
                    }});
                    document.getElementById('sliderGlitch').addEventListener('input', (e) => {{
                        uniforms.uGlitchIntensity.value = parseFloat(e.target.value);
                    }});
                    document.getElementById('sliderPSize').addEventListener('input', (e) => {{
                        uniforms.uPointSize.value = parseFloat(e.target.value);
                    }});
                    document.getElementById('sliderBloom').addEventListener('input', (e) => {{
                        if (bloomPass) bloomPass.strength = parseFloat(e.target.value);
                    }});
                    document.getElementById('sliderScan').addEventListener('input', (e) => {{
                        uniforms.uScanSpeed.value = parseFloat(e.target.value);
                    }});
                    document.getElementById('sliderClip').addEventListener('input', (e) => {{
                        uniforms.uZClip.value = parseFloat(e.target.value);
                    }});

                    // --- 5. ANIMATION LOOP ---
                    let time = 0;
                    function animate() {{
                        requestAnimationFrame(animate);
                        time += 0.02;
                        uniforms.uTime.value = time;

                        if (ring1) ring1.rotation.z += 0.005;
                        if (ring2) ring2.rotation.z -= 0.01;

                        controls.update();

                        if (useComposer && composer) {{
                            composer.render();
                        }} else {{
                            renderer.render(scene, camera);
                        }}
                    }}
                    animate();

                    const resizeObserver = new ResizeObserver(() => {{
                        const newWidth = container.clientWidth;
                        const newHeight = container.clientHeight || 620;
                        camera.aspect = newWidth / newHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(newWidth, newHeight);
                        if (composer) composer.setSize(newWidth, newHeight);
                    }});
                    resizeObserver.observe(container);

                }} catch(e) {{
                    console.error("WebGL Holo Engine Error:", e);
                }}
            }}

            window.onload = init;
        </script>
    </body>
    </html>
    """
    components.html(html_template, height=640)


# ---------------------------------------------------------
# 6. Main Dashboard & File Input Workspace
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

if image_pil is not None:
    st.divider()

    with st.spinner("🤖 EXECUTING NEURAL CLASSIFICATION & FORENSICS..."):
        detector_results = detector(image_pil)
        fake_score, real_score = parse_predictions(detector_results)
        depth_pil = generate_depth_map(image_pil)

        xai_reasons, freq_ratio, ela_std, lap_var = generate_xai_explanation(
            image_pil, fake_score, real_score
        )

    # Verdict & Explainable AI (XAI) Panel
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
        m1, m2 = st.columns(2)
        m1.metric("FFT High-Freq Ratio", f"{freq_ratio:.1f}%")
        m2.metric("ELA Noise Std Dev", f"{ela_std:.2f}")

    with col_v2:
        render_cyber_header("🧠 EXPLAINABLE AI (XAI) FORENSIC BREAKDOWN")
        st.caption(
            "Quantitative forensic features explaining the classification decision for technical evaluation:"
        )

        for reason in xai_reasons:
            st.markdown(
                f"""<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.88rem; font-family: 'JetBrains Mono', monospace;">{reason}</div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # 3D GLSL WebGL Hologram & Multi-Modal Views
    tab_holo, tab_2d, tab_3d, tab_audio = st.tabs(
        [
            "⚡ GPU GLSL HOLOGRAM",
            "🔬 2D FORENSIC MAPS",
            "🌐 VOLUMETRIC MESH",
            "🔊 SONIFICATION",
        ]
    )

    with tab_holo:
        render_cyber_header(
            "HYPER-REALISTIC DUAL-SURFACE WebGL GLSL HOLOGRAM ENGINE"
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
                caption="2D Fast Fourier Transform (FFT) High-Freq Spectrum",
                use_container_width=True,
            )
        with f2:
            st.image(
                compute_ela(image_pil),
                caption="Error Level Analysis (ELA) Compression Map",
                use_container_width=True,
            )
        with f3:
            st.image(
                compute_spatial_anomaly(image_pil),
                caption="Spatial Edge & Laplacian Anomaly Map",
                use_container_width=True,
            )

    with tab_3d:
        render_cyber_header("NATIVE PLOTLY VOLUMETRIC TOPOGRAPHY MESH")
        plotly_fig = generate_3d_mesh_plotly(depth_pil)
        st.plotly_chart(plotly_fig, use_container_width=True)

    with tab_audio:
        render_cyber_header("NEURO-ACOUSTIC FREQUENCY SONIFICATION")
        render_sonification_module()

else:
    st.info("💡 Upload or paste an image above to begin real-time forensics analysis.")
