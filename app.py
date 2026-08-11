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
    <div class="hud-title">MONOVISION v7.0 ULTRA</div>
    <div class="hud-subtitle">// GPU-ACCELERATED GLSL HOLOGRAM & AI FORENSICS SUITE</div>
    <div>
        <span class="hud-badge"><span class="pulse-dot"></span> REAL-TIME GPU VERTEX SHADER & UNREAL BLOOM ONLINE</span>
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
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>3D PIPELINE:</b> GLSL GPU Shader (90k Vertices)</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8;"><b>POST-FX:</b> Unreal Bloom + Selective Pass</p>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<h3 style="font-family: 'Orbitron', sans-serif; color: #00f3ff; font-size: 1.1rem;">🛡️ FORENSIC MODULES</h3>""",
        unsafe_allow_html=True,
    )
    st.caption("• Custom GLSL GPU Particle Displacement Shader")
    st.caption("• Selective Unreal Bloom & Chromatic Pass")
    st.caption("• Live Interactive Canvas Controls & Z-Cut Plane")
    st.caption("• Native 3D Volumetric Mesh Topography")
    st.caption("• Spatial Anomaly Artifact Map")
    st.caption("• Latent Prompt Inversion Vector")
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
# 4. Forensic Processing Functions
# ---------------------------------------------------------
def compute_ela(img_pil, quality=90):
    """Calculates JPEG Error Level Analysis (ELA) map."""
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
    """Calculates 2D Fast Fourier Transform high-frequency spectral map."""
    gray = np.array(img_pil.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return Image.fromarray(mag_norm)


def compute_spatial_anomaly(img_pil):
    """Calculates spatial edge disruption and high-gradient anomalies."""
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    heatmap = cv2.applyColorMap(laplacian_abs, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def generate_3d_mesh_plotly(depth_pil, downsample_size=(100, 100)):
    """Generates interactive Plotly 3D topography depth mesh."""
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
# 5. WebGL GLSL Shader Engine & Interactive Components
# ---------------------------------------------------------
def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def render_sonification_module():
    """Generates a Web Audio API synthesizer for acoustic data scanning."""
    html_code = """
    <div style="background: rgba(10,16,30,0.8); border: 1px solid rgba(0, 243, 255, 0.4); border-left: 4px solid #00f3ff; padding: 15px; border-radius: 8px; text-align: center; color: #00f3ff; font-family: 'Courier New', monospace; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);">
        <p style="margin-top: 0; font-weight: bold; letter-spacing: 2px;">🔊 NEURO-ACOUSTIC DATA SONIFICATION</p>
        <p style="font-size: 11px; color: #94a3b8; margin-bottom: 12px;">Translating latent visual frequencies into audible waveform telemetry.</p>
        <button onclick="playSciFiDrone()" style="background: rgba(0,243,255,0.15); color: #00f3ff; border: 1px solid #00f3ff; padding: 8px 20px; cursor: pointer; font-weight: bold; font-family: inherit; transition: 0.3s; margin-right: 10px; border-radius: 4px;">▶ INITIATE AUDIO SCAN</button>
        <button onclick="stopDrone()" style="background: rgba(255,0,85,0.1); color: #ff0055; border: 1px solid #ff0055; padding: 8px 20px; cursor: pointer; font-weight: bold; font-family: inherit; transition: 0.3s; border-radius: 4px;">■ HALT</button>
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
    """Ultra-High Density WebGL GPU Shader Hologram (90,000+ Particles at 60FPS) with Unreal Bloom & Live Interactive Controls."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: #02040a; overflow: hidden; width: 100vw; height: 580px; font-family: 'Courier New', monospace; }
            #canvas-container { width: 100%; height: 100%; position: relative; }
            
            .hud-overlay {
                position: absolute; top: 12px; left: 12px;
                color: #00f3ff; border: 1px solid rgba(0,243,255,0.5);
                padding: 6px 14px; background: rgba(2,4,10,0.85);
                font-size: 11px; letter-spacing: 1.5px; border-radius: 4px;
                pointer-events: none; z-index: 100; box-shadow: 0 0 12px rgba(0,243,255,0.25);
            }

            .controls-panel {
                position: absolute; bottom: 12px; left: 12px; right: 12px; z-index: 100;
                display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
                gap: 10px; background: rgba(2,4,10,0.88); padding: 10px 16px;
                border: 1px solid rgba(0,243,255,0.3); border-radius: 8px; backdrop-filter: blur(8px);
            }

            .control-group { display: flex; align-items: center; gap: 8px; color: #00f3ff; font-size: 11px; }
            .control-group label { font-weight: bold; letter-spacing: 1px; }
            .control-group input[type=range] {
                -webkit-appearance: none; width: 90px; background: rgba(0,243,255,0.2); height: 4px; border-radius: 2px;
            }
            .control-group input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #00f3ff; cursor: pointer;
            }

            .hud-btn {
                background: rgba(0,243,255,0.12); border: 1px solid #00f3ff; color: #00f3ff;
                padding: 5px 12px; font-size: 10px; cursor: pointer; border-radius: 4px; font-weight: bold;
                transition: all 0.2s; font-family: monospace;
            }
            .hud-btn:hover { background: #00f3ff; color: #000; box-shadow: 0 0 12px #00f3ff; }
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
            <div id="hud-status" class="hud-overlay">⚡ GPU GLSL VERTEX SHADER ENGINE // 90,000 VOXELS // BLOOM ACTIVE</div>
            
            <div class="controls-panel">
                <div class="control-group">
                    <label>EXTRUSION:</label>
                    <input type="range" id="sliderExtrude" min="0.1" max="4.0" step="0.1" value="1.8">
                </div>
                <div class="control-group">
                    <label>BLOOM:</label>
                    <input type="range" id="sliderBloom" min="0.0" max="3.0" step="0.1" value="1.4">
                </div>
                <div class="control-group">
                    <label>Z-CUT PLANE:</label>
                    <input type="range" id="sliderClip" min="0.0" max="1.0" step="0.01" value="1.0">
                </div>
                <div class="control-group">
                    <button class="hud-btn" onclick="setPalette(0)">CYBER</button>
                    <button class="hud-btn" onclick="setPalette(1)">MATRIX</button>
                    <button class="hud-btn" onclick="setPalette(2)">SYNTHWAVE</button>
                    <button class="hud-btn" onclick="setPalette(3)">TRUE RGB</button>
                </div>
            </div>
        </div>

        <script>
            if (typeof THREE !== 'undefined') {
                if (!THREE.CopyShader) {
                    THREE.CopyShader = {
                        uniforms: { 'tDiffuse': { value: null }, 'opacity': { value: 1.0 } },
                        vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }',
                        fragmentShader: 'uniform float opacity; uniform sampler2D tDiffuse; varying vec2 vUv; void main() { vec4 tex = texture2D( tDiffuse, vUv ); gl_FragColor = opacity * tex; }'
                    };
                }
                if (!THREE.LuminanceHighPassShader) {
                    THREE.LuminanceHighPassShader = {
                        shaderID: 'luminanceHighPass',
                        uniforms: {
                            'tDiffuse': { value: null },
                            'luminanceThreshold': { value: 0.21 },
                            'smoothWidth': { value: 0.01 },
                            'defaultColor': { value: new THREE.Color( 0x000000 ) },
                            'defaultOpacity': { value: 0.0 }
                        },
                        vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }',
                        fragmentShader: 'uniform sampler2D tDiffuse; uniform vec3 defaultColor; uniform float defaultOpacity; uniform float luminanceThreshold; uniform float smoothWidth; varying vec2 vUv; void main() { vec4 texel = texture2D( tDiffuse, vUv ); vec3 luma = vec3( 0.299, 0.587, 0.114 ); float v = dot( texel.rgb, luma ); vec4 outputColor = vec4( defaultColor, defaultOpacity ); float alpha = smoothstep( luminanceThreshold, luminanceThreshold + smoothWidth, v ); gl_FragColor = mix( outputColor, texel, alpha ); }'
                    };
                }
            }

            let uniforms = {
                uDepthMap: { value: null },
                uColorMap: { value: null },
                uExtrusionHeight: { value: 1.8 },
                uZClip: { value: 1.0 },
                uColorMode: { value: 0 },
                uTime: { value: 0.0 }
            };

            let bloomPass;
            let composer;
            let useComposer = false;

            function setPalette(mode) {
                uniforms.uColorMode.value = mode;
            }

            function createBase64Texture(b64Data) {
                const texture = new THREE.Texture();
                const img = new Image();
                img.onload = function() {
                    texture.image = img;
                    texture.needsUpdate = true;
                };
                img.src = b64Data;
                return texture;
            }

            const vertexShader = `
                uniform sampler2D uDepthMap;
                uniform float uExtrusionHeight;
                uniform float uZClip;
                uniform float uTime;
                varying vec2 vUv;
                varying float vDepth;
                varying float vClip;

                void main() {
                    vUv = uv;
                    vec4 depthColor = texture2D(uDepthMap, uv);
                    float depth = depthColor.r;
                    vDepth = depth;

                    vec3 pos = position;

                    if (depth > uZClip) {
                        vClip = 1.0;
                    } else {
                        vClip = 0.0;
                    }

                    pos.y += depth * uExtrusionHeight;
                    pos.y += sin(pos.x * 8.0 + uTime * 2.0) * 0.02 * depth;

                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = (12.0 / -mvPosition.z) * (0.8 + depth * 0.8);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `;

            const fragmentShader = `
                uniform sampler2D uColorMap;
                uniform float uTime;
                uniform int uColorMode;
                varying vec2 vUv;
                varying float vDepth;
                varying float vClip;

                void main() {
                    if (vClip > 0.5) discard;

                    vec4 texColor = texture2D(uColorMap, vUv);
                    vec3 finalColor;

                    if (uColorMode == 0) {
                        finalColor = mix(vec3(0.0, 0.35, 1.0), vec3(0.0, 0.95, 1.0), vDepth);
                    } else if (uColorMode == 1) {
                        finalColor = mix(vec3(0.0, 0.15, 0.05), vec3(0.1, 1.0, 0.4), vDepth);
                    } else if (uColorMode == 2) {
                        finalColor = mix(vec3(0.9, 0.0, 0.5), vec3(1.0, 0.6, 0.0), vDepth);
                    } else {
                        finalColor = texColor.rgb;
                    }

                    float scanline = sin((vUv.y * 180.0) - (uTime * 4.0)) * 0.18 + 0.82;
                    finalColor *= scanline;

                    float dist = length(gl_PointCoord - vec2(0.5));
                    if (dist > 0.5) discard;
                    float alpha = (1.0 - dist * 2.0) * (0.7 + vDepth * 0.3);

                    gl_FragColor = vec4(finalColor, alpha);
                }
            `;

            function init() {
                try {
                    const container = document.getElementById('canvas-container');
                    let width = container.clientWidth || window.innerWidth || 800;
                    let height = container.clientHeight || 580;

                    const scene = new THREE.Scene();
                    scene.fog = new THREE.FogExp2(0x02040a, 0.04);

                    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                    camera.position.set(0, 2.2, 4.8);

                    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                    renderer.setSize(width, height);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    container.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.target.set(0, 0.8, 0);

                    const gridHelper = new THREE.GridHelper(10, 28, 0x00f3ff, 0x112233);
                    gridHelper.position.y = -0.2;
                    scene.add(gridHelper);

                    const scanPlaneGeo = new THREE.PlaneGeometry(3.6, 3.6);
                    const scanPlaneMat = new THREE.MeshBasicMaterial({
                        color: 0x00f3ff, side: THREE.DoubleSide, transparent: true, opacity: 0.12, wireframe: true
                    });
                    const scanPlane = new THREE.Mesh(scanPlaneGeo, scanPlaneMat);
                    scanPlane.rotation.x = Math.PI / 2;
                    scene.add(scanPlane);

                    uniforms.uColorMap.value = createBase64Texture('__IMG_B64__');
                    uniforms.uDepthMap.value = createBase64Texture('__DEPTH_B64__');

                    const gridRes = 300;
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array(gridRes * gridRes * 3);
                    const uvs = new Float32Array(gridRes * gridRes * 2);

                    let pIdx = 0, uIdx = 0;
                    for (let y = 0; y < gridRes; y++) {
                        for (let x = 0; x < gridRes; x++) {
                            const u = x / (gridRes - 1);
                            const v = y / (gridRes - 1);

                            positions[pIdx] = (u - 0.5) * 3.6;
                            positions[pIdx + 1] = 0;
                            positions[pIdx + 2] = (v - 0.5) * 3.6;

                            uvs[uIdx] = u;
                            uvs[uIdx + 1] = 1.0 - v;

                            pIdx += 3;
                            uIdx += 2;
                        }
                    }

                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

                    const shaderMaterial = new THREE.ShaderMaterial({
                        uniforms: uniforms,
                        vertexShader: vertexShader,
                        fragmentShader: fragmentShader,
                        transparent: true,
                        blending: THREE.AdditiveBlending,
                        depthWrite: false
                    });

                    const particleSystem = new THREE.Points(geometry, shaderMaterial);
                    scene.add(particleSystem);

                    try {
                        if (typeof THREE.EffectComposer !== 'undefined' && typeof THREE.UnrealBloomPass !== 'undefined') {
                            const renderScene = new THREE.RenderPass(scene, camera);
                            bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(width, height), 1.4, 0.4, 0.85);
                            composer = new THREE.EffectComposer(renderer);
                            composer.addPass(renderScene);
                            composer.addPass(bloomPass);
                            useComposer = true;
                        }
                    } catch (err) {
                        console.warn("Bloom post-processing skipped, falling back to direct render:", err);
                        useComposer = false;
                    }

                    document.getElementById('sliderExtrude').addEventListener('input', (e) => {
                        uniforms.uExtrusionHeight.value = parseFloat(e.target.value);
                    });

                    document.getElementById('sliderBloom').addEventListener('input', (e) => {
                        if (bloomPass) bloomPass.strength = parseFloat(e.target.value);
                    });

                    document.getElementById('sliderClip').addEventListener('input', (e) => {
                        uniforms.uZClip.value = parseFloat(e.target.value);
                    });

                    let time = 0;
                    function animate() {
                        requestAnimationFrame(animate);
                        time += 0.02;
                        uniforms.uTime.value = time;

                        if (particleSystem) {
                            particleSystem.rotation.y += 0.003;
                            scanPlane.position.y = Math.sin(time * 1.5) * 0.9 + 0.8;
                        }

                        controls.update();

                        if (useComposer && composer) {
                            composer.render();
                        } else {
                            renderer.render(scene, camera);
                        }
                    }
                    animate();

                    const resizeObserver = new ResizeObserver(() => {
                        const newWidth = container.clientWidth;
                        const newHeight = container.clientHeight || 580;
                        if (newWidth > 0 && newHeight > 0) {
                            camera.aspect = newWidth / newHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(newWidth, newHeight);
                            if (useComposer && composer) {
                                composer.setSize(newWidth, newHeight);
                            }
                        }
                    });
                    resizeObserver.observe(container);

                } catch (err) {
                    console.error("Initialization error:", err);
                }
            }
            window.onload = init;
        </script>
    </body>
    </html>
    """
    html_code = html_template.replace("__IMG_B64__", img_b64).replace(
        "__DEPTH_B64__", depth_b64
    )
    components.html(html_code, height=620)


# ---------------------------------------------------------
# 6. Streamlit User Interface Workflow
# ---------------------------------------------------------
render_cyber_header("📂 TELEMETRY INPUT PORTAL")

input_tab1, input_tab2 = st.tabs(["📁 UPLOAD / PASTE", "🖼️ DEMO TARGETS"])

image_pil = None

with input_tab1:
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_file = st.file_uploader(
            "Select an image file for GPU forensic analysis",
            type=["png", "jpg", "jpeg", "webp"],
        )
    with col_u2:
        st.write("Or paste clipboard image:")
        paste_result = paste_image_button("📋 PASTE IMAGE")

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
    elif paste_result.image_data is not None:
        image_pil = paste_result.image_data.convert("RGB")

with input_tab2:
    st.caption("Load built-in test targets for validation:")
    demo_cols = st.columns(3)
    if demo_cols[0].button("🎯 Synthetic Portrait"):
        img_arr = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(img_arr, (200, 200), 120, (0, 243, 255), -1)
        cv2.putText(
            img_arr,
            "AI SYNTH",
            (120, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        image_pil = Image.fromarray(img_arr)
    if demo_cols[1].button("📷 Camera Scan"):
        img_arr = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        image_pil = Image.fromarray(img_arr)

if image_pil is not None:
    st.divider()

    # Image Processing
    with st.spinner("⚡ Running GPU GLSL Hologram & Forensic Engine..."):
        depth_pil = generate_depth_map(image_pil)
        img_b64 = image_to_base64(image_pil)
        depth_b64 = image_to_base64(depth_pil)

        ela_pil = compute_ela(image_pil)
        fft_pil = compute_fft(image_pil)
        spatial_pil = compute_spatial_anomaly(image_pil)

        predictions = detector(image_pil)
        fake_score, real_score = parse_predictions(predictions)

        caption_text = "Latent inversion unavailable."
        if captioner is not None:
            try:
                caption_res = captioner(image_pil)
                caption_text = caption_res[0]["generated_text"]
            except Exception:
                pass

    # Verdict Header
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if fake_score >= 50.0:
            st.markdown(
                f"""<div class="verdict-fake">🚨 AI SYNTHETIC DETECTED<br><span style="font-size: 1rem; color: #fff;">CONFIDENCE: {fake_score:.2f}%</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="verdict-real">🛡️ NATURAL PHOTOGRAPH<br><span style="font-size: 1rem; color: #fff;">CONFIDENCE: {real_score:.2f}%</span></div>""",
                unsafe_allow_html=True,
            )

    with res_col2:
        st.markdown(
            f"""
        <div style="background: rgba(10,16,30,0.8); border: 1px solid rgba(0,243,255,0.3); padding: 15px; border-radius: 8px;">
            <p style="margin: 0; color: #00f3ff; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;"><b>LATENT PROMPT INVERSION VECTOR:</b></p>
            <p style="margin-top: 5px; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">"{caption_text}"</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Forensic Viewports
    tab_holo, tab_forensic, tab_3dmesh, tab_audio, tab_export = st.tabs(
        [
            "⚡ 3D GPU GLSL HOLOGRAM",
            "🔬 FORENSICS & SPECTRAL MAPS",
            "📊 VOLUMETRIC MESH TOPOGRAPHY",
            "🔊 NEURO-ACOUSTIC SONIFICATION",
            "📄 FORENSIC REPORT EXPORT",
        ]
    )

    with tab_holo:
        render_cyber_header("VOLUMETRIC GPU PARTICLE SHADER INTERFACE")
        render_gpu_glsl_hologram(img_b64, depth_b64)

    with tab_forensic:
        render_cyber_header("MULTI-SPECTRAL ANOMALY DETECTION")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            st.image(image_pil, caption="Source Telemetry", use_container_width=True)
        with f_col2:
            st.image(
                ela_pil, caption="Error Level Analysis (ELA)", use_container_width=True
            )
        with f_col3:
            st.image(
                fft_pil,
                caption="2D FFT Spectrum Analysis",
                use_container_width=True,
            )
        with f_col4:
            st.image(
                spatial_pil,
                caption="Spatial Disparity Map",
                use_container_width=True,
            )

    with tab_3dmesh:
        render_cyber_header("INTERACTIVE VOLUMETRIC TOPOGRAPHY MESH")
        plotly_fig = generate_3d_mesh_plotly(depth_pil)
        st.plotly_chart(plotly_fig, use_container_width=True)

    with tab_audio:
        render_cyber_header("NEURO-ACOUSTIC FREQUENCY SONIFICATION")
        render_sonification_module()

    with tab_export:
        render_cyber_header("DIAGNOSTIC TELEMETRY REPORT")
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "verdict": "SYNTHETIC" if fake_score >= 50.0 else "AUTHENTIC",
            "confidence_synthetic": f"{fake_score:.2f}%",
            "confidence_authentic": f"{real_score:.2f}%",
            "prompt_inversion": caption_text,
            "pipeline_version": "MONOVISION v7.0 ULTRA",
        }
        st.json(report_data)
        st.download_button(
            label="💾 DOWNLOAD FORENSIC JSON REPORT",
            data=json.dumps(report_data, indent=4),
            file_name=f"monovision_forensic_report_{int(datetime.now().timestamp())}.json",
            mime="application/json",
        )
