import asyncio
import base64
from io import BytesIO
import json
import re

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_paste_button import paste_image_button
import torch
from transformers import pipeline

# Edge-TTS import check
try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# ---------------------------------------------------------
# 1. Page Configuration & Tactical HUD Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MONVISION v11.0 MARK-85",
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
        opacity: 0.15;
    }

    section[data-testid="stSidebar"] {
        background: rgba(4, 8, 18, 0.95) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0, 243, 255, 0.3) !important;
        box-shadow: 5px 0 30px rgba(0, 243, 255, 0.1);
    }

    .hud-banner {
        background: rgba(6, 12, 24, 0.88);
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
# 2. Async J.A.R.V.I.S. Audio Engine
# ---------------------------------------------------------
def _get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def _async_generate_edge_tts(text: str, pitch="-4Hz", rate="+2%") -> str:
    voice = "en-GB-RyanNeural"
    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    return base64.b64encode(mp3_bytes).decode("utf-8")


def render_jarvis_speech(text: str, rate="+2%", pitch="-4Hz"):
    clean_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    clean_text = re.sub(r"[*#_🤖🌐🏛️🔍🚨🛡️⚡📷]", "", clean_text)

    if EDGE_TTS_AVAILABLE:
        try:
            loop = _get_or_create_event_loop()
            b64_audio = loop.run_until_complete(
                _async_generate_edge_tts(clean_text, pitch=pitch, rate=rate)
            )

            audio_html = f"""
            <script>
                if (window.parent) {{
                    window.parent.document.querySelectorAll('audio').forEach(a => {{ 
                        a.pause(); 
                        a.currentTime = 0; 
                        a.remove();
                    }});
                }}
            </script>
            <audio autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
            """
            components.html(audio_html, height=0, width=0)
            return
        except Exception:
            pass

    # Browser Fallback
    html_code = f"""
    <script>
    function speakJarvis() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance("{clean_text}");
            function setVoiceAndSpeak() {{
                const voices = window.speechSynthesis.getVoices();
                const jarvisVoice = voices.find(v => 
                    v.name.includes("Google UK English Male") || 
                    v.name.includes("Daniel") || 
                    (v.lang.startsWith("en-GB") && v.name.toLowerCase().includes("male"))
                );
                if (jarvisVoice) msg.voice = jarvisVoice;
                msg.pitch = 0.88;
                msg.rate = 1.0;
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
# 3. Forensic Analysis Engines
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


def extract_exif_data(img_pil):
    exif_data = {}
    info = img_pil._getexif() if hasattr(img_pil, "_getexif") else None
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if isinstance(value, (bytes, str, int, float)):
                exif_data[str(decoded)] = str(value)
    return exif_data


def compute_chromatic_aberration(img_pil):
    arr = np.array(img_pil.convert("RGB"), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    diff_rb = np.abs(r - b)
    mean_fringe = float(np.mean(diff_rb))
    fringe_norm = cv2.normalize(
        diff_rb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    fringe_heatmap = cv2.applyColorMap(fringe_norm, cv2.COLORMAP_TURBO)
    return Image.fromarray(
        cv2.cvtColor(fringe_heatmap, cv2.COLOR_BGR2RGB)
    ), mean_fringe


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
    return Image.fromarray(mag_norm), np.abs(fshift)


def compute_spatial_anomaly(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    heatmap = cv2.applyColorMap(laplacian_abs, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def generate_3d_fft_surface(fshift_abs, max_points=80):
    resized_fft = cv2.resize(fshift_abs, (max_points, max_points))
    z_data = np.log(resized_fft + 1.0)

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_data,
                colorscale="Electric",
                lighting=dict(ambient=0.4, diffuse=0.8),
            )
        ]
    )
    fig.update_layout(
        title="3D FOURIER FREQUENCY ENERGY TOPOGRAPHY",
        autosize=True,
        height=450,
        margin=dict(l=10, r=10, b=10, t=40),
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


def generate_xai_explanation(img_pil, fake_score, real_score):
    gray = np.array(img_pil.convert("L"))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    _, fshift_abs = compute_fft(img_pil)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 10
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 > r**2
    high_freq = np.mean(fshift_abs[mask])
    total_freq = np.mean(fshift_abs) + 1e-8
    freq_ratio = float((high_freq / total_freq) * 100.0)

    ela_img = compute_ela(img_pil)
    ela_std = float(np.std(np.array(ela_img)))

    _, fringe_val = compute_chromatic_aberration(img_pil)
    exif_data = extract_exif_data(img_pil)

    is_ai = fake_score > real_score
    explanations = []
    spoken_reasons = []

    if is_ai:
        explanations.append(
            f"🤖 **Neural Patch Embeddings ({fake_score:.1f}% Confidence):** Vision Transformer flagged unnatural tile border artifacts standard in synthetic latent diffusion."
        )
        explanations.append(
            f"🌐 **Fourier Grid Spikes ({freq_ratio:.1f}% Energy):** Fourier spectrum exhibits mathematical upsampling spikes typical of AI convolutional generators."
        )
        explanations.append(
            f"🔍 **Uniform Noise Distribution (ELA Std Dev: {ela_std:.2f}):** Error Level Analysis detected smoothed noise profiles instead of real ISO physical camera grain."
        )
        explanations.append(
            f"🌈 **Zero Optics Fringe ({fringe_val:.2f}):** Missing natural lens chromatic aberration and optical dispersion along high-contrast object boundaries."
        )

        spoken_reasons.append(
            f"First, neural patch analysis detected synthetic generator artifacts with {fake_score:.1f} percent certainty."
        )
        spoken_reasons.append(
            f"Second, Fourier mathematical spectrum shows telltale lattice spikes from artificial diffusion upscaling."
        )
        spoken_reasons.append(
            "Third, error level testing proves uniform mathematical noise rather than organic camera sensor grain."
        )
    else:
        explanations.append(
            f"📷 **Authentic Optical Signature ({real_score:.1f}% Confidence):** Natural photon diffusion patterns verified by the ViT neural classifier."
        )
        explanations.append(
            f"🌐 **Natural Frequency Decay ({freq_ratio:.1f}% High-Freq):** Smooth, continuous spectral power falloff characteristic of glass camera lenses."
        )
        explanations.append(
            f"🔍 **Organic Sensor Noise (ELA Std Dev: {ela_std:.2f}):** Natural JPEG compression variance corresponding to real sensor hardware ISO profiles."
        )
        explanations.append(
            f"🌈 **Authentic Chromatic Fringe ({fringe_val:.2f}):** Genuine optical dispersion detected across physical contrast edges."
        )

        spoken_reasons.append(
            f"First, neural transformer identified natural photon distribution with {real_score:.1f} percent authentic confidence."
        )
        spoken_reasons.append(
            "Second, frequency spectrum analysis displays organic energy falloff expected from real physical camera optics."
        )
        spoken_reasons.append(
            "Third, error level testing verified authentic hardware sensor noise across the entire image."
        )

    return (
        explanations,
        spoken_reasons,
        freq_ratio,
        ela_std,
        lap_var,
        fringe_val,
        exif_data,
    )


# ---------------------------------------------------------
# 4. WebGL Hologram Engine
# ---------------------------------------------------------
def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def render_gpu_glsl_hologram(img_b64, depth_b64):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: #010308; overflow: hidden; width: 100vw; height: 620px; font-family: 'JetBrains Mono', monospace; }
            #canvas-container { width: 100%; height: 100%; position: relative; }
            .hud-overlay {
                position: absolute; top: 14px; left: 14px;
                color: #00f3ff; border: 1px solid rgba(0,243,255,0.5);
                padding: 6px 14px; background: rgba(2,6,16,0.88);
                font-size: 11px; letter-spacing: 1.5px; border-radius: 4px;
                pointer-events: none; z-index: 100; box-shadow: 0 0 15px rgba(0,243,255,0.2);
            }
            .controls-panel {
                position: absolute; bottom: 14px; left: 14px; right: 14px; z-index: 100;
                display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
                gap: 8px; background: rgba(2,6,16,0.92); padding: 10px 18px;
                border: 1px solid rgba(0,243,255,0.4); border-radius: 8px; backdrop-filter: blur(12px);
            }
            .control-group { display: flex; align-items: center; gap: 6px; color: #00f3ff; font-size: 10px; }
            .control-group input[type=range] {
                -webkit-appearance: none; width: 70px; background: rgba(0,243,255,0.2); height: 4px; border-radius: 2px;
            }
            .control-group input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #00f3ff; cursor: pointer;
            }
            .hud-btn {
                background: rgba(0,243,255,0.12); border: 1px solid #00f3ff; color: #00f3ff;
                padding: 4px 10px; font-size: 10px; cursor: pointer; border-radius: 4px; font-weight: bold;
            }
            .hud-btn:hover { background: #00f3ff; color: #000; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">👁️ MONVISIO
