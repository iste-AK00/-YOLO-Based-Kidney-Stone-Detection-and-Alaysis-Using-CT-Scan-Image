import base64
import glob
import io
import os
import re
import time
from fpdf import FPDF
from groq import Groq
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    .sidebar-datasets {
        text-align: left;
        margin-bottom: 0.6rem;
        margin-top: -0.8rem;
    }
    .sidebar-datasets .datasets-title {
        color: #7dd3fc;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
        display: block;
    }
    .sidebar-datasets .dataset-box {
        background: rgba(125, 211, 252, 0.08);
        border: 1px solid rgba(125, 211, 252, 0.3);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        color: #cbd5e1;
        font-size: 0.78rem;
    }
    .sidebar-datasets .dataset-box .dataset-item {
        padding: 0.25rem 0;
    }
    .sidebar-datasets .dataset-box .dataset-item:not(:last-child) {
        border-bottom: 1px solid rgba(125, 211, 252, 0.15);
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a4a);
        border: 1px solid #3a3a5a;
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(233, 69, 96, 0.18);
        border-color: rgba(233, 69, 96, 0.4);
    }
    .metric-card .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e94560;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #a8b2d8;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .status-detected {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(231, 76, 60, 0.35);
    }
    .status-clear {
        background: linear-gradient(135deg, #1a6b3a, #27ae60);
        color: white;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(39, 174, 96, 0.35);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e94560;
        border-left: 4px solid #e94560;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }

    .stFileUploader > div {
        border: 2px dashed #e94560 !important;
        border-radius: 12px !important;
        background: rgba(233, 69, 96, 0.04) !important;
        transition: background 0.18s ease, border-color 0.18s ease;
    }
    .stFileUploader > div:hover {
        background: rgba(233, 69, 96, 0.08) !important;
        border-color: #ff6b81 !important;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #a8b2d8 !important;
    }

    hr { border-color: #2a2a4a !important; }

    .info-box {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.35);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #7dd3fc;
        font-size: 0.9rem;
        line-height: 1.65;
    }
    .warn-box {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.35);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #fde047;
        font-size: 0.9rem;
    }

    .footer-box {
        background: rgba(0,0,0,0.25);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        color: #718096;
        font-size: 0.82rem;
        text-align: center;
    }
    .footer-creators {
        margin-top: 0.9rem;
        padding-top: 0.9rem;
        border-top: 1px solid #2a2a4a;
        color: #a8b2d8;
        font-size: 0.85rem;
    }
    .footer-creators .creators-title {
        color: #e94560;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 0.72rem;
        display: block;
        margin-bottom: 0.35rem;
    }
    .footer-creators .creator-name {
        display: inline-block;
        background: rgba(233, 69, 96, 0.08);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 50px;
        padding: 0.25rem 0.9rem;
        margin: 0.2rem;
        color: #e2e8f0;
        font-size: 0.82rem;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .footer-creators .creator-name:hover {
        background: rgba(233, 69, 96, 0.18);
        border-color: #e94560;
    }

    /* Hide only the Deploy button; keep the 3-dot menu visible */
    [data-testid="stAppDeployButton"] {
        display: none;
    }
    div[data-testid="stToolbarActions"] button[kind="header"] {
        display: none;
    }

    /* ---- Landing page ---- */
    .landing-wrap {
        position: relative;
        overflow: hidden;
        min-height: 40vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        border-radius: 16px;
        background: radial-gradient(circle at 50% 30%, rgba(233,69,96,0.06), transparent 60%);
    }
    .landing-bg-icon {
        position: absolute;
        top: 38%;
        right: 6%;
        transform: translateY(-50%);
        width: 150px;
        max-width: 22%;
        opacity: 0.55;
        z-index: 0;
        pointer-events: none;
    }
    .landing-logo {
        position: absolute;
        top: 38%;
        left: 6%;
        transform: translateY(-50%);
        width: 150px;
        height: 150px;
        border-radius: 50%;
        z-index: 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.45), 0 0 0 4px rgba(233,69,96,0.15);
    }
    .landing-content {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .landing-wrap h1 {
        color: #e94560;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }
    .landing-wrap .landing-sub {
        color: #a8b2d8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        max-width: 700px;
    }
    .landing-badges {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.7rem;
        margin-bottom: 0.5rem;
    }
    .landing-badge {
        background: rgba(125, 211, 252, 0.07);
        border: 1px solid rgba(125, 211, 252, 0.28);
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        color: #cbd5e1;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }
    .landing-badge:hover {
        transform: translateY(-2px);
        border-color: rgba(125, 211, 252, 0.6);
        background: rgba(125, 211, 252, 0.12);
    }
    .landing-badge b {
        color: #7dd3fc;
        font-weight: 700;
    }

    /* Responsive tweaks for small / narrow viewports */
    @media (max-width: 900px) {
        .landing-bg-icon, .landing-logo {
            display: none;
        }
        .landing-wrap h1 {
            font-size: 2.1rem;
        }
        .st-key-top_nav_bar {
            right: 1rem !important;
            top: 3.2rem !important;
        }
    }

    /* ---- About section (Project Overview / Features / Tech Stack) ---- */
    .about-wrap {
        max-width: 100%;
        margin: 1.5rem 0 2rem 0;
    }
    .about-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .about-card:hover {
        border-color: rgba(233, 69, 96, 0.35);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }
    .about-card h3 {
        color: #e94560;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .about-card p {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.75;
        margin: 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.9rem;
    }
    .feature-item {
        background: rgba(233, 69, 96, 0.06);
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.55;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }
    .feature-item:hover {
        transform: translateY(-3px);
        border-color: rgba(233, 69, 96, 0.5);
        background: rgba(233, 69, 96, 0.1);
        box-shadow: 0 8px 20px rgba(233, 69, 96, 0.15);
    }
    .feature-item b {
        color: #e94560;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 0.92rem;
    }
    .tech-stack-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .tech-groups-row {
        display: flex;
        flex-wrap: wrap;
        gap: 2.4rem;
        align-items: flex-start;
    }
    .tech-groups-row .tech-group-block {
        flex: 0 1 auto;
        min-width: 0;
    }
    .tech-pill {
        background: rgba(125, 211, 252, 0.08);
        border: 1px solid rgba(125, 211, 252, 0.3);
        border-radius: 50px;
        padding: 0.4rem 1rem;
        color: #7dd3fc;
        font-size: 0.82rem;
        font-weight: 600;
        transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
    }
    .tech-pill:hover {
        transform: translateY(-2px);
        background: rgba(125, 211, 252, 0.16);
        border-color: rgba(125, 211, 252, 0.55);
    }
    .tech-group-label {
        color: #a8b2d8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin: 0.9rem 0 0.5rem 0;
    }
    .tech-group-label:first-child {
        margin-top: 0;
    }

    /* ---------------------------------------------------------------
       Top-right nav bar, pinned beside Streamlit's native 3-dot menu.
       st.container(key="top_nav_bar") gets a stable ".st-key-top_nav_bar"
       class in modern Streamlit — that's what we target here.
    --------------------------------------------------------------- */
    .st-key-top_nav_bar {
        position: fixed !important;
        top: 0.55rem !important;
        right: 5rem !important;
        z-index: 999999 !important;
        width: auto !important;
        background: rgba(26, 26, 46, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        padding: 0.3rem 0.5rem !important;
        border-radius: 50px !important;
        border: 1px solid rgba(233, 69, 96, 0.35) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    }
    .st-key-top_nav_bar [data-testid="stHorizontalBlock"] {
        gap: 0.3rem !important;
    }
    .st-key-top_nav_bar [data-testid="stColumn"] {
        width: auto !important;
        min-width: fit-content !important;
        flex: 0 0 auto !important;
    }
    .st-key-top_nav_bar .stButton > button {
        border-radius: 50px !important;
        padding: 0.25rem 0.85rem !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        min-height: 0 !important;
        height: auto !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(233, 69, 96, 0.25) !important;
        color: #cbd5e1 !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="secondary"]:hover {
        background: rgba(233, 69, 96, 0.18) !important;
        border-color: #e94560 !important;
        color: #ffffff !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        border: 1px solid #e94560 !important;
        color: #ffffff !important;
    }

    /* Collapse the empty vertical space the nav row would otherwise leave behind */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-top_nav_bar) {
        margin: 0 !important;
        min-height: 0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "runs", "detect", "train")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


MODEL_INFO = {
    "name": "YOLO26x",
    "variant": "xl (extra-large)",
    "epochs": 50,
    "precision": 85.28,
    "recall": 80.31,
    "map50": 84.74,
    "map50_95": 41.32,
    "f1": 82.73,
}


def find_first_existing(*filenames):
    """Returns the first existing file path (inside RESULTS_DIR) among the filenames, or None."""
    for name in filenames:
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(path):
            return path
    return None



@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_ai_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def generate_ai_report(stone_count, avg_conf, severity_label, detections):
    """Sends detection summary to an LLM and returns a generated medical-style report."""
    client = get_ai_client()

    detection_lines = (
        "\n".join([
            f"- Stone #{d['Detection #']}: confidence {d['Conf %']},"
            f" area {d['Area %']}, size {d['Size (WxH)']},"
            f" location {d['Location (x1,y1)']}"
            for d in detections
        ])
        if detections
        else "No stones detected."
    )

    prompt = f"""
You are a radiology assistant. Based on the following kidney stone detection
results, write a professional medical report in Markdown format:

- Stones found: {stone_count}
- Average confidence: {avg_conf*100:.1f}%
- Severity: {severity_label}
- Per-stone details:
{detection_lines}

The report should have three sections:
1. Findings
2. Impression
3. Recommendation

End with a disclaimer stating that this report is for informational purposes
only, is not a final medical diagnosis, and that a qualified radiologist or
urologist should be consulted for an actual clinical decision.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _pdf_safe(text: str) -> str:
    """Strips markdown symbols and forces text into Latin-1 (core PDF fonts only support Latin-1)."""
    text = re.sub(r"[#*`_]", "", text)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(filename, stone_count, avg_conf, severity_label, detections, ai_report_text):
    """Builds a downloadable PDF summarizing detection results and the AI-generated report."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, "Kidney Stone Detection Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _pdf_safe(f"File: {filename}"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Stones Found: {stone_count}"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Average Confidence: {avg_conf*100:.1f}%"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Severity: {severity_label}"), ln=True)
    pdf.ln(4)

    if detections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detected Stones", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for d in detections:
            line = (
                f"Stone #{d['Detection #']}: confidence {d['Conf %']}, "
                f"area {d['Area %']}, size {d['Size (WxH)']}, "
                f"location {d['Location (x1,y1)']}"
            )
            pdf.multi_cell(0, 6, _pdf_safe(line))
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI-Generated Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _pdf_safe(ai_report_text))
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0, 5,
        _pdf_safe(
            "Disclaimer: This report is generated for informational and educational "
            "purposes only. It is not a certified medical diagnosis. Please consult a "
            "qualified radiologist or urologist for an actual clinical decision."
        ),
    )

    return bytes(pdf.output(dest="S"))


def get_severity(count: int, avg_conf: float) -> tuple[str, str]:
    if count == 0:
        return "No Stones Detected", "#27ae60"
    elif count == 1:
        if avg_conf >= 0.75:
            return "Single Stone – High Confidence", "#e67e22"
        else:
            return "Single Stone – Possible", "#f39c12"
    elif count <= 3:
        return "Multiple Stones – Moderate", "#e74c3c"
    else:
        return "Multiple Stones – Severe", "#c0392b"


FOOTER_HTML = """
<div class="footer-box">
    ⚕️ <b>Medical Disclaimer:</b> This application is intended for research and educational
    purposes only. It does not constitute medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare professional for medical decisions.
    <div class="footer-creators">
        <span class="creators-title">👥 Project Created By</span>
        <span class="creator-name">Ishteuk Ahmed Joy</span>
        <span class="creator-name">Israt Jahan</span>
        <span class="creator-name">Dipayon Nag</span>
        <span class="creator-name">Tanim</span>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Navigation state & UI (fixed, top-right, beside Streamlit's 3-dot menu)
# ---------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "home"


def go_to(page: str):
    st.session_state.nav_page = page
    st.rerun()


nav_bar = st.container(key="top_nav_bar")
with nav_bar:
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button(
            "🏠 Home",
            key="nav_home",
            type="primary" if st.session_state.nav_page == "home" else "secondary",
        ):
            go_to("home")
    with nav_col2:
        if st.button(
            "🧠 Model Info",
            key="nav_model",
            type="primary" if st.session_state.nav_page == "model" else "secondary",
        ):
            go_to("model")
    with nav_col3:
        if st.button(
            "📤 Upload & Detect",
            key="nav_upload",
            type="primary" if st.session_state.nav_page == "upload" else "secondary",
        ):
            go_to("upload")


# ---------------------------------------------------------------------------
# PAGE: HOME (landing)
# ---------------------------------------------------------------------------
if st.session_state.nav_page == "home":
    logo_b64 = get_base64_image(os.path.join(ASSETS_DIR, "logo.png"))
    bg_icon_b64 = get_base64_image(os.path.join(ASSETS_DIR, "kidney_bg.png"))

    bg_icon_html = (
        f'<img class="landing-bg-icon" src="data:image/png;base64,{bg_icon_b64}" />'
        if bg_icon_b64 else ""
    )
    logo_html = (
        f'<img class="landing-logo" src="data:image/png;base64,{logo_b64}" />'
        if logo_b64 else ""
    )

    st.markdown(
        f"""<div class="landing-wrap">
{logo_html}
{bg_icon_html}
<div class="landing-content">
<h1>🔬 Kidney Stone Detection System</h1>
<p class="landing-sub">An AI-assisted diagnostic support tool that detects kidney
stones from medical images (CT scans) using a custom-trained YOLO object-detection
model and generates automated radiology reports via LLMs.</p>
<div class="landing-badges">
<span class="landing-badge">🧬 <b>YOLO26x Model</b></span>
<span class="landing-badge">🎯 <b>85.3% Precision</b></span>
<span class="landing-badge">📊 <b>2,659 Training Images</b></span>
<span class="landing-badge">⚡ <b>Real-Time Detection</b></span>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Primary call-to-action sits right under the hero, above the About cards
    cta_top1, cta_top2, cta_top3 = st.columns([1, 1, 1])
    with cta_top2:
        if st.button("🚀 Start Detection", use_container_width=True, type="primary", key="cta_start"):
            go_to("upload")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """<div class="about-wrap">
<div class="about-card">
<h3>📌 Project Overview</h3>
<p>This project is developed as part of an undergraduate thesis in Computer
Science and Engineering (CSE). It leverages modern Computer Vision techniques
to automate the detection, localization, and classification of kidney stones,
helping clinicians with faster and more reliable initial screening.</p>
</div>

<div class="about-card">
<h3>✨ Key Features</h3>
<div class="feature-grid">
<div class="feature-item"><b>🎯 Real-Time Detection</b>Accurate detection of kidney stones with bounding box overlays.</div>
<div class="feature-item"><b>📈 Performance Evaluation</b>High precision and mAP metrics on benchmark datasets.</div>
<div class="feature-item"><b>🖥️ Interactive Web Interface</b>Streamlit-based UI for seamless image upload and visualization.</div>
<div class="feature-item"><b>📝 AI Medical Report Generation</b>Automated summary reports powered by LLM integration.</div>
</div>
</div>

<div class="about-card">
<h3>🛠️ Tech Stack &amp; Libraries</h3>
<div class="tech-groups-row">
<div class="tech-group-block">
<div class="tech-group-label">Language</div>
<div class="tech-stack-row">
<span class="tech-pill">Python</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Frameworks</div>
<div class="tech-stack-row">
<span class="tech-pill">Streamlit</span>
<span class="tech-pill">PyTorch</span>
<span class="tech-pill">Ultralytics YOLO</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Libraries</div>
<div class="tech-stack-row">
<span class="tech-pill">OpenCV</span>
<span class="tech-pill">PIL</span>
<span class="tech-pill">NumPy</span>
<span class="tech-pill">Pandas</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Model Architecture</div>
<div class="tech-stack-row">
<span class="tech-pill">YOLO Series (Custom Trained)</span>
</div>
</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# PAGE: MODEL INFO
# ---------------------------------------------------------------------------
if st.session_state.nav_page == "model":
    st.markdown(
        """
    <div class="main-header">
        <h1>🧠 About The Model</h1>
        <p>Details on the computer vision model powering this detection system</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Hide the sidebar entirely on this page — full-width layout, dataset details live in the Dataset Info tab
    st.markdown(
        "<style>section[data-testid='stSidebar'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

    tab_model, tab_arch, tab_perf, tab_dataset = st.tabs([
        "📋 Model Summary",
        "🏗️ Architecture",
        "📈 Performance Evaluation",
        "📊 Dataset Info",
    ])

    # --- Model Summary ---
    with tab_model:
        st.markdown('<div class="section-title">Model Used</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Property": ["Model", "Variant", "Framework", "Task", "Class(es)", "Epochs Trained"],
            "Value": [
                MODEL_INFO["name"],
                MODEL_INFO["variant"],
                "Ultralytics YOLO",
                "Object Detection",
                "stone",
                str(MODEL_INFO["epochs"]),
            ],
        }).set_index("Property"))

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">How Detection Works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            1. The uploaded image is passed to the model.<br>
            2. The model predicts candidate bounding boxes with confidence scores.<br>
            3. Boxes below the <b>Confidence Threshold</b> are discarded.<br>
            4. Overlapping boxes are merged using <b>Non-Maximum Suppression (IoU)</b>.<br>
            5. Remaining boxes are drawn on the image and summarized in the results table.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">AI Report Generation</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            After detection, an LLM (Llama 3.3 70B via Groq) can generate a
            structured, radiology-style report summarizing the findings —
            for informational purposes only.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Architecture ---
    with tab_arch:
        st.markdown('<div class="section-title">YOLO26 Architecture</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        This system uses <b>{MODEL_INFO['name']}</b> — the <b>xl (extra-large)</b> variant of
        <b>YOLO26</b>, the latest generation in the YOLO (You Only Look Once) family of
        single-stage, anchor-free object detectors. It is fine-tuned specifically to identify
        kidney stones in CT-scan imagery, processing the entire image in a single forward pass
        and predicting all bounding boxes and confidence scores simultaneously.
        </div>
        """, unsafe_allow_html=True)

        arch_img_path = os.path.join(ASSETS_DIR, "architecture.png")
        if os.path.isfile(arch_img_path):
            st.image(arch_img_path, caption="YOLO26 Architecture Diagram", use_container_width=True)
        else:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Architecture diagram not found. Upload an image below,
                or place one at <code>assets/architecture.png</code> in the
                project folder — it will then show up here automatically.
            </div>
            """, unsafe_allow_html=True)
            uploaded_arch_img = st.file_uploader(
                "Upload Architecture Diagram",
                type=["png", "jpg", "jpeg"],
                key="arch_uploader",
            )
            if uploaded_arch_img is not None:
                st.image(uploaded_arch_img, caption="YOLO26 Architecture Diagram", use_container_width=True)

        st.markdown("""
        The network is organized into three stages — **Backbone**, **Neck**, and **Head**:

        #### 1. Backbone (Feature Extraction)
        The input image (640×640×3) is passed through a stack of strided **Conv** layers
        (each halving the spatial resolution: P1 → P5) interleaved with **C3k2** blocks —
        CSP-style blocks made of stacked 3×3 convolutions that extract increasingly abstract
        features while keeping the parameter count efficient. At the deepest stage (P5), two
        specialized blocks refine the features further:
        - **SPPF (Spatial Pyramid Pooling – Fast)** — pools features at multiple receptive
          field sizes so the network sees both local detail and global context.
        - **C2PSA (Parallel Spatial Attention)** — an attention block that lets the network
          focus on the most relevant spatial regions, useful for small, low-contrast objects
          such as kidney stones.

        #### 2. Neck (Multi-Scale Feature Fusion)
        The neck follows a **PAN-FPN** (Path Aggregation + Feature Pyramid) design:
        - A **top-down path** upsamples deep, semantically-rich features (P5) and concatenates
          them with shallower, higher-resolution features (P4, P3), refined at each step by C3k2 blocks.
        - A **bottom-up path** then re-downsamples these fused features back through P4 and P5,
          so every scale ends up with both fine spatial detail and strong semantic context.

        This fusion is what allows the model to detect kidney stones of very different sizes —
        from tiny sub-5 mm stones to large clusters — within the same image.

        #### 3. Head (Multi-Scale Detection)
        Three parallel **Detect** heads operate on the fused P3, P4, and P5 feature maps
        (80×80, 40×40, and 20×20 respectively), each independently predicting bounding boxes,
        objectness, and class confidence at that scale. Detecting at three resolutions
        simultaneously means small stones (best seen at P3) and larger ones (best seen at P5)
        are both captured reliably in a single pass.

        #### Model Scaling
        YOLO26 defines five variants (**n, s, m, l, xl**) scaled by a **depth multiplier (d)**,
        **width multiplier (w)**, and **max channel cap (mc)**. This project uses the **xl**
        variant — the deepest and widest configuration — trading some inference speed for the
        highest accuracy, which is appropriate for an offline/near-real-time diagnostic-support
        tool rather than a strict real-time video pipeline.

        #### Why YOLO26 specifically (vs. earlier YOLO versions)
        Independent latency-vs-accuracy benchmarks (COCO mAP50-95 vs. TensorRT FP16 latency)
        show YOLO26 achieving a better accuracy-per-millisecond trade-off than YOLO11, YOLOv10,
        YOLOv9, and YOLOv8 across every model size — at any given latency budget, YOLO26 reaches
        a higher mAP. This makes it a strong choice for a medical imaging task where both
        detection accuracy (missing a stone is costly) and reasonable inference speed matter.
        """)

        perf_img_path = os.path.join(ASSETS_DIR, "performance_comparison.png")
        if os.path.isfile(perf_img_path):
            st.markdown('<div class="section-title" style="margin-top:1rem;">YOLO Version Comparison</div>', unsafe_allow_html=True)
            st.image(
                perf_img_path,
                caption="YOLO26 vs. YOLO11 / YOLOv10 / YOLOv9 / YOLOv8 — latency vs. COCO mAP50-95",
                use_container_width=True,
            )
        else:
            st.caption(
                "Optional: place a latency-vs-accuracy comparison chart at "
                "`assets/performance_comparison.png` to display it here."
            )

    # --- Performance Evaluation ---
    with tab_perf:
        st.markdown('<div class="section-title">Metrics Summary</div>', unsafe_allow_html=True)
        st.caption(f"Results on the validation set after {MODEL_INFO['epochs']} epochs of training.")

        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        for col, label, val in zip(
            [pm1, pm2, pm3, pm4, pm5],
            ["Precision", "Recall", "mAP@50", "mAP@50-95", "F1-Score"],
            [MODEL_INFO["precision"], MODEL_INFO["recall"], MODEL_INFO["map50"], MODEL_INFO["map50_95"], MODEL_INFO["f1"]],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.7rem;">{val:.1f}%</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Result Plots</div>', unsafe_allow_html=True)
        st.caption(f"Source: `{RESULTS_DIR}`")

        result_plots = {
            "Confusion Matrix": ["confusion_matrix.png"],
            "Confusion Matrix (Normalized)": ["confusion_matrix_normalized.png"],
            "Overall Results (loss / mAP curves)": ["results.png"],
            "Precision-Recall Curve": ["BoxPR_curve.png", "PR_curve.png"],
            "F1 Curve": ["BoxF1_curve.png", "F1_curve.png"],
            "Precision Curve": ["BoxP_curve.png", "P_curve.png"],
            "Recall Curve": ["BoxR_curve.png", "R_curve.png"],
        }

        any_found = False
        plot_cols = st.columns(2)
        i = 0
        for label, filenames in result_plots.items():
            found_path = find_first_existing(*filenames)
            if found_path:
                any_found = True
                with plot_cols[i % 2]:
                    st.markdown(f"**{label}**")
                    st.image(found_path, use_container_width=True)
                i += 1

        if not any_found:
            st.markdown("""
            <div class="warn-box">
                ⚠️ No training result plots were found. Please verify the
                files exist under <code>runs/detect/train/</code>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sample Validation Predictions</div>', unsafe_allow_html=True)
        val_pred_paths = sorted(
            glob.glob(os.path.join(RESULTS_DIR, "val_batch*_pred.jpg"))
        )
        if val_pred_paths:
            vp_cols = st.columns(min(3, len(val_pred_paths)))
            for idx, vp_path in enumerate(val_pred_paths):
                with vp_cols[idx % len(vp_cols)]:
                    st.image(vp_path, use_container_width=True)
        else:
            st.caption("No validation prediction samples found.")

    # --- Dataset Info (placed after Performance Evaluation) ---
    with tab_dataset:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown('<div class="section-title">Dataset Sources</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="info-box">
                Two public kidney-stone object-detection datasets were merged to build the
                training set:
                <ul>
                    <li><b>Kidney Stone Images with Bounding Box Annotations</b> (Kaggle,
                    by Safura Hajiheidari) — ~1,300 CT-scan images with YOLO-format bounding
                    box annotations, itself sourced from Roboflow Universe.</li>
                    <li><b>Kidney Stone Detection</b> (Roboflow Universe, by East West
                    University) — 1,299 CT-scan images with bounding box annotations.</li>
                </ul>
                Because both datasets originate from overlapping Roboflow sources, the merged
                collection was de-duplicated and consolidated before training.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d2:
            st.markdown('<div class="section-title">Annotation & Classes</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="info-box">
                <b>Target Class:</b> <code>stone</code> (single-class object detection)<br><br>
                Bounding boxes are represented using normalized YOLO coordinate format:
                <code>[class_id, x_center, y_center, width, height]</code>.<br>
                Both source datasets provide CT-scan images with kidney stones annotated
                across a range of sizes, shapes, and positions within the urinary system.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Distribution Summary</div>', unsafe_allow_html=True)
        st.caption("Final merged dataset used for training: 2,659 images total.")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">2,290</div>
                    <div class="metric-label">Training Images (86.1%)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_d2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">123</div>
                    <div class="metric-label">Validation Images (4.6%)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_d3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">246</div>
                    <div class="metric-label">Test Images (9.3%)</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Citations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem;">
        S. Hajiheidari, "Kidney Stone Images with Bounding Box Annotations," Kaggle, 2023.
        Available: kaggle.com/datasets/safurahajiheidari/kidney-stone-images<br><br>
        East West University, "Kidney Stone Detection Dataset," Roboflow Universe, 2023.
        Available: universe.roboflow.com/east-west-university-9frzq/kidney-stone-detection-wfjba
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
    <div class="warn-box" style="margin-top:1.4rem;">
        ⚠️ This model is a research/educational tool. It is <b>not</b> a certified
        diagnostic device. Detection results and confidence scores should always
        be verified by a qualified radiologist or urologist.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# PAGE: UPLOAD & DETECT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are discarded.",
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression overlap threshold.",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#a8b2d8;font-size:0.85rem;'>This application is intended for "
        "research and educational purposes only. It does not constitute medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare professional for "
        "medical decisions.</p>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="main-header">
    <h1>🔬 Kidney Stone Detection System</h1>
</div>
""",
    unsafe_allow_html=True,
)

with st.spinner("Loading model…"):
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIFF",
)

if uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()

# Reset the previously generated AI report whenever a new/different image is uploaded
current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.get("last_uploaded_file_id") != current_file_id:
    st.session_state.pop("ai_report", None)
    st.session_state["last_uploaded_file_id"] = current_file_id

image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

with st.spinner("Running detection…"):
    try:
        start = time.perf_counter()
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        elapsed_s = time.perf_counter() - start
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        st.stop()

result = results[0]
boxes = result.boxes

detections = []
if boxes is not None and len(boxes) > 0:
    for box in boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, f"class_{cls}")
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        area_pct = (w * h) / (image.width * image.height) * 100
        detections.append({
            "Detection #": len(detections) + 1,
            "Label": label,
            "Confidence": conf,
            "Conf %": f"{conf*100:.1f}%",
            "Area %": f"{area_pct:.2f}%",
            "Location (x1,y1)": f"({int(x1)}, {int(y1)})",
            "Size (WxH)": f"{int(w)}×{int(h)} px",
        })

stone_count = len(detections)
avg_conf = float(np.mean([d["Confidence"] for d in detections])) if detections else 0.0
max_conf = float(np.max([d["Confidence"] for d in detections])) if detections else 0.0
min_conf = float(np.min([d["Confidence"] for d in detections])) if detections else 0.0
severity_label, severity_color = get_severity(stone_count, avg_conf)

if stone_count > 0:
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    display_image = Image.fromarray(annotated_rgb)
    display_caption = (
        f"{uploaded_file.name}  ·  {image.width}×{image.height} px  · "
        f" {stone_count} stone(s) detected"
    )
else:
    display_image = image
    display_caption = (
        f"{uploaded_file.name}  ·  {image.width}×{image.height} px  ·  No stones"
        " detected"
    )

col_img, col_analysis = st.columns([1.05, 1], gap="large")

with col_img:
    st.markdown(
        f'<div class="section-title">{"Detection Result" if stone_count > 0 else "Uploaded Image"}</div>',
        unsafe_allow_html=True,
    )
    st.image(display_image, use_container_width=True)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    file_size_kb = buf.tell() / 1024

with col_analysis:
    st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)

    if stone_count > 0:
        st.markdown('<span class="status-detected">⚠️ Kidney Stone(s) Detected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-clear">✅ No Stones Detected</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stone_count}</div>
                <div class="metric-label">Stones Found</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_conf*100:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed_s:.2f}s</div>
                <div class="metric-label">Inference Time</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div style="background:rgba(0,0,0,0.2);border-left:4px solid {severity_color};
                border-radius:8px;padding:0.8rem 1rem;margin:0.5rem 0 1rem 0;">
        <span style="color:{severity_color};font-weight:700;">Severity Assessment</span><br>
        <span style="color:#ddd;font-size:0.95rem;">{severity_label}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if stone_count > 0:
        st.markdown(
            f"""
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">
                <div style="background:#1e2a3a;border-radius:8px;padding:0.5rem 0.9rem;flex:1;min-width:100px;text-align:center;">
                    <div style="color:#7dd3fc;font-size:0.78rem;text-transform:uppercase;">Max Conf</div>
                    <div style="color:#e2e8f0;font-weight:700;">{max_conf*100:.1f}%</div>
                </div>
                <div style="background:#1e2a3a;border-radius:8px;padding:0.5rem 0.9rem;flex:1;min-width:100px;text-align:center;">
                    <div style="color:#7dd3fc;font-size:0.78rem;text-transform:uppercase;">Min Conf</div>
                    <div style="color:#e2e8f0;font-weight:700;">{min_conf*100:.1f}%</div>
                </div>
                <div style="background:#1e2a3a;border-radius:8px;padding:0.5rem 0.9rem;flex:1;min-width:100px;text-align:center;">
                    <div style="color:#7dd3fc;font-size:0.78rem;text-transform:uppercase;">Threshold</div>
                    <div style="color:#e2e8f0;font-weight:700;">{conf_threshold*100:.0f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- AI Report Generation (full width, no side gap) ----
if stone_count > 0:
    st.markdown("---")
    header_col, control_col = st.columns([3, 1])
    with header_col:
        st.markdown('<div class="section-title">AI-Generated Report</div>', unsafe_allow_html=True)
    with control_col:
        report_columns = st.selectbox(
            "Layout",
            options=[1, 2, 3],
            index=1,
            format_func=lambda n: f"{n} Column{'s' if n > 1 else ''}",
            label_visibility="collapsed",
        )

    if st.button("🧾 Generate AI Report", use_container_width=True):
        with st.spinner("Generating AI report..."):
            try:
                ai_report = generate_ai_report(stone_count, avg_conf, severity_label, detections)
                st.session_state["ai_report"] = ai_report
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    if "ai_report" in st.session_state:
        report_html = st.session_state["ai_report"]
        st.markdown(
            f"""
                <style>
                .ai-report-columns {{
                    column-count: {report_columns};
                    column-gap: 2.2rem;
                    width: 100%;
                }}
                .ai-report-columns h1, .ai-report-columns h2, .ai-report-columns h3 {{
                    color: #e94560;
                    break-after: avoid;
                }}
                .ai-report-columns p, .ai-report-columns li {{
                    color: #e2e8f0;
                    break-inside: avoid;
                }}
                .ai-report-columns ul, .ai-report-columns ol {{
                    padding-left: 1.2rem;
                }}
                </style>
                """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ai-report-columns">\n\n{report_html}\n\n</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        try:
            pdf_bytes = build_pdf_report(
                uploaded_file.name, stone_count, avg_conf, severity_label, detections, report_html
            )
            st.download_button(
                label="⬇️ Download Report as PDF",
                data=pdf_bytes,
                file_name=f"kidney_stone_report_{uploaded_file.name.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

st.markdown("---")
st.markdown('<div class="section-title">Detailed Detection Results</div>', unsafe_allow_html=True)

if detections:
    df = pd.DataFrame(detections)[[
        "Detection #",
        "Label",
        "Conf %",
        "Area %",
        "Location (x1,y1)",
        "Size (WxH)",
    ]]
    st.dataframe(
        df.style.set_properties(**{
            "background-color": "#1e1e2e",
            "color": "#e2e8f0",
            "border": "1px solid #2a2a4a",
        }).set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#0f3460"),
                ("color", "#e94560"),
                ("font-weight", "bold"),
            ],
        }]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title" style="margin-top:1.2rem;">Confidence per Detection</div>',
        unsafe_allow_html=True,
    )
    chart_df = (
        pd.DataFrame({
            "Detection": [f"Stone #{d['Detection #']}" for d in detections],
            "Confidence (%)": [round(d["Confidence"] * 100, 2) for d in detections],
        })
        .sort_values("Confidence (%)", ascending=False)
        .set_index("Detection")
    )
    st.bar_chart(chart_df, color="#e94560", height=260)

else:
    st.markdown(
        f"""
    <div class="info-box">
        ✅ No kidney stones were detected at the current confidence threshold
        (<b>{conf_threshold*100:.0f}%</b>). Try lowering the threshold in the
        sidebar if you believe stones may be present.
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div class="section-title">Full Analysis Report</div>', unsafe_allow_html=True)

r1, r2 = st.columns(2, gap="medium")
with r1:
    st.markdown("**Image Information**")
    st.table(
        pd.DataFrame({
            "Property": ["Filename", "Dimensions", "Color Mode", "File Size"],
            "Value": [
                uploaded_file.name,
                f"{image.width} × {image.height} px",
                image.mode,
                f"{file_size_kb:.1f} KB",
            ],
        }).set_index("Property")
    )

with r2:
    st.markdown("**Model & Detection Info**")
    st.table(
        pd.DataFrame({
            "Property": [
                "Conf Threshold",
                "IoU Threshold",
                "Inference Time",
                "Stones Detected",
                "Avg Confidence",
            ],
            "Value": [
                f"{conf_threshold*100:.0f}%",
                f"{iou_threshold*100:.0f}%",
                f"{elapsed_s:.2f} s",
                str(stone_count),
                f"{avg_conf*100:.1f}%" if stone_count else "N/A",
            ],
        }).set_index("Property")
    )

st.markdown("---")
st.markdown(FOOTER_HTML, unsafe_allow_html=True)