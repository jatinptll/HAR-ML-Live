import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import socket
import sys
import time
import warnings
import streamlit.components.v1 as components

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAR · Activity Recognizer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  /* --- Base --- */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  .main { background: #0a0e1a; color: #e8eaf6; }
  [data-testid="stAppViewContainer"] { background: #0a0e1a; }
  [data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e2a4a;
  }

  /* --- Headings --- */
  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
  
  /* Hero title */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0;
  }

  .hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #4a5580;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.25rem;
  }

  /* --- Activity Result Card --- */
  .activity-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #2a3560;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
  }

  .activity-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
  }

  .activity-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4a5580;
    margin-bottom: 0.5rem;
  }

  .activity-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #e8eaf6;
    margin: 0.25rem 0;
  }

  .confidence-pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #60a5fa;
    margin-top: 0.5rem;
  }

  .risk-card {
    background: #111827;
    border: 1px solid #1e2a4a;
    border-radius: 12px;
    padding: 1.1rem;
    margin: 1rem 0;
  }

  .risk-title {
    font-size: 1rem;
    font-weight: 800;
    color: #e8eaf6;
    margin-bottom: 0.35rem;
  }

  .risk-copy {
    font-size: 0.82rem;
    color: #7b86b6;
    line-height: 1.55;
  }

  .risk-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
  }

  .risk-metrics .metric-card {
    min-width: 0;
    padding: 0.85rem 0.65rem;
  }

  .risk-metrics .metric-value {
    font-size: clamp(1rem, 2.6vw, 1.35rem);
    line-height: 1.15;
    overflow-wrap: anywhere;
  }

  .risk-metrics .metric-label {
    letter-spacing: 1.8px;
  }

  @media (max-width: 760px) {
    .risk-metrics {
      grid-template-columns: 1fr;
    }
  }

  .mode-note {
    border: 1px solid #21304f;
    background: rgba(17, 24, 39, 0.62);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    color: #7b86b6;
    font-size: 0.82rem;
    line-height: 1.55;
    margin-bottom: 1rem;
  }

  /* --- Metric cards --- */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
  }

  .metric-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1e2a4a;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
  }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
  }

  .metric-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a5580;
    margin-top: 0.2rem;
  }

  /* --- Slider customization --- */
  .stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
  }

  /* --- Section headers --- */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4a5580;
    border-bottom: 1px solid #1e2a4a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* --- Activity icons --- */
  .activity-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
  }

  /* --- Status indicator --- */
  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* --- Hide Streamlit branding --- */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIVITIES = [
    "Walking", "Walking Upstairs", "Walking Downstairs",
    "Sitting", "Standing", "Laying"
]

ACTIVITY_ICONS = {
    "Walking": "🚶",
    "Walking Upstairs": "🪜",
    "Walking Downstairs": "⬇️",
    "Sitting": "🪑",
    "Standing": "🧍",
    "Laying": "🛏️",
}

ACTIVITY_COLORS = {
    "Walking": "#60a5fa",
    "Walking Upstairs": "#a78bfa",
    "Walking Downstairs": "#f472b6",
    "Sitting": "#34d399",
    "Standing": "#fbbf24",
    "Laying": "#fb923c",
}

LIVE_REFRESH_COMPONENT = components.declare_component(
    "live_refresh_tick",
    path=os.path.join(os.path.dirname(__file__), "components", "live_refresh"),
)

BASE_DIR = os.path.dirname(__file__)
PHONE_BRIDGE_PATH = os.path.join(BASE_DIR, "phone_motion_latest.json")
PHONE_BRIDGE_PORT = 8765

FALL_LEVELS = {
    "Normal": {"color": "#34d399", "icon": "✅"},
    "Abnormal": {"color": "#fbbf24", "icon": "⚠️"},
    "Fall likely": {"color": "#fb7185", "icon": "🚨"},
}

# Default sensor presets for each activity
ACTIVITY_PRESETS = {
    "Walking":            {"acc_x": 0.28, "acc_y": -0.02, "acc_z": 0.09, "gyro_x": 0.12, "gyro_y": 0.05, "gyro_z": 0.03},
    "Walking Upstairs":   {"acc_x": 0.22, "acc_y": 0.10,  "acc_z": 0.12, "gyro_x": 0.18, "gyro_y": 0.08, "gyro_z": 0.04},
    "Walking Downstairs": {"acc_x": 0.20, "acc_y": -0.12, "acc_z": 0.10, "gyro_x": 0.15, "gyro_y": -0.06,"gyro_z": 0.03},
    "Sitting":            {"acc_x": 0.00, "acc_y": 0.00,  "acc_z":-0.08, "gyro_x": 0.00, "gyro_y": 0.00, "gyro_z": 0.00},
    "Standing":           {"acc_x": 0.00, "acc_y": 0.00,  "acc_z":-0.07, "gyro_x": 0.00, "gyro_y": 0.00, "gyro_z": 0.00},
    "Laying":             {"acc_x":-0.01, "acc_y": 0.00,  "acc_z": 0.97, "gyro_x": 0.00, "gyro_y": 0.00, "gyro_z": 0.00},
}

PRESET_BUTTON_LABELS = {
    "Walking": "Walking",
    "Walking Upstairs": "Upstairs",
    "Walking Downstairs": "Downstairs",
    "Sitting": "Sitting",
    "Standing": "Standing",
    "Laying": "Laying",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        # Auto-train if model not found
        import train_model
        train_model.train_and_save()
    with open(model_path, "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact["model"]
scaler = artifact["scaler"]
model_accuracy = artifact["accuracy"]
model_dataset = artifact.get("dataset", "UCI HAR Dataset")
model_feature_names = artifact.get("feature_names", [])
model_training_source = artifact.get("training_source", "actual UCI HAR dataset")

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, motion=None):
    """Build the compact feature vector used by the real UCI HAR-trained model."""
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    if motion:
        acc_x_std = safe_float(motion.get("acc_x_std"), 0)
        acc_y_std = safe_float(motion.get("acc_y_std"), 0)
        acc_z_std = safe_float(motion.get("acc_z_std"), 0)
        gyro_x_std = safe_float(motion.get("gyro_x_std"), 0)
        gyro_y_std = safe_float(motion.get("gyro_y_std"), 0)
        gyro_z_std = safe_float(motion.get("gyro_z_std"), 0)
        acc_rms = safe_float(motion.get("acc_rms_g"), acc_mag)
        gyro_rms = safe_float(motion.get("gyro_rms"), gyro_mag)
        peak_body = safe_float(motion.get("peak_body_g"), acc_mag)
        peak_jerk = safe_float(motion.get("peak_jerk_gs"), 0)
        stillness = safe_float(motion.get("stillness_score"), 0)
    else:
        dynamic_scale = 0.18 if acc_mag > 0.15 or gyro_mag > 0.08 else 0.03
        acc_x_std = abs(acc_x) * dynamic_scale
        acc_y_std = abs(acc_y) * dynamic_scale
        acc_z_std = abs(acc_z) * dynamic_scale
        gyro_x_std = abs(gyro_x) * dynamic_scale
        gyro_y_std = abs(gyro_y) * dynamic_scale
        gyro_z_std = abs(gyro_z) * dynamic_scale
        acc_rms = acc_mag
        gyro_rms = gyro_mag
        peak_body = acc_mag
        peak_jerk = 0.0
        stillness = 1.0 if acc_mag < 0.14 and gyro_mag < 0.18 else 0.0

    features = np.array(
        [
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            acc_x_std,
            acc_y_std,
            acc_z_std,
            gyro_x_std,
            gyro_y_std,
            gyro_z_std,
            acc_rms,
            gyro_rms,
            peak_body,
            peak_jerk,
            clamp(stillness, 0, 1),
        ],
        dtype=float,
    )

    return features.reshape(1, -1)


def predict_activity(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, motion=None):
    """Return predicted activity and probability distribution."""
    features = build_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, motion)
    features_scaled = scaler.transform(features)
    
    pred_idx = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]
    
    return ACTIVITIES[pred_idx], probs


def render_motion_metrics(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """Render derived motion magnitude metrics for the current input window."""
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    is_dynamic = acc_mag > 0.15 or gyro_mag > 0.08
    motion_type = "Dynamic" if is_dynamic else "Static"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{acc_mag:.3f}g</div>
            <div class="metric-label">Acc Magnitude</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{gyro_mag:.3f}</div>
            <div class="metric-label">Gyro Magnitude</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:{'#60a5fa' if is_dynamic else '#34d399'}">{motion_type}</div>
            <div class="metric-label">Motion Type</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_manual_preset(activity):
    """Push a preset into the manual simulator slider widget state."""
    st.session_state["last_preset"] = activity
    for key, value in ACTIVITY_PRESETS[activity].items():
        st.session_state[key] = float(value)


def safe_float(value, default=0.0):
    """Coerce component values into finite floats."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def get_lan_ip():
    """Best-effort IP that phones on the same hotspot can try opening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "localhost"


def normalize_live_payload(payload):
    """Convert the browser motion payload into model-ready sensor values."""
    payload = payload or {}
    return {
        "running": bool(payload.get("running", False)),
        "error": payload.get("error"),
        "samples": int(safe_float(payload.get("samples"), 0)),
        "sample_rate_hz": safe_float(payload.get("sample_rate_hz"), 0),
        "acc_x": clamp(safe_float(payload.get("acc_x"), 0), -1.0, 1.0),
        "acc_y": clamp(safe_float(payload.get("acc_y"), 0), -1.0, 1.0),
        "acc_z": clamp(safe_float(payload.get("acc_z"), 0), -1.0, 1.0),
        "gyro_x": clamp(safe_float(payload.get("gyro_x"), 0), -2.0, 2.0),
        "gyro_y": clamp(safe_float(payload.get("gyro_y"), 0), -2.0, 2.0),
        "gyro_z": clamp(safe_float(payload.get("gyro_z"), 0), -2.0, 2.0),
        "acc_x_std": safe_float(payload.get("acc_x_std"), 0),
        "acc_y_std": safe_float(payload.get("acc_y_std"), 0),
        "acc_z_std": safe_float(payload.get("acc_z_std"), 0),
        "gyro_x_std": safe_float(payload.get("gyro_x_std"), 0),
        "gyro_y_std": safe_float(payload.get("gyro_y_std"), 0),
        "gyro_z_std": safe_float(payload.get("gyro_z_std"), 0),
        "acc_rms_g": safe_float(payload.get("acc_rms_g"), 0),
        "gyro_rms": safe_float(payload.get("gyro_rms"), 0),
        "peak_body_g": safe_float(payload.get("peak_body_g"), 0),
        "peak_total_g": safe_float(payload.get("peak_total_g"), 0),
        "peak_jerk_gs": safe_float(payload.get("peak_jerk_gs"), 0),
        "stillness_score": clamp(safe_float(payload.get("stillness_score"), 0), 0, 1),
        "seconds_since_impact": payload.get("seconds_since_impact"),
        "received_at": payload.get("received_at"),
        "source": payload.get("source", "streamlit_component"),
    }


def load_bridge_motion(max_age_seconds=6):
    """Read the latest phone collector payload written by phone_sensor_bridge.py."""
    if not os.path.exists(PHONE_BRIDGE_PATH):
        return normalize_live_payload({"source": "phone_bridge"})

    try:
        with open(PHONE_BRIDGE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return normalize_live_payload({"source": "phone_bridge"})

    motion = normalize_live_payload(payload)
    received_at = safe_float(motion.get("received_at"), 0)
    motion["fresh"] = bool(received_at and (time.time() - received_at <= max_age_seconds))
    motion["age_seconds"] = time.time() - received_at if received_at else None
    return motion


def get_impact_age(motion):
    seconds_since_impact = motion.get("seconds_since_impact")
    return safe_float(seconds_since_impact, 999) if seconds_since_impact is not None else 999


def is_motion_settled(motion):
    return (
        motion["stillness_score"] >= 0.62
        and motion["acc_rms_g"] < 0.22
        and motion["gyro_rms"] < 0.26
    )


def normalize_risk_window(motion):
    """
    Normalize the risk payload without hiding the current meter values.
    The collector now sends rolling-window peaks, so the meter can behave naturally.
    """
    return dict(motion or normalize_live_payload(None))


def assess_fall_risk(motion):
    """
    Rule-based layer for abnormal motion on top of the activity classifier.
    It looks for impact strength, jerk, and whether the current motion has recovered.
    """
    motion = normalize_risk_window(motion)
    impact_score = max(
        motion["peak_body_g"] / 1.7,
        max(motion["peak_total_g"] - 1.0, 0) / 1.6,
    )
    jerk_score = motion["peak_jerk_gs"] / 12.0
    stillness_score = motion["stillness_score"]

    impact_age = get_impact_age(motion)
    recent_impact = impact_age <= 6
    watch_window = impact_age <= 14
    strong_impact = motion["peak_body_g"] >= 1.45 or motion["peak_total_g"] >= 2.45
    hard_jerk = motion["peak_jerk_gs"] >= 10
    post_impact_still = stillness_score >= 0.72
    current_motion_settled = is_motion_settled(motion)

    if impact_age <= 6:
        decay = 1.0
    elif impact_age <= 18:
        decay = 1.0 - ((impact_age - 6) / 12)
    else:
        decay = 0.0

    raw_score = clamp((impact_score * 0.52) + (jerk_score * 0.38), 0, 1)
    composite = raw_score * decay

    if recent_impact and strong_impact and hard_jerk and post_impact_still:
        level = "Fall likely"
        message = "Hard impact followed by low movement. Check the person immediately."
        composite = max(composite, 0.86)
    elif (strong_impact or hard_jerk) and recent_impact:
        level = "Abnormal"
        message = "Unusual spike detected. Keep recording to see if movement stabilizes."
        composite = max(composite, 0.62)
    elif (strong_impact or hard_jerk) and watch_window and not current_motion_settled:
        level = "Abnormal"
        message = "Motion is still unstable after the spike."
        composite = max(composite, 0.42)
    else:
        level = "Normal"
        if strong_impact or hard_jerk:
            message = "Motion has stabilized after the previous spike."
        else:
            message = "No impact pattern detected in the current phone motion window."
        composite = min(composite, 0.18)

    return level, composite * 100, message


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">HAR<br>Sensor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Activity Recognizer v2.0</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input Mode",
        ["Live Phone Sensor", "Manual Simulator"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">Quick Presets</div>', unsafe_allow_html=True)
    if input_mode == "Live Phone Sensor":
        st.markdown("""
        <div style="font-size:0.72rem; color:#4a5580; line-height:1.7; margin-bottom:0.75rem">
        Presets remain available in manual simulator mode. Live mode reads your phone IMU directly.
        </div>
        """, unsafe_allow_html=True)
    
    preset_cols = st.columns(2)
    selected_preset = None
    for i, act in enumerate(ACTIVITIES):
        col = preset_cols[i % 2]
        with col:
            if st.button(
                f"{ACTIVITY_ICONS[act]} {PRESET_BUTTON_LABELS[act]}",
                key=f"preset_{i}",
                width="stretch"
            ):
                selected_preset = act

    if selected_preset:
        apply_manual_preset(selected_preset)

    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:0.5rem">
        <div class="metric-value">{model_accuracy*100:.1f}%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:0.75rem; color:#4a5580; margin-top:1rem; line-height:1.6">
        <b style="color:#6b7db3">Dataset:</b> Actual UCI HAR<br>
        <b style="color:#6b7db3">Model:</b> Random Forest<br>
        <b style="color:#6b7db3">Features:</b> {len(model_feature_names) or 17} live-compatible features<br>
        <b style="color:#6b7db3">Classes:</b> 6 activities<br>
        <b style="color:#6b7db3">Sensor:</b> Smartphone IMU<br>
        <b style="color:#6b7db3">Safety Layer:</b> Impact + jerk + stillness
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#4a5580; line-height:1.7">
    Classifies human activities using accelerometer and gyroscope data from smartphone sensors — the core technology behind fitness trackers, fall detection, and smart wearables.
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
col_hero, col_status = st.columns([3, 1])
with col_hero:
    st.markdown('<div class="hero-title">Human Activity<br>Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live Phone IMU · Fall Detection · 6-Class Classifier</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style="text-align:right; margin-top:1rem; font-family:'Space Mono',monospace; font-size:0.75rem; color:#4a5580">
        <span class="status-dot"></span>MODEL LIVE
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layout: sliders | result ──────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">📡 Sensor Inputs</div>', unsafe_allow_html=True)
    
    # Apply preset if selected
    default = ACTIVITY_PRESETS.get(
        selected_preset or st.session_state.get("last_preset", "Walking"),
        ACTIVITY_PRESETS["Walking"]
    )

    is_live_mode = input_mode == "Live Phone Sensor"
    live_motion = None

    if is_live_mode:
        LIVE_REFRESH_COMPONENT(key="live_refresh", default=0)
        bridge_motion = load_bridge_motion()
        use_bridge = bridge_motion.get("fresh") and bridge_motion["samples"] >= 4
        live_motion = bridge_motion if use_bridge else normalize_live_payload({"source": "phone_bridge"})

        if live_motion["error"]:
            st.warning(live_motion["error"])

        has_live_samples = live_motion["samples"] >= 4
        acc_x = live_motion["acc_x"] if has_live_samples else 0.0
        acc_y = live_motion["acc_y"] if has_live_samples else 0.0
        acc_z = live_motion["acc_z"] if has_live_samples else 0.0
        gyro_x = live_motion["gyro_x"] if has_live_samples else 0.0
        gyro_y = live_motion["gyro_y"] if has_live_samples else 0.0
        gyro_z = live_motion["gyro_z"] if has_live_samples else 0.0

        if not has_live_samples:
            st.info("Waiting for phone motion samples. The activity prediction will update after a few sensor events.")

        st.markdown("**Live Model Inputs** `(rolling window)`")
        sensor_cols = st.columns(2)
        with sensor_cols[0]:
            st.metric("Acc X", f"{acc_x:.3f}g")
            st.metric("Acc Y", f"{acc_y:.3f}g")
            st.metric("Acc Z", f"{acc_z:.3f}g")
        with sensor_cols[1]:
            st.metric("Gyro X", f"{gyro_x:.3f}")
            st.metric("Gyro Y", f"{gyro_y:.3f}")
            st.metric("Gyro Z", f"{gyro_z:.3f}")

        render_motion_metrics(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

        lan_ip = get_lan_ip()
        st.markdown(f"""
        <div class="mode-note">
        Phone collector: <b style="color:#e8eaf6">http://{lan_ip}:{PHONE_BRIDGE_PORT}</b><br>
        Streamlit dashboard: <b style="color:#e8eaf6">http://{lan_ip}:8501</b><br>
        Active feed: <b style="color:#e8eaf6">{'Phone collector bridge' if use_bridge else 'Waiting for phone collector'}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("**Body Acceleration** `(g-force, body component)`")
        acc_x = st.slider("X-axis (forward/backward)", -1.0, 1.0, float(default["acc_x"]), 0.01, key="acc_x",
                          help="Forward-backward body acceleration (gravity removed)")
        acc_y = st.slider("Y-axis (side to side)",     -1.0, 1.0, float(default["acc_y"]), 0.01, key="acc_y",
                          help="Left-right body acceleration")
        acc_z = st.slider("Z-axis (up/down)",           -1.0, 1.0, float(default["acc_z"]), 0.01, key="acc_z",
                          help="Up-down body acceleration")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Angular Velocity** `(rad/s, gyroscope)`")
        gyro_x = st.slider("Gyro X (pitch)",  -2.0, 2.0, float(default["gyro_x"]), 0.01, key="gyro_x",
                           help="Pitch angular velocity")
        gyro_y = st.slider("Gyro Y (roll)",   -2.0, 2.0, float(default["gyro_y"]), 0.01, key="gyro_y",
                           help="Roll angular velocity")
        gyro_z = st.slider("Gyro Z (yaw)",    -2.0, 2.0, float(default["gyro_z"]), 0.01, key="gyro_z",
                           help="Yaw angular velocity")
        render_motion_metrics(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)


with right_col:
    st.markdown('<div class="section-header">🎯 Prediction</div>', unsafe_allow_html=True)
    
    # Run prediction
    predicted, probs = predict_activity(
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        live_motion if is_live_mode else None,
    )
    confidence = max(probs) * 100
    icon = ACTIVITY_ICONS[predicted]
    color = ACTIVITY_COLORS[predicted]
    
    st.markdown(f"""
    <div class="activity-card">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{color},{color}88)"></div>
        <span class="activity-icon">{icon}</span>
        <div class="activity-label">Detected Activity</div>
        <div class="activity-name">{predicted}</div>
        <div class="confidence-pill">{confidence:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    if is_live_mode:
        risk_motion = normalize_risk_window(live_motion or normalize_live_payload(None))
        fall_level, fall_score, fall_message = assess_fall_risk(risk_motion)
        fall_meta = FALL_LEVELS[fall_level]
        st.markdown(f"""
        <div class="risk-card" style="border-color:{fall_meta['color']}66">
            <div class="risk-title" style="color:{fall_meta['color']}">{fall_meta['icon']} {fall_level}</div>
            <div class="risk-copy">{fall_message}</div>
            <div class="risk-metrics">
                <div class="metric-card">
                    <div class="metric-value" style="color:{fall_meta['color']}">{fall_score:.0f}%</div>
                    <div class="metric-label">Risk Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_motion.get('peak_total_g', 0):.2f}g</div>
                    <div class="metric-label">Impact Peak</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_motion.get('peak_jerk_gs', 0):.1f}</div>
                    <div class="metric-label">Jerk Peak</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence bar chart
    sorted_idx = np.argsort(probs)[::-1]
    
    fig = go.Figure()
    for i in sorted_idx:
        act = ACTIVITIES[i]
        prob = probs[i] * 100
        bar_color = ACTIVITY_COLORS[act] if act == predicted else "#1e2a4a"
        text_color = "#e8eaf6" if act == predicted else "#4a5580"
        
        fig.add_trace(go.Bar(
            x=[prob],
            y=[f"{ACTIVITY_ICONS[act]} {act}"],
            orientation="h",
            marker=dict(
                color=bar_color,
                line=dict(width=0),
            ),
            text=f"{prob:.1f}%",
            textposition="inside" if prob > 15 else "outside",
            textfont=dict(color=text_color, family="Space Mono", size=11),
            showlegend=False,
        ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 105],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#6b7db3", family="Space Mono", size=11),
            autorange="reversed",
        ),
        bargap=0.25,
        font=dict(family="Syne"),
    )
    
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── Radar chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Sensor Signal Profile</div>', unsafe_allow_html=True)

radar_col, heatmap_col = st.columns([1, 1], gap="large")

with radar_col:
    categories = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    values_raw = [acc_x, acc_y, acc_z, gyro_x / 2, gyro_y / 2, gyro_z / 2]
    values_norm = [(v + 1) / 2 for v in values_raw]  # normalise to 0–1

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba(96,165,250,0.15)",
        line=dict(color="#60a5fa", width=2),
        name="Current Reading",
    ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickfont=dict(color="#6b7db3", size=11, family="Space Mono"),
                gridcolor="#1e2a4a",
                linecolor="#1e2a4a",
            ),
            gridshape="circular",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_radar, width="stretch", config={"displayModeBar": False})

with heatmap_col:
    # Probability heatmap across all activities
    all_probs_matrix = [probs]  # single reading; show as horizontal band
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=[probs * 100],
        x=ACTIVITIES,
        y=["Current"],
        colorscale=[
            [0.0, "#0a0e1a"],
            [0.3, "#1e2a4a"],
            [0.6, "#3b4fa8"],
            [1.0, "#60a5fa"],
        ],
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#4a5580", size=10, family="Space Mono"),
            title=dict(text="%", font=dict(color="#4a5580")),
            bgcolor="rgba(0,0,0,0)",
        ),
        text=[[f"{p:.1f}%" for p in probs * 100]],
        texttemplate="%{text}",
        textfont=dict(size=11, family="Space Mono", color="#e8eaf6"),
    ))
    
    fig_heat.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=60),
        xaxis=dict(
            tickfont=dict(color="#6b7db3", size=10, family="Space Mono"),
            tickangle=-30,
        ),
        yaxis=dict(showticklabels=False),
        font=dict(family="Syne"),
    )
    st.plotly_chart(fig_heat, width="stretch", config={"displayModeBar": False})


# ── Feature importance insight ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Model Feature Importance (Top 6 Sensor Channels)</div>', unsafe_allow_html=True)

importance = model.feature_importances_[:6]
feat_names = ["Body Acc X", "Body Acc Y", "Body Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
importance_pct = importance / importance.sum() * 100
sorted_feat = sorted(zip(feat_names, importance_pct), key=lambda x: x[1], reverse=True)

fig_imp = go.Figure()
for feat, imp in sorted_feat:
    fig_imp.add_trace(go.Bar(
        x=[feat], y=[imp],
        marker=dict(
            color=f"rgba(167,139,250,{0.4 + imp/100})",
            line=dict(width=0)
        ),
        text=f"{imp:.1f}%",
        textposition="outside",
        textfont=dict(color="#a78bfa", family="Space Mono", size=11),
        showlegend=False,
    ))

fig_imp.update_layout(
    height=220,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=20, b=10),
    xaxis=dict(
        showgrid=False,
        tickfont=dict(color="#6b7db3", family="Space Mono", size=11),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1e2a4a",
        tickfont=dict(color="#4a5580", size=10),
        zeroline=False,
    ),
    bargap=0.35,
    font=dict(family="Syne"),
)
st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1.5rem; border-top:1px solid #1e2a4a">
  <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#2a3560; letter-spacing:3px">
    ACTUAL UCI HAR DATASET · LIVE PHONE SENSOR · FALL DETECTION · RANDOM FOREST CLASSIFIER
  </span>
</div>
""", unsafe_allow_html=True)
