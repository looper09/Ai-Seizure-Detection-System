import streamlit as st
import mne
import numpy as np
import pandas as pd
import joblib
import time
import os
import tempfile
from datetime import datetime
from scipy import signal, stats

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FS          = 256
WINDOW_SEC  = 5
WINDOW_SIZE = FS * WINDOW_SEC

COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ',  'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
]

KNOWN_SEIZURES = {
    'chb01_03.edf': 2996, 'chb01_04.edf': 1467,
    'chb03_02.edf': 1088, 'chb03_04.edf': 1440,
    'chb06_04.edf': 582,  'chb06_09.edf': 360,
    'chb08_05.edf': 1656, 'chb08_11.edf': 1200,
    'chb12_08.edf': 1440, 'chb12_09.edf': 1200,
}

st.set_page_config(
    page_title="NeuroGuard — Seizure Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0e1a; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a1628 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-title { font-size: 2rem; font-weight: 600; color: #e2e8f0; margin: 0; letter-spacing: -0.5px; }
.hero-sub { font-size: 0.9rem; color: #718096; margin: 0.25rem 0 0; font-family: 'DM Mono', monospace; }
.hero-badge {
    background: rgba(72,187,120,0.15);
    border: 1px solid rgba(72,187,120,0.4);
    color: #68d391;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
}

/* Metric cards */
.metric-card {
    background: #0d1b3e;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.35); }
.metric-value { font-size: 1.8rem; font-weight: 600; color: #63b3ed; font-family: 'DM Mono', monospace; }
.metric-label { font-size: 0.75rem; color: #718096; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

/* Status banners */
.seizure-alert {
    background: rgba(245,101,101,0.12);
    border: 1px solid rgba(245,101,101,0.5);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #fc8181;
    font-weight: 500;
    font-size: 1.05rem;
    animation: pulse-border 1.5s ease-in-out infinite;
}
.normal-status {
    background: rgba(72,187,120,0.1);
    border: 1px solid rgba(72,187,120,0.35);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #68d391;
    font-weight: 500;
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(245,101,101,0.5); }
    50%       { border-color: rgba(245,101,101,0.9); }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid rgba(99,179,237,0.1);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p { color: #a0aec0 !important; font-size: 0.85rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b3e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #718096;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.2) !important;
    color: #63b3ed !important;
}

/* Hint box */
.hint-box {
    background: #0d1b3e;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #718096;
    font-family: 'DM Mono', monospace;
    line-height: 1.8;
}
.hint-box strong { color: #63b3ed; }

/* Timeline bar */
.timeline-wrap {
    background: #0d1b3e;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #718096;
}
.history-seizure { color: #fc8181; }
.history-normal  { color: #68d391; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1b3e;
    border: 1px dashed rgba(99,179,237,0.3);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TENSORFLOW IMPORT
# ─────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'total_seizures' not in st.session_state:
    st.session_state.total_seizures = 0

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    errors = []
    for f in ['best_rf_tuned_model.pkl', 'scaler.pkl']:
        if not os.path.exists(f):
            errors.append(f)
    if errors:
        raise FileNotFoundError(
            f"Missing model files: {', '.join(errors)}\n"
            "Run the training notebook first to generate them."
        )
    rf  = joblib.load('best_rf_tuned_model.pkl')
    scl = joblib.load('scaler.pkl')
    cnn = None
    if TF_AVAILABLE:
        for fname in ['advanced_model.keras', 'advanced_model.h5', 'best_cnn.keras']:
            if os.path.exists(fname):
                try:
                    cnn = tf.keras.models.load_model(fname)
                    break
                except Exception:
                    pass
    metrics = None
    if os.path.exists('model_metrics.pkl'):
        try:
            metrics = joblib.load('model_metrics.pkl')
        except Exception:
            pass
    return rf, cnn, scl, metrics

try:
    rf_model, cnn_model, scaler, model_metrics = load_assets()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION 
# ─────────────────────────────────────────────
def extract_features(window_flat, window_size=WINDOW_SIZE, sfreq=FS):
    """
    Extract statistical + band-power features from a flattened window.
    window_flat: 1D array of shape (window_size * n_channels,)
    Returns: 1D feature vector
    """
    n_ch = len(window_flat) // window_size
    w3d  = window_flat.reshape(window_size, n_ch)
    feats = []
    for ch in range(n_ch):
        d = w3d[:, ch].astype(np.float64)
        # Time-domain
        feats += [
            d.mean(), d.std(), d.var(),
            np.percentile(d, 25), np.percentile(d, 75),
            float(stats.skew(d)), float(stats.kurtosis(d)),
            d.min(), d.max(),
        ]
        # Band power
        freqs, psd = signal.welch(d, sfreq, nperseg=min(sfreq, window_size))
        feats += [
            float(psd[(freqs >= 0.5) & (freqs < 4 )].mean()),  # Delta
            float(psd[(freqs >= 4  ) & (freqs < 8 )].mean()),  # Theta
            float(psd[(freqs >= 8  ) & (freqs < 13)].mean()),  # Alpha
            float(psd[(freqs >= 13 ) & (freqs < 30)].mean()),  # Beta
            float(psd[(freqs >= 30 ) & (freqs < 50)].mean()),  # Gamma
        ]
    return np.array(feats, dtype=np.float32)

# ─────────────────────────────────────────────
# EEG PREPROCESSING 
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess_eeg(file_bytes, filename, start_second):
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=False, verbose=False)
        valid_picks = [ch for ch in COMMON_CHANNELS if ch in raw.ch_names]
        if not valid_picks:
            return None, None, 0
        raw.pick(valid_picks)
        
        raw.load_data()  
        raw.filter(l_freq=0.5, h_freq=40.0, method='fir', verbose=False)

        n_ch      = len(raw.ch_names)
        start_s   = int(start_second * FS)
        end_s     = start_s + WINDOW_SIZE

        if end_s > raw.n_times:
            return None, None, n_ch

        # ← Slice from memory
        data, _  = raw[:, start_s:end_s]
        
        # ⚠️ CRITICAL MATRIX FLIP FIX (.T added here) ⚠️
        window_flat = data.T.flatten().astype(np.float32)

        # Scale
        window_scaled = scaler.transform(window_flat.reshape(1, -1))[0]

        # Features for RF
        features = extract_features(window_scaled)

        return window_scaled, features, n_ch

    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(window_scaled, features, n_ch, model_type):
    if model_type == "Random Forest":
        pred  = rf_model.predict(features.reshape(1, -1))[0]
        proba = rf_model.predict_proba(features.reshape(1, -1))[0]
        conf  = float(max(proba)) * 100
        return int(pred), conf, float(proba[1]) * 100

    if TF_AVAILABLE and cnn_model:
        cnn_input = window_scaled.reshape(1, WINDOW_SIZE, n_ch)
        prob = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
        
        # ⚠️ INCREASED SENSITIVITY FIX (Threshold lowered to 0.3) ⚠️
        pred = 1 if prob > 0.30 else 0
        conf = (prob if pred == 1 else 1 - prob) * 100
        
        return pred, conf, prob * 100

    # Fallback to RF
    pred  = rf_model.predict(features.reshape(1, -1))[0]
    proba = rf_model.predict_proba(features.reshape(1, -1))[0]
    conf  = float(max(proba)) * 100
    return int(pred), conf, float(proba[1]) * 100

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div style="font-size:2.5rem">🧠</div>
  <div style="flex:1">
    <div class="hero-title">NeuroGuard</div>
    <div class="hero-sub">Epileptic Seizure Detection · CHB-MIT · BSAI-IV</div>
  </div>
  <div class="hero-badge">● SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_type = st.selectbox(
        "Active Model",
        ["Random Forest", "1D-CNN"],
        help="Random Forest uses engineered features. 1D-CNN learns directly from raw EEG.",
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    if models_loaded and model_metrics:
        rf_m  = model_metrics.get('random_forest', {})
        cnn_m = model_metrics.get('cnn_1d', {})
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RF Accuracy",  f"{rf_m.get('test_acc', 0):.2%}")
            st.metric("RF AUC",       f"{rf_m.get('roc_auc', 0):.3f}")
        with c2:
            st.metric("CNN Accuracy", f"{cnn_m.get('test_acc', 0):.2%}")
            st.metric("CNN AUC",      f"{cnn_m.get('roc_auc', 0):.3f}")
        st.metric("RF F1 (seizure)",  f"{rf_m.get('f1_score', 0):.4f}")
        st.metric("CNN F1 (seizure)", f"{cnn_m.get('f1_score', 0):.4f}")

        if model_metrics.get('dataset_info'):
            d = model_metrics['dataset_info']
            st.markdown("---")
            st.markdown("### 🗂️ Dataset Info")
            st.markdown(f"""
- **Total windows:** {d.get('total_windows', 'N/A'):,}
- **Normal:** {d.get('normal_samples', 'N/A'):,}
- **Seizure:** {d.get('seizure_samples', 'N/A'):,}
- **Patients:** {d.get('num_files', 'N/A')}
- **Split:** GroupShuffleSplit (patient-aware)
            """)
    elif not models_loaded:
        st.error("Models not loaded. Run the notebook first.")

    st.markdown("---")
    # Live session stats
    st.markdown("### 📡 Session Stats")
    st.metric("Windows Scanned",   st.session_state.total_scanned)
    st.metric("Seizures Detected", st.session_state.total_seizures)
    if st.session_state.total_scanned > 0:
        rate = st.session_state.total_seizures / st.session_state.total_scanned * 100
        st.metric("Detection Rate", f"{rate:.1f}%")

    st.markdown("---")
    st.markdown("### 👥 Developed by")
    st.markdown("**Asad Ali Asim** \n**Muhammad Daniyal Khan** \nBSAI-IV")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if not models_loaded:
    st.error(f"""
    **Models not found.**
    {model_error}

    Steps to fix:
    1. Open `Model_training_fixed.ipynb`
    2. Set your `data_dir` path in Cell 1
    3. Run all cells (takes 30–60 min for full CHB-MIT)
    4. Restart this app — model files will be in the same folder
    """)
    st.stop()

# ── File upload ──
st.markdown("#### Upload Patient EDF File")
uploaded = st.file_uploader("", type=["edf"], label_visibility="collapsed")

if not uploaded:
    st.info("Upload a `.edf` file from the CHB-MIT dataset to begin analysis.")

    # Known seizure times hint
    st.markdown("""
    <div class="hint-box">
    <strong>Known seizure timestamps for testing:</strong><br>
    chb01_03.edf → ~2996s &nbsp;|&nbsp; chb01_04.edf → ~1467s<br>
    chb03_02.edf → ~1088s &nbsp;|&nbsp; chb03_04.edf → ~1440s<br>
    chb06_04.edf → ~582s  &nbsp;|&nbsp; chb06_09.edf → ~360s<br>
    chb08_05.edf → ~1656s &nbsp;|&nbsp; chb08_11.edf → ~1200s<br>
    chb12_08.edf → ~1440s &nbsp;|&nbsp; chb12_09.edf → ~1200s
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = uploaded.read()
fname      = uploaded.name
st.success(f"✓ Loaded: `{fname}`  ({len(file_bytes)/1024/1024:.1f} MB)")

# Auto-hint if known file
if fname in KNOWN_SEIZURES:
    st.info(f"💡 Known seizure in `{fname}` at **~{KNOWN_SEIZURES[fname]}s** — use this as your start time.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎛️  Manual Scanner", "📡  Live Feed", "📋  History & Export"])

# ══════════════════════════════════════════════
# TAB 1 — MANUAL SCANNER
# ══════════════════════════════════════════════
with tab1:
    st.markdown("##### Analyse a specific time window")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        start_time = st.number_input(
            "Start time (seconds)", min_value=0, value=0, step=5,
            help="The model analyses a 5-second window beginning at this timestamp."
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Prediction", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("Preprocessing EEG window..."):
            w_scaled, feats, n_ch = preprocess_eeg(file_bytes, fname, start_time)

        if feats is None:
            st.error("Selected time exceeds file length. Choose a smaller start time.")
        else:
            pred, conf, seizure_prob = predict(w_scaled, feats, n_ch, model_type)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Result",       "🚨 SEIZURE" if pred == 1 else "✅ NORMAL")
            m2.metric("Confidence",   f"{conf:.1f}%")
            m3.metric("Seizure Prob", f"{seizure_prob:.1f}%")
            m4.metric("Window",       f"{start_time}s – {start_time+5}s")

            # Status banner
            if pred == 1:
                st.markdown(f"""
                <div class="seizure-alert">
                🚨 SEIZURE DETECTED at {start_time}s–{start_time+5}s
                &nbsp;·&nbsp; Confidence: {conf:.1f}%
                &nbsp;·&nbsp; Immediate clinical review required
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-status">
                ✅ Normal brain activity detected ({start_time}s–{start_time+5}s)
                &nbsp;·&nbsp; Confidence: {conf:.1f}%
                </div>""", unsafe_allow_html=True)

            # Log to history
            st.session_state.history.append({
                'Timestamp':   datetime.now().strftime('%H:%M:%S'),
                'File':        fname,
                'Window':      f"{start_time}s–{start_time+5}s",
                'Model':       model_type,
                'Prediction':  'SEIZURE' if pred == 1 else 'NORMAL',
                'Confidence':  f"{conf:.1f}%",
                'Seizure Prob':f"{seizure_prob:.1f}%",
            })
            st.session_state.total_scanned  += 1
            st.session_state.total_seizures += pred

# ══════════════════════════════════════════════
# TAB 2 — LIVE FEED
# ══════════════════════════════════════════════
with tab2:
    st.markdown("##### Continuous monitoring — scans rolling 5-second windows every second")

    col_p, col_q = st.columns([2, 1])
    with col_p:
        live_start = st.number_input(
            "Simulation start (seconds)", min_value=0, value=1430, step=10,
            help="Try 1430s with chb03_04.edf to see a seizure detected live."
        )
    with col_q:
        scan_dur = st.slider("Scan duration (windows)", 5, 30, 15)

    st.caption("💡 Try 1430s with `chb03_04.edf` to watch a seizure get caught in real-time.")

    if st.button("▶️  Start Live Monitoring", type="primary"):
        feed_ph  = st.empty()
        prog_ph  = st.empty()
        status_ph = st.empty()

        seizure_caught = False
        for i, current_t in enumerate(range(live_start, live_start + scan_dur)):
            prog_ph.progress((i + 1) / scan_dur, text=f"Scanning window {i+1}/{scan_dur}")

            with st.spinner(""):
                w_scaled, feats, n_ch = preprocess_eeg(file_bytes, fname, current_t)

            if feats is None:
                status_ph.warning(f"Window at {current_t}s exceeds file length — stopping.")
                break

            pred, conf, seizure_prob = predict(w_scaled, feats, n_ch, model_type)

            st.session_state.history.append({
                'Timestamp':    datetime.now().strftime('%H:%M:%S'),
                'File':         fname,
                'Window':       f"{current_t}s–{current_t+5}s",
                'Model':        model_type,
                'Prediction':   'SEIZURE' if pred == 1 else 'NORMAL',
                'Confidence':   f"{conf:.1f}%",
                'Seizure Prob': f"{seizure_prob:.1f}%",
            })
            st.session_state.total_scanned  += 1
            st.session_state.total_seizures += pred

            with feed_ph.container():
                st.markdown(f"**Current window:** `{current_t}s → {current_t+5}s`")
                if pred == 1:
                    st.markdown(f"""
                    <div class="seizure-alert">
                    🚨 ALARM — SEIZURE at {current_t}s · Confidence: {conf:.1f}% · Halting scan
                    </div>""", unsafe_allow_html=True)
                    st.toast("CRITICAL: Seizure detected!", icon="🚨")
                    seizure_caught = True
                else:
                    st.markdown(f"""
                    <div class="normal-status">
                    ✅ {current_t}s — Normal &nbsp;·&nbsp; Confidence: {conf:.1f}%
                    </div>""", unsafe_allow_html=True)

            if seizure_caught:
                break

            time.sleep(0.8)

        prog_ph.empty()
        if not seizure_caught:
            status_ph.success(f"✅ Scan complete — no seizure detected across {scan_dur} windows.")

# ══════════════════════════════════════════════
# TAB 3 — HISTORY & EXPORT
# ══════════════════════════════════════════════
with tab3:
    if not st.session_state.history:
        st.info("No predictions made yet. Use the Manual Scanner or Live Feed first.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # Summary metrics
        total   = len(df)
        n_seiz  = (df['Prediction'] == 'SEIZURE').sum()
        n_norm  = total - n_seiz

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Scanned",    total)
        s2.metric("Seizures Found",   n_seiz)
        s3.metric("Normal Windows",   n_norm)
        s4.metric("Seizure Rate",     f"{n_seiz/total*100:.1f}%" if total else "0%")

        st.markdown("---")

        # Color-coded table
        def highlight(row):
            if row['Prediction'] == 'SEIZURE':
                return ['background-color: rgba(245,101,101,0.08); color: #fc8181'] * len(row)
            return ['color: #68d391'] * len(row)

        st.dataframe(
            df.style.apply(highlight, axis=1),
            use_container_width=True,
            height=400,
        )

        # Export
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name=f"seizure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️  Clear History", use_container_width=True):
            st.session_state.history         = []
            st.session_state.total_scanned   = 0
            st.session_state.total_seizures  = 0
            st.rerun()