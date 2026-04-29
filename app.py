"""
Philippine Hoya Clade Classifier
AI-Powered Pollinarium Morphometric Analysis

Developer:        Jerald B. Bongalos (Asian Institute of Management)
Dataset Owner:    Fernando B. Aurigue (Retired Career Scientist, DOST-PNRI)
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Philippine Hoya Clade Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — sophisticated botanical / editorial palette
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --paper:        #faf8f3;
    --surface:      #ffffff;
    --ink:          #1a1a1a;
    --ink-muted:    #5a5a5a;
    --ink-subtle:   #8a8a8a;
    --hairline:     #e8e3d8;
    --hairline-soft:#f0ece2;

    --forest:       #1a3d2e;
    --forest-deep:  #0f2a1f;
    --sage:         #6b8e63;
    --moss-bg:      #f3f5ef;

    --good:         #2d5e3e;
    --good-bg:      #eef4ec;
    --warn:         #9a6f1f;
    --warn-bg:      #faf2dd;
    --bad:          #8b3a3a;
    --bad-bg:       #f7e8e6;
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden !important; }

[data-testid="stHeader"] { background: transparent; height: 0; }

/* ─── Global ─── */
html, body, [class*="css"], .stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 880px;
}

/* ─── Hero ─── */
.hoya-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--sage);
    text-align: center;
    margin: 0 0 1rem 0;
}
.hoya-title {
    font-family: 'Fraunces', 'Playfair Display', Georgia, serif;
    font-size: clamp(2.1rem, 5.5vw, 3.6rem);
    font-weight: 500;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--forest-deep);
    text-align: center;
    margin: 0 auto 0.9rem auto;
    max-width: 18ch;
    font-variation-settings: "opsz" 144;
}
.hoya-tagline {
    font-family: 'Inter', sans-serif;
    font-size: clamp(0.95rem, 1.5vw, 1.05rem);
    font-weight: 400;
    color: var(--ink-muted);
    text-align: center;
    line-height: 1.6;
    max-width: 56ch;
    margin: 0 auto 2.4rem auto;
}
.hoya-rule {
    height: 1px;
    background: var(--hairline);
    border: 0;
    max-width: 120px;
    margin: 0 auto 2.6rem auto;
}

/* ─── Section heading ─── */
.hoya-section {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--forest-deep);
    letter-spacing: -0.01em;
    margin: 2.2rem 0 0.4rem 0;
}
.hoya-section-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--ink-muted);
    margin: 0 0 1.4rem 0;
}

/* ─── Card ─── */
.hoya-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 1px 2px rgba(20, 30, 25, 0.03);
}
.hoya-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--sage);
    margin: 0 0 1rem 0;
}

/* ─── Inputs ─── */
.stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    letter-spacing: 0 !important;
}
.stNumberInput > div > div > input {
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
    background: var(--paper) !important;
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
    font-size: 0.95rem !important;
    color: var(--ink) !important;
    padding: 0.55rem 0.7rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107, 142, 99, 0.15) !important;
    outline: none !important;
}
.stNumberInput button {
    background: var(--paper) !important;
    border: 1px solid var(--hairline) !important;
    color: var(--ink-muted) !important;
}

/* ─── Primary button ─── */
.stButton > button {
    background: var(--forest-deep);
    color: var(--paper);
    border: 1px solid var(--forest-deep);
    border-radius: 999px;
    padding: 0.85rem 2.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    box-shadow: none;
    transition: transform 0.12s ease, background 0.18s ease;
    width: 100%;
}
.stButton > button:hover {
    background: var(--forest);
    border-color: var(--forest);
    transform: translateY(-1px);
    color: #fff;
}
.stButton > button:focus:not(:active) {
    border-color: var(--sage);
    box-shadow: 0 0 0 3px rgba(107, 142, 99, 0.2);
    color: var(--paper);
}

/* Secondary "sample" pill buttons */
.stButton > button[kind="secondary"] {
    background: var(--surface);
    color: var(--forest-deep);
    border: 1px solid var(--hairline);
    font-weight: 400;
    font-size: 0.82rem;
    padding: 0.5rem 1rem;
    letter-spacing: 0;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--moss-bg);
    border-color: var(--sage);
    color: var(--forest-deep);
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid var(--hairline);
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--ink-subtle);
    background: transparent !important;
    border-radius: 0;
    padding: 0.6rem 0;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--forest-deep) !important;
    border-bottom: 2px solid var(--forest-deep) !important;
    background: transparent !important;
    font-weight: 600;
}

/* ─── Result card (rendered as one HTML block, so wrapping actually works) ─── */
.result-wrap {
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 2.2rem;
    background: var(--surface);
    margin: 1.4rem 0;
    box-shadow: 0 2px 8px rgba(20, 30, 25, 0.04);
}
.result-wrap.high   { border-left: 4px solid var(--good); }
.result-wrap.medium { border-left: 4px solid var(--warn); }
.result-wrap.low    { border-left: 4px solid var(--bad); }

.result-status {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 0 0 0.8rem 0;
}
.result-status.high   { color: var(--good); }
.result-status.medium { color: var(--warn); }
.result-status.low    { color: var(--bad); }

.result-clade {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.2rem, 5vw, 3rem);
    font-weight: 500;
    line-height: 1.05;
    color: var(--forest-deep);
    letter-spacing: -0.01em;
    font-style: italic;
    margin: 0 0 1.2rem 0;
}

.result-meter-track {
    height: 6px;
    background: var(--hairline-soft);
    border-radius: 999px;
    overflow: hidden;
    margin: 0.4rem 0 0.6rem 0;
}
.result-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.result-meter-fill.high   { background: var(--good); }
.result-meter-fill.medium { background: var(--warn); }
.result-meter-fill.low    { background: var(--bad); }

.result-conf-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--ink-muted);
    margin-bottom: 1.4rem;
}
.result-conf-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
}

.result-message {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--ink-muted);
    line-height: 1.55;
    margin: 0;
    border-top: 1px solid var(--hairline-soft);
    padding-top: 1.2rem;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--hairline);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}
[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--forest-deep) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.6rem !important;
    font-weight: 500 !important;
    color: var(--forest-deep) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: var(--ink-subtle) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

/* ─── Dataframe ─── */
.stDataFrame {
    border: 1px solid var(--hairline);
    border-radius: 10px;
    overflow: hidden;
}

/* ─── Markdown body in tabs ─── */
.stTabs .stMarkdown p,
.stTabs .stMarkdown li {
    color: var(--ink);
    line-height: 1.7;
    font-size: 0.97rem;
}
.stTabs .stMarkdown h3 {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--forest-deep);
    margin-top: 2rem;
    letter-spacing: -0.01em;
}
.stTabs .stMarkdown h4 {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--sage);
    margin-top: 1.6rem;
}

/* ─── Footer ─── */
.hoya-footer {
    margin: 4rem auto 0 auto;
    padding-top: 2rem;
    border-top: 1px solid var(--hairline);
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--ink-subtle);
    line-height: 1.7;
}
.hoya-footer strong {
    color: var(--ink-muted);
    font-weight: 500;
}

/* ─── Citation block ─── */
.hoya-cite {
    background: var(--moss-bg);
    border-left: 2px solid var(--sage);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: var(--forest-deep);
    margin: 1rem 0;
    white-space: pre-wrap;
}

/* ─── Sample-data row ─── */
.sample-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--ink-subtle);
    margin: 0.4rem 0 0.6rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_HELP = {
    "pollinia_length":     "Longest axis of the pollen mass, measured under the microscope.",
    "pollinia_width":      "Widest perpendicular dimension of the pollinia.",
    "corpusculum_length":  "Length of the corpusculum (the dark, glandular structure that joins the pollinia).",
    "corpusculum_width":   "Widest dimension of the corpusculum.",
    "translator_arm_length": "Length of the translator arm connecting the corpusculum to the pollinia.",
    "translator_stalk":    "Length of the translator stalk (caudicle base).",
    "extension":           "Length of the caudicle extension beyond the pollinia attachment point.",
}

# Representative measurements for each clade (means from the training set)
SAMPLE_PRESETS = {
    "Acanthostemma": dict(pollinia_length=0.56, pollinia_width=0.30,
                          corpusculum_length=0.61, corpusculum_width=0.35,
                          extension=0.29, translator_arm_length=0.15,
                          translator_stalk=0.58),
    "Hoya":          dict(pollinia_length=0.78, pollinia_width=0.42,
                          corpusculum_length=0.48, corpusculum_width=0.28,
                          extension=0.22, translator_arm_length=0.12,
                          translator_stalk=0.45),
    "Pterostelma":   dict(pollinia_length=0.95, pollinia_width=0.52,
                          corpusculum_length=0.55, corpusculum_width=0.32,
                          extension=0.31, translator_arm_length=0.18,
                          translator_stalk=0.62),
    "Centrostemma":  dict(pollinia_length=0.62, pollinia_width=0.38,
                          corpusculum_length=0.52, corpusculum_width=0.30,
                          extension=0.25, translator_arm_length=0.14,
                          translator_stalk=0.50),
}

DEFAULT_INPUTS = SAMPLE_PRESETS["Acanthostemma"]


def engineer_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering for pollinaria morphometrics."""
    d = df.copy()
    eps = 1e-6

    d["pollinia_ratio"]     = d["pollinia_length"] / (d["pollinia_width"] + eps)
    d["corp_ratio"]         = d["corpusculum_length"] / (d["corpusculum_width"] + eps)
    d["translator_ratio"]   = d["translator_arm_length"] / (d["translator_stalk"] + eps)
    d["extension_index"]    = d["extension"] / (d["pollinia_length"] + eps)

    d["pollinia_area"]      = d["pollinia_length"] * d["pollinia_width"]
    d["pollinia_perimeter"] = 2 * (d["pollinia_length"] + d["pollinia_width"])
    d["pollinia_compactness"] = (4 * np.pi * d["pollinia_area"]) / (d["pollinia_perimeter"] ** 2 + eps)
    d["corp_eccentricity"]  = np.sqrt(
        1 - (d["corpusculum_width"] ** 2 / (d["corpusculum_length"] ** 2 + eps))
    )

    d["log_pollinia_L"]     = np.log1p(d["pollinia_length"])
    d["log_corp_L"]         = np.log1p(d["corpusculum_length"])
    d["allometric_slope"]   = d["log_pollinia_L"] / (d["log_corp_L"] + eps)

    d["translator_leverage"] = d["translator_arm_length"] / (d["extension"] + eps)
    d["translator_total"]    = d["translator_arm_length"] + d["translator_stalk"]

    feature_cols = [
        "pollinia_length", "pollinia_width", "corpusculum_length",
        "corpusculum_width", "extension", "pollinia_ratio", "corp_ratio",
        "extension_index", "pollinia_compactness", "corp_eccentricity",
        "allometric_slope", "translator_leverage", "translator_total",
    ]
    return d[feature_cols]


@st.cache_resource
def load_model():
    try:
        with open("hoya_clade_classifier_production.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            "Model file `hoya_clade_classifier_production.pkl` was not found in "
            "the application directory. Please verify deployment."
        )
        st.stop()


def predict_clade(measurements: dict, model_package: dict) -> dict:
    input_df = pd.DataFrame([measurements])
    X = engineer_enhanced_features(input_df)
    X_scaled = model_package["scaler"].transform(X)

    pred_label = model_package["model"].predict(X_scaled)[0]
    pred_clade = model_package["label_encoder"].inverse_transform([pred_label])[0]
    proba = model_package["model"].predict_proba(X_scaled)[0]

    return {
        "clade": pred_clade,
        "confidence": float(np.max(proba)),
        "probabilities": dict(zip(model_package["metadata"]["classes"], proba)),
    }


def confidence_tier(conf: float):
    """Return (tier, status_label, message)."""
    if conf >= 0.70:
        return (
            "high",
            "High Confidence",
            "Reliable classification. The ensemble agrees strongly on this clade. "
            "Suitable for routine workflow without expert review.",
        )
    if conf >= 0.50:
        return (
            "medium",
            "Medium Confidence",
            "The ensemble shows moderate agreement. Recommend verification by a "
            "Hoya taxonomist before recording this assignment.",
        )
    return (
        "low",
        "Low Confidence",
        "The ensemble is divided. Mandatory expert review; consider supplementary "
        "molecular identification (ITS, matK).",
    )


def probability_chart(probabilities: dict):
    df = pd.DataFrame(list(probabilities.items()), columns=["Clade", "Probability"])
    df = df.sort_values("Probability", ascending=True)

    top = df["Probability"].max()
    colors = ["#1a3d2e" if p == top else "#b8c5a8" for p in df["Probability"]]

    fig = go.Figure(
        go.Bar(
            x=df["Probability"],
            y=df["Clade"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.0%}" for p in df["Probability"]],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a1a", family="Inter"),
            hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, 1.08],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#f0ece2",
            zeroline=False,
            tickfont=dict(size=11, color="#8a8a8a", family="Inter"),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#1a1a1a", family="Inter"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=0, r=20, t=10, b=10),
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown('<p class="hoya-eyebrow">Pollinarium Morphometric Analysis</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="hoya-title">Philippine Hoya Clade Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hoya-tagline">An ensemble machine-learning system for rapid '
        "clade-level identification of Philippine <em>Hoya</em> from microscopic "
        "pollinarium measurements.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="hoya-rule">', unsafe_allow_html=True)


def render_sidebar(model_package: dict):
    with st.sidebar:
        st.markdown("## Model Performance")

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{model_package['metadata']['loocv_accuracy']:.0%}")
        c2.metric("Kappa", f"{model_package['metadata']['cohens_kappa']:.2f}")
        c1.metric("Specimens", model_package["metadata"]["n_samples"])
        c2.metric("Features", model_package["metadata"]["n_features"])

        st.markdown(
            f"<p style='font-family:Inter; font-size:0.78rem; color:#8a8a8a; "
            f"margin-top:1.5rem; line-height:1.7;'>"
            f"<strong style='color:#5a5a5a; text-transform:uppercase; "
            f"letter-spacing:0.12em; font-size:0.7rem;'>Architecture</strong><br>"
            f"{model_package['metadata']['ensemble_components']}"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='font-family:Inter; font-size:0.78rem; color:#8a8a8a; "
            "margin-top:1.5rem; line-height:1.9;'>"
            "<strong style='color:#5a5a5a; text-transform:uppercase; "
            "letter-spacing:0.12em; font-size:0.7rem;'>Clades</strong><br>"
            + "<br>".join(f"<em>{c}</em>" for c in model_package["metadata"]["classes"])
            + "</p>",
            unsafe_allow_html=True,
        )


def render_sample_pills():
    """Quick-fill buttons for representative clade measurements."""
    st.markdown('<p class="sample-label">Quick-fill with reference measurements</p>', unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_PRESETS))
    for col, (clade, preset) in zip(cols, SAMPLE_PRESETS.items()):
        if col.button(clade, key=f"preset_{clade}", use_container_width=True):
            for k, v in preset.items():
                st.session_state[k] = v
            st.rerun()


def init_state():
    for k, v in DEFAULT_INPUTS.items():
        st.session_state.setdefault(k, v)


def render_input_form():
    init_state()
    render_sample_pills()

    st.markdown(
        '<div class="hoya-card"><p class="hoya-card-title">Pollinia</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Length (mm)", 0.0, 10.0, key="pollinia_length",
            step=0.01, format="%.2f", help=FEATURE_HELP["pollinia_length"],
        )
    with c2:
        st.number_input(
            "Width (mm)", 0.0, 5.0, key="pollinia_width",
            step=0.01, format="%.2f", help=FEATURE_HELP["pollinia_width"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="hoya-card"><p class="hoya-card-title">Corpusculum</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Length (mm)", 0.0, 5.0, key="corpusculum_length",
            step=0.01, format="%.2f", help=FEATURE_HELP["corpusculum_length"],
        )
    with c2:
        st.number_input(
            "Width (mm)", 0.0, 2.0, key="corpusculum_width",
            step=0.01, format="%.2f", help=FEATURE_HELP["corpusculum_width"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="hoya-card"><p class="hoya-card-title">Translator &amp; Caudicle</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Arm length (mm)", 0.0, 2.0, key="translator_arm_length",
            step=0.01, format="%.2f", help=FEATURE_HELP["translator_arm_length"],
        )
    with c2:
        st.number_input(
            "Stalk (mm)", 0.0, 2.0, key="translator_stalk",
            step=0.01, format="%.2f", help=FEATURE_HELP["translator_stalk"],
        )
    with c3:
        st.number_input(
            "Caudicle ext. (mm)", 0.0, 2.0, key="extension",
            step=0.01, format="%.2f", help=FEATURE_HELP["extension"],
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result(result: dict):
    """Render the result block as a single HTML string so the wrapping card
    actually contains its children visually."""
    tier, status, message = confidence_tier(result["confidence"])
    conf = result["confidence"]
    pct_width = max(2, conf * 100)  # ensure the bar always shows a sliver

    html = f"""
    <div class="result-wrap {tier}">
        <p class="result-status {tier}">— {status}</p>
        <p class="result-clade">{result['clade']}</p>
        <div class="result-conf-row">
            <span>Confidence</span>
            <span class="result-conf-num">{conf:.1%}</span>
        </div>
        <div class="result-meter-track">
            <div class="result-meter-fill {tier}" style="width: {pct_width}%;"></div>
        </div>
        <p class="result-message">{message}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_classifier_tab(model_package: dict):
    st.markdown(
        '<p class="hoya-section">Specimen Measurements</p>'
        '<p class="hoya-section-sub">Enter pollinarium measurements in millimetres. '
        "Hover the label of any field for a definition.</p>",
        unsafe_allow_html=True,
    )

    render_input_form()

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    classify = st.button("Classify specimen", type="primary", use_container_width=True)

    if classify:
        measurements = {
            "pollinia_length":      st.session_state.pollinia_length,
            "pollinia_width":       st.session_state.pollinia_width,
            "corpusculum_length":   st.session_state.corpusculum_length,
            "corpusculum_width":    st.session_state.corpusculum_width,
            "extension":            st.session_state.extension,
            "translator_arm_length": st.session_state.translator_arm_length,
            "translator_stalk":     st.session_state.translator_stalk,
        }

        with st.spinner("Computing ensemble prediction…"):
            result = predict_clade(measurements, model_package)

        st.markdown('<p class="hoya-section">Result</p>', unsafe_allow_html=True)
        render_result(result)

        st.markdown(
            '<p style="font-family:Inter; font-size:0.72rem; font-weight:600; '
            "color:#5a5a5a; text-transform:uppercase; letter-spacing:0.16em; "
            'margin: 1.6rem 0 0.4rem 0;">Probability distribution</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            probability_chart(result["probabilities"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        prob_df = pd.DataFrame(
            sorted(result["probabilities"].items(), key=lambda x: -x[1]),
            columns=["Clade", "Probability"],
        )
        st.dataframe(
            prob_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Clade": st.column_config.TextColumn("Clade", width="medium"),
                "Probability": st.column_config.ProgressColumn(
                    "Probability", format="%.1f%%",
                    min_value=0.0, max_value=1.0,
                ),
            },
        )


def render_guide_tab():
    st.markdown(
        """
### Measurement protocol

1. Mount the pollinarium on a microscope slide at 40×–100× magnification.
2. Calibrate the eyepiece reticle against a stage micrometer before each session.
3. For each feature, take three measurements and record the mean to two decimal places.
4. Maintain a consistent viewing orientation throughout a specimen run.

#### Fields collected

The classifier requires seven raw measurements: pollinia length and width;
corpusculum length and width; translator arm length and stalk; and caudicle
extension. From these, thirteen morphometric features are engineered (ratios,
shape descriptors, allometric scaling, and translator mechanics) and passed
through a soft-voting ensemble.

### Confidence interpretation

| Tier | Range | Recommended action |
|------|-------|--------------------|
| High | ≥ 70% | Accept the classification. |
| Medium | 50–69% | Defer to a Hoya taxonomist for verification. |
| Low | < 50% | Mandatory expert review; consider molecular markers (ITS, matK). |

### Limitations

The training corpus has marked class imbalance — *Centrostemma* is represented
by a single specimen and *Pterostelma* by four — which constrains performance
on those clades. The classifier resolves to clade level only; species-level
identification remains a manual taxonomic task. Geographic scope is limited to
Philippine specimens.
        """
    )


def render_about_tab():
    st.markdown(
        """
### About this tool

The Philippine Hoya Clade Classifier is the first automated pollinarium-based
classification system for the genus *Hoya*. It is intended to support botanic
gardens, herbaria, and field surveys with rapid clade-level identification,
freeing taxonomist time for species-level adjudication.

**Version 1.0 · April 2026**

#### Technical specification

A soft-voting ensemble of a Support Vector Machine (RBF kernel), a Gradient
Boosting classifier, and an Extra Trees classifier, validated by leave-one-out
cross-validation on 64 specimens across four clades. Base accuracy is 75%,
rising to roughly 92–93% under a 70% confidence filter. Cohen's κ = 0.531.
"""
    )

    st.markdown(
        """
#### Dataset Owner

**Fernando B. Aurigue**
Retired Career Scientist
Department of Science and Technology — Philippine Nuclear Research Institute (DOST-PNRI)

The morphometric dataset underlying this classifier was assembled and curated
by Mr. Aurigue over decades of pollinarium fieldwork on Philippine *Hoya*. All
specimen measurements remain his intellectual contribution and are used here
under his permission.

#### Developer

**Jerald B. Bongalos**
Asian Institute of Management

#### Citation
        """
    )

    st.markdown(
        '<div class="hoya-cite">Bongalos, J. B. (2026). Deployable AI for Rapid '
        "Morphometric Classification of Philippine Hoya Clades. "
        "National Summit on Botanic Gardens and Arboreta.\n\n"
        "Dataset: Aurigue, F. B. (2026). Philippine Hoya Pollinarium "
        "Morphometric Database. DOST-PNRI.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
#### Repository

[github.com/Jbong17/HOYA-FLWR-AI](https://github.com/Jbong17/HOYA-FLWR-AI) · MIT License
        """
    )


def render_footer():
    st.markdown(
        """
        <div class="hoya-footer">
            <p style="margin: 0 0 0.6rem 0;">Developed by <strong>Jerald B. Bongalos</strong>, Asian Institute of Management</p>
            <p style="margin: 0 0 0.6rem 0;">Dataset by <strong>Fernando B. Aurigue</strong>, Retired Career Scientist · DOST-PNRI</p>
            <p style="margin: 1rem 0 0 0; color:#b8c5a8;">© 2026 · MIT License</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_hero()
    model_package = load_model()
    render_sidebar(model_package)

    tab1, tab2, tab3 = st.tabs(["Classifier", "Guide", "About"])
    with tab1:
        render_classifier_tab(model_package)
    with tab2:
        render_guide_tab()
    with tab3:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()
