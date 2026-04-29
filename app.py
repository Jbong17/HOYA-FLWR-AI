"""
Philippine Hoya Clade Classifier
AI-Powered Pollinarium Morphometric Analysis

Developer:        Jerald B. Bongalos (Asian Institute of Management)
Dataset Owner:    Fernando B. Aurigue (Retired Career Scientist, DOST-PNRI)
"""

import base64
import datetime as _dt
import io
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

GITHUB_REPO = "Jbong17/HOYA-FLWR-AI"
GITHUB_API = "https://api.github.com"
SUBMISSIONS_LOG_PATH = "submissions/predictions_log.csv"


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
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

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
/* Hero block — centered, emblem on top */
.hoya-hero {
    text-align: center;
    margin: 0 auto 2.8rem auto;
}
.hoya-emblem {
    display: block;
    width: clamp(110px, 14vw, 150px);
    height: auto;
    margin: 0 auto 1.6rem auto;
}

/* When using the real photographic logo via st.image(), constrain
   width and remove Streamlit's default column padding for a tight look. */
.hoya-logo-wrap {
    text-align: center;
    margin: 0 auto 1.4rem auto;
}
.hoya-logo-wrap [data-testid="stImage"] {
    margin: 0 auto;
}
.hoya-logo-wrap [data-testid="stImage"] img {
    max-width: clamp(220px, 38vw, 380px);
    height: auto;
    margin: 0 auto;
    display: block;
}
.hoya-eyebrow {
    /* legacy class — kept for any non-hero usage; not in current hero */
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--sage);
    text-align: center;
    margin: 0 auto 1.1rem auto;
}
.hoya-title {
    /* Sized so "Philippine Hoya Clade Classifier" stays on ONE LINE across
       every reasonable viewport (~360px phone → ~1400px desktop). */
    font-family: 'Cormorant Garamond', 'EB Garamond', Garamond, Georgia, serif;
    font-size: clamp(1.25rem, 4vw, 2.6rem);
    font-weight: 600;
    line-height: 1.08;
    letter-spacing: -0.015em;
    color: var(--forest-deep);
    text-align: center;
    margin: 0.4rem auto 0.6rem auto;
    white-space: nowrap;
    overflow: visible;
}
.hoya-title em {
    font-style: italic;
    font-weight: 500;
}
.hoya-subtitle {
    /* Same Garamond family + size as title, differentiated by italic +
       sage color for editorial hierarchy without breaking visual unity. */
    font-family: 'Cormorant Garamond', 'EB Garamond', Garamond, Georgia, serif;
    font-size: clamp(1.25rem, 4vw, 2.6rem);
    font-weight: 400;
    font-style: italic;
    line-height: 1.12;
    letter-spacing: -0.01em;
    color: var(--sage);
    text-align: center;
    margin: 0 auto 1.4rem auto;
    white-space: nowrap;
    overflow: visible;
}
.hoya-tagline {
    font-family: 'Inter', sans-serif;
    font-size: clamp(1.02rem, 1.6vw, 1.15rem);
    font-weight: 400;
    color: var(--ink-muted);
    text-align: center;
    line-height: 1.65;
    max-width: 58ch;
    margin: 0 auto 2.2rem auto;
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
    font-family: 'EB Garamond', 'Cormorant Garamond', Garamond, Georgia, serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--forest-deep);
    letter-spacing: -0.005em;
    margin: 2.4rem 0 0.5rem 0;
    line-height: 1.2;
}
.hoya-section-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: var(--ink-muted);
    line-height: 1.6;
    margin: 0 0 1.5rem 0;
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
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--sage);
    margin: 0 0 1rem 0;
}

/* ─── Inputs ─── */
.stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    letter-spacing: 0 !important;
    line-height: 1.4 !important;
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
    font-family: 'EB Garamond', 'Cormorant Garamond', Garamond, Georgia, serif;
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
    font-family: 'EB Garamond', 'Cormorant Garamond', Garamond, Georgia, serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--forest-deep) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'EB Garamond', 'Cormorant Garamond', Garamond, Georgia, serif !important;
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
    line-height: 1.75;
    font-size: 1.05rem;
    font-family: 'Inter', sans-serif;
}
.stTabs .stMarkdown strong {
    color: var(--ink);
    font-weight: 600;
}
.stTabs .stMarkdown em {
    color: var(--ink);
}
.stTabs .stMarkdown h3 {
    font-family: 'EB Garamond', 'Cormorant Garamond', Garamond, Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--forest-deep);
    margin-top: 2.2rem;
    margin-bottom: 0.6rem;
    letter-spacing: -0.005em;
    line-height: 1.25;
}
.stTabs .stMarkdown h4 {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--sage);
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
}
.stTabs .stMarkdown table {
    font-size: 0.98rem;
    border-collapse: collapse;
    margin: 1rem 0;
}
.stTabs .stMarkdown th {
    background: var(--moss-bg);
    font-weight: 600;
    color: var(--forest-deep);
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--hairline);
    text-align: left;
}
.stTabs .stMarkdown td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--hairline-soft);
}

/* ─── Footer ─── */
.hoya-footer {
    margin: 4rem auto 0 auto;
    padding-top: 2.4rem;
    border-top: 1px solid var(--hairline);
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--ink-subtle);
    line-height: 1.75;
}
.hoya-footer p {
    margin: 0 0 0.5rem 0;
}
.hoya-footer strong {
    color: var(--ink-muted);
    font-weight: 500;
}
.hoya-footer-logo {
    display: block;
    width: clamp(72px, 8vw, 88px);
    height: auto;
    margin: 0 auto 1rem auto;
    opacity: 0.95;
}
.hoya-footer-divider {
    width: 60px;
    height: 1px;
    background: var(--hairline);
    border: 0;
    margin: 0.4rem auto 1.4rem auto;
}
.hoya-footer-meta {
    margin-top: 1.4rem !important;
    font-size: 0.78rem;
    color: #b8c5a8;
    letter-spacing: 0.02em;
}
.hoya-footer-meta strong {
    color: var(--sage);
    font-weight: 600;
    letter-spacing: 0.05em;
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
    "pollinia_length":
        "Pollinium — the compact, waxy mass of pollen grains. Measure the "
        "longest axis (proximal to distal pole) of one pollinium under the "
        "microscope. Each Hoya pollinarium contains two pollinia; record the "
        "mean of left and right.",
    "pollinia_width":
        "Widest dimension of the pollinium, measured perpendicular to its long "
        "axis at the broadest point.",
    "corpusculum_length":
        "Corpusculum — the dark, sclerotised gland at the apex of the "
        "pollinarium that clasps the pollinator. Measure from the proximal tip "
        "(where the translator arms emerge) to the distal apex along the "
        "central axis.",
    "corpusculum_width":
        "Widest dimension of the corpusculum, measured perpendicular to its "
        "long axis. Diagnostic for several Hoya clades.",
    "translator_arm_length":
        "Translator arm — the short flexible structure connecting each "
        "pollinium to the corpusculum. Measure from the corpusculum margin to "
        "the point of attachment on the pollinium.",
    "translator_stalk":
        "Translator stalk (caudicle) — the proximal portion of the translator "
        "between the corpusculum and the translator arm. Measure along its "
        "central axis.",
    "extension":
        "Caudicle extension — the slender extension of the caudicle that "
        "protrudes beyond the pollinia attachment point. Measure from the "
        "pollinium insertion to the distal terminus.",
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
# HISTORY (session-scoped log of every classification this browser session)
# ──────────────────────────────────────────────────────────────────────────────
HISTORY_COLUMNS = [
    "timestamp_utc",
    "predicted_clade", "confidence",
    "pollinia_length", "pollinia_width",
    "corpusculum_length", "corpusculum_width",
    "translator_arm_length", "translator_stalk", "extension",
    "prob_Acanthostemma", "prob_Centrostemma", "prob_Hoya", "prob_Pterostelma",
]


def append_history(measurements: dict, result: dict) -> None:
    entry = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "predicted_clade": result["clade"],
        "confidence": round(float(result["confidence"]), 4),
        **{k: round(float(v), 3) for k, v in measurements.items()},
        **{f"prob_{k}": round(float(v), 4) for k, v in result["probabilities"].items()},
    }
    st.session_state.history.append(entry)


def history_dataframe() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.DataFrame(st.session_state.history)
    # Order columns predictably; missing prob_* (e.g. Centrostemma) get 0
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[HISTORY_COLUMNS]


def history_csv_bytes() -> bytes:
    return history_dataframe().to_csv(index=False).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# GITHUB INTEGRATION (Issues for review queue, file commits for log)
# ──────────────────────────────────────────────────────────────────────────────
def _github_token() -> str | None:
    """Read the GitHub PAT from Streamlit secrets. Returns None if not set."""
    try:
        return st.secrets["github_token"]
    except (KeyError, FileNotFoundError):
        return None
    except Exception:
        return None


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_create_issue(title: str, body: str, labels: list[str]) -> tuple[bool, str]:
    """Create a GitHub Issue. Returns (ok, url-or-error-message)."""
    token = _github_token()
    if not token:
        return False, (
            "GitHub integration is not configured. The app maintainer needs to "
            "add a `github_token` secret in Streamlit Cloud → Settings → Secrets."
        )
    try:
        r = requests.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
            headers=_gh_headers(token),
            json={"title": title, "body": body, "labels": labels},
            timeout=15,
        )
    except requests.RequestException as exc:
        return False, f"Network error contacting GitHub: {exc}"
    if r.status_code == 201:
        return True, r.json()["html_url"]
    return False, f"GitHub API returned {r.status_code}: {r.text[:200]}"


def github_get_file_sha(path: str) -> tuple[str | None, str | None]:
    """Return (sha, current_b64_content) for a file; (None, None) if missing."""
    token = _github_token()
    if not token:
        return None, None
    try:
        r = requests.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
            headers=_gh_headers(token),
            timeout=10,
        )
    except requests.RequestException:
        return None, None
    if r.status_code == 200:
        data = r.json()
        return data.get("sha"), data.get("content")
    return None, None


def github_commit_file(path: str, content_bytes: bytes, message: str) -> tuple[bool, str]:
    """Create or update a file in the repo. Returns (ok, url-or-error)."""
    token = _github_token()
    if not token:
        return False, (
            "GitHub integration is not configured. The app maintainer needs to "
            "add a `github_token` secret in Streamlit Cloud → Settings → Secrets."
        )
    sha, _ = github_get_file_sha(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
            headers=_gh_headers(token),
            json=payload,
            timeout=20,
        )
    except requests.RequestException as exc:
        return False, f"Network error contacting GitHub: {exc}"
    if r.status_code in (200, 201):
        return True, r.json()["content"]["html_url"]
    return False, f"GitHub API returned {r.status_code}: {r.text[:200]}"


def build_review_issue(measurements: dict, result: dict,
                       proposed_label: str, notes: str) -> tuple[str, str, list[str]]:
    """Build (title, body, labels) for a submission Issue."""
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    body = f"""## Specimen Submission for Expert Review

**Submitted (UTC):** `{ts}`

### Measurements (mm)

| Field | Value |
| --- | --- |
| Pollinia length | `{measurements['pollinia_length']:.2f}` |
| Pollinia width | `{measurements['pollinia_width']:.2f}` |
| Corpusculum length | `{measurements['corpusculum_length']:.2f}` |
| Corpusculum width | `{measurements['corpusculum_width']:.2f}` |
| Translator arm length | `{measurements['translator_arm_length']:.2f}` |
| Translator stalk | `{measurements['translator_stalk']:.2f}` |
| Caudicle extension | `{measurements['extension']:.2f}` |

### Model output

- **Predicted clade:** {result['clade']}
- **Confidence:** {result['confidence']:.1%}

#### Probability distribution

| Clade | Probability |
| --- | --- |
""" + "\n".join(f"| {k} | {v:.1%} |"
                for k, v in sorted(result["probabilities"].items(),
                                   key=lambda x: -x[1])) + f"""

### Submitter's proposed label

> **{proposed_label}**

### Submitter's notes

{notes.strip() if notes and notes.strip() else "_(none provided)_"}

---

> :warning: **Pending taxonomist verification.** Do not include in the training
> dataset until a Hoya specialist has reviewed and confirmed the proposed clade.
> Close this issue with comment `verified` to mark approved for the next
> retraining cycle, or `rejected` with reasoning to dismiss.
"""
    title = f"Submission: proposed {proposed_label} (model: {result['clade']}, {result['confidence']:.0%})"
    labels = ["submission", "needs-review"]
    return title, body, labels


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
HERO_EMBLEM_SVG = """
<svg class="hoya-emblem" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Philippine Hoya Clade Classifier emblem">
  <!-- Concentric medallion rings -->
  <circle cx="80" cy="80" r="74" fill="none" stroke="#6b8e63" stroke-width="1" opacity="0.4"/>
  <circle cx="80" cy="80" r="68" fill="none" stroke="#1a3d2e" stroke-width="0.6" opacity="0.5"/>

  <!-- Five-petal stylised Hoya flower (5-fold radial symmetry) -->
  <g transform="translate(80 80)" fill="#1a3d2e">
    <ellipse cx="0" cy="-22" rx="7" ry="16" />
    <ellipse cx="0" cy="-22" rx="7" ry="16" transform="rotate(72)" />
    <ellipse cx="0" cy="-22" rx="7" ry="16" transform="rotate(144)" />
    <ellipse cx="0" cy="-22" rx="7" ry="16" transform="rotate(216)" />
    <ellipse cx="0" cy="-22" rx="7" ry="16" transform="rotate(288)" />
    <!-- Inner star (corona) -->
    <g fill="#9a6f1f">
      <ellipse cx="0" cy="-9" rx="2.6" ry="6" />
      <ellipse cx="0" cy="-9" rx="2.6" ry="6" transform="rotate(72)" />
      <ellipse cx="0" cy="-9" rx="2.6" ry="6" transform="rotate(144)" />
      <ellipse cx="0" cy="-9" rx="2.6" ry="6" transform="rotate(216)" />
      <ellipse cx="0" cy="-9" rx="2.6" ry="6" transform="rotate(288)" />
    </g>
    <circle r="2.4" fill="#1a3d2e"/>
  </g>

  <!-- Microscope scale-bar ticks (right side) -->
  <g stroke="#6b8e63" stroke-width="1" stroke-linecap="round">
    <line x1="146" y1="62" x2="151" y2="62"/>
    <line x1="143" y1="72" x2="151" y2="72" stroke-width="1.4"/>
    <line x1="146" y1="82" x2="151" y2="82"/>
    <line x1="143" y1="92" x2="151" y2="92" stroke-width="1.4"/>
    <line x1="146" y1="102" x2="151" y2="102"/>
  </g>

  <!-- Subtle leaf flourish (left) -->
  <path d="M 9,82 Q 16,72 30,76 Q 22,84 9,82 Z"
        fill="#6b8e63" opacity="0.55"/>
  <path d="M 12,80 L 28,77" stroke="#1a3d2e" stroke-width="0.6" opacity="0.5" fill="none"/>
</svg>
"""


LOGO_PATH = "Logo No Background Hoya.png"
NUKLEYO_LOGO_PATH = "Nukleyo DS Logo.png"


@st.cache_data
def _nukleyo_logo_data_uri() -> str | None:
    """Read the Nukleyo logo once and cache it as a base64 data URI.
    Returns None if the file is not in the working directory."""
    if not os.path.exists(NUKLEYO_LOGO_PATH):
        return None
    try:
        with open(NUKLEYO_LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def render_hero():
    """Hero with photographic logo (if present) or SVG emblem fallback."""
    if os.path.exists(LOGO_PATH):
        # Center the logo via three-column layout. The wrapper div lets us
        # target the st.image with custom CSS for sizing and removes
        # Streamlit's default column gutters from interfering visually.
        st.markdown('<div class="hoya-logo-wrap">', unsafe_allow_html=True)
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fallback: tasteful SVG emblem
        st.markdown(
            f'<div style="text-align:center; margin-bottom:1.4rem;">{HERO_EMBLEM_SVG}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="hoya-hero">'
        '<h1 class="hoya-title">Philippine <em>Hoya</em> Clade Classifier</h1>'
        '<p class="hoya-subtitle">Pollinarium Morphometric Analysis</p>'
        '<p class="hoya-tagline">An ensemble machine-learning system for rapid '
        'clade-level identification of Philippine <em>Hoya</em> from microscopic '
        'pollinarium measurements.</p>'
        '<hr class="hoya-rule">'
        '</div>',
        unsafe_allow_html=True,
    )


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
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_measurements", None)


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

        # Persist across reruns so the submission expander stays interactive
        st.session_state.last_result = result
        st.session_state.last_measurements = measurements
        # Auto-log to session history
        append_history(measurements, result)

    # Render the result block from session state so form interactions
    # (e.g. typing in the submission expander) don't make it disappear.
    if st.session_state.get("last_result") is not None:
        result = st.session_state.last_result
        measurements = st.session_state.last_measurements

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

        render_submission_section(measurements, result, model_package)


def render_submission_section(measurements: dict, result: dict, model_package: dict):
    """Lets the user contribute this specimen to the next retraining cycle,
    gated by expert review (creates a GitHub Issue tagged 'needs-review')."""
    st.markdown(
        '<p style="font-family:Inter; font-size:0.72rem; font-weight:600; '
        "color:#5a5a5a; text-transform:uppercase; letter-spacing:0.16em; "
        'margin: 2rem 0 0.4rem 0;">Contribute to dataset</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Submit this specimen for expert review", expanded=False):
        st.markdown(
            "If you have an expert-confirmed identification of this specimen, "
            "you may submit it as a candidate for the next training cycle. "
            "Submissions are queued for taxonomist verification by the dataset "
            "owner before any inclusion."
        )

        classes = list(model_package["metadata"]["classes"])
        # Default the dropdown to the model's prediction so a user agreeing
        # with the model just clicks submit.
        try:
            default_idx = classes.index(result["clade"])
        except ValueError:
            default_idx = 0

        proposed_label = st.selectbox(
            "Confirmed clade (your expert identification)",
            classes,
            index=default_idx,
            key="submission_proposed_label",
            help="Select the clade you (or a consulted taxonomist) have confirmed for this specimen.",
        )
        notes = st.text_area(
            "Notes (optional)",
            key="submission_notes",
            height=90,
            placeholder="Collection locality, herbarium voucher number, "
                        "expert who verified, observations…",
        )

        token_present = _github_token() is not None
        if not token_present:
            st.info(
                "**Expert-review submissions are not yet enabled on this deployment.** "
                "The maintainer needs to add a `github_token` secret in "
                "Streamlit Cloud → Settings → Secrets. Until then, you can still "
                "download your session history as CSV from the **History** tab."
            )

        submit_clicked = st.button(
            "Submit for review",
            key="submit_review_btn",
            disabled=not token_present,
            use_container_width=True,
        )

        if submit_clicked:
            with st.spinner("Opening review issue on GitHub…"):
                title, body, labels = build_review_issue(
                    measurements, result, proposed_label, notes,
                )
                ok, info = github_create_issue(title, body, labels)
            if ok:
                st.success(
                    f"Submission opened for review. Tracked at "
                    f"[{info.split('/')[-2]}#{info.split('/')[-1]}]({info})."
                )
            else:
                st.error(info)


def render_history_tab():
    """Browser-session log of every classification + sync-to-repo + CSV download."""
    st.markdown(
        '<p class="hoya-section">Session History</p>'
        '<p class="hoya-section-sub">Every classification you run in this browser '
        "session is logged here. Download the log as CSV at any time, or sync the "
        "current session to the repository for permanent record.</p>",
        unsafe_allow_html=True,
    )

    history = st.session_state.get("history", [])
    n = len(history)

    if n == 0:
        st.markdown(
            '<div class="hoya-card" style="text-align:center; padding:2.4rem 1rem;">'
            '<p style="font-family:Inter; color:#8a8a8a; margin:0; font-size:0.95rem;">'
            "No classifications yet. Run a prediction in the <strong>Classifier</strong> "
            "tab to start logging."
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    # Summary line
    high = sum(1 for h in history if h["confidence"] >= 0.70)
    med = sum(1 for h in history if 0.50 <= h["confidence"] < 0.70)
    low = sum(1 for h in history if h["confidence"] < 0.50)
    st.markdown(
        f'<p style="font-family:Inter; font-size:0.88rem; color:#5a5a5a; '
        f'margin: 0 0 1rem 0;">'
        f"<strong>{n}</strong> classification{'s' if n != 1 else ''} this session  ·  "
        f'<span style="color:#2d5e3e;">{high} high</span>  ·  '
        f'<span style="color:#9a6f1f;">{med} medium</span>  ·  '
        f'<span style="color:#8b3a3a;">{low} low</span>'
        f"</p>",
        unsafe_allow_html=True,
    )

    df = history_dataframe()
    # Display a compact, human-readable view
    display_df = df[[
        "timestamp_utc", "predicted_clade", "confidence",
        "pollinia_length", "pollinia_width",
        "corpusculum_length", "corpusculum_width",
        "translator_arm_length", "translator_stalk", "extension",
    ]].copy()
    display_df = display_df.iloc[::-1].reset_index(drop=True)  # newest first

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp_utc": st.column_config.TextColumn("Time (UTC)", width="medium"),
            "predicted_clade": st.column_config.TextColumn("Clade", width="medium"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%", min_value=0.0, max_value=1.0,
            ),
            "pollinia_length": st.column_config.NumberColumn("P. length", format="%.2f"),
            "pollinia_width":  st.column_config.NumberColumn("P. width",  format="%.2f"),
            "corpusculum_length": st.column_config.NumberColumn("C. length", format="%.2f"),
            "corpusculum_width":  st.column_config.NumberColumn("C. width",  format="%.2f"),
            "translator_arm_length": st.column_config.NumberColumn("T. arm", format="%.2f"),
            "translator_stalk": st.column_config.NumberColumn("T. stalk", format="%.2f"),
            "extension": st.column_config.NumberColumn("Caud.", format="%.2f"),
        },
    )

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.download_button(
            "Download CSV",
            data=history_csv_bytes(),
            file_name=f"hoya_history_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        token_present = _github_token() is not None
        sync_clicked = st.button(
            "Sync to repository",
            disabled=not token_present,
            help="Commit this session's log as a CSV to the GitHub repository "
                 "(requires github_token secret).",
            use_container_width=True,
        )

    with c3:
        clear_clicked = st.button("Clear history", use_container_width=True)

    if not token_present:
        st.markdown(
            '<p style="font-family:Inter; font-size:0.78rem; color:#8a8a8a; '
            'margin: 0.4rem 0 0 0;">Repository sync is disabled: the maintainer '
            "has not configured the <code>github_token</code> secret.</p>",
            unsafe_allow_html=True,
        )

    if sync_clicked:
        with st.spinner("Committing log to GitHub…"):
            ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
            ok, info = github_commit_file(
                SUBMISSIONS_LOG_PATH,
                history_csv_bytes(),
                f"chore: sync session predictions log ({ts}, n={n})",
            )
        if ok:
            st.success(f"Synced. View [predictions log on GitHub]({info}).")
        else:
            st.error(info)

    if clear_clicked:
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_measurements = None
        st.rerun()


def render_guide_tab():
    st.markdown(
        """
### Pollinarium anatomy

The *Hoya* **pollinarium** is the entire pollen-dispersal unit of a single
flower — a compound structure that detaches from the column as a whole and
attaches to the body of a visiting pollinator. It comprises four
morphologically distinct parts, each measurable under a compound microscope.

#### Pollinia

The **pollinia** *(singular: pollinium)* are the two compact, waxy masses of
pollen grains. Unlike loose pollen, the grains here are bound together into a
coherent unit by a sticky elastoviscin matrix, so the entire pollinium is
transferred as a single object during pollination. The pollinia are typically
elongate-ellipsoidal, flattened laterally, and translucent yellow to amber
under transmitted light. Their length and width are diagnostic at the clade
level: *Acanthostemma* tends to have shorter, broader pollinia; *Pterostelma*
characteristically larger and more elongate.

#### Corpusculum

The **corpusculum** is the small, dark, sclerotised (hardened, melanised)
clip-like gland at the distal apex of the pollinarium. It is the structure
that mechanically attaches to the pollinator — a bee's tongue, leg, or
mouthparts — when the insect probes the flower. Once attached, the entire
pollinarium is yanked free from the column and carried to the next bloom. The
corpusculum's length and width, and the ratio between them (eccentricity),
are highly diagnostic: it can be nearly circular in some clades and markedly
elongate in others.

#### Translator (caudicle)

The **translator** is the connecting tissue between the corpusculum and the
pollinia. In *Hoya* it has two components measured separately:

- **Translator arms** — short flexible bridges, one from each side of the
  corpusculum, each attaching to one pollinium.
- **Translator stalk** *(also called the caudicle proper)* — the proximal
  portion of the translator extending from the corpusculum before the arms
  diverge. The stalk's mechanical properties (length, flexibility) determine
  the geometry of pollinarium handover.

#### Caudicle extension

The **caudicle extension** is the slender continuation of the caudicle that
projects beyond the point where the pollinia insert. Functionally, it acts as
a lever arm: when the pollinarium is dehydrated by the warmth of the
pollinator's body, the extension flexes and rotates the pollinia into
alignment with the receptive stigmatic chamber of the next flower. Its length
relative to the pollinia (the *extension index*) is a key engineered feature
in this classifier.

---

### Measurement protocol

1. Mount the pollinarium on a microscope slide at **40× to 100×**
   magnification with a coverslip; for greatest precision, immerse in a drop
   of glycerine.
2. Calibrate the eyepiece reticle against a **stage micrometer** at the start
   of each session — the calibration drifts with objective changes.
3. For each feature, take **three measurements** and record the mean to two
   decimal places. This averages out reticle-reading parallax.
4. Maintain a consistent viewing orientation throughout the specimen run —
   pollinaria are three-dimensional and apparent dimensions change with tilt.

#### Fields collected

The classifier requires seven raw measurements: pollinia length and width;
corpusculum length and width; translator arm length and stalk; and caudicle
extension. From these, thirteen morphometric features are engineered — ratios,
shape descriptors (compactness, eccentricity), allometric scaling
(log-log relationships), and translator mechanics — and passed through a
soft-voting ensemble.

---

### Confidence interpretation

| Tier | Range | Recommended action |
|------|-------|--------------------|
| **High** | ≥ 70% | Accept the classification. Suitable for routine workflow. |
| **Medium** | 50–69% | Defer to a *Hoya* taxonomist for verification before recording. |
| **Low** | < 50% | Mandatory expert review; consider molecular markers (ITS, matK). |

---

### Limitations

The training corpus has marked class imbalance — *Centrostemma* is represented
by a single specimen and *Pterostelma* by four — which constrains performance
on those clades. The classifier resolves to **clade level only**;
species-level identification remains a manual taxonomic task. Geographic
scope is limited to Philippine specimens.
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
    """Footer with Nukleyo ownership mark, attribution, and copyright.

    NOTE: the HTML is built as a single flush-left string to avoid
    Streamlit's markdown parser interpreting indented lines as code blocks
    (any 4+ space indent triggers <pre><code> wrapping even with
    unsafe_allow_html=True, which would render the HTML as literal text)."""
    logo_uri = _nukleyo_logo_data_uri()
    logo_html = (
        f'<img class="hoya-footer-logo" src="{logo_uri}" alt="Nukleyo Decision Science"/>'
        '<hr class="hoya-footer-divider"/>'
        if logo_uri else ""
    )

    html = (
        '<div class="hoya-footer">'
        f'{logo_html}'
        '<p>Developed by <strong>Jerald B. Bongalos</strong>, Asian Institute of Management</p>'
        '<p>Dataset by <strong>Fernando B. Aurigue</strong>, Retired Career Scientist · DOST-PNRI</p>'
        '<p class="hoya-footer-meta">© 2026 · MIT License · '
        'An initiative of <strong>NUKLEYO DECISION SCIENCE</strong></p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_hero()
    model_package = load_model()
    render_sidebar(model_package)

    tab1, tab2, tab3, tab4 = st.tabs(["Classifier", "History", "Guide", "About"])
    with tab1:
        render_classifier_tab(model_package)
    with tab2:
        render_history_tab()
    with tab3:
        render_guide_tab()
    with tab4:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()
