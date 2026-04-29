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
import streamlit.components.v1 as components

GITHUB_REPO = "Jbong17/HOYA-FLWR-AI"
GITHUB_API = "https://api.github.com"
SUBMISSIONS_LOG_PATH = "submissions/predictions_log.csv"


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# Look for the moth-pollinarium icon under a few common upload names.
# GitHub's "Upload files" flow keeps whatever name the file had on the
# user's computer, so we try the most likely candidates in order rather
# than forcing them to rename.
_PWA_ICON_CANDIDATES = (
    "desktop_icon.png",
    "image.png",
    "icon.png",
    "Logo No Background Hoya.png",
)


def _detect_pwa_icon() -> str | None:
    for path in _PWA_ICON_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


PWA_ICON_PATH = _detect_pwa_icon()
# Fall back to the microscope emoji if no icon file is present.
_PAGE_ICON = PWA_ICON_PATH or "🔬"

st.set_page_config(
    page_title="Philippine Hoya Clade Classifier",
    page_icon=_PAGE_ICON,
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

/* ─── Pollinarium microscope watermark (very subtle, ~6% opacity) ─── */
/* Fixed-position pseudo-element behind all content. Image hosted in the
   repo and served by GitHub's raw CDN. If the file is missing the
   browser silently fails the request and the page renders without it. */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url('https://raw.githubusercontent.com/Jbong17/HOYA-FLWR-AI/main/pollinarium_watermark.JPG');
    background-size: 50% auto;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity: 0.06;
    pointer-events: none;
    z-index: 0;
}
.stApp > * {
    position: relative;
    z-index: 1;
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
    /* Garamond italic in sage. Bumped substantially over the title size
       for legibility — the italic sage style makes it visually recede
       otherwise. */
    font-family: 'Cormorant Garamond', 'EB Garamond', Garamond, Georgia, serif;
    font-size: clamp(1.85rem, 5.3vw, 3.3rem);
    font-weight: 500;
    font-style: italic;
    line-height: 1.15;
    letter-spacing: -0.01em;
    color: var(--sage);
    text-align: center;
    margin: 0 auto 1.4rem auto;
    white-space: nowrap;
    overflow: visible;
}
/* Defensively use !important on the centered hero text so Streamlit's
   default `[data-testid="stMarkdownContainer"] p` rules cannot override. */
.hoya-hero,
.hoya-hero p,
.hoya-hero h1 {
    text-align: center !important;
}
.hoya-tagline {
    font-family: 'Inter', sans-serif !important;
    font-size: clamp(1.02rem, 1.6vw, 1.15rem) !important;
    font-weight: 400 !important;
    color: var(--ink-muted) !important;
    text-align: center !important;
    line-height: 1.65 !important;
    max-width: 58ch !important;
    margin: 0 auto 2.2rem auto !important;
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

/* ─── Card (legacy class — kept for compatibility) ─── */
.hoya-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 1px 2px rgba(20, 30, 25, 0.03);
}

/* Style Streamlit's native bordered container to match the .hoya-card
   aesthetic. The bordered container properly wraps its child Streamlit
   components (number inputs, columns, etc.) which a manual <div> in
   markdown cannot do. */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--hairline) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    padding: 1.5rem 1.6rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 1px 2px rgba(20, 30, 25, 0.03) !important;
}

.hoya-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--sage);
    margin: 0 0 0.55rem 0;
}
.hoya-card-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.94rem;
    font-weight: 400;
    color: var(--ink-muted);
    line-height: 1.55;
    margin: 0 0 1.25rem 0;
}
.hoya-card-desc em {
    font-style: italic;
    color: var(--ink);
    font-weight: 500;
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

/* ─── Primary button (also applied to download button) ─── */
.stButton > button,
.stDownloadButton > button {
    background: var(--forest-deep) !important;
    color: var(--paper) !important;
    border: 1px solid var(--forest-deep) !important;
    border-radius: 999px !important;
    padding: 0.85rem 2.4rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    box-shadow: none !important;
    transition: transform 0.12s ease, background 0.18s ease;
    width: 100% !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--forest) !important;
    border-color: var(--forest) !important;
    transform: translateY(-1px);
    color: #ffffff !important;
}
.stButton > button:focus:not(:active),
.stDownloadButton > button:focus:not(:active) {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107, 142, 99, 0.2) !important;
    color: var(--paper) !important;
}

/* Inner button content (label spans) — Streamlit wraps button text */
.stButton > button p,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: var(--paper) !important;
    font-weight: 500 !important;
}

/* Secondary "sample" pill buttons */
.stButton > button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--forest-deep) !important;
    border: 1px solid var(--hairline) !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    letter-spacing: 0 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--moss-bg) !important;
    border-color: var(--sage) !important;
    color: var(--forest-deep) !important;
}
.stButton > button[kind="secondary"] p,
.stButton > button[kind="secondary"] span {
    color: var(--forest-deep) !important;
}

/* ─── Force light theme on native Streamlit widgets ─── */
/* Streamlit's native widgets (dataframe, expander, selectbox, textarea)
   respect prefers-color-scheme and go dark on users in dark mode. The
   .streamlit/config.toml file should pin theme=light, but these CSS
   overrides are belt-and-suspenders for any widget that escapes. */

/* DataFrame / table — wrapper only.
   IMPORTANT: do NOT override background or color on the inner elements
   ([role="grid"], [role="row"], [role="gridcell"], child <div>s).
   Streamlit's st.dataframe uses Glide Data Grid which paints content
   to a <canvas>; the ARIA elements sit above the canvas for screen
   readers but are transparent. Forcing them opaque hides the canvas
   data. Theme (light/dark) for the canvas is driven by config.toml. */
[data-testid="stDataFrame"] {
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Expander — header bar and body */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 12px !important;
    margin: 0.6rem 0 !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    background: var(--moss-bg) !important;
    color: var(--forest-deep) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0.85rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] details > summary:hover {
    background: #e9eee2 !important;
}
[data-testid="stExpander"] details[open] > summary {
    border-bottom: 1px solid var(--hairline) !important;
}
[data-testid="stExpander"] details > div:last-child,
[data-testid="stExpanderDetails"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    padding: 1.2rem !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] span {
    color: var(--ink) !important;
}

/* Selectbox */
[data-baseweb="select"] {
    background: var(--paper) !important;
}
[data-baseweb="select"] > div {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    color: var(--ink) !important;
    background: transparent !important;
}
/* Selectbox dropdown menu */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    color: var(--ink) !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background: var(--moss-bg) !important;
}

/* Textarea */
[data-testid="stTextArea"] textarea,
.stTextArea textarea,
textarea {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextArea"] label {
    color: var(--ink) !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}

/* Text input */
[data-testid="stTextInput"] input,
.stTextInput input {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
}

/* ─── File uploader — light card with dashed border ─── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--sage) !important;
    border-radius: 12px !important;
    padding: 1.4rem !important;
    color: var(--ink-muted) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--ink-muted) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--paper) !important;
    color: var(--forest-deep) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 999px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.2rem !important;
    width: auto !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--moss-bg) !important;
    border-color: var(--sage) !important;
}
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span {
    color: var(--forest-deep) !important;
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

/* Color the active-tab underline indicator (BaseWeb's [data-baseweb="tab-highlight"])
   forest-green instead of Streamlit's default red. Belt-and-suspenders alongside
   the primaryColor setting in .streamlit/config.toml. */
.stTabs [data-baseweb="tab-highlight"],
[data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    background-color: var(--forest-deep) !important;
    background: var(--forest-deep) !important;
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
/* New layout: Nukleyo logo on the LEFT, attribution stack on the RIGHT */
.hoya-footer-attribution {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--ink-subtle);
    line-height: 1.7;
    padding-top: 0.4rem;
    padding-bottom: 2rem;
}
.hoya-footer-attribution p {
    margin: 0 0 0.7rem 0 !important;
    text-align: left !important;
}
.hoya-footer-attribution strong {
    color: var(--ink-muted);
    font-weight: 600;
}

/* Fallback: centered text-only block (used if Nukleyo logo file missing) */
.hoya-footer-text {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--ink-subtle);
    line-height: 1.75;
    padding-bottom: 2rem;
}
.hoya-footer-text p {
    margin: 0 0 0.5rem 0 !important;
    text-align: center !important;
}
.hoya-footer-text strong {
    color: var(--ink-muted);
    font-weight: 500;
}

/* Copyright/initiative line — small, light, sage brand */
.hoya-footer-meta {
    margin-top: 1.2rem !important;
    font-size: 0.82rem !important;
    color: #b8c5a8 !important;
    letter-spacing: 0.02em;
}
.hoya-footer-meta strong {
    color: var(--sage) !important;
    font-weight: 600 !important;
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

/* ─── Mobile responsive (≤ 640 px wide — phones in portrait) ─── */
/* The bulk of the design uses clamp() and is already viewport-fluid, but
   three specific things need overrides on phones:
     1. The hero title/subtitle have `white-space: nowrap` so they stay
        on one line on desktop; on a narrow phone that overflows the
        viewport horizontally — let them wrap instead.
     2. Streamlit's st.columns() renders as [data-testid="stHorizontalBlock"]
        and stays horizontal at every viewport. On phones the 2- and 3-column
        input layouts squeeze each input to ~100 px wide. Stack them.
     3. Tabs and footer-logo column need slight tightening for narrow widths. */
@media (max-width: 640px) {
    /* Title and subtitle: allow wrapping; remove nowrap-induced overflow */
    .hoya-title,
    .hoya-subtitle {
        white-space: normal !important;
        max-width: 100% !important;
        padding: 0 0.4rem;
    }

    /* Block container: slightly tighter side padding on phones */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
    }

    /* Stack ALL st.columns vertically on phones */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Footer: center the attribution text once it stacks below the logo */
    .hoya-footer-attribution,
    .hoya-footer-attribution p {
        text-align: center !important;
    }
    .hoya-footer-logo-wrap [data-testid="stImage"] img {
        margin: 0 auto 1rem auto !important;
    }

    /* Tabs: tighter gap so 5 tabs fit without wrapping past the right edge */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem !important;
        padding: 0.5rem 0 !important;
    }

    /* Cards: slightly tighter padding inside */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1.1rem 1rem !important;
    }

    /* Section headings: shrink a touch so they don't dominate the screen */
    .hoya-section {
        font-size: 1.4rem !important;
    }
    .hoya-section-sub {
        font-size: 0.92rem !important;
    }

    /* Result clade name in the result card */
    .result-clade {
        font-size: clamp(2rem, 9vw, 2.5rem) !important;
    }

    /* Pill hyperlinks: 2 per row instead of 4 */
    .hoya-pill-link {
        flex-basis: calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 0.8rem !important;
    }
    .hoya-pills-row {
        gap: 0.6rem !important;
    }
}

/* Even tighter — very narrow (iPhone SE 1st-gen, etc., ≤ 380 px) */
@media (max-width: 380px) {
    .hoya-title {
        font-size: 1.6rem !important;
        line-height: 1.15 !important;
    }
    .hoya-subtitle {
        font-size: 1.3rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem !important;
    }
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

/* ─── Clade hyperlink pills ─── */
/* Real <a href> anchors styled to look like the previous st.button pills.
   Each link scrolls to an "About <clade>" section anchored at the bottom
   of the Classifier tab. */
.hoya-pills-row {
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    align-items: stretch;
    margin: 0.4rem 0 1.6rem 0;
    flex-wrap: wrap;
}
.hoya-pill-link {
    flex: 1 1 0;
    min-width: 140px;
    text-align: center;
    padding: 0.7rem 1.2rem;
    background: var(--surface) !important;
    color: var(--forest-deep) !important;
    border: 1px solid var(--hairline);
    border-radius: 999px;
    font-family: 'Inter', sans-serif;
    font-size: 0.96rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: background 0.15s ease, border-color 0.15s ease,
                transform 0.12s ease, box-shadow 0.15s ease;
    cursor: pointer;
}
.hoya-pill-link:hover {
    background: var(--moss-bg) !important;
    border-color: var(--sage);
    color: var(--forest-deep) !important;
    text-decoration: none !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(20, 30, 25, 0.06);
}
.hoya-pill-link:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px rgba(107, 142, 99, 0.25);
    border-color: var(--sage);
}
.hoya-pill-link:active {
    transform: translateY(0);
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
# BATCH CLASSIFICATION (CSV upload → per-row prediction → CSV download)
# ──────────────────────────────────────────────────────────────────────────────
BATCH_REQUIRED_COLS = [
    "pollinia_length", "pollinia_width",
    "corpusculum_length", "corpusculum_width",
    "translator_arm_length", "translator_stalk", "extension",
]
BATCH_TEMPLATE_COLS = ["specimen_id"] + BATCH_REQUIRED_COLS


def build_template_csv() -> bytes:
    """Template CSV with 3 example rows representing different clades."""
    template = pd.DataFrame(
        [
            ["JBA-2026-001", 0.56, 0.30, 0.61, 0.35, 0.15, 0.58, 0.29],
            ["JBA-2026-002", 0.78, 0.42, 0.48, 0.28, 0.12, 0.45, 0.22],
            ["JBA-2026-003", 0.95, 0.52, 0.55, 0.32, 0.18, 0.62, 0.31],
        ],
        columns=BATCH_TEMPLATE_COLS,
    )
    return template.to_csv(index=False).encode("utf-8")


def validate_batch_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Returns (is_valid, error_message). specimen_id is optional."""
    missing = [c for c in BATCH_REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    if len(df) == 0:
        return False, "CSV is empty (no data rows)."
    for col in BATCH_REQUIRED_COLS:
        try:
            pd.to_numeric(df[col], errors="raise")
        except Exception:
            return False, (
                f"Column '{col}' contains non-numeric values. "
                "All measurement columns must be numeric (mm)."
            )
    if df[BATCH_REQUIRED_COLS].isnull().any().any():
        return False, (
            "Some measurement values are missing. Every row must have a "
            "value for all 7 measurement columns."
        )
    return True, ""


def classify_batch(df: pd.DataFrame, model_package: dict) -> pd.DataFrame:
    """Run prediction on each row. Returns a results DataFrame."""
    results = []
    progress = st.progress(0, text="Classifying batch…")
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        specimen_id = (
            str(row["specimen_id"]) if "specimen_id" in df.columns
            and pd.notna(row.get("specimen_id"))
            else f"row-{i + 1}"
        )
        measurements = {c: float(row[c]) for c in BATCH_REQUIRED_COLS}
        try:
            result = predict_clade(measurements, model_package)
            entry = {
                "specimen_id": specimen_id,
                **{c: round(measurements[c], 3) for c in BATCH_REQUIRED_COLS},
                "predicted_clade": result["clade"],
                "confidence": round(float(result["confidence"]), 4),
                **{
                    f"prob_{k}": round(float(v), 4)
                    for k, v in result["probabilities"].items()
                },
            }
        except Exception as exc:
            entry = {
                "specimen_id": specimen_id,
                **{c: round(measurements[c], 3) for c in BATCH_REQUIRED_COLS},
                "predicted_clade": "ERROR",
                "confidence": 0.0,
                "error": str(exc)[:120],
            }
        results.append(entry)
        progress.progress((i + 1) / n)
    progress.empty()
    return pd.DataFrame(results)


def append_batch_to_history(results_df: pd.DataFrame, model_package: dict) -> int:
    """Append every successful row from a batch to st.session_state.history.
    Returns number of rows added."""
    classes = model_package["metadata"]["classes"]
    added = 0
    for _, row in results_df.iterrows():
        if row.get("predicted_clade") == "ERROR":
            continue
        measurements = {c: float(row[c]) for c in BATCH_REQUIRED_COLS}
        result = {
            "clade": row["predicted_clade"],
            "confidence": float(row["confidence"]),
            "probabilities": {
                k: float(row[f"prob_{k}"])
                for k in classes
                if f"prob_{k}" in row
            },
        }
        append_history(measurements, result)
        added += 1
    return added


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

    # Note: inline style="text-align:center" is added to the tagline because
    # some Streamlit themes inject default `[data-testid="stMarkdownContainer"] p`
    # rules with high specificity that defeat even !important class rules.
    # Inline styles beat all selector-based rules so this is bulletproof.
    st.markdown(
        '<div class="hoya-hero" style="text-align:center;">'
        '<h1 class="hoya-title" style="text-align:center;">'
        'Philippine <em>Hoya</em> Clade Classifier</h1>'
        '<p class="hoya-subtitle" style="text-align:center;">'
        'Pollinarium Morphometric Analysis</p>'
        '<p class="hoya-tagline" '
        'style="text-align:center; max-width:58ch; margin:0 auto 2.2rem auto;">'
        'An ensemble machine-learning system for rapid '
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


CLADE_BLURBS = {
    "Acanthostemma": {
        "subtitle": "Broad-leafed clade with fleshy, recurved corona",
        "n": "24 specimens (37.5%)",
        "body": (
            "Acanthostemma species typically have **fleshy, often recurved coronal "
            "lobes** giving the corona an upright profile (whence the name "
            "*acantho-* 'thorn'). Pollinaria tend toward shorter, broader "
            "pollinia with a relatively robust corpusculum. The translator "
            "arms are short and the caudicle extension is moderate.\n\n"
            "**Examples:** *H. carnosa*-aligned forms, *H. cumingiana*."
        ),
    },
    "Hoya": {
        "subtitle": "The type clade — defines the genus",
        "n": "35 specimens (54.7%)",
        "body": (
            "The **type clade** of the genus, with the most diverse species "
            "count in Philippine collections. Pollinia tend to be more "
            "elongate than those of *Acanthostemma*, with a wide morphometric "
            "range across the corpusculum. This is the classifier's strongest "
            "performance class (~80% recall under LOOCV).\n\n"
            "**Examples:** *H. cagayanensis*, *H. paziae*, *H. siariae*."
        ),
    },
    "Pterostelma": {
        "subtitle": "'Winged corona' group with large pollinaria",
        "n": "4 specimens (6.3%)",
        "body": (
            "From Greek *πτερόν* 'wing' + *στέμμα* 'crown' — the corona has "
            "thin, wing-like flanges projecting laterally. **Pollinaria are "
            "characteristically larger** than the other clades, with elongate "
            "pollinia and a long translator stalk and caudicle extension. "
            "The classifier's *translator_leverage* and *extension_index* "
            "engineered features carry strong weight here.\n\n"
            "**Examples:** *H. mindorensis*, *H. siamica*."
        ),
    },
    "Centrostemma": {
        "subtitle": "Rare — only one specimen in this dataset",
        "n": "1 specimen (1.6%)",
        "body": (
            "From Greek *κέντρον* 'centre' + *στέμμα* 'crown' — strongly "
            "developed central coronal axis with reduced lateral lobes, "
            "producing an upright, almost columnar corona. **Severely "
            "underrepresented** in the corpus; predictions for this clade "
            "should be treated as exploratory and require taxonomist "
            "verification.\n\n"
            "**Examples:** *H. multiflora* (under some treatments), *H. lyi*."
        ),
    },
}


def render_pwa_meta_tags():
    """Inject PWA-related <meta> and <link> tags into the parent document
    head so the app is installable as a desktop/mobile PWA via Chrome,
    Edge, and Safari with a custom icon.

    Critical detail for iPhone: iOS Safari reads <link rel="apple-touch-icon">
    when 'Add to Home Screen' is tapped. Without that specific link, iOS
    falls back to a screenshot of the page (or Streamlit's default brand
    mark). The favicon set via st.set_page_config doesn't substitute.

    The icon URL points at GitHub's raw CDN so it's a stable absolute URL
    regardless of how Streamlit is hosting the file at the moment."""
    icon_url = (
        "https://raw.githubusercontent.com/Jbong17/HOYA-FLWR-AI/main/"
        + (PWA_ICON_PATH.replace(" ", "%20") if PWA_ICON_PATH else "")
    )

    components.html(
        f"""
        <script>
        (function() {{
            const parent = window.parent;
            if (parent.__hoya_pwa_setup) return;
            parent.__hoya_pwa_setup = true;

            const head = parent.document.head;

            const addMeta = (name, content) => {{
                if (head.querySelector('meta[name="' + name + '"]')) return;
                const m = parent.document.createElement('meta');
                m.setAttribute('name', name);
                m.setAttribute('content', content);
                head.appendChild(m);
            }};

            const addLink = (rel, href, sizes) => {{
                let sel = 'link[rel="' + rel + '"]';
                if (sizes) sel += '[sizes="' + sizes + '"]';
                if (head.querySelector(sel)) return;
                const l = parent.document.createElement('link');
                l.setAttribute('rel', rel);
                l.setAttribute('href', href);
                if (sizes) l.setAttribute('sizes', sizes);
                head.appendChild(l);
            }};

            const iconUrl = "{icon_url}";

            // Browser-chrome theme colour on mobile + standalone PWA window
            addMeta('theme-color', '#1a3d2e');
            // Friendly name for installed-app surfaces (dock, taskbar, share sheets)
            addMeta('application-name', 'Hoya Classifier');
            // Apple iOS / iPadOS / macOS Safari add-to-home-screen support
            addMeta('apple-mobile-web-app-capable', 'yes');
            addMeta('apple-mobile-web-app-status-bar-style', 'default');
            addMeta('apple-mobile-web-app-title', 'Hoya Classifier');

            // CRITICAL for iPhone: apple-touch-icon link tag tells iOS Safari
            // which image to use for the home-screen icon. Without this, iOS
            // falls back to a webpage screenshot or Streamlit's default mark.
            if (iconUrl && iconUrl.endsWith('main/') === false) {{
                addLink('apple-touch-icon', iconUrl);
                addLink('apple-touch-icon', iconUrl, '180x180');
                addLink('apple-touch-icon', iconUrl, '152x152');
                addLink('apple-touch-icon', iconUrl, '120x120');
                // Standard <link rel="icon"> for desktop browsers PWA install
                addLink('icon', iconUrl);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def render_cross_tab_nav_script():
    """Inject JavaScript that turns the .hoya-pill-link anchors in the
    Classifier tab into real cross-tab navigation: clicking a pill activates
    the Guide tab, then scrolls to the matching #about-<clade> anchor.

    Why JavaScript: Streamlit's st.tabs are switched client-side via clicking
    the [role=tab] button. Plain HTML anchor links can't trigger that switch
    on their own — when the target is hidden in a non-active tab, the browser
    has nothing visible to scroll to.

    Why a guard: the components.html iframe is reloaded on every Streamlit
    rerun. Without the `__hoya_clade_nav_setup` guard on window.parent, we'd
    attach a new click listener every rerun and the navigation would fire
    multiple times on a single click."""
    components.html(
        """
        <script>
        (function() {
            const parent = window.parent;
            if (parent.__hoya_clade_nav_setup) return;
            parent.__hoya_clade_nav_setup = true;

            const doc = parent.document;

            doc.addEventListener('click', function(e) {
                // Match the clicked element OR any ancestor pill link
                const link = e.target.closest && e.target.closest('.hoya-pill-link');
                if (!link) return;

                e.preventDefault();
                e.stopPropagation();

                const href = link.getAttribute('href') || '';
                const anchorId = href.startsWith('#') ? href.slice(1) : href;

                // Find the Guide tab button — Streamlit tabs render as
                // [role="tab"] elements with the visible label as text.
                const tabs = doc.querySelectorAll('[data-baseweb="tab"]');
                let guideTab = null;
                tabs.forEach(function(tab) {
                    if (tab.textContent.trim() === 'Guide') guideTab = tab;
                });

                if (guideTab) {
                    // Click only if not already active. aria-selected is
                    // "true" on the active tab.
                    if (guideTab.getAttribute('aria-selected') !== 'true') {
                        guideTab.click();
                    }
                }

                // After the tab content renders, scroll to the anchor.
                // Two retries spaced apart in case Streamlit's tab switch
                // takes longer than the first delay.
                const scrollToAnchor = function(retries) {
                    const target = doc.getElementById(anchorId);
                    if (target) {
                        target.scrollIntoView({behavior: 'smooth', block: 'start'});
                    } else if (retries > 0) {
                        setTimeout(function() { scrollToAnchor(retries - 1); }, 200);
                    }
                };
                setTimeout(function() { scrollToAnchor(3); }, 250);
            }, true);  // capture phase, so we beat any default link handlers
        })();
        </script>
        """,
        height=0,
    )


def render_clade_pill_links():
    """Four real <a href> hyperlinks styled as pills. Each link scrolls
    the page to its corresponding 'About <clade>' anchor section rendered
    by render_clade_info_anchored() at the bottom of the Classifier tab.

    Note on anchor matching: Streamlit auto-generates anchor IDs from
    markdown header text using a slug rule (lowercase, spaces → dashes).
    Each href below must match the slug of the corresponding ### header
    rendered later in the same tab.
    """
    st.markdown(
        '<p class="sample-label">About the four clades — tap a name to learn more</p>'
        '<div class="hoya-pills-row">'
        '<a class="hoya-pill-link" href="#about-acanthostemma">Acanthostemma</a>'
        '<a class="hoya-pill-link" href="#about-the-hoya-clade">Hoya</a>'
        '<a class="hoya-pill-link" href="#about-pterostelma">Pterostelma</a>'
        '<a class="hoya-pill-link" href="#about-centrostemma">Centrostemma</a>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_clade_info_anchored():
    """Detailed clade info section. Renders the four 'About <clade>' anchored
    subsections that the pill hyperlinks scroll to. Each ### header gets a
    Streamlit-auto-generated anchor ID matching the href in
    render_clade_pill_links()."""
    st.markdown(
        '<p class="hoya-section">About the four clades</p>'
        '<p class="hoya-section-sub">Click any clade pill above to scroll '
        "directly to its description. Each entry covers diagnostic "
        "morphology, dataset representation, and representative species.</p>",
        unsafe_allow_html=True,
    )

    headings = {
        "Acanthostemma": "About Acanthostemma",
        "Hoya":          "About the Hoya clade",
        "Pterostelma":   "About Pterostelma",
        "Centrostemma":  "About Centrostemma",
    }

    for clade, info in CLADE_BLURBS.items():
        st.markdown(
            f"### {headings[clade]}\n\n"
            f"*{info['subtitle']}* · **{info['n']}**\n\n"
            f"{info['body']}\n\n"
            "---"
        )


def init_state():
    for k, v in DEFAULT_INPUTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_measurements", None)


def render_input_form():
    init_state()
    render_clade_pill_links()

    # ── Pollinia ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<p class="hoya-card-title">Pollinia</p>'
            '<p class="hoya-card-desc">The two compact, waxy pollen masses '
            'transferred as a single unit during pollination. Their length '
            'and width are diagnostic across <em>Hoya</em> clades.</p>',
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

    # ── Corpusculum ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<p class="hoya-card-title">Corpusculum</p>'
            '<p class="hoya-card-desc">The dark, sclerotised gland at the apex '
            'of the pollinarium that physically clips onto a visiting pollinator '
            'and carries the pollinia to the next flower.</p>',
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

    # ── Translator & Caudicle ─────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            '<p class="hoya-card-title">Translator &amp; Caudicle</p>'
            '<p class="hoya-card-desc">The connecting structures between '
            'corpusculum and pollinia: a flexible <em>arm</em>, a proximal '
            '<em>stalk</em> (caudicle), and a slender <em>caudicle '
            'extension</em> that levers the pollinia into alignment with '
            'the next stigma.</p>',
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


def render_batch_tab(model_package: dict):
    """Bulk classification via CSV upload."""
    st.markdown(
        '<p class="hoya-section">Batch Classification</p>'
        '<p class="hoya-section-sub">Classify many specimens at once. Download '
        'the template, fill in your measurements (one specimen per row), '
        'upload it back, and download the results as a single CSV with '
        'predicted clade, confidence, and full probability distribution.</p>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("batch_results", None)

    # ── Step 1 — template ─────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:Inter; font-size:0.78rem; font-weight:600; '
        "color:#5a5a5a; text-transform:uppercase; letter-spacing:0.16em; "
        'margin: 1.6rem 0 0.6rem 0;">1 · Get the template</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download CSV template",
            data=build_template_csv(),
            file_name="hoya_classifier_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.markdown(
            '<p style="font-family:Inter; font-size:0.92rem; color:#5a5a5a; '
            'line-height:1.6; margin:0; padding-top:0.4rem;">'
            "8 columns · 3 example rows · UTF-8 encoded. The "
            "<code>specimen_id</code> column is optional but recommended; "
            "the seven measurement columns are required, all in millimetres."
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Step 2 — upload ───────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:Inter; font-size:0.78rem; font-weight:600; '
        "color:#5a5a5a; text-transform:uppercase; letter-spacing:0.16em; "
        'margin: 2rem 0 0.6rem 0;">2 · Upload your CSV</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse the CSV: {exc}")
        return

    ok, msg = validate_batch_df(df)
    if not ok:
        st.error(msg)
        return

    n = len(df)
    st.markdown(
        f'<p style="font-family:Inter; font-size:0.95rem; color:#5a5a5a; '
        f'margin: 0.6rem 0 0.8rem 0;"><strong>{n}</strong> '
        f'specimen{"s" if n != 1 else ""} loaded and validated.</p>',
        unsafe_allow_html=True,
    )

    if st.button(
        f"Classify {n} specimen{'s' if n != 1 else ''}",
        type="primary",
        use_container_width=True,
    ):
        results_df = classify_batch(df, model_package)
        st.session_state.batch_results = results_df
        # Auto-append to history (per user choice in setup)
        added = append_batch_to_history(results_df, model_package)
        st.success(
            f"Classification complete. {added} of {n} rows added to "
            f"the **History** tab automatically."
        )

    # ── Step 3 — results ──────────────────────────────────────────────────
    results_df = st.session_state.get("batch_results")
    if results_df is None or len(results_df) == 0:
        return

    st.markdown(
        '<p style="font-family:Inter; font-size:0.78rem; font-weight:600; '
        "color:#5a5a5a; text-transform:uppercase; letter-spacing:0.16em; "
        'margin: 2rem 0 0.6rem 0;">3 · Results</p>',
        unsafe_allow_html=True,
    )

    # Summary
    n_rows = len(results_df)
    n_ok = int((results_df["predicted_clade"] != "ERROR").sum())
    n_high = int(((results_df["confidence"] >= 0.70)
                  & (results_df["predicted_clade"] != "ERROR")).sum())
    n_med = int(((results_df["confidence"] >= 0.50)
                 & (results_df["confidence"] < 0.70)
                 & (results_df["predicted_clade"] != "ERROR")).sum())
    n_low = int(((results_df["confidence"] < 0.50)
                 & (results_df["predicted_clade"] != "ERROR")).sum())
    n_err = n_rows - n_ok

    st.markdown(
        f'<p style="font-family:Inter; font-size:0.92rem; color:#5a5a5a; '
        f'margin: 0 0 1rem 0;">'
        f"<strong>{n_rows}</strong> classified  ·  "
        f'<span style="color:#2d5e3e;">{n_high} high</span>  ·  '
        f'<span style="color:#9a6f1f;">{n_med} medium</span>  ·  '
        f'<span style="color:#8b3a3a;">{n_low} low</span>'
        + (f'  ·  <span style="color:#8b3a3a;">{n_err} error</span>'
           if n_err else "")
        + "</p>",
        unsafe_allow_html=True,
    )

    # Display — multiply 0–1 probability/confidence values by 100 so the
    # ProgressColumn / NumberColumn format strings produce real percentages
    # (e.g. 85.0% not 0.8%).
    display_df = results_df.copy()
    for col in ["confidence", "prob_Acanthostemma", "prob_Centrostemma",
                "prob_Hoya", "prob_Pterostelma"]:
        if col in display_df.columns:
            display_df[col] = display_df[col] * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "specimen_id": st.column_config.TextColumn("Specimen ID", width="medium"),
            "predicted_clade": st.column_config.TextColumn("Clade", width="small"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%",
                min_value=0.0, max_value=100.0,
            ),
            "pollinia_length": st.column_config.NumberColumn("P. length", format="%.2f"),
            "pollinia_width":  st.column_config.NumberColumn("P. width",  format="%.2f"),
            "corpusculum_length": st.column_config.NumberColumn("C. length", format="%.2f"),
            "corpusculum_width":  st.column_config.NumberColumn("C. width",  format="%.2f"),
            "translator_arm_length": st.column_config.NumberColumn("T. arm", format="%.2f"),
            "translator_stalk": st.column_config.NumberColumn("T. stalk", format="%.2f"),
            "extension": st.column_config.NumberColumn("Caud.", format="%.2f"),
            "prob_Acanthostemma": st.column_config.NumberColumn("Acanthostemma", format="%.1f%%"),
            "prob_Centrostemma":  st.column_config.NumberColumn("Centrostemma",  format="%.1f%%"),
            "prob_Hoya":          st.column_config.NumberColumn("Hoya",          format="%.1f%%"),
            "prob_Pterostelma":   st.column_config.NumberColumn("Pterostelma",   format="%.1f%%"),
        },
    )

    # Download button
    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    cdl1, cdl2 = st.columns([1, 1])
    with cdl1:
        st.download_button(
            "Download results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"hoya_batch_results_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cdl2:
        if st.button("Clear results", use_container_width=True):
            st.session_state.batch_results = None
            st.rerun()


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
    # Multiply 0–1 confidence by 100 so format "%.1f%%" produces a real
    # percentage label (84.9% rather than 0.8%).
    display_df["confidence"] = display_df["confidence"] * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp_utc": st.column_config.TextColumn("Time (UTC)", width="medium"),
            "predicted_clade": st.column_config.TextColumn("Clade", width="medium"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%", min_value=0.0, max_value=100.0,
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
    # The anchored "About the four clades" section — destination of the
    # pill hyperlinks rendered in the Classifier tab. Cross-tab navigation
    # is handled by JavaScript injected by render_cross_tab_nav_script().
    render_clade_info_anchored()

    # Rest of the guide content
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

The training corpus has marked class imbalance — *Centrostemma* is
represented by a single specimen and *Pterostelma* by four — which
constrains performance on those clades. The classifier resolves to **clade
level only**; species-level identification remains a manual taxonomic task.
Geographic scope is limited to Philippine specimens.
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
    """Footer with Nukleyo logo anchored at the top-left of the attribution
    block (developer credentials, dataset attribution, copyright)."""
    # Hairline divider above the footer
    st.markdown(
        '<hr style="border:0; border-top:1px solid #e8e3d8; margin:4rem 0 2rem 0;">',
        unsafe_allow_html=True,
    )

    if os.path.exists(NUKLEYO_LOGO_PATH):
        # Two-column: logo on left, attribution text on right
        col_logo, col_text = st.columns([1, 5])
        with col_logo:
            st.image(NUKLEYO_LOGO_PATH, width=100)
        with col_text:
            st.markdown(
                '<div class="hoya-footer-attribution">'
                '<p><strong>Jerald B. Bongalos</strong>, PhD in Data Science<br>'
                'Aboitiz School of Innovation, Technology and Entrepreneurship '
                '| Asian Institute of Management</p>'
                '<p>Dataset by <strong>Fernando B. Aurigue</strong>, '
                'Retired Career Scientist · DOST-PNRI</p>'
                '<p class="hoya-footer-meta">© 2026 · MIT License · '
                'An initiative of <strong>NUKLEYO DECISION SCIENCE</strong></p>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        # No-logo fallback: render attribution centered, full-width
        html = (
            '<div class="hoya-footer-text">'
            '<p style="text-align:center;"><strong>Jerald B. Bongalos</strong>, '
            'PhD in Data Science, Aboitiz School of Innovation, Technology '
            'and Entrepreneurship | Asian Institute of Management</p>'
            '<p style="text-align:center;">Dataset by '
            '<strong>Fernando B. Aurigue</strong>, Retired Career Scientist · DOST-PNRI</p>'
            '<p class="hoya-footer-meta" style="text-align:center;">'
            '© 2026 · MIT License · '
            'An initiative of <strong>NUKLEYO DECISION SCIENCE</strong></p>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_cross_tab_nav_script()
    render_pwa_meta_tags()
    render_hero()
    model_package = load_model()
    render_sidebar(model_package)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Classifier", "Batch", "History", "Guide", "About"]
    )
    with tab1:
        render_classifier_tab(model_package)
    with tab2:
        render_batch_tab(model_package)
    with tab3:
        render_history_tab()
    with tab4:
        render_guide_tab()
    with tab5:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()
