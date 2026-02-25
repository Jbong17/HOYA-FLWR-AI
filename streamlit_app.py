import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
import datetime
import json

# --- PRE-CONFIG & ELEGANT CUSTOM STYLING ---
st.set_page_config(page_title="Hoya Morpho-ID", page_icon="🌸", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400&display=swap');
    .stApp { background-color: #FAFAF8; color: #4A5D4E; font-family: 'Lato', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: #2E4D32; font-style: italic; }
    
    /* Elegant Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; background-color: #E8EDE7; padding: 15px 15px 0px 15px; border-radius: 20px 20px 0 0; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 15px 15px 0 0; color: #7A8B7C; border: none; font-size: 16px; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF !important; color: #2E4D32 !important; border-bottom: 4px solid #A3B18A !important; font-weight: 700; }
    
    /* Result Cards */
    .result-card { background: white; border-radius: 25px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #E9EDE9; margin-bottom: 20px; color: #2E4D32; }
    
    /* Legible Table Styling */
    [data-testid="stTable"] { background-color: white; border-radius: 15px; overflow: hidden; }
    [data-testid="stTable"] td { color: #2E4D32 !important; font-size: 14px; }
    [data-testid="stTable"] th { background-color: #588157 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL ENGINE ---
@st.cache_data
def get_hoya_data():
    # Dataset remains the same as previous stable version
    data = {'species': ['lazaroi', 'laut', 'bulacanensis', 'dalanesiae', 'cumingiana', 'diversifolia', 'ardamosana', 'brevialata', 'multiflora', 'monetteae', 'angustifolia', 'citrina', 'linea', 'blanca', 'linea', 'ciliatifolia', 'castillione', 'laoagensis', 'brittonii', 'pubicalyx', 'buotii', 'vitellinoides', 'densifolia', 'camphorifolia', 'camphorifolia', 'cutilipensis', 'kentiana', 'bangbangensis', 'wayetii', 'wayetii', 'cutispicellana', 'salacae', 'tricolor', 'bicolensis', 'obscura', 'obscura', 'obscura', 'obscura', 'flagellata', 'camphorifolia', 'samoensis', 'pottsii', 'cerata', 'cerata', 'cerata', 'albiflora', 'stagensis', 'obscura', 'golamcoana', 'odorata', 'benguetensis', 'benguetensis', 'malae', 'ruthiae', 'benguetensis', 'litoralis', 'mcgregorii', 'bensianii', 'chloroleuca', 'biakensis', 'pottsii', 'benguetensis', 'benguetensis', 'edoroana'], 'pollinia_length': [0.56, 1.05, 0.51, 0.59, 0.85, 3.8, 5.6, 0.34, 5, 7.8, 4.5, 4.2, 4.3, 0.36, 2.7, 3, 0.26, 0.37, 0.7, 0.6, 0.69, 0.6, 0.82, 0.31, 0.4, 0.36, 0.48, 0.42, 0.39, 0.39, 0.35, 0.29, 0.36, 0.24, 0.35, 0.32, 0.35, 0.31, 0.41, 0.46, 0.48, 0.88, 0.83, 0.94, 0.51, 0.87, 0.9, 0.75, 0.72, 0.7, 0.57, 0.45, 0.85, 1.05, 0.39, 0.54, 0.46, 0.93, 0.46, 0.46, 0.36, 0.55, 0.54, 0.66], 'pollinia_width': [0.3, 0.5, 0.2, 0.24, 0.24, 1.4, 2.4, 0.17, 2.9, 2, 1.8, 1.8, 1.4, 0.15, 1.1, 1.3, 0.11, 0.15, 0.28, 0.24, 0.24, 0.24, 0.2, 0.13, 0.16, 0.18, 0.18, 0.17, 0.17, 0.16, 0.17, 0.12, 0.15, 0.11, 0.15, 0.14, 0.15, 0.12, 0.16, 0.22, 0.21, 0.25, 0.25, 0.28, 0.2, 0.25, 0.23, 0.3, 0.28, 0.25, 0.23, 0.2, 0.27, 0.28, 0.16, 0.2, 0.17, 0.19, 0.17, 0.17, 0.17, 0.22, 0.26, 0.26], 'corpusculum_length': [0.61, 0.96, 0.12, 0.22, 0.28, 1.9, 2.6, 0.15, 4.6, 2.5, 2.3, 1.9, 1.6, 0.1, 1.2, 1.1, 0.07, 0.11, 0.3, 0.38, 0.39, 0.26, 0.23, 0.1, 0.16, 0.14, 0.14, 0.14, 0.17, 0.16, 0.22, 0.09, 0.1, 0.09, 0.12, 0.12, 0.16, 0.13, 0.16, 0.24, 0.2, 0.42, 0.42, 0.45, 0.19, 0.33, 0.4, 0.35, 0.43, 0.4, 0.18, 0.16, 0.4, 0.47, 0.17, 0.23, 0.16, 0.11, 0.19, 0.2, 0.23, 0.18, 0.18, 0.43], 'clade': ['Acanthostemma', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex']}
    return pd.DataFrame(data)

def train_engine(df):
    trio = ['pollinia_length', 'pollinia_width', 'corpusculum_length']
    X = np.log1p(df[trio]); y = df['clade']
    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)
    return model, scaler, trio

def get_probs(model, scaled_in):
    d = model.decision_function(scaled_in)
    if d.ndim == 1: p = 1 / (1 + np.exp(-d)); return np.column_stack([1-p, p])
    e = np.exp(d); return e / np.sum(e, axis=1, keepdims=True)

df = get_hoya_data(); model, scaler, trio = train_engine(df)
if 'test_log' not in st.session_state: st.session_state.test_log = []

# --- TABS ---
st.write("<h1 style='text-align: center;'>🌿 Hoya Morpho-ID: Precision Taxonomy</h1>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Research Philosophy", "🔬 Single Sample Diagnostic Facility", "📊 Batch Processing", "📜 Historical Registry"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown("""
    <div style='background-color: #F3F5F1; padding: 25px; border-radius: 20px; border-left: 5px solid #588157; margin-bottom: 25px;'>
        <i>"In the delicate world of Hoya taxonomy, identification is often a dense thicket of morphological overlap. 
        This project seeks to harmonize Biological Geometry with Artificial Intelligence to preserve the heritage 
        of Philippine flora."</i>
    </div>
    """, unsafe_allow_html=True)
    
    col_n, col_m = st.columns([3, 2])
    with col_n:
        st.write("### 📜 The App Background")
        st.write("Taxonomic classification within the Philippine *Hoya* genus is traditionally a resource-intensive endeavor...")
        
        with st.expander("🔍 View the 34 Original Morphometric Features"):
            st.markdown("""
            * **Core Dimensions (11):** Pollinia Length, Pollinia Width, Corpusculum Length, Corpusculum Width, Shoulder Width, Waist Width, Hips Width, Base Extension, Translator Arm Length, Translator Stalk, Total Structure Length.
            * **Geometric Ratios (11):** Pollinia Aspect Ratio, Corpusculum Aspect Ratio, P/C Length Ratio, P/C Width Ratio, Shoulder-to-Waist Taper, Waist-to-Hips Ratio, Translator-to-Pollinia Ratio, Extension-to-Length Ratio, Shoulder-to-Hips Ratio, Stalk-to-Arm Ratio, Total Width Index.
            * **Derived Estimates (12):** Estimated Pollinia Area, Estimated Corpusculum Area, Relative Shoulder Position, Log-transformed Lengths (P & C), Sqrt-Area transformations, Squared Width deviations, Length-Width interaction terms, Normalized Volumetric Indices, and Asymmetry Coefficients.
            """)

        st.write("### ✨ The 'Golden Trio' Revelation")
        st.info("1. **Pollinia Length** | 2. **Pollinia Width** | 3. **Corpusculum Length**")

    with col_m:
        st.write("### 🔬 Modeling Performance")
        st.write(f"**Sample Size (n):** 64 Specimens")
        st.markdown("<div style='background:white; padding:20px; border-radius:15px; border:1px solid #ddd;'><h4>Scientific Integrity</h4><b>Accuracy: 91.67%</b><br><small>LOOCV validation on high-confidence samples.</small></div>", unsafe_allow_html=True)

# --- TAB 2: SINGLE SAMPLE DIAGNOSTIC FACILITY ---
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.write("### Input Micrometrics")
        p_l = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85)
        p_w = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24)
        c_l = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.28)
        if st.button("Generate Diagnostic Report"):
            s_in = scaler.transform(np.log1p([[p_l, p_w, c_l]]))
            probs = get_probs(model, s_in)
            conf, pred = np.max(probs), model.classes_[np.argmax(probs)]
            st.session_state.last_res = {"pred": pred, "conf": conf, "data": [p_l, p_w, c_l]}
            st.session_state.test_log.append({"Timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "Sample": "Manual", "P_L": p_l, "P_W": p_w, "C_L": c_l, "Result": pred, "Confidence": f"{conf:.1%}"})

    with col_out:
        if 'last_res' in st.session_state:
            r = st.session_state.last_res
            st.markdown(f"<div class='result-card'><h3>Diagnostic Outcome: {r['pred']}</h3><b>Confidence: {r['conf']:.1%}</b></div>", unsafe_allow_html=True)
            fig = px.scatter_3d(df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', color='clade', template="plotly_white")
            fig.add_trace(go.Scatter3d(x=[r['data'][0]], y=[r['data'][1]], z=[r['data'][2]], mode='markers', marker=dict(size=10, color='red', symbol='cross')))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: BATCH PROCESSING ---
with tab3:
    up = st.file_uploader("Upload Batch CSV", type="csv")
    if up:
        b_df = pd.read_csv(up)
        b_sc = scaler.transform(np.log1p(b_df[trio]))
        b_pr = get_probs(model, b_sc)
        b_df['Prediction'] = [model.classes_[i] for i in np.argmax(b_pr, axis=1)]
        b_df['Confidence'] = np.max(b_pr, axis=1)
        st.write("### Batch Analysis Results")
        st.dataframe(b_df.style.format({'Confidence': '{:.1%}'}).background_gradient(cmap='Greens'))
        # Logging batch
        for _, row in b_df.iterrows():
            st.session_state.test_log.append({"Timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "Sample": row.get('id', 'Batch'), "P_L": row[trio[0]], "P_W": row[trio[1]], "C_L": row[trio[2]], "Result": row['Prediction'], "Confidence": f"{row['Confidence']:.1%}"})

# --- TAB 4: HISTORICAL REGISTRY (Legible & Exportable) ---
with tab4:
    st.write("## 📜 Historical Session Registry")
    if st.session_state.test_log:
        log_df = pd.DataFrame(st.session_state.test_log)
        st.table(log_df)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Download CSV Log", log_df.to_csv(index=False), "hoya_logs.csv", "text/csv")
        with c2:
            st.download_button("📥 Download JSON Log", json.dumps(st.session_state.test_log), "hoya_logs.json", "application/json")
        with c3:
            if st.button("🗑️ Clear Registry"): st.session_state.test_log = []; st.rerun()
    else:
        st.info("Registry is currently empty.")
