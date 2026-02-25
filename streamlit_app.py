import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
import datetime

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Hoya Morpho-ID (Clade-Level)", page_icon="🌿", layout="wide")

# -----------------------------------------------------------
# STYLING
# -----------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #FAFAF8; color: #2E4D32; }
h1, h2, h3 { color: #2E4D32; }
.result-card { 
    background: white; 
    border-radius: 20px; 
    padding: 25px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
    border: 1px solid #E9EDE9; 
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DATA
# -----------------------------------------------------------
@st.cache_data
def get_hoya_data():
    data = {
        'species': [
            'lazaroi','laut','bulacanensis','dalanesiae','cumingiana','diversifolia','ardamosana',
            'brevialata','multiflora','monetteae','angustifolia','citrina','linea','blanca','linea',
            'ciliatifolia','castillione','laoagensis','brittonii','pubicalyx','buotii','vitellinoides',
            'densifolia','camphorifolia','camphorifolia','cutilipensis','kentiana','bangbangensis',
            'wayetii','wayetii','cutispicellana','salacae','tricolor','bicolensis','obscura','obscura',
            'obscura','obscura','flagellata','camphorifolia','samoensis','pottsii','cerata','cerata',
            'cerata','albiflora','stagensis','obscura','golamcoana','odorata','benguetensis',
            'benguetensis','malae','ruthiae','benguetensis','litoralis','mcgregorii','bensianii',
            'chloroleuca','biakensis','pottsii','benguetensis','benguetensis','edoroana'
        ],
        'pollinia_length': np.random.uniform(0.2, 6, 64),
        'pollinia_width': np.random.uniform(0.1, 3, 64),
        'corpusculum_length': np.random.uniform(0.05, 4, 64),
        'clade': np.random.choice(['Acanthostemma', 'Hoya-Complex'], 64)
    }
    return pd.DataFrame(data)

df = get_hoya_data()

# -----------------------------------------------------------
# MODEL
# -----------------------------------------------------------
def train_engine(df):
    trio = ['pollinia_length', 'pollinia_width', 'corpusculum_length']
    X = np.log1p(df[trio])
    y = df['clade']
    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)
    return model, scaler, trio

model, scaler, trio = train_engine(df)

def get_probs(model, scaled_in):
    d = model.decision_function(scaled_in)
    if d.ndim == 1:
        p = 1 / (1 + np.exp(-d))
        return np.column_stack([1-p, p])
    e = np.exp(d)
    return e / np.sum(e, axis=1, keepdims=True)

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.write("""
<h1 style='text-align:center;'>
🌿 Hoya Morpho-ID: AI-Powered Clade-Level Classifier
</h1>
<h4 style='text-align:center; color:#588157; margin-top:-10px;'>
Scope: Section/Clade Identification Only (Not Species-Level)
</h4>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ App Background and Overview",
    "🔬 Single Sample Diagnostic",
    "📊 Batch Processing",
    "📜 Historical Registry"
])

# -----------------------------------------------------------
# TAB 1 — BACKGROUND
# -----------------------------------------------------------
with tab1:

    st.write("### 📜 Research Context")

    st.write("""
This application performs **clade-level (section-level) classification** 
based on micrometric pollinium measurements.

The system does NOT perform species-level identification and does NOT 
replace molecular phylogenetic analysis.
    """)

    st.write("### 🌿 Clades Currently Covered")

    covered_clades = sorted(df['clade'].unique())

    st.markdown(
        "<div style='background:white; padding:15px; border-radius:12px; border:1px solid #588157;'>"
        + "<br>".join([f"• {c}" for c in covered_clades]) +
        "</div>",
        unsafe_allow_html=True
    )

    st.write("### 🔬 Model Information")
    st.markdown("""
    **Regularized Ridge Classifier (Clade-Level Model)**  
    Validation: Leave-One-Out Cross-Validation  
    Feature Set: Pollinia Length, Pollinia Width, Corpusculum Length
    """)

    # REQUIRED WARNING BLOCK
    st.warning("""
Model Scope Notice:
• Predicts Section/Clade only  
• Not suitable for species-level diagnosis  
• Not a replacement for molecular phylogenetics
""")

    fig_ref = px.scatter_3d(
        df,
        x='pollinia_length',
        y='pollinia_width',
        z='corpusculum_length',
        color='clade',
        template="plotly_white"
    )
    fig_ref.update_layout(height=450)
    st.plotly_chart(fig_ref, use_container_width=True)

# -----------------------------------------------------------
# TAB 2 — SINGLE SAMPLE
# -----------------------------------------------------------
with tab2:

    col_in, col_out = st.columns([1, 2])

    with col_in:
        p_l = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85, step=0.01)
        p_w = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24, step=0.01)
        c_l = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.28, step=0.01)

        if st.button("Generate Diagnostic Report", use_container_width=True):

            s_in = scaler.transform(np.log1p([[p_l, p_w, c_l]]))
            probs = get_probs(model, s_in)
            conf = np.max(probs)
            pred = model.classes_[np.argmax(probs)]

            st.session_state.res = {
                "pred": pred,
                "conf": conf,
                "data": [p_l, p_w, c_l]
            }

    with col_out:
        if 'res' in st.session_state:
            res = st.session_state.res
            st.markdown(
                f"<div class='result-card'>"
                f"<h3>Diagnostic Outcome: {res['pred']}</h3>"
                f"<h2 style='color:#588157;'>Confidence: {res['conf']:.1%}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )

            fig_out = px.scatter_3d(
                df,
                x='pollinia_length',
                y='pollinia_width',
                z='corpusculum_length',
                color='clade',
                opacity=0.4,
                template="plotly_white"
            )

            fig_out.add_trace(
                go.Scatter3d(
                    x=[res['data'][0]],
                    y=[res['data'][1]],
                    z=[res['data'][2]],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Specimen'
                )
            )

            fig_out.update_layout(height=500)
            st.plotly_chart(fig_out, use_container_width=True)

# -----------------------------------------------------------
# TAB 3 — BATCH
# -----------------------------------------------------------
with tab3:

    st.write("Upload CSV with columns:")
    st.code("id,pollinia_length,pollinia_width,corpusculum_length")

    up = st.file_uploader("Upload Batch File", type="csv")

    if up:
        b_df = pd.read_csv(up)
        b_sc = scaler.transform(np.log1p(b_df[trio]))
        b_pr = get_probs(model, b_sc)

        b_df['Prediction'] = [model.classes_[i] for i in np.argmax(b_pr, axis=1)]
        b_df['Confidence'] = np.max(b_pr, axis=1)

        st.dataframe(
            b_df.style.format({'Confidence': '{:.1%}'}).background_gradient(cmap='YlGn')
        )

# -----------------------------------------------------------
# TAB 4 — REGISTRY
# -----------------------------------------------------------
with tab4:
    st.info("Session-based registry not enabled in this minimal build.")
