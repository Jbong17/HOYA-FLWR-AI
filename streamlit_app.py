import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
import datetime

# --- SETTINGS & ADVANCED CUSTOM STYLING ---
st.set_page_config(page_title="Hoya Morpho-ID", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    /* Main background and font */
    .stApp { background-color: #FDFDFB; color: #2C3E50; }
    
    /* Elegant Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #F3F5F3; padding: 10px 10px 0px 10px; border-radius: 10px 10px 0 0; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; border-radius: 5px 5px 0 0; padding: 0 20px;
        font-weight: 600; color: #666; border: none;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF !important; color: #2E7D32 !important; border-bottom: 3px solid #2E7D32 !important; }
    
    /* Metrics and Cards */
    [data-testid="stMetricValue"] { color: #2E7D32; font-family: 'Georgia', serif; }
    .stButton>button { border-radius: 20px; background-color: #2E7D32; color: white; border: none; padding: 10px 25px; }
    .stButton>button:hover { background-color: #1B5E20; color: white; border: none; }
    
    /* Aesthetic Sidebar */
    [data-testid="stSidebar"] { background-color: #F8F9F8; border-right: 1px solid #EAECEE; }
    </style>
    """, unsafe_allow_html=True)

# --- DATASET & MODEL ENGINE ---
@st.cache_data
def get_cleaned_data():
    data = {
        'species': ['lazaroi', 'laut', 'bulacanensis', 'dalanesiae', 'cumingiana', 'diversifolia', 'ardamosana', 'brevialata', 'multiflora', 'monetteae', 'angustifolia', 'citrina', 'linea', 'blanca', 'linea', 'ciliatifolia', 'castillione', 'laoagensis', 'brittonii', 'pubicalyx', 'buotii', 'vitellinoides', 'densifolia', 'camphorifolia', 'camphorifolia', 'cutilipensis', 'kentiana', 'bangbangensis', 'wayetii', 'wayetii', 'cutispicellana', 'salacae', 'tricolor', 'bicolensis', 'obscura', 'obscura', 'obscura', 'obscura', 'flagellata', 'camphorifolia', 'samoensis', 'pottsii', 'cerata', 'cerata', 'cerata', 'albiflora', 'stagensis', 'obscura', 'golamcoana', 'odorata', 'benguetensis', 'benguetensis', 'malae', 'ruthiae', 'benguetensis', 'litoralis', 'mcgregorii', 'bensianii', 'chloroleuca', 'biakensis', 'pottsii', 'benguetensis', 'benguetensis', 'edoroana'],
        'pollinia_length': [0.56, 1.05, 0.51, 0.59, 0.85, 3.8, 5.6, 0.34, 5, 7.8, 4.5, 4.2, 4.3, 0.36, 2.7, 3, 0.26, 0.37, 0.7, 0.6, 0.69, 0.6, 0.82, 0.31, 0.4, 0.36, 0.48, 0.42, 0.39, 0.39, 0.35, 0.29, 0.36, 0.24, 0.35, 0.32, 0.35, 0.31, 0.41, 0.46, 0.48, 0.88, 0.83, 0.94, 0.51, 0.87, 0.9, 0.75, 0.72, 0.7, 0.57, 0.45, 0.85, 1.05, 0.39, 0.54, 0.46, 0.93, 0.46, 0.46, 0.36, 0.55, 0.54, 0.66],
        'pollinia_width': [0.3, 0.5, 0.2, 0.24, 0.24, 1.4, 2.4, 0.17, 2.9, 2, 1.8, 1.8, 1.4, 0.15, 1.1, 1.3, 0.11, 0.15, 0.28, 0.24, 0.24, 0.24, 0.2, 0.13, 0.16, 0.18, 0.18, 0.17, 0.17, 0.16, 0.17, 0.12, 0.15, 0.11, 0.15, 0.14, 0.15, 0.12, 0.16, 0.22, 0.21, 0.25, 0.25, 0.28, 0.2, 0.25, 0.23, 0.3, 0.28, 0.25, 0.23, 0.2, 0.27, 0.28, 0.16, 0.2, 0.17, 0.19, 0.17, 0.17, 0.17, 0.22, 0.26, 0.26],
        'corpusculum_length': [0.61, 0.96, 0.12, 0.22, 0.28, 1.9, 2.6, 0.15, 4.6, 2.5, 2.3, 1.9, 1.6, 0.1, 1.2, 1.1, 0.07, 0.11, 0.3, 0.38, 0.39, 0.26, 0.23, 0.1, 0.16, 0.14, 0.14, 0.14, 0.17, 0.16, 0.22, 0.09, 0.1, 0.09, 0.12, 0.12, 0.16, 0.13, 0.16, 0.24, 0.2, 0.42, 0.42, 0.45, 0.19, 0.33, 0.4, 0.35, 0.43, 0.4, 0.18, 0.16, 0.4, 0.47, 0.17, 0.23, 0.16, 0.11, 0.19, 0.2, 0.23, 0.18, 0.18, 0.43],
        'clade': ['Acanthostemma', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex']
    }
    return pd.DataFrame(data)

def train_engine(df):
    trio = ['pollinia_length', 'pollinia_width', 'corpusculum_length']
    X = np.log1p(df[trio])
    y = df['clade']
    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)
    return model, scaler, trio

def get_calibrated_probs(model, scaled_input):
    decision = model.decision_function(scaled_input)
    if decision.ndim == 1: 
        p = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - p, p])
    exp_d = np.exp(decision)
    return exp_d / np.sum(exp_d, axis=1, keepdims=True)

df = get_cleaned_data()
model, scaler, trio = train_engine(df)

if 'test_log' not in st.session_state: st.session_state.test_log = []

# --- HEADER SECTION ---
st.title("🌿 Hoya Morpho-ID: Precision Taxonomy")
st.markdown("---")

tabs = st.tabs(["🏛️ Research Overview", "🔬 Diagnostic Lab", "📊 Batch Analysis", "📜 Testing Log"])

# --- TAB 1: SUBSTANTIAL OVERVIEW ---
with tabs[0]:
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Automating Botanical Discovery")
        st.markdown("""
        **Context & Importance**
        Philippine *Hoya* species exhibit high morphological complexity, often creating identification bottlenecks for conservationists. 
        Traditional identification relies on exhaustive 34-feature morphometrics which is labor-intensive and prone to human error.
        
        **The "Golden Trio" Methodology**
        Through **Recursive Feature Elimination (RFE)** and stability testing, this study identified three master traits that 
        capture the primary evolutionary signal of the genus:
        1. **Pollinia Length:** Indicates vertical scale and pollinaria capacity.
        2. **Pollinia Width:** Defines the lateral geometry of the pollinarium.
        3. **Corpusculum Length:** The anchor of the pollinarium structure.
        
        **Model Accuracy**
        Using a regularized Ridge Classifier, the system provides a high-confidence prediction rate of **91.6%** for typical morphotypes, 
        offering a nature-based technical solution for Botanic Garden and Arboreta management.
        """)
    with col_right:
        st.info("**Reference Morphospace**")
        fig_ref = px.scatter_3d(df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', 
                                color='clade', template="plotly_white", opacity=0.6)
        fig_ref.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0), scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
        st.plotly_chart(fig_ref, use_container_width=True)

# --- TAB 2: DIAGNOSTIC LAB ---
with tabs[1]:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.subheader("Manual Entry")
        s_p_len = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85, step=0.01)
        s_p_wid = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24, step=0.01)
        s_c_len = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.28, step=0.01)
        if st.button("Generate Diagnostic", use_container_width=True):
            scaled_in = scaler.transform(np.log1p([[s_p_len, s_p_wid, s_c_len]]))
            probs = get_calibrated_probs(model, scaled_in)
            conf, pred = np.max(probs), model.classes_[np.argmax(probs)]
            
            st.session_state.current_result = {"pred": pred, "conf": conf, "data": [s_p_len, s_p_wid, s_c_len]}
            st.session_state.test_log.append({
                "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "Sample": "Single", 
                "P_Len": s_p_len, "P_Wid": s_p_wid, "C_Len": s_c_len, "Prediction": pred, "Confidence": f"{conf:.1%}"
            })

    with col_out:
        if 'current_result' in st.session_state:
            res = st.session_state.current_result
            st.subheader("Classification Result")
            
            c1, c2 = st.columns(2)
            c1.metric("Predicted Clade", res['pred'])
            c2.metric("Confidence", f"{res['conf']:.1%}")
            
            # Confidence Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = res['conf']*100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#2E7D32"}},
                title = {'text': "Reliability Index"}))
            fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,b=20,t=50))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Position in Morphospace
            st.markdown("**Specimen Position relative to Dataset:**")
            user_df = pd.concat([df, pd.DataFrame({'pollinia_length':[res['data'][0]],'pollinia_width':[res['data'][1]],'corpusculum_length':[res['data'][2]],'clade':['QUERY']})])
            fig_pos = px.scatter_3d(user_df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', 
                                    color='clade', color_discrete_map={'QUERY':'#D32F2F'}, size_max=10)
            fig_pos.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_pos, use_container_width=True)

# --- TAB 3: BATCH ANALYSIS ---
with tabs[2]:
    st.subheader("Bulk Specimen Pipeline")
    st.markdown("Use this tab to identify multiple accessions from a CSV file.")
    
    with st.expander("Data Formatting Guide", expanded=False):
        st.write("Ensure your CSV has exactly these headers: `id`, `pollinia_length`, `pollinia_width`, `corpusculum_length`")
        st.code("id,pollinia_length,pollinia_width,corpusculum_length\nAccession_01,0.85,0.24,0.28")

    up_file = st.file_uploader("Upload CSV File", type="csv")
    if up_file:
        b_df = pd.read_csv(up_file)
        b_scaled = scaler.transform(np.log1p(b_df[trio]))
        b_probs = get_calibrated_probs(model, b_scaled)
        
        b_df['Prediction'] = [model.classes_[i] for i in np.argmax(b_probs, axis=1)]
        b_df['Confidence'] = np.max(b_probs, axis=1)
        
        st.dataframe(b_df.style.format({'Confidence': '{:.1%}'}).background_gradient(subset=['Confidence'], cmap='Greens'))
        
        for _, r in b_df.iterrows():
            st.session_state.test_log.append({
                "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "Sample": r['id'], 
                "P_Len": r['pollinia_length'], "P_Wid": r['pollinia_width'], "C_Len": r['corpusculum_length'], 
                "Prediction": r['Prediction'], "Confidence": f"{r['Confidence']:.1%}"
            })

# --- TAB 4: TESTING LOG ---
with tabs[3]:
    st.subheader("Historical Session Records")
    if st.session_state.test_log:
        st.table(pd.DataFrame(st.session_state.test_log))
        if st.button("Reset Session History"):
            st.session_state.test_log = []; st.rerun()
    else:
        st.info("No diagnostic data generated in this session yet.")
