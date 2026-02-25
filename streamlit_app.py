import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
import datetime

# --- PRE-CONFIG & ELEGANT CUSTOM STYLING ---
st.set_page_config(page_title="Hoya Morpho-ID", page_icon="🌸", layout="wide")

# Custom CSS for an "Effeminate Botanical" Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400&display=swap');

    /* Main Background - Soft Linen/Cream */
    .stApp { background-color: #FAFAF8; color: #4A5D4E; font-family: 'Lato', sans-serif; }
    
    /* Headers - Elegant Serif */
    h1, h2, h3 { font-family: 'Playfair Display', serif; color: #2E4D32; font-style: italic; }
    
    /* Elegant Tab Styling - Soft Sage & White */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 15px; background-color: #E8EDE7; padding: 15px 15px 0px 15px; 
        border-radius: 20px 20px 0 0; 
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 15px 15px 0 0; padding: 0 30px;
        font-weight: 400; color: #7A8B7C; border: none; font-size: 16px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #FFFFFF !important; color: #2E4D32 !important; 
        border-bottom: 4px solid #A3B18A !important; font-weight: 700;
    }
    
    /* Fancy Cards for Results */
    .result-card {
        background: white; border-radius: 25px; padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #E9EDE9;
        margin-bottom: 20px;
    }
    
    /* Buttons - Soft Forest Green */
    .stButton>button { 
        border-radius: 30px; background-color: #588157; color: white; 
        border: none; padding: 12px 40px; font-weight: 300; letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #3A5A40; color: white; transform: translateY(-2px); }
    
    /* Log Table Legibility Fix */
    [data-testid="stTable"] { color: #2E4D32 !important; background-color: white; border-radius: 15px; }
    th { background-color: #A3B18A !important; color: white !important; font-family: 'Lato', sans-serif; }
    
    /* Input Fields */
    input { border-radius: 15px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATASET ENGINE ---
@st.cache_data
def get_hoya_data():
    data = {
        'species': ['lazaroi', 'laut', 'bulacanensis', 'dalanesiae', 'cumingiana', 'diversifolia', 'ardamosana', 'brevialata', 'multiflora', 'monetteae', 'angustifolia', 'citrina', 'linea', 'blanca', 'linea', 'ciliatifolia', 'castillione', 'laoagensis', 'brittonii', 'pubicalyx', 'buotii', 'vitellinoides', 'densifolia', 'camphorifolia', 'camphorifolia', 'cutilipensis', 'kentiana', 'bangbangensis', 'wayetii', 'wayetii', 'cutispicellana', 'salacae', 'tricolor', 'bicolensis', 'obscura', 'obscura', 'obscura', 'obscura', 'flagellata', 'camphorifolia', 'samoensis', 'pottsii', 'cerata', 'cerata', 'cerata', 'albiflora', 'stagensis', 'obscura', 'golamcoana', 'odorata', 'benguetensis', 'benguetensis', 'malae', 'ruthiae', 'benguetensis', 'litoralis', 'mcgregorii', 'bensianii', 'chloroleuca', 'biakensis', 'pottsii', 'benguetensis', 'benguetensis', 'edoroana'],
        'pollinia_length': [0.56, 1.05, 0.51, 0.59, 0.85, 3.8, 5.6, 0.34, 5, 7.8, 4.5, 4.2, 4.3, 0.36, 2.7, 3, 0.26, 0.37, 0.7, 0.6, 0.69, 0.6, 0.82, 0.31, 0.4, 0.36, 0.48, 0.42, 0.39, 0.39, 0.35, 0.29, 0.36, 0.24, 0.35, 0.32, 0.35, 0.31, 0.41, 0.46, 0.48, 0.88, 0.83, 0.94, 0.51, 0.87, 0.9, 0.75, 0.72, 0.7, 0.57, 0.45, 0.85, 1.05, 0.39, 0.54, 0.46, 0.93, 0.46, 0.46, 0.36, 0.55, 0.54, 0.66],
        'pollinia_width': [0.3, 0.5, 0.2, 0.24, 0.24, 1.4, 2.4, 0.17, 2.9, 2, 1.8, 1.8, 1.4, 0.15, 1.1, 1.3, 0.11, 0.15, 0.28, 0.24, 0.24, 0.24, 0.2, 0.13, 0.16, 0.18, 0.18, 0.17, 0.17, 0.16, 0.17, 0.12, 0.15, 0.11, 0.15, 0.14, 0.15, 0.12, 0.16, 0.22, 0.21, 0.25, 0.25, 0.28, 0.2, 0.25, 0.23, 0.3, 0.28, 0.25, 0.23, 0.2, 0.27, 0.28, 0.16, 0.2, 0.17, 0.19, 0.17, 0.17, 0.17, 0.22, 0.26, 0.26],
        'corpusculum_length': [0.61, 0.96, 0.12, 0.22, 0.28, 1.9, 2.6, 0.15, 4.6, 2.5, 2.3, 1.9, 1.6, 0.1, 1.2, 1.1, 0.07, 0.11, 0.3, 0.38, 0.39, 0.26, 0.23, 0.1, 0.16, 0.14, 0.14, 0.14, 0.17, 0.16, 0.22, 0.09, 0.1, 0.09, 0.12, 0.12, 0.16, 0.13, 0.16, 0.24, 0.2, 0.42, 0.42, 0.45, 0.19, 0.33, 0.4, 0.35, 0.43, 0.4, 0.18, 0.16, 0.4, 0.47, 0.17, 0.23, 0.16, 0.11, 0.19, 0.2, 0.23, 0.18, 0.18, 0.43],
        'clade': ['Acanthostemma', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex']
    }
    return pd.DataFrame(data)

def train_model(df):
    trio = ['pollinia_length', 'pollinia_width', 'corpusculum_length']
    X = np.log1p(df[trio])
    y = df['clade']
    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)
    return model, scaler, trio

def get_probs(model, scaled_input):
    decision = model.decision_function(scaled_input)
    if decision.ndim == 1: 
        p = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - p, p])
    exp_d = np.exp(decision)
    return exp_d / np.sum(exp_d, axis=1, keepdims=True)

df = get_hoya_data()
model, scaler, trio = train_model(df)

if 'test_log' not in st.session_state: st.session_state.test_log = []

# --- APP LAYOUT ---
st.write(f"<h1 style='text-align: center; margin-bottom: 30px;'>🌿 Hoya Morpho-ID: Precision Taxonomy</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🌸 Research Philosophy", "🔬 Diagnostic Studio", "📊 Batch Processing", "📜 Historical Registry"])

# --- TAB 1: SUBSTANTIAL & ELEGANT OVERVIEW ---
with tab1:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.write("## The Art of Botanical Precision")
        st.markdown("""
        In the delicate world of *Hoya* taxonomy, identification is often a dense thicket of 
        morphological overlap. This project seeks to harmonize **Biological Geometry** with 
        **Artificial Intelligence** to preserve the heritage of Philippine flora.
        
        ### The "Golden Trio" Revelation
        Traditional taxonomic audits rely on 34 distinct micrometric variables—a process that 
        is both labor-intensive and prone to subjectivity. Our research, powered by **Recursive 
        Feature Elimination (RFE)**, has distilled this complexity into a **Golden Trio** of master 
        traits that serve as the primary evolutionary signatures for the genus:
        
        * **Pollinia Length:** The vertical architecture of the pollen mass.         * **Pollinia Width:** The lateral breadth, defining the geometric clade-signature.
        * **Corpusculum Length:** The anchor point that dictates structural stability.
        
        ### Scientific Integrity
        By utilizing a **Regularized Ridge Classifier**, we achieve a high-fidelity prediction 
        accuracy of **91.6%**. This provides Botanic Garden curators and Arboreta managers with 
        a nature-based technical solution to catalogue diversity with unprecedented speed.
        """)
    with col_r:
        st.write("### Reference Morphospace")
        fig_ref = px.scatter_3d(df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', 
                                color='clade', color_discrete_sequence=px.colors.qualitative.Antique,
                                template="plotly_white", opacity=0.7)
        fig_ref.update_layout(height=500, margin=dict(l=0,r=0,b=0,t=0), scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)))
        st.plotly_chart(fig_ref, use_container_width=True)

# --- TAB 2: FANCY DIAGNOSTIC STUDIO ---
with tab2:
    st.write("## Specimen Identification Studio")
    col_in, col_spacer, col_out = st.columns([1.5, 0.2, 3])
    
    with col_in:
        st.write("### Specimen Metrics")
        s_p_len = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85)
        s_p_wid = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24)
        s_c_len = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.28)
        
        if st.button("Unveil Identity", use_container_width=True):
            scaled_in = scaler.transform(np.log1p([[s_p_len, s_p_wid, s_c_len]]))
            probs = get_probs(model, scaled_in)
            conf, pred = np.max(probs), model.classes_[np.argmax(probs)]
            
            st.session_state.single_res = {"pred": pred, "conf": conf, "coords": [s_p_len, s_p_wid, s_c_len]}
            st.session_state.test_log.append({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Sample": "Manual Input", "P-Length": s_p_len, "P-Width": s_p_wid, 
                "C-Length": s_c_len, "Clade Prediction": pred, "Reliability": f"{conf:.1%}"
            })

    with col_out:
        if 'single_res' in st.session_state:
            res = st.session_state.single_res
            st.write(f"<div class='result-card'>", unsafe_allow_html=True)
            
            st.write("### Classification Discovery")
            m1, m2 = st.columns(2)
            m1.metric("Clade Identity", res['pred'])
            m2.metric("Reliability Index", f"{res['conf']:.1%}")
            
            # Fancy Gauge
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number", value = res['conf']*100,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#588157"}, 
                         'steps': [{'range': [0, 70], 'color': "#F4F4F2"}, {'range': [70, 100], 'color': "#DCE4DC"}]},
                title = {'text': "Confidence Index (%)", 'font': {'size': 18, 'family': 'Lato'}}))
            fig_g.update_layout(height=280, margin=dict(l=30,r=30,b=20,t=40), paper_bgcolor="white")
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.write("#### Specimen Orientation")
            user_df = pd.concat([df, pd.DataFrame({'pollinia_length':[res['coords'][0]],'pollinia_width':[res['coords'][1]],'corpusculum_length':[res['coords'][2]],'clade':['NEW SPECIMEN']})])
            fig_v = px.scatter_3d(user_df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', 
                                  color='clade', color_discrete_map={'NEW SPECIMEN':'#A6172D'}, size_max=12, template="plotly_white")
            fig_v.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_v, use_container_width=True)
            st.write(f"</div>", unsafe_allow_html=True)
        else:
            st.info("Awaiting specimen data for identification...")

# --- TAB 3: BATCH PROCESSING ---
with tab3:
    st.write("## Batch Repository Analysis")
    st.markdown("For large collections, upload your curated dataset in CSV format.")
    
    col_up, col_guide = st.columns([2, 1])
    with col_guide:
        st.write("### 📜 Format Guide")
        st.markdown("""
        1. **id**: Sample name
        2. **pollinia_length**: (mm)
        3. **pollinia_width**: (mm)
        4. **corpusculum_length**: (mm)
        """)
        
    with col_up:
        up_file = st.file_uploader("Upload Accession Data", type="csv")
        if up_file:
            b_df = pd.read_csv(up_file)
            b_scaled = scaler.transform(np.log1p(b_df[trio]))
            b_probs = get_probs(model, b_scaled)
            
            b_df['Clade Prediction'] = [model.classes_[i] for i in np.argmax(b_probs, axis=1)]
            b_df['Reliability'] = np.max(b_probs, axis=1)
            
            st.write("### Processed Accessions")
            st.dataframe(b_df.style.format({'Reliability': '{:.1%}'}).background_gradient(subset=['Reliability'], cmap='YlGn'))
            
            for _, r in b_df.iterrows():
                st.session_state.test_log.append({
                    "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Sample": r['id'], "P-Length": r['pollinia_length'], "P-Width": r['pollinia_width'], 
                    "C-Length": r['corpusculum_length'], "Clade Prediction": r['Clade Prediction'], "Reliability": f"{r['Reliability']:.1%}"
                })

# --- TAB 4: HISTORICAL REGISTRY (Legibility Fixed) ---
with tab4:
    st.write("## Session Historical Registry")
    if st.session_state.test_log:
        registry_df = pd.DataFrame(st.session_state.test_log)
        st.table(registry_df)
        
        if st.button("Purge Registry History"):
            st.session_state.test_log = []; st.rerun()
    else:
        st.info("The registry is currently empty.")
