import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
import datetime

# --- SETTINGS & STYLE ---
st.set_page_config(page_title="Hoya Morpho-ID", page_icon="🌿", layout="wide")

# Custom CSS for "Artistic" Botanical UI
st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f0;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #2e7d32 !important; color: white !important; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL ENGINE ---
@st.cache_data
def get_data():
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

df = get_data()
model, scaler, trio = train_model(df)

def get_probs(model, scaled_input):
    decision = model.decision_function(scaled_input)
    # FIX: Handle binary vs multiclass decision function shapes
    if decision.ndim == 1: 
        # Binary Case: convert to 2D probabilities
        p = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - p, p])
    else:
        # Multiclass Case: Softmax
        exp_d = np.exp(decision)
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)

# --- SESSION STATE FOR LOGGING ---
if 'test_log' not in st.session_state:
    st.session_state.test_log = []

# --- APP LAYOUT ---
st.title("🌿 Hoya Morpho-ID: Precision Taxonomy")

tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Overview", "🔬 Single Sample", "📊 Batch Processing", "📜 Testing Log"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.header("Project Background")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### Conservation Through Precision
        Developed for the *National Summit on Botanic Gardens and Arboreta*, this application 
        leverages machine learning to solve the taxonomic bottleneck in Philippine *Hoya* collections.
        
        **The "Golden Trio" Approach:**
        Research identified that 31 out of 34 morphological features were redundant. By focusing 
        exclusively on **Pollinia Length**, **Pollinia Width**, and **Corpusculum Length**, we achieved 
        high-fidelity classification without destructive sampling.
        """)
    with col_b:
        st.success("**Model Performance**\n\n- Baseline Accuracy: 71.8%\n- High-Confidence Accuracy: 91.6%\n- Model: Regularized Ridge (L2)")
    
    st.divider()
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Hoya_linearis_flower.jpg/640px-Hoya_linearis_flower.jpg", caption="Hoya species morphology")

# --- TAB 2: SINGLE SAMPLE ---
with tab2:
    st.header("Individual Specimen Diagnostic")
    with st.expander("Instructions", expanded=False):
        st.write("Enter measurements obtained via micrometer. The model will analyze the log-proportions and return a clade prediction.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        s_p_len = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85)
        s_p_wid = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24)
        s_c_len = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.28)
        run_btn = st.button("Run Diagnostic", use_container_width=True)

    if run_btn:
        input_data = np.log1p(np.array([[s_p_len, s_p_wid, s_c_len]]))
        scaled_in = scaler.transform(input_data)
        probs = get_probs(model, scaled_in)
        conf = np.max(probs)
        pred = model.classes_[np.argmax(probs)]
        
        # Log result
        st.session_state.test_log.append({
            "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sample": "Manual Input", "P_Len": s_p_len, "P_Wid": s_p_wid, 
            "C_Len": s_c_len, "Prediction": pred, "Confidence": f"{conf:.2%}"
        })

        with col2:
            if conf >= 0.70:
                st.success(f"**Identified Clade: {pred}**")
                st.metric("Confidence Score", f"{conf:.1%}")
            else:
                st.warning(f"**Ambiguous Morphotype** (Confidence: {conf:.1%})")
            
            # 3D Visual
            temp_df = pd.concat([df, pd.DataFrame({'pollinia_length':[s_p_len],'pollinia_width':[s_p_wid],'corpusculum_length':[s_c_len],'clade':['QUERY']})])
            fig = px.scatter_3d(temp_df, x='pollinia_length', y='pollinia_width', z='corpusculum_length', color='clade', 
                                color_discrete_map={'QUERY':'#ff0000'})
            fig.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: BATCH PROCESSING ---
with tab3:
    st.header("Batch Specimen Analysis")
    st.markdown("Download the template, fill in your measurements, and upload for bulk classification.")
    
    template = pd.DataFrame(columns=['sample_id', 'pollinia_length', 'pollinia_width', 'corpusculum_length'])
    st.download_button("Download CSV Template", template.to_csv(index=False), "template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload filled CSV", type="csv")
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        try:
            # Calculation
            batch_X = np.log1p(batch_df[trio])
            batch_scaled = scaler.transform(batch_X)
            batch_probs = get_probs(model, batch_scaled)
            
            batch_df['Prediction'] = [model.classes_[i] for i in np.argmax(batch_probs, axis=1)]
            batch_df['Confidence'] = np.max(batch_probs, axis=1)
            batch_df['Status'] = batch_df['Confidence'].apply(lambda x: "High" if x >= 0.7 else "Low/Ambiguous")
            
            st.dataframe(batch_df.style.background_gradient(subset=['Confidence'], cmap='Greens'))
            
            # Log bulk
            for _, row in batch_df.iterrows():
                st.session_state.test_log.append({
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Sample": row['sample_id'], "P_Len": row['pollinia_length'], 
                    "P_Wid": row['pollinia_width'], "C_Len": row['corpusculum_length'], 
                    "Prediction": row['Prediction'], "Confidence": f"{row['Confidence']:.2%}"
                })
        except Exception as e:
            st.error(f"Error: Ensure CSV columns match template exactly. Detail: {e}")

# --- TAB 4: TESTING LOG ---
with tab4:
    st.header("Session History")
    if st.session_state.test_log:
        log_df = pd.DataFrame(st.session_state.test_log)
        st.dataframe(log_df, use_container_width=True)
        if st.button("Clear Log"):
            st.session_state.test_log = []
            st.rerun()
    else:
        st.info("No diagnostics run yet this session.")
