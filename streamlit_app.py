
"""
===============================================================================
FlwrAI: Hoya Species Classifier - FIXED UI VERSION
Clean, readable interface with proper contrast
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import plotly.graph_objects as go

st.set_page_config(
    page_title="FlwrAI - Hoya Classifier",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# IMPROVED CSS - CLEAN & READABLE
# ============================================================================

st.markdown("""
    <style>
    /* Force light theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fef5ff 0%, #f0f9ff 50%, #fffbf0 100%);
    }
    
    /* Remove dark theme */
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #d946ef, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #6b21a8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Section containers - WHITE BACKGROUNDS */
    .overview-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
        margin-bottom: 2rem;
        border: 2px solid #e9d5ff;
    }
    
    .input-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
        border: 2px solid #fce7f3;
        min-height: 650px;
    }
    
    .output-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        border: 2px solid #dbeafe;
        min-height: 650px;
    }
    
    /* Section headers - DARK TEXT */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #8b5cf6;
        padding-bottom: 0.5rem;
    }
    
    /* Prediction box - CLEAN */
    .prediction-result {
        background: linear-gradient(135deg, #fae8ff 0%, #fbcfe8 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #d946ef;
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .species-name {
        font-size: 2.8rem;
        font-weight: 800;
        color: #86198f;
        font-style: italic;
        margin: 0.5rem 0;
    }
    
    .confidence-text {
        font-size: 2rem;
        font-weight: 700;
        color: #7c3aed;
        margin: 0.5rem 0;
    }
    
    /* Info boxes - READABLE */
    .info-card {
        background: #dbeafe;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        color: #1e3a8a;
        font-size: 1rem;
    }
    
    .success-card {
        background: #d1fae5;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        color: #065f46;
        font-size: 1rem;
    }
    
    .warning-card {
        background: #fed7aa;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        color: #78350f;
        font-size: 1rem;
    }
    
    /* Metric cards */
    .metric-box {
        background: linear-gradient(135deg, #fae8ff 0%, #e0e7ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #c084fc;
        margin: 0.5rem 0;
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #7c3aed;
        margin: 0;
    }
    
    .metric-text {
        font-size: 1rem;
        font-weight: 600;
        color: #4c1d95;
        margin: 0.3rem 0 0 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, #d946ef, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 2rem;
        border: none;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);
    }
    
    /* Input fields - DARK TEXT */
    .stNumberInput label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stNumberInput input {
        border: 2px solid #e9d5ff !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        color: #1e293b !important;
        background: white !important;
    }
    
    /* Expanders - CLEAN */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #fae8ff 0%, #e0e7ff 100%) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        font-size: 1.1rem !important;
        border: 2px solid #c084fc !important;
    }
    
    /* Tabs - READABLE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #fae8ff;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #1e293b;
        border: 2px solid #e9d5ff;
        font-size: 1.05rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #d946ef, #8b5cf6);
        color: white;
        border-color: #d946ef;
    }
    
    /* Remove extra margins */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Text readability */
    p, li, span {
        color: #1e293b;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5 {
        color: #1e293b;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS (SAME AS BEFORE)
# ============================================================================

@st.cache_resource
def load_model():
    try:
        with gzip.open('hoya_model_compressed.pkl.gz', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found!")
        st.stop()

def engineer_features_single(measurements):
    features = measurements.copy()
    features['pollinia_ratio'] = measurements['pollinia_length'] / (measurements['pollinia_width'] + 1e-6)
    features['corpusculum_ratio'] = measurements['corpusculum_length'] / (measurements['corpusculum_width'] + 1e-6)
    features['corp_to_poll_length'] = measurements['corpusculum_length'] / (measurements['pollinia_length'] + 1e-6)
    features['corp_to_poll_width'] = measurements['corpusculum_width'] / (measurements['pollinia_width'] + 1e-6)
    features['pollinia_area'] = measurements['pollinia_length'] * measurements['pollinia_width']
    features['corpusculum_area'] = measurements['corpusculum_length'] * measurements['corpusculum_width']
    features['shoulder_to_waist'] = measurements['shoulder'] / (measurements['waist'] + 1e-6)
    features['waist_to_hips'] = measurements['waist'] / (measurements['hips'] + 1e-6)
    features['translator_total'] = measurements['translator_arm_length'] + measurements['translator_stalk']
    features['translator_ratio'] = measurements['translator_arm_length'] / (measurements['translator_stalk'] + 1e-6)
    features['extension_to_length'] = measurements['extension'] / (measurements['pollinia_length'] + 1e-6)
    features['pollinia_length_sq'] = measurements['pollinia_length'] ** 2
    features['corpusculum_length_sq'] = measurements['corpusculum_length'] ** 2
    features['pollinia_area_sq'] = features['pollinia_area'] ** 2
    features['poll_corp_interaction'] = measurements['pollinia_length'] * measurements['corpusculum_length']
    features['body_shape_index'] = (measurements['shoulder'] + measurements['hips']) / (measurements['waist'] + 1e-6)
    features['translator_complexity'] = features['translator_ratio'] * features['translator_total']
    features['pollinia_volume_proxy'] = features['pollinia_area'] * measurements['pollinia_width']
    features['corpusculum_volume_proxy'] = features['corpusculum_area'] * measurements['corpusculum_width']
    features['overall_size_index'] = np.log1p(features['pollinia_area'] + features['corpusculum_area'])
    features['size_variance'] = features['pollinia_area'] / (features['corpusculum_area'] + 1e-6)
    features['total_length_to_width'] = (measurements['pollinia_length'] + measurements['corpusculum_length']) / (measurements['pollinia_width'] + measurements['corpusculum_width'] + 1e-6)
    features['body_symmetry'] = abs(measurements['shoulder'] - measurements['hips']) / (measurements['waist'] + 1e-6)
    features['morphological_index'] = features['pollinia_ratio'] * features['corpusculum_ratio'] * features['translator_ratio']
    return features

def predict_species(measurements, model_package):
    features_dict = engineer_features_single(measurements)
    df_features = pd.DataFrame([features_dict])
    X_input = df_features[model_package['feature_names']].values
    X_scaled = model_package['scaler'].transform(X_input)
    prediction_encoded = model_package['model'].predict(X_scaled)[0]
    probabilities = model_package['model'].predict_proba(X_scaled)[0]
    predicted_species = model_package['label_encoder'].inverse_transform([prediction_encoded])[0]
    species_names = model_package['label_encoder'].classes_
    species_probs = list(zip(species_names, probabilities))
    species_probs.sort(key=lambda x: x[1], reverse=True)
    return predicted_species, species_probs, probabilities

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence", 'font': {'size': 22, 'color': '#1e293b'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#7c3aed'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#64748b'},
            'bar': {'color': '#d946ef'},
            'bgcolor': '#f8fafc',
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b'}
    )
    return fig

def create_probability_chart(species_probs, top_n=8):
    top_species = species_probs[:top_n]
    species_names = [f"Hoya {sp[0]}" for sp in top_species]
    probabilities = [sp[1] * 100 for sp in top_species]
    colors = ['#d946ef', '#c026d3', '#a21caf', '#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95']
    
    fig = go.Figure(go.Bar(
        y=species_names[::-1],
        x=probabilities[::-1],
        orientation='h',
        marker=dict(color=colors[:len(species_names)][::-1]),
        text=[f"{p:.1f}%" for p in probabilities[::-1]],
        textposition='outside',
        textfont=dict(size=13, color='#1e293b', weight='bold')
    ))
    
    fig.update_layout(
        title=dict(text="Top Candidate Species", font=dict(size=20, color='#1e293b', weight='bold')),
        xaxis_title="Confidence (%)",
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], gridcolor='#e2e8f0', title_font=dict(color='#1e293b')),
        yaxis=dict(title_font=dict(color='#1e293b')),
        font=dict(size=13, color='#1e293b')
    )
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">🌺 FlwrAI: Hoya Species Classifier 🌸</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Species Identification from Pollinaria Morphometrics</p>', unsafe_allow_html=True)
    
    model_package = load_model()
    
    # ========================================================================
    # TOP: OVERVIEW SECTION
    # ========================================================================
    
    st.markdown('<div class="overview-box">', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 About Tool", "🧠 Training", "📊 Performance", "⚠️ Usage & Limits"])
    
    with tab1:
        col1, col2, col3 = st.columns([2.5, 1, 1])
        
        with col1:
            st.markdown("### What is FlwrAI?")
            st.markdown("""
                **FlwrAI** uses advanced machine learning to identify *Hoya* species from pollinaria measurements.
                
                **How it works:**
                1. 📏 Measure pollinaria under microscope
                2. 📝 Input 10 measurements (mm)
                3. 🤖 AI analyzes with 34 features
                4. 🎯 Get species prediction + confidence
                
                **Features:**
                - ⚡ Instant identification
                - 🎯 49 species coverage
                - 📊 Confidence scoring
                - 🌐 No installation needed
                """)
        
        with col2:
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["n_species"]}</p><p class="metric-text">Species</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["n_features"]}</p><p class="metric-text">Features</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["accuracy"]*100:.0f}%</p><p class="metric-text">Accuracy</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-box"><p class="metric-number">64</p><p class="metric-text">Samples</p></div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Dataset 📚")
            st.markdown("""
                - **Source:** PNRI Philippines
                - **Specimens:** 64 pollinaria
                - **Species:** 49 Hoya species
                - **Features:** 10 raw + 24 engineered
                
                ### Processing 🔧
                - Feature engineering
                - Robust scaling
                - SMOTE balancing
                - Stratified splitting
                """)
        
        with col2:
            st.markdown("### Models 🤖")
            st.markdown(f"""
                - 🌲 Random Forest
                - 🚀 XGBoost
                - ⚡ LightGBM
                - 🐱 CatBoost
                - 🧠 Neural Network
                - 🗳️ Voting Ensembles
                
                **Best:** {model_package['model_name'].replace('_', ' ').title()}
                """)
    
    with tab3:
        if model_package.get('results') is not None:
            st.markdown("### 🏆 Top Models")
            results_df = model_package['results'].nlargest(5, 'accuracy')[['model', 'accuracy']]
            results_df['model'] = results_df['model'].apply(lambda x: x.replace('_', ' ').title())
            results_df['accuracy'] = results_df['accuracy'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(results_df, hide_index=True, use_container_width=True)
        else:
            st.info("Performance metrics available in training notebook")
    
    with tab4:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 📏 How to Use")
            st.markdown("""
                1. Measure pollinaria under microscope
                2. Input 10 measurements (right panel →)
                3. Click "IDENTIFY SPECIES"
                4. View results (left panel ←)
                """)
        
        with col2:
            st.markdown('<div class="warning-card"><b>⚠️ LIMITATIONS</b><br><br>• Research prototype only<br>• Small dataset (64 samples)<br>• NOT for commercial use<br>• Expert verification required<br>• Philippines-focused</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # BOTTOM: INPUT (RIGHT) & OUTPUT (LEFT)
    # ========================================================================
    
    col_left, col_right = st.columns([1.4, 1])
    
    # RIGHT: INPUT
    with col_right:
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📝 Manual Input</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card"><b>📏 All values in millimeters (mm)</b></div>', unsafe_allow_html=True)
        
        measurements = {}
        
        with st.expander("🌸 Pollinia", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                measurements['pollinia_length'] = st.number_input("Length", 0.0, 10.0, 0.5, 0.01, key='pl')
            with c2:
                measurements['pollinia_width'] = st.number_input("Width", 0.0, 10.0, 0.2, 0.01, key='pw')
        
        with st.expander("🔬 Corpusculum", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                measurements['corpusculum_length'] = st.number_input("Length", 0.0, 10.0, 0.3, 0.01, key='cl')
            with c2:
                measurements['corpusculum_width'] = st.number_input("Width", 0.0, 10.0, 0.15, 0.01, key='cw')
        
        with st.expander("⚖️ Body", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                measurements['shoulder'] = st.number_input("Shoulder", 0.0, 5.0, 0.1, 0.01)
            with c2:
                measurements['waist'] = st.number_input("Waist", 0.0, 5.0, 0.08, 0.01)
            with c3:
                measurements['hips'] = st.number_input("Hips", 0.0, 5.0, 0.05, 0.01)
        
        with st.expander("🔗 Other", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                measurements['extension'] = st.number_input("Extension", 0.0, 5.0, 0.15, 0.01)
            with c2:
                measurements['translator_arm_length'] = st.number_input("Trans Arm", 0.0, 5.0, 0.05, 0.01)
            with c3:
                measurements['translator_stalk'] = st.number_input("Trans Stalk", 0.0, 5.0, 0.08, 0.01)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🌺 IDENTIFY SPECIES 🌺", type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LEFT: OUTPUT
    with col_left:
        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🎯 Analysis Report</h2>', unsafe_allow_html=True)
        
        if predict_button:
            predicted_species, species_probs, all_probs = predict_species(measurements, model_package)
            st.session_state['prediction'] = predicted_species
            st.session_state['species_probs'] = species_probs
        
        if 'prediction' in st.session_state:
            predicted_species = st.session_state['prediction']
            species_probs = st.session_state['species_probs']
            confidence = species_probs[0][1]
            
            st.markdown(f'<div class="prediction-result"><p style="margin:0; color:#64748b; font-size:1.1rem;">PREDICTED SPECIES</p><p class="species-name">Hoya {predicted_species}</p><p class="confidence-text">{confidence*100:.1f}% Confidence</p></div>', unsafe_allow_html=True)
            
            if confidence >= 0.75:
                st.markdown('<div class="success-card"><b>✅ HIGH CONFIDENCE</b> - Strong match to species profile</div>', unsafe_allow_html=True)
            elif confidence >= 0.50:
                st.markdown('<div class="warning-card"><b>⚠️ MODERATE CONFIDENCE</b> - Review alternatives below</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card"><b>⚠️ LOW CONFIDENCE</b> - Consult expert botanist</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            with c2:
                st.markdown(f"""
                    ### Quick Stats
                    **Top 3 Candidates:**
                    1. Hoya {species_probs[0][0]} ({species_probs[0][1]*100:.1f}%)
                    2. Hoya {species_probs[1][0]} ({species_probs[1][1]*100:.1f}%)
                    3. Hoya {species_probs[2][0]} ({species_probs[2][1]*100:.1f}%)
                    """)
            
            st.plotly_chart(create_probability_chart(species_probs, 8), use_container_width=True)
            
            with st.expander("📋 All Probabilities"):
                prob_df = pd.DataFrame(species_probs, columns=['Species', 'Probability'])
                prob_df['Probability'] = (prob_df['Probability'] * 100).apply(lambda x: f"{x:.2f}%")
                prob_df['Species'] = 'Hoya ' + prob_df['Species']
                prob_df.insert(0, 'Rank', range(1, len(prob_df) + 1))
                st.dataframe(prob_df, hide_index=True, height=350)
        else:
            st.markdown('<div class="info-card" style="text-align:center; padding:3rem;"><h3>👉 Enter measurements</h3><p style="font-size:1.1rem;">Input pollinaria measurements in the right panel and click "IDENTIFY SPECIES"</p><p style="font-size:2.5rem; margin-top:1rem;">🌺 🔬 ✨</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div style="text-align:center; color:#64748b; padding:2rem; font-size:0.95rem;">Made with 💜 | v2.0 | Research & Education Only | 🌺 🌸 🌼 🌻 🌷</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
