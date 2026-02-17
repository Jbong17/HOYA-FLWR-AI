
"""
===============================================================================
STREAMLIT WEB APPLICATION
AI-Assisted Hoya Species Classification using Pollinaria Measurement
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="FlowrAI - Hoya Classifier",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - FLOWER & COLORFUL THEME
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Quicksand:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #ffeef8 0%, #e7f5ff 50%, #fff4e6 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-family: 'Quicksand', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #ff6b9d, #c768dd, #8e54e9, #4e54c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 1rem 0;
        animation: fadeInDown 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #8e54e9;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .emoji-decoration {
        font-size: 2.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .overview-section {
        background: linear-gradient(135deg, #ffffff 0%, #fef6ff 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(142, 84, 233, 0.15);
        margin-bottom: 2rem;
        border: 2px solid rgba(255, 107, 157, 0.2);
    }
    
    .input-panel {
        background: linear-gradient(135deg, #fff4f0 0%, #fffbf5 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(255, 107, 157, 0.15);
        border: 2px solid rgba(255, 184, 184, 0.3);
        min-height: 700px;
    }
    
    .output-panel {
        background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(78, 84, 200, 0.15);
        border: 2px solid rgba(142, 84, 233, 0.3);
        min-height: 700px;
    }
    
    .section-title {
        font-family: 'Quicksand', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #8e54e9;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #ff6b9d;
        padding-bottom: 0.5rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ffeef8 0%, #fff0f6 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #ff6b9d;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(255, 107, 157, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .species-name {
        font-size: 2.5rem;
        font-weight: 700;
        color: #c768dd;
        font-style: italic;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-score {
        font-size: 2rem;
        font-weight: 600;
        color: #8e54e9;
        text-align: center;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e7f5ff 0%, #f0f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4e54c8;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff4e6 0%, #fff9f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #ffa94d;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e7f9f5 0%, #f0fff4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #51cf66;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #ff6b9d, #c768dd, #8e54e9);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 30px;
        box-shadow: 0 6px 12px rgba(142, 84, 233, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(142, 84, 233, 0.4);
    }
    
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #ffc9d9;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #fff0f6 0%, #fef6ff 100%);
        border-radius: 12px;
        font-weight: 600;
        color: #8e54e9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid #ffc9d9;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #c768dd;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8e54e9;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #fff0f6 0%, #fef6ff 100%);
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        color: #8e54e9;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #ff6b9d, #c768dd);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    try:
        with open('hoya_advanced_model.pkl', 'rb') as f:
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
        title={'text': "Confidence", 'font': {'size': 20, 'color': '#8e54e9'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#c768dd'}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#ff6b9d"},
            'steps': [
                {'range': [0, 50], 'color': '#ffe0e6'},
                {'range': [50, 75], 'color': '#ffc9d9'},
                {'range': [75, 100], 'color': '#ffb3c6'}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_probability_chart(species_probs, top_n=8):
    top_species = species_probs[:top_n]
    species_names = [f"Hoya {sp[0]}" for sp in top_species]
    probabilities = [sp[1] * 100 for sp in top_species]
    colors = ['#ff6b9d', '#ff8bb3', '#ffabc9', '#c768dd', '#d88ae7', '#8e54e9', '#9d6eea', '#ac87eb']
    
    fig = go.Figure(go.Bar(
        y=species_names[::-1],
        x=probabilities[::-1],
        orientation='h',
        marker=dict(color=colors[:len(species_names)][::-1]),
        text=[f"{p:.1f}%" for p in probabilities[::-1]],
        textposition='auto',
        textfont=dict(size=12, color='white')
    ))
    
    fig.update_layout(
        title="Top Candidate Species",
        xaxis_title="Confidence (%)",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100])
    )
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("<div style='text-align: center;'><span class='emoji-decoration'>🌺</span> <span class='emoji-decoration'>🌸</span> <span class='emoji-decoration'>🌼</span> <span class='emoji-decoration'>🌻</span> <span class='emoji-decoration'>🌷</span></div>", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🌺 FlwrAI: Hoya Species Classifier 🌸</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Species Identification from Pollinaria Morphometrics ✨</p>', unsafe_allow_html=True)
    
    model_package = load_model()
    
    # ========================================================================
    # TOP HORIZONTAL HALF: OVERVIEW SECTION
    # ========================================================================
    
    st.markdown('<div class="overview-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📖 Overview & Documentation</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌟 About Tool", "🧠 Training Background", "📊 Model Performance", "⚠️ How to Use & Limitations"])
    
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
                ### What is FlowrAI? 🤔
                
                **FlowrAI** uses advanced machine learning to identify *Hoya* species from pollinaria measurements. 
                Simply input 10 microscopic measurements and get instant species predictions with confidence scores!
                
                **Key Features:**
                - ⚡ Instant AI-powered identification
                - 🎯 Trained on 49 Philippine Hoya species  
                - 📈 Confidence scoring with uncertainty quantification
                - 🌐 No installation required - works in your browser
                """)
        
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model_package["n_species"]}</div><div class="metric-label">Species Covered</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model_package["n_features"]}</div><div class="metric-label">AI Features</div></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model_package["accuracy"]*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">64</div><div class="metric-label">Training Samples</div></div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                ### Dataset 📚
                - **Source:** Philippine National Research Institute (PNRI)
                - **Specimens:** 64 pollinaria samples
                - **Species:** 49 Philippine Hoya species
                - **Raw Features:** 10 morphometric measurements
                - **Engineered Features:** 24 additional AI-derived features
                - **Total Features:** 34 for classification
                
                ### Data Processing 🔧
                - Advanced feature engineering (polynomial & interaction terms)
                - Robust scaling (outlier-resistant normalization)
                - SMOTE for class balance (where applicable)
                - Stratified train/test splitting
                """)
        
        with col2:
            st.markdown("""
                ### Models Trained 🤖
                
                Our ensemble includes:
                - 🌲 **Random Forest** - 300 trees
                - 🚀 **XGBoost** - Gradient boosting
                - ⚡ **LightGBM** - Fast boosting
                - 🐱 **CatBoost** - Categorical boosting  
                - 🧠 **Neural Network** - Deep learning
                - 🗳️ **Voting Ensembles** - Soft & hard
                - 📦 **Bagging** - Bootstrap aggregating
                
                **Best Model:** {model_name}
                
                ### Validation ✅
                - 5-fold cross-validation
                - Probability calibration
                - Uncertainty quantification
                """.format(model_name=model_package['model_name'].replace('_', ' ').title()))
    
    with tab3:
        if model_package.get('results') is not None and not model_package['results'].empty:
            results_df = model_package['results'].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Top 5 Models by Accuracy")
                top_5 = results_df.nlargest(5, 'accuracy')[['model', 'accuracy', 'mean_confidence']]
                top_5['model'] = top_5['model'].apply(lambda x: x.replace('_', ' ').title())
                top_5['accuracy'] = top_5['accuracy'].apply(lambda x: f"{x*100:.2f}%")
                top_5['mean_confidence'] = top_5['mean_confidence'].apply(lambda x: f"{x:.3f}")
                top_5.columns = ['Model', 'Test Accuracy', 'Avg Confidence']
                st.dataframe(top_5, hide_index=True, use_container_width=True)
                
                st.markdown("""
                    **Performance Metrics Explained:**
                    - **Accuracy:** % of correct predictions
                    - **Confidence:** Model certainty (0-1)
                    - **Entropy:** Uncertainty (lower = better)
                    """)
            
            with col2:
                st.markdown("#### 📈 Champion Model Statistics")
                best_stats = results_df[results_df['model'] == model_package['model_name']].iloc[0]
                
                st.markdown(f"""
                    <div class="success-box">
                    <b>Selected Model:</b> {model_package['model_name'].replace('_', ' ').title()}<br><br>
                    <b>Test Accuracy:</b> {best_stats['accuracy']*100:.2f}%<br>
                    <b>Mean Confidence:</b> {best_stats['mean_confidence']:.4f}<br>
                    <b>Mean Entropy:</b> {best_stats['mean_entropy']:.4f}<br><br>
                    <small>Selected via cross-validation and calibration analysis</small>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📊 Model Visualization")
                st.info("Radar plots and performance charts were generated during training. See training notebook for full visualizations.")
        else:
            st.warning("Detailed performance metrics not available in current model package.")
    
    with tab4:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
                ### 📏 How to Use This Tool
                
                **Step 1: Prepare Your Sample**
                - Extract pollinarium carefully from flower
                - Mount on microscope slide
                - Use calibrated microscope with measurement capability
                
                **Step 2: Take Measurements**
                All measurements in millimeters (mm):
                1. **Pollinia:** Length and width of pollen masses
                2. **Corpusculum:** Length and width of central structure
                3. **Body:** Shoulder, waist, and hips measurements
                4. **Translator:** Arm and stalk lengths
                5. **Extension:** Extended portion length
                
                **Step 3: Input & Analyze**
                - Enter measurements in RIGHT panel →
                - Click "IDENTIFY SPECIES" button
                - View results in LEFT panel ←
                
                **Step 4: Interpret Results**
                - Check confidence score (aim for >75%)
                - Review alternative species
                - Download report for records
                """)
        
        with col2:
            st.markdown("""
                <div class="warning-box">
                <h4>⚠️ CRITICAL LIMITATIONS</h4>
                
                <b>This is a PROTOTYPE:</b><br>
                ❌ NOT for commercial use<br>
                ❌ NOT a replacement for experts<br>
                ❌ NOT 100% accurate<br><br>
                
                <b>Data Constraints:</b><br>
                • Small dataset (64 specimens)<br>
                • Some species: only 1 sample<br>
                • Geographic bias (Philippines)<br>
                • Many Hoya species not included<br><br>
                
                <b>Requirements:</b><br>
                • Calibrated microscope needed<br>
                • Fresh samples preferred<br>
                • Proper extraction technique<br>
                • Measurement precision critical<br><br>
                
                <b>✅ USE FOR:</b><br>
                • Preliminary screening<br>
                • Educational purposes<br>
                • Research exploration<br><br>
                
                <b>⚠️ ALWAYS:</b><br>
                • Verify with expert botanists<br>
                • Check morphological keys<br>
                • Consider multiple candidates<br>
                • Use responsible judgment
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # BOTTOM HALF: VERTICAL SPLIT - LEFT (OUTPUT) | RIGHT (INPUT)
    # ========================================================================
    
    col_left, col_right = st.columns([1.3, 1])
    
    # RIGHT PANEL: INPUT SECTION
    with col_right:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">📝 Manual Input</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box"><b>📏 All measurements in millimeters (mm)</b><br>Use calibrated microscope | Precision: 0.01 mm</div>', unsafe_allow_html=True)
        
        measurements = {}
        
        with st.expander("🌸 Pollinia Measurements", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                measurements['pollinia_length'] = st.number_input("Length (mm)", 0.0, 10.0, 0.5, 0.01, key='pl')
            with col2:
                measurements['pollinia_width'] = st.number_input("Width (mm)", 0.0, 10.0, 0.2, 0.01, key='pw')
        
        with st.expander("🔬 Corpusculum Measurements", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                measurements['corpusculum_length'] = st.number_input("Length (mm)", 0.0, 10.0, 0.3, 0.01, key='cl')
            with col2:
                measurements['corpusculum_width'] = st.number_input("Width (mm)", 0.0, 10.0, 0.15, 0.01, key='cw')
        
        with st.expander("⚖️ Body Architecture", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                measurements['shoulder'] = st.number_input("Shoulder", 0.0, 5.0, 0.1, 0.01)
            with col2:
                measurements['waist'] = st.number_input("Waist", 0.0, 5.0, 0.08, 0.01)
            with col3:
                measurements['hips'] = st.number_input("Hips", 0.0, 5.0, 0.05, 0.01)
        
        with st.expander("🔗 Translator & Extension", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                measurements['extension'] = st.number_input("Extension", 0.0, 5.0, 0.15, 0.01)
            with col2:
                measurements['translator_arm_length'] = st.number_input("Trans. Arm", 0.0, 5.0, 0.05, 0.01)
            with col3:
                measurements['translator_stalk'] = st.number_input("Trans. Stalk", 0.0, 5.0, 0.08, 0.01)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🌺 IDENTIFY SPECIES 🌺", type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LEFT PANEL: OUTPUT SECTION
    with col_left:
        st.markdown('<div class="output-panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">🎯 Analysis Report & Visualization</h2>', unsafe_allow_html=True)
        
        if predict_button:
            with st.spinner("🔮 AI analyzing pollinaria morphometrics..."):
                predicted_species, species_probs, all_probs = predict_species(measurements, model_package)
            st.session_state['prediction'] = predicted_species
            st.session_state['species_probs'] = species_probs
            st.session_state['measurements'] = measurements
        
        if 'prediction' in st.session_state:
            predicted_species = st.session_state['prediction']
            species_probs = st.session_state['species_probs']
            confidence = species_probs[0][1]
            
            # Main prediction
            st.markdown(f'<div class="prediction-box"><div style="text-align: center;"><p style="font-size: 1.1rem; color: #8e54e9; margin: 0;">🌺 PREDICTED SPECIES 🌺</p><p class="species-name">Hoya {predicted_species}</p><p class="confidence-score">{confidence*100:.1f}% Confidence</p></div></div>', unsafe_allow_html=True)
            
            # Confidence interpretation
            if confidence >= 0.75:
                st.markdown('<div class="success-box"><b>✅ HIGH CONFIDENCE</b><br>Strong match to species profile. Measurements align well with trained data.</div>', unsafe_allow_html=True)
            elif confidence >= 0.50:
                st.markdown('<div class="warning-box"><b>⚠️ MODERATE CONFIDENCE</b><br>Likely correct but verify with alternatives below. Expert confirmation recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box"><b>⚠️ LOW CONFIDENCE</b><br>High uncertainty. Review multiple candidates and consult expert botanist.</div>', unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            with col2:
                st.markdown(f"""
                    ### Quick Stats
                    - **Rank 1:** Hoya {species_probs[0][0]} ({species_probs[0][1]*100:.1f}%)
                    - **Rank 2:** Hoya {species_probs[1][0]} ({species_probs[1][1]*100:.1f}%)
                    - **Rank 3:** Hoya {species_probs[2][0]} ({species_probs[2][1]*100:.1f}%)
                    
                    **Recommendation:**  
                    {('Proceed with identification' if confidence >= 0.75 else 'Verify with expert' if confidence >= 0.50 else 'Consult expert botanist')}
                    """)
            
            st.plotly_chart(create_probability_chart(species_probs, 8), use_container_width=True)
            
            # Full probability table
            with st.expander("📋 View All Species Probabilities"):
                prob_df = pd.DataFrame(species_probs, columns=['Species', 'Probability'])
                prob_df['Probability'] = (prob_df['Probability'] * 100).apply(lambda x: f"{x:.2f}%")
                prob_df['Species'] = 'Hoya ' + prob_df['Species']
                prob_df.insert(0, 'Rank', range(1, len(prob_df) + 1))
                st.dataframe(prob_df, hide_index=True, use_container_width=True, height=350)
            
            # Export
            results_dict = {
                'Predicted_Species': f"Hoya {predicted_species}",
                'Confidence_Percentage': f"{confidence*100:.2f}%",
                'Top_5_Alternatives': [f"Hoya {sp[0]} ({sp[1]*100:.2f}%)" for sp in species_probs[1:6]],
                'Input_Measurements_mm': st.session_state['measurements'],
                'Timestamp': pd.Timestamp.now().isoformat(),
                'Model_Used': model_package['model_name']
            }
            
            st.download_button(
                label="📥 Download Analysis Report (JSON)",
                data=pd.DataFrame([results_dict]).to_json(orient='records', indent=2),
                file_name=f"hoya_classification_{predicted_species}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.markdown('<div class="info-box" style="text-align: center; padding: 4rem 2rem;"><h3 style="color: #8e54e9;">👉 Enter measurements in the right panel</h3><p style="font-size: 1.1rem;">Fill in all 10 pollinaria measurements and click <b>"IDENTIFY SPECIES"</b> to get AI-powered analysis!</p><br><p style="font-size: 3.5rem;">🌺 🔬 ✨</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div style="text-align: center; color: #8e54e9; padding: 2rem; font-family: Quicksand;"><p>Made with 💜 by AI Research Team | v2.0 Advanced | For Research & Education Only<br><span style="font-size: 1.5rem;">🌺 🌸 🌼 🌻 🌷</span></p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
