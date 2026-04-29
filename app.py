"""
Hoya Clade Classifier - Professional Web Interface
AI-Powered Pollinarium Morphometric Analysis for Philippine Hoya Conservation

Developed by: Jerald B. Bongalos
Dataset Provider: Fernando B. Aurigue (Retired Career Scientist, DOST-PNRI)
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Philippine Hoya Clade Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS - SOPHISTICATED BOTANICAL THEME
# ================================
st.markdown("""
<style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
    }
    
    /* Main container */
    .main {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #1b5e20;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #558b2f;
        text-align: center;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #2e7d32;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #81c784;
    }
    
    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #2e7d32;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.15);
    }
    
    .result-medium {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 6px solid #f57c00;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 124, 0, 0.15);
    }
    
    .result-low {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #c62828;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(198, 40, 40, 0.15);
    }
    
    /* Predicted clade display */
    .clade-display {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 1rem 0;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    .confidence-display {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .confidence-high { color: #2e7d32; }
    .confidence-medium { color: #f57c00; }
    .confidence-low { color: #c62828; }
    
    /* Status messages */
    .status-message {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f8e9 0%, #dcedc8 100%);
        border-right: 2px solid #aed581;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        box-shadow: 0 6px 24px rgba(46, 125, 50, 0.4);
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #558b2f;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        color: #1b5e20;
        border: 2px solid #81c784;
        font-weight: 600;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border: 2px solid #c5e1a5;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #66bb6a;
        box-shadow: 0 0 0 3px rgba(102, 187, 106, 0.1);
    }
    
    /* Metric cards in sidebar */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #558b2f;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def engineer_enhanced_features(df):
    """Comprehensive feature engineering for pollinaria morphometrics"""
    d = df.copy()
    
    d['pollinia_ratio'] = d['pollinia_length'] / (d['pollinia_width'] + 1e-6)
    d['corp_ratio'] = d['corpusculum_length'] / (d['corpusculum_width'] + 1e-6)
    d['translator_ratio'] = d['translator_arm_length'] / (d['translator_stalk'] + 1e-6)
    d['extension_index'] = d['extension'] / (d['pollinia_length'] + 1e-6)
    
    d['pollinia_area'] = d['pollinia_length'] * d['pollinia_width']
    d['pollinia_perimeter'] = 2 * (d['pollinia_length'] + d['pollinia_width'])
    d['pollinia_compactness'] = (4 * np.pi * d['pollinia_area']) / (d['pollinia_perimeter']**2 + 1e-6)
    d['corp_eccentricity'] = np.sqrt(1 - (d['corpusculum_width']**2 / (d['corpusculum_length']**2 + 1e-6)))
    
    d['log_pollinia_L'] = np.log1p(d['pollinia_length'])
    d['log_corp_L'] = np.log1p(d['corpusculum_length'])
    d['allometric_slope'] = d['log_pollinia_L'] / (d['log_corp_L'] + 1e-6)
    
    d['translator_leverage'] = d['translator_arm_length'] / (d['extension'] + 1e-6)
    d['translator_total'] = d['translator_arm_length'] + d['translator_stalk']
    
    feature_cols = [
        'pollinia_length', 'pollinia_width', 'corpusculum_length',
        'corpusculum_width', 'extension', 'pollinia_ratio', 'corp_ratio',
        'extension_index', 'pollinia_compactness', 'corp_eccentricity',
        'allometric_slope', 'translator_leverage', 'translator_total'
    ]
    
    return d[feature_cols]


@st.cache_resource
def load_model():
    """Load the pre-trained model package"""
    try:
        with open('hoya_clade_classifier_production.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Ensure 'hoya_clade_classifier_production.pkl' is in the directory.")
        st.stop()


def predict_clade(measurements, model_package):
    """Make prediction from measurements"""
    input_df = pd.DataFrame([measurements])
    X_input = engineer_enhanced_features(input_df)
    X_scaled = model_package['scaler'].transform(X_input)
    
    pred_label = model_package['model'].predict(X_scaled)[0]
    pred_clade = model_package['label_encoder'].inverse_transform([pred_label])[0]
    proba = model_package['model'].predict_proba(X_scaled)[0]
    
    return {
        'clade': pred_clade,
        'confidence': np.max(proba),
        'probabilities': dict(zip(model_package['metadata']['classes'], proba))
    }


def create_probability_chart(probabilities):
    """Create elegant probability visualization"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Clade', 'Probability'])
    df = df.sort_values('Probability', ascending=True)
    
    colors = ['#1b5e20' if p == df['Probability'].max() else '#66bb6a' for p in df['Probability']]
    
    fig = go.Figure(go.Bar(
        x=df['Probability'], y=df['Clade'], orientation='h',
        marker=dict(color=colors, line=dict(color='#2e7d32', width=2)),
        text=[f"{p:.1%}" for p in df['Probability']], textposition='auto',
        textfont=dict(size=14, family='Inter', color='white', weight='bold'),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Classification Probability Distribution",
                  font=dict(size=18, family='Playfair Display', color='#1b5e20')),
        xaxis=dict(title="Probability", tickformat='.0%', range=[0, 1]),
        yaxis=dict(title="", tickfont=dict(size=13, family='Inter', color='#2e7d32')),
        plot_bgcolor='rgba(255, 255, 255, 0.9)', paper_bgcolor='rgba(0, 0, 0, 0)',
        height=300, margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">Philippine Hoya Clade Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Pollinarium Morphometric Analysis</p>', unsafe_allow_html=True)
    
    model_package = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="font-family: \'Playfair Display\', serif; color: #1b5e20;">Model Performance</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{model_package['metadata']['loocv_accuracy']:.1%}")
        col1.metric("Features", model_package['metadata']['n_features'])
        col2.metric("Kappa", f"{model_package['metadata']['cohens_kappa']:.3f}")
        col2.metric("Samples", model_package['metadata']['n_samples'])
        
        st.markdown("---")
        st.markdown("**Ensemble Architecture:**")
        st.info(model_package['metadata']['ensemble_components'])
        
        st.markdown("---")
        st.markdown("**Classification Clades:**")
        for clade in model_package['metadata']['classes']:
            st.markdown(f"• *{clade}*")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔬 **Classifier**", "📖 **User Guide**", "ℹ️ **About**"])
    
    with tab1:
        st.markdown('<p class="section-header">Pollinarium Measurements</p>', unsafe_allow_html=True)
        st.markdown("Enter microscopic measurements of your *Hoya* pollinarium (millimeters):")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔹 Pollinia")
            poll_len = st.number_input("Length (mm)", 0.0, 10.0, 0.56, 0.01, format="%.2f")
            poll_wid = st.number_input("Width (mm)", 0.0, 5.0, 0.30, 0.01, format="%.2f")
            
            st.markdown("#### 🔹 Corpusculum")
            corp_len = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.61, 0.01, format="%.2f")
            corp_wid = st.number_input("Corpusculum Width (mm)", 0.0, 2.0, 0.35, 0.01, format="%.2f")
            
            st.markdown("#### 🔹 Anatomy")
            shoulder = st.number_input("Shoulder (mm)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
            waist = st.number_input("Waist (mm)", 0.0, 1.0, 0.24, 0.01, format="%.2f")
            hips = st.number_input("Hips (mm)", 0.0, 1.0, 0.14, 0.01, format="%.2f")
        
        with col2:
            st.markdown("#### 🔹 Translator")
            trans_arm = st.number_input("Arm Length (mm)", 0.0, 2.0, 0.15, 0.01, format="%.2f")
            trans_stalk = st.number_input("Stalk (mm)", 0.0, 2.0, 0.58, 0.01, format="%.2f")
            
            st.markdown("#### 🔹 Extension")
            extension = st.number_input("Caudicle Extension (mm)", 0.0, 2.0, 0.29, 0.01, format="%.2f")
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 Classify Specimen", type="primary", use_container_width=True):
                result = predict_clade({
                    'pollinia_length': poll_len, 'pollinia_width': poll_wid,
                    'corpusculum_length': corp_len, 'corpusculum_width': corp_wid,
                    'shoulder': shoulder, 'waist': waist, 'hips': hips,
                    'extension': extension, 'translator_arm_length': trans_arm,
                    'translator_stalk': trans_stalk
                }, model_package)
                
                # Confidence level
                if result['confidence'] >= 0.70:
                    conf_class, result_class, icon, msg = "confidence-high", "result-high", "✅", "High Confidence — Reliable Classification"
                elif result['confidence'] >= 0.50:
                    conf_class, result_class, icon, msg = "confidence-medium", "result-medium", "⚠️", "Medium Confidence — Expert Review Recommended"
                else:
                    conf_class, result_class, icon, msg = "confidence-low", "result-low", "❌", "Low Confidence — Requires Expert Verification"
                
                # Results
                st.markdown("---")
                st.markdown('<p class="section-header">Classification Results</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="{result_class}">', unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown(f'<p style="font-size: 4rem; text-align: center; margin: 0;">{icon}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="clade-display">{result["clade"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence-display {conf_class}">{result["confidence"]:.1%}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="status-message">{msg}</p>', unsafe_allow_html=True)
                
                with col_res2:
                    st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Table
                st.markdown("##### 📊 Detailed Probabilities")
                prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Clade', 'Probability'])
                prob_df = prob_df.sort_values('Probability', ascending=False)
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("##### 💡 Recommendation")
                
                if result['confidence'] >= 0.70:
                    st.success(f"**✓ ACCEPT** — Classify as ***{result['clade']}***. Proceed with research workflow.")
                elif result['confidence'] >= 0.50:
                    st.warning(f"**⚠ REVIEW** — Classification as ***{result['clade']}*** requires taxonomist verification.")
                else:
                    st.error(f"**✗ EXPERT REQUIRED** — Mandatory review. Consider molecular identification.")
    
    with tab2:
        st.markdown('<p class="section-header">User Guide</p>', unsafe_allow_html=True)
        st.markdown("""
        ### 🔬 Measurement Protocol
        
        1. **Specimen Prep:** Mount pollinarium on microscope slide (40x–100x)
        2. **Calibration:** Verify microscope scale regularly
        3. **Precision:** Use 2 decimal places, take 3 measurements (average)
        4. **Orientation:** Maintain consistent view angle
        
        ### 📏 Confidence Interpretation
        
        | Level | Range | Action |
        |-------|-------|--------|
        | High | ≥70% | Accept classification |
        | Medium | 50-69% | Expert review |
        | Low | <50% | Mandatory verification |
        
        ### ⚠️ Limitations
        
        - Class imbalance: *Centrostemma* (1 sample), *Pterostelma* (4 samples)
        - Morphological convergence possible
        - Clade-level only (not species)
        - Philippine specimens (geographic scope)
        """)
    
    with tab3:
        st.markdown('<p class="section-header">About</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Philippine Hoya Clade Classifier
            
            **Version:** 1.0 | **Release:** April 2026
            
            First automated pollinarium classification system for *Hoya* species.
            Enables rapid clade-level identification for conservation and research.
            
            #### Technical Specs
            
            - **Algorithm:** Soft Voting Ensemble (SVM + GradBoost + ExtraTrees)
            - **Accuracy:** 75.0% base, ~92-93% filtered (τ≥0.70)
            - **Kappa:** 0.531 (excellent agreement)
            - **Data:** 64 specimens, 4 clades
            - **Validation:** Leave-One-Out Cross-Validation
            
            #### Dataset Provider
            
            **Fernando B. Aurigue**  
            Retired Career Scientist  
            DOST Philippine Nuclear Research Institute (PNRI)
            
            #### Developer
            
            **Jerald B. Bongalos**  
            PhD Candidate, Data Science  
            Asian Institute of Management
            
            #### Citation
            
            ```
            Bongalos, J. B. (2026). Deployable AI for Rapid 
            Morphometric Classification of Philippine Hoya Clades. 
            National Summit on Botanic Gardens and Arboreta.
            
            Dataset: Aurigue, F. B. (2026). Philippine Hoya 
            Pollinarium Morphometric Database. DOST-PNRI.
            ```
            
            **GitHub:** [Jbong17/HOYA-FLWR-AI](https://github.com/Jbong17/HOYA-FLWR-AI)  
            **License:** MIT | **Conference:** May 25-29, 2026
            """)
        
        with col2:
            st.info("**Stats**\n\n📊 75% Accuracy\n🎯 0.531 Kappa\n🔬 64 Specimens\n🌿 4 Clades\n⚡ 13 Features")
            st.success("**Impact**\n\n• Botanic gardens\n• Herbaria\n• Field surveys\n• Conservation")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; font-family: 'Inter', sans-serif;">
        <p style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 600; color: #1b5e20;">
            Preserving Philippine Biodiversity Through AI
        </p>
        <p style="color: #558b2f; margin: 0.3rem 0;">
            <strong>Jerald B. Bongalos</strong> | Asian Institute of Management
        </p>
        <p style="color: #689f38; margin: 0.3rem 0;">
            Dataset: <strong>Fernando B. Aurigue</strong> (Retired Career Scientist, DOST-PNRI)
        </p>
        <p style="color: #7cb342; font-size: 0.85rem; margin-top: 1rem;">
            © 2026 | MIT License | National Summit on Botanic Gardens and Arboreta
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
