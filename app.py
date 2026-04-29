"""
Hoya Clade Classifier - Web Interface
Streamlit application for interactive pollinarium classification
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Hoya Clade Classifier",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def engineer_enhanced_features(df):
    """
    Comprehensive feature engineering for pollinaria morphometrics
    """
    d = df.copy()
    
    # Original ratios
    d['pollinia_ratio'] = d['pollinia_length'] / (d['pollinia_width'] + 1e-6)
    d['corp_ratio'] = d['corpusculum_length'] / (d['corpusculum_width'] + 1e-6)
    d['translator_ratio'] = d['translator_arm_length'] / (d['translator_stalk'] + 1e-6)
    d['extension_index'] = d['extension'] / (d['pollinia_length'] + 1e-6)
    
    # Shape descriptors
    d['pollinia_area'] = d['pollinia_length'] * d['pollinia_width']
    d['pollinia_perimeter'] = 2 * (d['pollinia_length'] + d['pollinia_width'])
    d['pollinia_compactness'] = (4 * np.pi * d['pollinia_area']) / (d['pollinia_perimeter']**2 + 1e-6)
    d['corp_eccentricity'] = np.sqrt(1 - (d['corpusculum_width']**2 / (d['corpusculum_length']**2 + 1e-6)))
    
    # Allometric scaling
    d['log_pollinia_L'] = np.log1p(d['pollinia_length'])
    d['log_corp_L'] = np.log1p(d['corpusculum_length'])
    d['allometric_slope'] = d['log_pollinia_L'] / (d['log_corp_L'] + 1e-6)
    
    # Translator mechanics
    d['translator_leverage'] = d['translator_arm_length'] / (d['extension'] + 1e-6)
    d['translator_total'] = d['translator_arm_length'] + d['translator_stalk']
    
    # Select features
    feature_cols = [
        'pollinia_length', 'pollinia_width', 'corpusculum_length',
        'corpusculum_width', 'extension',
        'pollinia_ratio', 'corp_ratio', 'extension_index',
        'pollinia_compactness', 'corp_eccentricity',
        'allometric_slope',
        'translator_leverage', 'translator_total'
    ]
    
    return d[feature_cols]


@st.cache_resource
def load_model():
    """Load the pre-trained model package"""
    try:
        with open('hoya_clade_classifier_production.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'hoya_clade_classifier_production.pkl' is in the same directory.")
        st.stop()


def predict_clade(measurements, model_package):
    """Make prediction from measurements"""
    # Create DataFrame
    input_df = pd.DataFrame([measurements])
    
    # Engineer features
    X_input = engineer_enhanced_features(input_df)
    
    # Scale
    X_scaled = model_package['scaler'].transform(X_input)
    
    # Predict
    pred_label = model_package['model'].predict(X_scaled)[0]
    pred_clade = model_package['label_encoder'].inverse_transform([pred_label])[0]
    
    # Get probabilities
    proba = model_package['model'].predict_proba(X_scaled)[0]
    confidence = np.max(proba)
    all_proba = dict(zip(model_package['metadata']['classes'], proba))
    
    return {
        'clade': pred_clade,
        'confidence': confidence,
        'probabilities': all_proba
    }


def create_probability_chart(probabilities):
    """Create interactive probability bar chart"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Clade', 'Probability'])
    df = df.sort_values('Probability', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Clade'],
        orientation='h',
        marker=dict(
            color=df['Probability'],
            colorscale='Greens',
            showscale=False
        ),
        text=[f"{p:.1%}" for p in df['Probability']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Probability",
        yaxis_title="Clade",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🌺 Philippine Hoya Clade Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Pollinarium Morphometric Analysis</p>', unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Accuracy (LOOCV)", f"{model_package['metadata']['loocv_accuracy']:.1%}")
        st.metric("Cohen's Kappa", f"{model_package['metadata']['cohens_kappa']:.3f}")
        st.metric("Training Samples", model_package['metadata']['n_samples'])
        st.metric("Features", model_package['metadata']['n_features'])
        
        st.markdown("---")
        st.markdown("**Ensemble Components:**")
        st.markdown(model_package['metadata']['ensemble_components'])
        
        st.markdown("---")
        st.markdown("**Clades:**")
        for clade in model_package['metadata']['classes']:
            st.markdown(f"• {clade}")
        
        st.markdown("---")
        st.info(f"**Model Date:** {model_package['metadata']['training_date']}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔬 Classifier", "📖 Guide", "ℹ️ About"])
    
    with tab1:
        st.header("Pollinarium Measurements")
        st.markdown("Enter the microscopic measurements of your Hoya pollinarium specimen:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pollinia")
            pollinia_length = st.number_input("Length (mm)", min_value=0.0, max_value=10.0, value=0.56, step=0.01, format="%.2f")
            pollinia_width = st.number_input("Width (mm)", min_value=0.0, max_value=5.0, value=0.30, step=0.01, format="%.2f")
            
            st.subheader("Corpusculum")
            corpusculum_length = st.number_input("Corpusculum Length (mm)", min_value=0.0, max_value=5.0, value=0.61, step=0.01, format="%.2f")
            corpusculum_width = st.number_input("Corpusculum Width (mm)", min_value=0.0, max_value=2.0, value=0.35, step=0.01, format="%.2f")
            
            st.subheader("Corpusculum Anatomy")
            shoulder = st.number_input("Shoulder (mm)", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
            waist = st.number_input("Waist (mm)", min_value=0.0, max_value=1.0, value=0.24, step=0.01, format="%.2f")
            hips = st.number_input("Hips (mm)", min_value=0.0, max_value=1.0, value=0.14, step=0.01, format="%.2f")
        
        with col2:
            st.subheader("Translator")
            translator_arm_length = st.number_input("Arm Length (mm)", min_value=0.0, max_value=2.0, value=0.15, step=0.01, format="%.2f")
            translator_stalk = st.number_input("Stalk (mm)", min_value=0.0, max_value=2.0, value=0.58, step=0.01, format="%.2f")
            
            st.subheader("Extension")
            extension = st.number_input("Caudicle Extension (mm)", min_value=0.0, max_value=2.0, value=0.29, step=0.01, format="%.2f")
        
        # Predict button
        if st.button("🔍 Classify Specimen", type="primary", use_container_width=True):
            measurements = {
                'pollinia_length': pollinia_length,
                'pollinia_width': pollinia_width,
                'corpusculum_length': corpusculum_length,
                'corpusculum_width': corpusculum_width,
                'shoulder': shoulder,
                'waist': waist,
                'hips': hips,
                'extension': extension,
                'translator_arm_length': translator_arm_length,
                'translator_stalk': translator_stalk
            }
            
            # Make prediction
            result = predict_clade(measurements, model_package)
            
            # Display results
            st.markdown("---")
            st.header("🎯 Classification Results")
            
            # Confidence styling
            if result['confidence'] >= 0.70:
                conf_class = "confidence-high"
                status_icon = "✅"
                status_msg = "High Confidence - Reliable Classification"
            elif result['confidence'] >= 0.50:
                conf_class = "confidence-medium"
                status_icon = "⚠️"
                status_msg = "Medium Confidence - Consider Expert Review"
            else:
                conf_class = "confidence-low"
                status_icon = "❌"
                status_msg = "Low Confidence - Expert Review Recommended"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"### {status_icon} Predicted Clade")
                st.markdown(f'<p style="font-size: 2rem; color: #2E7D32; margin: 0;"><strong>{result["clade"]}</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<p class="{conf_class}" style="font-size: 1.5rem; margin: 0;">{result["confidence"]:.1%}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: #666; margin-top: 0.5rem;">{status_msg}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Probability chart
                fig = create_probability_chart(result['probabilities'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            st.subheader("📊 Detailed Probability Distribution")
            prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Clade', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=False)
            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Deployment recommendations
            st.markdown("---")
            st.subheader("💡 Recommendation")
            
            if result['confidence'] >= 0.70:
                st.success(f"""
                **Accept Classification**: The specimen can be classified as **{result['clade']}** with high confidence.
                
                **Next Steps:**
                - Record classification in database
                - No expert review required
                - Continue with conservation/research workflow
                """)
            elif result['confidence'] >= 0.50:
                st.warning(f"""
                **Review Recommended**: The classification as **{result['clade']}** should be verified.
                
                **Next Steps:**
                - Flag for expert taxonomist review
                - Check top 2-3 probabilities for alternative identifications
                - Consider additional morphological features
                """)
            else:
                st.error(f"""
                **Expert Review Required**: Low confidence classification.
                
                **Next Steps:**
                - Mandatory expert taxonomist review
                - Consider molecular identification (ITS, matK)
                - Re-examine specimen for measurement errors
                - May represent morphological convergence case
                """)
    
    with tab2:
        st.header("📖 User Guide")
        
        st.markdown("""
        ### How to Use This Classifier
        
        #### 1. Prepare Your Specimen
        - Mount pollinarium on microscope slide
        - Use appropriate magnification (typically 40x-100x)
        - Ensure clear view of all structures
        
        #### 2. Take Measurements
        - **Pollinia**: Measure length and width of pollen masses
        - **Corpusculum**: Central adhesive disc dimensions
        - **Translator Arms/Stalk**: Connecting structures
        - **Extension**: Caudicle extension length
        - **Anatomy**: Shoulder, waist, hips of corpusculum
        
        #### 3. Input Data
        - Enter all measurements in millimeters (mm)
        - Use 2 decimal precision when possible
        - Double-check measurements for accuracy
        
        #### 4. Interpret Results
        - **High Confidence (≥70%)**: Accept classification
        - **Medium Confidence (50-69%)**: Consider expert review
        - **Low Confidence (<50%)**: Require expert review
        
        ### Measurement Tips
        
        - Use calibrated microscope with measurement scale
        - Take multiple measurements and use average
        - Ensure pollinarium is properly oriented
        - Record any unusual morphological features
        
        ### Known Limitations
        
        - **Class Imbalance**: Limited training data for Centrostemma (1 sample) and Pterostelma (4 samples)
        - **Morphological Convergence**: Some species (e.g., H. tsangii) exhibit convergent evolution
        - **Clade-Level Only**: Current model classifies to clade, not species
        - **Philippine Focus**: Trained on Philippine Hoya specimens
        """)
    
    with tab3:
        st.header("ℹ️ About")
        
        st.markdown("""
        ### Philippine Hoya Clade Classifier
        
        **Version**: 1.0  
        **Release Date**: April 2026  
        **Developer**: Jerald B. Bongalos, Asian Institute of Management
        
        ### Project Background
        
        This AI-powered classification system addresses a critical gap in Philippine biodiversity conservation.
        The Philippines hosts over 150 Hoya species, many endemic, but traditional identification requires
        expert taxonomists and is time-consuming.
        
        ### Technical Details
        
        - **Algorithm**: Soft Voting Ensemble (SVM + Gradient Boosting + Extra Trees)
        - **Training Data**: 64 pollinarium specimens across 4 clades
        - **Validation**: Leave-One-Out Cross-Validation (LOOCV)
        - **Accuracy**: 75.0% base, ~92-93% with confidence filtering
        
        ### Performance Metrics
        
        - **LOOCV Accuracy**: 75.00%
        - **Cohen's Kappa**: 0.531 (excellent agreement)
        - **Improvement**: +3.12pp over baseline single models
        
        ### Citation
        
        If you use this tool in research, please cite:
        
        ```
        Bongalos, J. B. (2026). Deployable AI for Rapid Morphometric 
        Classification of Philippine Hoya Clades. National Summit on 
        Botanic Gardens and Arboreta.
        ```
        
        ### Contact
        
        - **GitHub**: [Jbong17/HOYA-FLWR-AI](https://github.com/Jbong17/HOYA-FLWR-AI)
        - **Institution**: Asian Institute of Management
        - **Program**: PhD in Data Science
        
        ### Acknowledgments
        
        - AIM Faculty and Advisers
        - National University Manila (Co-adviser for thesis)
        - Philippine National Research Institute (PNRI)
        - Philippine Botanic Gardens and Arboreta
        
        ### License
        
        MIT License - Open source for research and conservation purposes
        
        ---
        
        **⭐ Star the GitHub repository if you find this useful!**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌺 Preserving Philippine Biodiversity Through AI 🌺</p>
        <p style="font-size: 0.9rem;">Developed by Jerald B. Bongalos | Asian Institute of Management | 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
