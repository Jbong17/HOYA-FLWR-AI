
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

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Special+Elite&display=swap');

    /* Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fef5ff 0%, #f0f9ff 50%, #fffbf0 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Block padding */
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.5rem;
        max-width: 96%;
    }

    /* Apply typewriter font ONLY to non-interactive elements */
    h1, h2, h3, p, li, .main-title, .subtitle,
    .section-header, .species-name, .confidence-text {
        font-family: 'Courier Prime', 'Courier New', monospace !important;
    }

    /* Main title */
    .main-title {
        font-size: 3.2rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #d946ef, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0 0.3rem 0;
        padding: 0;
    }

    .subtitle {
        font-size: 1.4rem !important;
        color: #6b21a8;
        text-align: center;
        margin: 0 0 1rem 0;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.9rem !important;
        font-weight: 700;
        color: #1e293b !important;
        margin: 0 0 1rem 0;
        padding: 0 0 0.5rem 0;
        border-bottom: 3px solid #8b5cf6;
    }

    /* Overview box */
    .overview-box {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
        margin-bottom: 1.2rem;
        border: 2px solid #e9d5ff;
    }

    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #fae8ff, #fbcfe8);
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid #d946ef;
        margin: 1rem 0;
        text-align: center;
    }

    .species-name {
        font-size: 2.8rem !important;
        font-weight: 700;
        color: #86198f !important;
        font-style: italic;
        margin: 0.4rem 0;
    }

    .confidence-text {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #7c3aed !important;
        margin: 0.3rem 0;
    }

    /* Info cards */
    .info-card {
        background: #dbeafe;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        color: #1e3a8a !important;
        font-size: 1.1rem;
        margin: 0.8rem 0;
    }

    .success-card {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        color: #065f46 !important;
        font-size: 1.1rem;
        margin: 0.8rem 0;
    }

    .warning-card {
        background: #fed7aa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        color: #78350f !important;
        font-size: 1.1rem;
        margin: 0.8rem 0;
    }

    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #fae8ff, #e0e7ff);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #c084fc;
        margin: 0.4rem 0;
    }

    .metric-number {
        font-size: 2.8rem !important;
        font-weight: 800;
        color: #7c3aed !important;
        margin: 0;
        line-height: 1.1;
        font-family: 'Courier Prime', monospace !important;
    }

    .metric-text {
        font-size: 1.1rem !important;
        font-weight: 600;
        color: #4c1d95 !important;
        margin: 0.2rem 0 0 0;
    }

    /* FIX EXPANDERS - White background, dark text */
    details {
        background: #faf5ff !important;
        border: 2px solid #c084fc !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
    }

    details summary {
        background: linear-gradient(135deg, #ede9fe, #fae8ff) !important;
        color: #1e293b !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        list-style: none !important;
        font-family: 'Courier Prime', monospace !important;
    }

    details summary::-webkit-details-marker { display: none !important; }

    details[open] summary {
        border-bottom: 2px solid #c084fc !important;
        border-radius: 8px 8px 0 0 !important;
    }

    /* Streamlit expander override */
    [data-testid="stExpander"] {
        border: 2px solid #c084fc !important;
        border-radius: 12px !important;
        background: #faf5ff !important;
        margin-bottom: 0.6rem !important;
    }

    [data-testid="stExpander"] > div:first-child {
        background: linear-gradient(135deg, #ede9fe, #fae8ff) !important;
        border-radius: 10px 10px 0 0 !important;
    }

    /* Expander header text - FORCE DARK */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {
        color: #1e293b !important;
        background: linear-gradient(135deg, #ede9fe, #fae8ff) !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        font-family: 'Courier Prime', monospace !important;
    }

    /* Input labels - FORCE DARK */
    label, .stNumberInput label, [data-testid="stNumberInput"] label {
        color: #1e293b !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        font-family: 'Courier Prime', monospace !important;
    }

    /* Input boxes */
    [data-testid="stNumberInput"] input {
        border: 2px solid #c084fc !important;
        border-radius: 8px !important;
        font-size: 1.15rem !important;
        color: #1e293b !important;
        background: white !important;
        font-family: 'Courier Prime', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: #fae8ff !important;
        border-radius: 10px 10px 0 0 !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 1.1rem !important;
        padding: 10px 20px !important;
        font-family: 'Courier Prime', monospace !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #d946ef, #8b5cf6) !important;
        color: white !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(120deg, #d946ef, #8b5cf6) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        padding: 0.9rem 2rem !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100% !important;
        font-family: 'Courier Prime', monospace !important;
    }

    /* General text sizing */
    p { font-size: 1.05rem !important; color: #1e293b !important; }
    li { font-size: 1.05rem !important; color: #1e293b !important; }
    h3 { font-size: 1.4rem !important; color: #1e293b !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
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
        title={'text': "Confidence", 'font': {'size': 22, 'color': '#1e293b', 'family': 'Courier Prime'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#7c3aed', 'family': 'Courier Prime'}},
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
        height=300,
        margin=dict(l=30, r=30, t=70, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b', 'family': 'Courier Prime'}
    )
    return fig

def create_probability_chart(species_probs, top_n=8):
    top_species = species_probs[:top_n]
    # Full names with proper spacing for readability
    species_names = [f"H. {sp[0]}" for sp in top_species]
    probabilities = [sp[1] * 100 for sp in top_species]
    colors = ['#d946ef', '#c026d3', '#a21caf', '#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95']

    fig = go.Figure(go.Bar(
        y=species_names[::-1],
        x=probabilities[::-1],
        orientation='h',
        marker=dict(color=colors[:len(species_names)][::-1]),
        text=[f"  {p:.1f}%" for p in probabilities[::-1]],
        textposition='outside',
        textfont=dict(size=15, color='#1e293b', family='Courier Prime')
    ))

    fig.update_layout(
        title=dict(
            text="Top Candidate Species",
            font=dict(size=22, color='#1e293b', family='Courier Prime')
        ),
        xaxis_title="Confidence (%)",
        height=420,
        margin=dict(l=160, r=60, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, max(probabilities) * 1.3],
            gridcolor='#e2e8f0',
            tickfont=dict(size=13, color='#1e293b', family='Courier Prime')
        ),
        yaxis=dict(
            tickfont=dict(size=14, color='#1e293b', family='Courier Prime'),
            automargin=True
        ),
        font=dict(size=14, color='#1e293b', family='Courier Prime')
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-title">🌺 FlwrAI: Hoya Species Classifier 🌸</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Identification from Pollinaria Morphometrics</p>', unsafe_allow_html=True)

    model_package = load_model()

    # =====================================================================
    # TOP: OVERVIEW
    # =====================================================================
    st.markdown('<div class="overview-box">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 About", "🧠 Training", "📊 Performance", "⚠️ Limits"])

    with tab1:
        col1, col2, col3 = st.columns([2.5, 1, 1])
        with col1:
            st.markdown("### What is FlwrAI?")
            st.markdown("**FlwrAI** uses machine learning to identify *Hoya* species from pollinaria measurements. Input 10 microscopic measurements and receive an AI prediction with confidence score.")
        with col2:
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["n_species"]}</p><p class="metric-text">Species</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["n_features"]}</p><p class="metric-text">Features</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box"><p class="metric-number">{model_package["accuracy"]*100:.0f}%</p><p class="metric-text">Accuracy</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-box"><p class="metric-number">64</p><p class="metric-text">Samples</p></div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset:** 64 specimens, 49 species (PNRI Philippines)")
            st.markdown("**Features:** 10 raw + 24 engineered = 34 total")
        with col2:
            st.markdown("**Models:** RF, XGBoost, LightGBM, CatBoost, MLP, Voting")
            st.markdown(f"**Best Model:** {model_package['model_name'].replace('_', ' ').title()}")

    with tab3:
        if model_package.get('results') is not None:
            results_df = model_package['results'].nlargest(5, 'accuracy')[['model', 'accuracy']]
            results_df['model'] = results_df['model'].apply(lambda x: x.replace('_', ' ').title())
            results_df['accuracy'] = results_df['accuracy'].apply(lambda x: f"{x*100:.1f}%")
            results_df.columns = ['Model', 'Accuracy']
            st.dataframe(results_df, hide_index=True, use_container_width=True)
        else:
            st.info("Performance metrics available in training notebook.")

    with tab4:
        st.markdown('<div class="warning-card">⚠️ <b>Research prototype only.</b> Small dataset (64 samples). Expert verification always required. Not for commercial use.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================================
    # BOTTOM: Input LEFT | Output RIGHT
    # =====================================================================
    col_input, col_output = st.columns([1, 1.4])

    # LEFT: INPUT PANEL
    with col_input:
        st.markdown('<h2 class="section-header">📝 Manual Input</h2>', unsafe_allow_html=True)
        st.markdown('<div class="info-card">📏 All measurements in millimeters (mm)</div>', unsafe_allow_html=True)

        measurements = {}

        with st.expander("🌸 Pollinia Measurements", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                measurements['pollinia_length'] = st.number_input("Length (mm)", 0.0, 10.0, 0.5, 0.01, key='pl')
            with c2:
                measurements['pollinia_width'] = st.number_input("Width (mm)", 0.0, 10.0, 0.2, 0.01, key='pw')

        with st.expander("🔬 Corpusculum Measurements", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                measurements['corpusculum_length'] = st.number_input("Length (mm)", 0.0, 10.0, 0.3, 0.01, key='cl')
            with c2:
                measurements['corpusculum_width'] = st.number_input("Width (mm)", 0.0, 10.0, 0.15, 0.01, key='cw')

        with st.expander("⚖️ Body Architecture", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                measurements['shoulder'] = st.number_input("Shoulder", 0.0, 5.0, 0.1, 0.01)
            with c2:
                measurements['waist'] = st.number_input("Waist", 0.0, 5.0, 0.08, 0.01)
            with c3:
                measurements['hips'] = st.number_input("Hips", 0.0, 5.0, 0.05, 0.01)

        with st.expander("🔗 Translator & Extension", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                measurements['extension'] = st.number_input("Extension", 0.0, 5.0, 0.15, 0.01)
            with c2:
                measurements['translator_arm_length'] = st.number_input("Trans. Arm", 0.0, 5.0, 0.05, 0.01)
            with c3:
                measurements['translator_stalk'] = st.number_input("Trans. Stalk", 0.0, 5.0, 0.08, 0.01)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🌺 IDENTIFY SPECIES", type="primary")

    # RIGHT: OUTPUT PANEL
    with col_output:
        st.markdown('<h2 class="section-header">🎯 Analysis Report</h2>', unsafe_allow_html=True)

        if predict_button:
            predicted_species, species_probs, all_probs = predict_species(measurements, model_package)
            st.session_state['prediction'] = predicted_species
            st.session_state['species_probs'] = species_probs

        if 'prediction' in st.session_state:
            predicted_species = st.session_state['prediction']
            species_probs = st.session_state['species_probs']
            confidence = species_probs[0][1]

            st.markdown(
                f'<div class="prediction-result">'
                f'<p style="margin:0; color:#64748b; font-size:1rem;">PREDICTED SPECIES</p>'
                f'<p class="species-name">Hoya {predicted_species}</p>'
                f'<p class="confidence-text">{confidence*100:.1f}% Confidence</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            if confidence >= 0.75:
                st.markdown('<div class="success-card"><b>✅ HIGH CONFIDENCE</b> — Strong match to species profile.</div>', unsafe_allow_html=True)
            elif confidence >= 0.50:
                st.markdown('<div class="warning-card"><b>⚠️ MODERATE CONFIDENCE</b> — Review alternative candidates below.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card"><b>⚠️ LOW CONFIDENCE</b> — Consult expert botanist for verification.</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
            with c2:
                rank1 = f"Hoya {species_probs[0][0]} ({species_probs[0][1]*100:.1f}%)"
                rank2 = f"Hoya {species_probs[1][0]} ({species_probs[1][1]*100:.1f}%)"
                rank3 = f"Hoya {species_probs[2][0]} ({species_probs[2][1]*100:.1f}%)"
                st.markdown("**Top 3 Candidates:**")
                st.markdown(f"1. {rank1}")
                st.markdown(f"2. {rank2}")
                st.markdown(f"3. {rank3}")

            st.plotly_chart(create_probability_chart(species_probs, 8), use_container_width=True)

            with st.expander("📋 View All Species Probabilities"):
                prob_df = pd.DataFrame(species_probs, columns=['Species', 'Probability'])
                prob_df['Probability'] = (prob_df['Probability'] * 100).apply(lambda x: f"{x:.2f}%")
                prob_df['Species'] = 'Hoya ' + prob_df['Species']
                prob_df.insert(0, 'Rank', range(1, len(prob_df) + 1))
                st.dataframe(prob_df, hide_index=True, height=300, use_container_width=True)

        else:
            st.markdown(
                '<div class="info-card" style="text-align:center; padding:2.5rem;">'
                '<h3>👈 Enter your measurements</h3>'
                '<p>Fill in the pollinaria measurements on the left and click IDENTIFY SPECIES to get AI predictions.</p>'
                '<p style="font-size:2rem; margin-top:1rem;">🌺 🔬 ✨</p>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div style="text-align:center; color:#64748b; padding:1rem; font-size:1rem; font-family: Courier Prime, monospace;">v2.0 | Research & Education Only | 🌺 🌸 🌼</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
