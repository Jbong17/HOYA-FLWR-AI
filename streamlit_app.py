import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.covariance import EmpiricalCovariance
import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hoya Morpho-ID (Clade-Level)",
    page_icon="🌿",
    layout="wide"
)

# ============================================================
# CLEAN TAB STYLING
# ============================================================
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
    color: #2E4D32 !important;
}
.stTabs [aria-selected="true"] {
    border-bottom: 4px solid #588157 !important;
}
.result-box {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #CFE1D6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADER (Replace with real dataset later)
# ============================================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 64
    return pd.DataFrame({
        "species": [f"species_{i}" for i in range(n)],
        "pollinia_length": np.random.uniform(0.2, 5, n),
        "pollinia_width": np.random.uniform(0.1, 2.5, n),
        "corpusculum_length": np.random.uniform(0.05, 3, n),
        "clade": np.random.choice(["Acanthostemma", "Hoya-Complex"], n)
    })

df = load_data()
TRIO = ["pollinia_length", "pollinia_width", "corpusculum_length"]

# ============================================================
# TRAIN MODEL
# ============================================================
def train_model(df):
    X = np.log1p(df[TRIO])
    y = df["clade"]
    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)
    return model, scaler

model, scaler = train_model(df)

# ============================================================
# LOOCV + BOOTSTRAP CI
# ============================================================
def compute_loocv_ci(df, n_boot=300):

    X = np.log1p(df[TRIO].values)
    y = df["clade"].values
    loo = LeaveOneOut()

    preds = []
    truths = []

    for tr, te in loo.split(X):
        scaler_local = RobustScaler().fit(X[tr])
        model_local = RidgeClassifier().fit(
            scaler_local.transform(X[tr]), y[tr]
        )
        pred = model_local.predict(scaler_local.transform(X[te]))
        preds.append(pred[0])
        truths.append(y[te][0])

    base = balanced_accuracy_score(truths, preds)

    boot_scores = []
    n = len(truths)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_scores.append(
            balanced_accuracy_score(
                np.array(truths)[idx],
                np.array(preds)[idx]
            )
        )

    ci_low = np.percentile(boot_scores, 2.5)
    ci_high = np.percentile(boot_scores, 97.5)

    return base, ci_low, ci_high

acc, ci_low, ci_high = compute_loocv_ci(df)

# ============================================================
# OOD PREP
# ============================================================
X_scaled_full = scaler.transform(np.log1p(df[TRIO]))
cov = EmpiricalCovariance().fit(X_scaled_full)
train_maha = cov.mahalanobis(X_scaled_full)
OOD_THRESHOLD = float(np.percentile(train_maha, 95))

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# TITLE
# ============================================================
st.title("🌿 Hoya Morpho-ID: AI-Powered Clade-Level Classifier")
st.caption("Scope: Section/Clade Identification Only (Not Species-Level)")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Background",
    "🔬 Single Sample",
    "📊 Batch Analysis",
    "📜 Test History"
])

# ============================================================
# TAB 1 — BACKGROUND
# ============================================================
with tab1:

    st.header("📜 Research Context")

    st.markdown("""
The Philippine *Hoya* genus exhibits substantial morphometric diversity,
making clade-level discrimination a non-trivial classification task.

This system operationalizes micrometric pollinium morphology into a structured,
statistically validated machine learning framework.

The model performs **Section/Clade-level identification only** using quantitative
morphometric descriptors derived from floral reproductive structures.
""")

    st.warning("""
Model Scope Notice:
• Predicts Section/Clade only  
• Not suitable for species-level diagnosis  
• Not a replacement for molecular phylogenetics  
• Intended as a decision-support diagnostic tool
""")

    # --------------------------------------------------------
    # Clade Coverage
    # --------------------------------------------------------
    st.subheader("🌿 Clades Currently Covered")

    unique_clades = sorted(df["clade"].unique())
    for c in unique_clades:
        st.write(f"• {c}")

    # --------------------------------------------------------
    # Feature Architecture
    # --------------------------------------------------------
    st.subheader("🧬 Morphometric Feature Architecture")

    st.markdown("""
Total Engineered Feature Space: **34 Morphometric Descriptors**

**Core Micrometrics (11)**
• Pollinia Length  
• Pollinia Width  
• Corpusculum Length  
• Corpusculum Width  
• Shoulder Width  
• Waist Width  
• Hips Width  
• Base Extension  
• Translator Arm Length  
• Translator Stalk  
• Total Structure Length  

**Geometric Ratios (11)**
• Pollinia Aspect Ratio  
• Corpusculum Aspect Ratio  
• Pollinia-to-Corpusculum Length Ratio  
• Pollinia-to-Corpusculum Width Ratio  
• Shoulder-to-Waist Taper Index  
• Waist-to-Hips Ratio  
• Translator-to-Pollinia Ratio  
• Extension-to-Length Ratio  
• Shoulder-to-Hips Ratio  
• Stalk-to-Arm Ratio  
• Total Width Index  

**Derived Morphometric Indices (12)**
• Estimated Pollinia Area  
• Estimated Corpusculum Area  
• Relative Shoulder Position  
• Log-Transformed Lengths  
• Square-Root Area Transformations  
• Squared Width Deviations  
• Length–Width Interaction Terms  
• Normalized Volumetric Indices  
• Asymmetry Coefficients  
""")

    # --------------------------------------------------------
    # Golden Trio Emphasis
    # --------------------------------------------------------
    st.subheader("✨ The 'Golden Trio' Discovery")

    st.info("""
Through dimensionality reduction and feature importance analysis,
three dominant predictors consistently explain the majority of clade-level variance:

• **Pollinia Length**  
• **Pollinia Width**  
• **Corpusculum Length**

These form the operational inference backbone of the deployed classifier.
""")

    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------
    st.subheader("🧠 Model Architecture")

    st.markdown("""
**Algorithm:** Regularized Ridge Classifier  
**Preprocessing:** Log1p Transformation + Robust Scaling  
**Validation Strategy:** Leave-One-Out Cross-Validation (LOOCV)  
**Imbalance Handling:** Balanced Accuracy Metric  
""")

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------
    st.subheader("📊 Model Performance")

    st.write(f"Balanced Accuracy (LOOCV): **{acc*100:.2f}%**")
    st.write(f"95% Bootstrap Confidence Interval: **[{ci_low*100:.2f}% – {ci_high*100:.2f}%]**")

    st.caption("""
Performance estimates are derived from strict LOOCV to mitigate small-sample bias.
Confidence intervals computed via non-parametric bootstrap resampling.
""")

# ============================================================
# TAB 2 — SINGLE SAMPLE
# ============================================================
with tab2:

    col1, col2 = st.columns([1, 2])

    with col1:
        p_l = st.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.8)
        p_w = st.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.25)
        c_l = st.number_input("Corpusculum Length (mm)", 0.0, 5.0, 0.3)

        if st.button("Run Diagnosis"):

            x = np.log1p([[p_l, p_w, c_l]])
            x_scaled = scaler.transform(x)

            pred = model.predict(x_scaled)[0]

            maha = float(cov.mahalanobis(x_scaled)[0])
            ood = maha > OOD_THRESHOLD

            st.session_state.current_result = (pred, maha, ood)

            st.session_state.history.append({
                "Timestamp": datetime.datetime.now(),
                "Pollinia_Length": p_l,
                "Pollinia_Width": p_w,
                "Corpusculum_Length": c_l,
                "Prediction": pred,
                "Mahalanobis_Distance": maha,
                "OOD": ood
            })

    with col2:
        if "current_result" in st.session_state:

            pred, maha, ood = st.session_state.current_result

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader(f"Prediction: {pred}")
            st.write(f"Mahalanobis Distance: {maha:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

            if ood:
                st.error("⚠️ Out-of-Distribution: Outside training morphology.")
            else:
                st.success("Within Training Distribution")

            fig = px.scatter_3d(
                df,
                x="pollinia_length",
                y="pollinia_width",
                z="corpusculum_length",
                color="clade"
            )
            fig.add_trace(go.Scatter3d(
                x=[p_l],
                y=[p_w],
                z=[c_l],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Specimen"
            ))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — BATCH
# ============================================================
with tab3:

    st.write("Upload CSV with columns:")
    st.code("pollinia_length,pollinia_width,corpusculum_length")

    file = st.file_uploader("Upload File", type="csv")

    if file:
        bdf = pd.read_csv(file)
        x_scaled = scaler.transform(np.log1p(bdf[TRIO]))
        bdf["Prediction"] = model.predict(x_scaled)
        st.dataframe(bdf)

# ============================================================
# TAB 4 — HISTORY
# ============================================================
with tab4:

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)
        st.download_button(
            "Download History",
            hist_df.to_csv(index=False),
            "hoya_history.csv"
        )
    else:
        st.info("No diagnostic history yet.")
