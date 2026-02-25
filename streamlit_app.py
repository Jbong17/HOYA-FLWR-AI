import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.covariance import EmpiricalCovariance
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax
import datetime
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hoya Morpho-ID (Clade-Level)",
    page_icon="🌿",
    layout="wide"
)

# ============================================================
# CLEAN UI
# ============================================================
st.title("🌿 Hoya Morpho-ID: AI-Powered Clade-Level Classifier")
st.caption("Scope: Section/Clade Identification Only (Not Species-Level)")

# ============================================================
# PRODUCTION DATA LOADING
# ============================================================
@st.cache_data
def load_dataset():

    # Priority 1: production CSV in repo
    if os.path.exists("hoya_dataset.csv"):
        df = pd.read_csv("hoya_dataset.csv")
        return df

    # Fallback demo dataset
    np.random.seed(42)
    n = 64
    return pd.DataFrame({
        "pollinia_length": np.random.uniform(0.2, 5, n),
        "pollinia_width": np.random.uniform(0.1, 2.5, n),
        "corpusculum_length": np.random.uniform(0.05, 3, n),
        "clade": np.random.choice(["Acanthostemma", "Hoya-Complex"], n)
    })

df = load_dataset()
TRIO = ["pollinia_length", "pollinia_width", "corpusculum_length"]

# ============================================================
# TRAIN MODEL
# ============================================================
@st.cache_resource
def train_model(df):

    X = np.log1p(df[TRIO])
    y = df["clade"]

    scaler = RobustScaler().fit(X)
    model = RidgeClassifier(alpha=1.0).fit(scaler.transform(X), y)

    return model, scaler

model, scaler = train_model(df)

# ============================================================
# LOOCV + CI
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
        pred = model_local.predict(
            scaler_local.transform(X[te])
        )
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

    return base, np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5)

acc, ci_low, ci_high = compute_loocv_ci(df)

# ============================================================
# CALIBRATION
# ============================================================
def compute_probabilities(model, scaler, X_raw):

    X_scaled = scaler.transform(np.log1p(X_raw))
    decision = model.decision_function(X_scaled)

    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])

    probs = softmax(decision, axis=1)
    return probs

def calibration_plot():

    probs = compute_probabilities(model, scaler, df[TRIO])
    y_true = (df["clade"] == model.classes_[1]).astype(int)

    frac_pos, mean_pred = calibration_curve(
        y_true, probs[:,1], n_bins=5
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_pred,
        y=frac_pos,
        mode="lines+markers",
        name="Model"
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode="lines",
        name="Perfect"
    ))

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency"
    )

    return fig

# ============================================================
# CENTROIDS
# ============================================================
X_scaled_full = scaler.transform(np.log1p(df[TRIO]))
centroids_scaled = pd.DataFrame(X_scaled_full, columns=TRIO)
centroids_scaled["clade"] = df["clade"]
centroids_scaled = centroids_scaled.groupby("clade").mean()

centroids_raw = np.expm1(centroids_scaled)

# ============================================================
# OOD DETECTION
# ============================================================
cov = EmpiricalCovariance().fit(X_scaled_full)
train_maha = cov.mahalanobis(X_scaled_full)
OOD_THRESHOLD = float(np.percentile(train_maha, 95))

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Background",
    "🔬 Single Sample",
    "📊 Calibration & Model",
    "📜 History"
])

# ============================================================
# BACKGROUND
# ============================================================
with tab1:

    st.header("Model Information")

    st.write(f"Balanced Accuracy (LOOCV): {acc*100:.2f}%")
    st.write(f"95% CI: [{ci_low*100:.2f}% – {ci_high*100:.2f}%]")

    st.write("""
Golden Trio:
• Pollinia Length  
• Pollinia Width  
• Corpusculum Length  

Full Morphometric Framework:
• 34 engineered features (core + ratios + derived indices)
""")

    st.warning("""
Model Scope Notice:
• Predicts Section/Clade only  
• Not suitable for species-level diagnosis  
• Not a replacement for molecular phylogenetics
""")

# ============================================================
# SINGLE SAMPLE
# ============================================================
with tab2:

    col1, col2 = st.columns([1,2])

    with col1:
        p_l = st.number_input("Pollinia Length", 0.0, 10.0, 0.8)
        p_w = st.number_input("Pollinia Width", 0.0, 5.0, 0.25)
        c_l = st.number_input("Corpusculum Length", 0.0, 5.0, 0.3)

        if st.button("Diagnose"):

            X_input = np.array([[p_l, p_w, c_l]])
            probs = compute_probabilities(model, scaler, X_input)
            pred = model.classes_[np.argmax(probs)]

            maha = float(
                cov.mahalanobis(
                    scaler.transform(np.log1p(X_input))
                )[0]
            )

            # Bayesian uncertainty (Dirichlet approximation)
            alpha = probs[0] * len(df)
            posterior_mean = alpha / np.sum(alpha)
            entropy = -np.sum(posterior_mean * np.log(posterior_mean + 1e-9))

            st.session_state.current = (pred, probs[0], maha, entropy)

            st.session_state.history.append({
                "Timestamp": datetime.datetime.now(),
                "Prediction": pred,
                "Mahalanobis": maha,
                "Entropy": entropy
            })

    with col2:
        if "current" in st.session_state:

            pred, probs, maha, entropy = st.session_state.current

            st.subheader(f"Prediction: {pred}")
            st.write(f"Mahalanobis Distance: {maha:.3f}")
            st.write(f"Predictive Entropy (Uncertainty): {entropy:.3f}")

            if maha > OOD_THRESHOLD:
                st.error("⚠️ Out-of-Distribution")
            else:
                st.success("Within Training Distribution")

            # 3D Plot with centroid overlay
            fig = px.scatter_3d(
                df,
                x="pollinia_length",
                y="pollinia_width",
                z="corpusculum_length",
                color="clade",
                opacity=0.4
            )

            # Add centroids
            for clade in centroids_raw.index:
                fig.add_trace(go.Scatter3d(
                    x=[centroids_raw.loc[clade, "pollinia_length"]],
                    y=[centroids_raw.loc[clade, "pollinia_width"]],
                    z=[centroids_raw.loc[clade, "corpusculum_length"]],
                    mode="markers",
                    marker=dict(size=8, symbol="diamond"),
                    name=f"{clade} Centroid"
                ))

            fig.add_trace(go.Scatter3d(
                x=[p_l], y=[p_w], z=[c_l],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Specimen"
            ))

            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CALIBRATION TAB
# ============================================================
with tab3:
    st.header("Calibration & Reliability")
    st.plotly_chart(calibration_plot(), use_container_width=True)

# ============================================================
# HISTORY
# ============================================================
with tab4:

    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist)
        st.download_button(
            "Download CSV",
            hist.to_csv(index=False),
            "hoya_history.csv"
        )
    else:
        st.info("No diagnostic history yet.")
