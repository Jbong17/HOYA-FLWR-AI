import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import RobustScaler

# --- DATASET EMBEDDING ---
# We include the data here so the app is "Plug and Play" on GitHub
def load_hoya_data():
    data = {
        'species': ['lazaroi', 'laut', 'bulacanensis', 'dalanesiae', 'cumingiana', 'diversifolia', 'ardamosana', 'brevialata', 'multiflora', 'monetteae', 'angustifolia', 'citrina', 'linea', 'blanca', 'linea', 'ciliatifolia', 'castillione', 'laoagensis', 'brittonii', 'pubicalyx', 'buotii', 'vitellinoides', 'densifolia', 'camphorifolia', 'camphorifolia', 'cutilipensis', 'kentiana', 'bangbangensis', 'wayetii', 'wayetii', 'cutispicellana', 'salacae', 'tricolor', 'bicolensis', 'obscura', 'obscura', 'obscura', 'obscura', 'flagellata', 'camphorifolia', 'samoensis', 'pottsii', 'cerata', 'cerata', 'cerata', 'albiflora', 'stagensis', 'obscura', 'golamcoana', 'odorata', 'benguetensis', 'benguetensis', 'malae', 'ruthiae', 'benguetensis', 'litoralis', 'mcgregorii', 'bensianii', 'chloroleuca', 'biakensis', 'pottsii', 'benguetensis', 'benguetensis', 'edoroana'],
        'pollinia_length': [0.56, 1.05, 0.51, 0.59, 0.85, 3.8, 5.6, 0.34, 5, 7.8, 4.5, 4.2, 4.3, 0.36, 2.7, 3, 0.26, 0.37, 0.7, 0.6, 0.69, 0.6, 0.82, 0.31, 0.4, 0.36, 0.48, 0.42, 0.39, 0.39, 0.35, 0.29, 0.36, 0.24, 0.35, 0.32, 0.35, 0.31, 0.41, 0.46, 0.48, 0.88, 0.83, 0.94, 0.51, 0.87, 0.9, 0.75, 0.72, 0.7, 0.57, 0.45, 0.85, 1.05, 0.39, 0.54, 0.46, 0.93, 0.46, 0.46, 0.36, 0.55, 0.54, 0.66],
        'pollinia_width': [0.3, 0.5, 0.2, 0.24, 0.24, 1.4, 2.4, 0.17, 2.9, 2, 1.8, 1.8, 1.4, 0.15, 1.1, 1.3, 0.11, 0.15, 0.28, 0.24, 0.24, 0.24, 0.2, 0.13, 0.16, 0.18, 0.18, 0.17, 0.17, 0.16, 0.17, 0.12, 0.15, 0.11, 0.15, 0.14, 0.15, 0.12, 0.16, 0.22, 0.21, 0.25, 0.25, 0.28, 0.2, 0.25, 0.23, 0.3, 0.28, 0.25, 0.23, 0.2, 0.27, 0.28, 0.16, 0.2, 0.17, 0.19, 0.17, 0.17, 0.17, 0.22, 0.26, 0.26],
        'corpusculum_length': [0.61, 0.96, 0.12, 0.22, 0.28, 1.9, 2.6, 0.15, 4.6, 2.5, 2.3, 1.9, 1.6, 0.1, 1.2, 1.1, 0.07, 0.11, 0.3, 0.38, 0.39, 0.26, 0.23, 0.1, 0.16, 0.14, 0.14, 0.14, 0.17, 0.16, 0.22, 0.09, 0.1, 0.09, 0.12, 0.12, 0.16, 0.13, 0.16, 0.24, 0.2, 0.42, 0.42, 0.45, 0.19, 0.33, 0.4, 0.35, 0.43, 0.4, 0.18, 0.16, 0.4, 0.47, 0.17, 0.23, 0.16, 0.11, 0.19, 0.2, 0.23, 0.18, 0.18, 0.43],
        'clade': ['Acanthostemma', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Hoya-Complex', 'Acanthostemma', 'Acanthostemma', 'Hoya-Complex']
    }
    return pd.DataFrame(data)

# --- APP CONFIG ---
st.set_page_config(page_title="Hoya Morpho-ID", page_icon="🌿", layout="wide")

st.title("🌿 Hoya Clade Diagnostic Tool")
st.markdown("""
This tool uses a **Ridge Classification** model optimized via **Recursive Feature Elimination** to predict Hoya clades based on the "Golden Trio" of pollinarium measurements.
Developed for the *National Summit on Botanic Gardens and Arboreta*.
""")

# --- LOAD & TRAIN ---
df = load_hoya_data()
trio = ['pollinia_length', 'pollinia_width', 'corpusculum_length']

# Pre-processing (Log + Scale)
X = np.log1p(df[trio])
y = df['clade']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = RidgeClassifier(alpha=1.0)
model.fit(X_scaled, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Micrometer Measurements")
p_len = st.sidebar.number_input("Pollinia Length (mm)", 0.0, 10.0, 0.85, 0.01)
p_wid = st.sidebar.number_input("Pollinia Width (mm)", 0.0, 5.0, 0.24, 0.01)
c_len = st.sidebar.number_input("Corp. Length (mm)", 0.0, 5.0, 0.28, 0.01)

# --- PREDICTION LOGIC ---
# Transform input
raw_in = np.array([[p_len, p_wid, c_len]])
log_in = np.log1p(raw_in)
scaled_in = scaler.transform(log_in)

# Ridge Probabilities (via Decision Function Softmax)
decision = model.decision_function(scaled_in)
probs = np.exp(decision) / np.sum(np.exp(decision), axis=1)
conf = np.max(probs)
pred = model.predict(scaled_in)[0]

# --- DISPLAY RESULTS ---
col_res, col_plot = st.columns([1, 2])

with col_res:
    st.subheader("Diagnostic Result")
    if conf >= 0.70:
        st.success(f"**Predicted Clade:** {pred}")
        st.metric("Confidence Score", f"{conf:.1%}")
    else:
        st.warning(f"**Result: Ambiguous Morphotype**")
        st.write(f"Confidence ({conf:.1%}) is below the taxonomic threshold (70%).")
    
    st.write("---")
    st.write("**Probability Distribution:**")
    prob_df = pd.DataFrame({'Clade': model.classes_, 'Probability': probs[0]})
    st.bar_chart(prob_df.set_index('Clade'))

with col_plot:
    st.subheader("3D Morphospace Visualization")
    # Add the user's point to the plot
    user_point = pd.DataFrame({
        'pollinia_length': [p_len], 'pollinia_width': [p_wid], 
        'corpusculum_length': [c_len], 'clade': ['USER INPUT'], 'species': ['Current Query']
    })
    plot_df = pd.concat([df, user_point])
    
    fig = px.scatter_3d(plot_df, x='pollinia_length', y='pollinia_width', z='corpusculum_length',
                        color='clade', symbol='clade', opacity=0.7,
                        color_discrete_map={'USER INPUT': '#000000'},
                        labels={'pollinia_length': 'P-Length', 'pollinia_width': 'P-Width', 'corpusculum_length': 'C-Length'})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

st.info("**Reference Note:** H. angustifolia samples are known to exhibit high morphological overlap with the Hoya-Complex.")
