# 🌺 HOYA-FLWR-AI: Deployable AI for Philippine Hoya Clade Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy: 75%](https://img.shields.io/badge/Accuracy-75%25-success.svg)]()
[![Cohen's Kappa: 0.531](https://img.shields.io/badge/Kappa-0.531-success.svg)]()

> **Automated morphometric classification of Philippine Hoya clades using ensemble machine learning on pollinarium features**

Developed by **Jerald B. Bongalos** | Asian Institute of Management  
Presented at: National Summit on Botanic Gardens and Arboreta 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project presents the **first automated classification system for Hoya pollinaria**, addressing a critical gap in Philippine biodiversity conservation. Traditional Hoya identification requires expert taxonomists and is time-consuming. Our AI-powered system enables rapid, accurate clade-level classification using microscopic pollinarium measurements.

### **Problem Statement**

- 🌿 **150+ Hoya species** in the Philippines (many endemic)
- 🔬 **Expert scarcity**: Few taxonomists available
- ⏱️ **Slow identification**: Manual morphological analysis takes hours
- 🎯 **Critical need**: Rapid screening for botanic gardens and conservation

### **Solution**

An ensemble machine learning classifier that:
- ✅ Achieves **75% base accuracy** (LOOCV)
- ✅ Reaches **~92-93% accuracy** with confidence filtering (τ=0.70)
- ✅ Processes specimens in **seconds** vs hours
- ✅ Provides interpretable confidence scores

---

## ⚡ Key Features

### **Model Capabilities**

- **Multi-Clade Classification**: Distinguishes 4 major Philippine Hoya clades:
  - *Acanthostemma* (24 specimens)
  - *Centrostemma* (1 specimen)
  - *Hoya* (35 specimens)
  - *Pterostelma* (4 specimens)

- **Ensemble Architecture**: Soft Voting Classifier combining:
  - Support Vector Machine (RBF kernel)
  - Gradient Boosting
  - Extra Trees

- **Robust Feature Engineering**: 13 morphometric features including:
  - Golden Trio: pollinia length/width, corpusculum length
  - Shape descriptors: compactness, eccentricity
  - Allometric scaling: log-log relationships
  - Translator mechanics: leverage ratios

- **Production-Ready Deployment**:
  - Confidence thresholding for reliability
  - Probability distributions for transparency
  - Flagging system for expert review

---

## 📊 Performance

### **Model Comparison Results**

| Model | Accuracy | Cohen's Kappa | Status |
|-------|----------|---------------|--------|
| **Soft Voting Ensemble** | **75.00%** | **0.531** | ✅ **Production** |
| Weighted Ensemble | 73.44% | 0.510 | ✅ Alternative |
| Ridge Classifier | 71.88% | 0.485 | ✅ Baseline |
| XGBoost | 71.88% | 0.470 | - |
| Gradient Boosting | 71.88% | 0.491 | - |
| Hard Voting | 71.88% | 0.497 | - |
| Extra Trees | 64.06% | 0.335 | - |
| SVM (RBF) | 56.25% | 0.273 | - |

### **Performance Evolution**

```
Initial Model (34 features):    57.81% ❌ Overfitted
Ridge (Golden Trio):            71.88% ✅ Optimized  
Soft Voting Ensemble:           75.00% 🎯 Best
With Confidence Filter (τ=0.70): ~92-93% 🚀 Deployment
```

**Key Achievement**: +3.12 percentage point improvement over baseline through ensemble methods

### **Per-Clade Performance** (Soft Voting)

- **Acanthostemma**: ~79% recall, 73-83% precision
- **Hoya**: ~80% recall, 74-78% precision  
- **Pterostelma**: Limited (only 4 samples)
- **Centrostemma**: N/A (only 1 sample - LOOCV impossible)

---

## 🚀 Installation

### **Prerequisites**

- Python 3.8+
- pip or conda

### **Clone Repository**

```bash
git clone https://github.com/Jbong17/HOYA-FLWR-AI.git
cd HOYA-FLWR-AI
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Download Pre-trained Model**

The production model (`hoya_clade_classifier_production.pkl`) is available in the repository.

---

## ⚡ Quick Start

### **Python API**

```python
import pickle
import pandas as pd
import numpy as np

# Load model package
with open('hoya_clade_classifier_production.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
label_encoder = model_package['label_encoder']

# Example: Classify a specimen
measurements = {
    'pollinia_length': 0.56,
    'pollinia_width': 0.30,
    'corpusculum_length': 0.61,
    'corpusculum_width': 0.35,
    'extension': 0.29,
    'translator_arm_length': 0.15,
    'translator_stalk': 0.58,
    'shoulder': 0.20,
    'waist': 0.24,
    'hips': 0.14
}

# Engineer features (see notebook for full function)
X_input = engineer_enhanced_features(pd.DataFrame([measurements]))

# Scale and predict
X_scaled = scaler.transform(X_input)
prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

# Get results
clade = label_encoder.inverse_transform([prediction])[0]
confidence = np.max(probabilities)

print(f"Predicted Clade: {clade}")
print(f"Confidence: {confidence:.1%}")

# Deployment decision
if confidence >= 0.70:
    print("✅ High confidence - auto-accept")
else:
    print("⚠️ Low confidence - flag for expert review")
```

### **Web Interface** (Coming Soon)

```bash
streamlit run app.py
```

---

## 📁 Dataset

### **Composition**

- **Total Specimens**: 64
- **Species Represented**: 40+ Philippine Hoya species
- **Measurement Protocol**: Microscopic pollinarium morphometrics
- **Feature Dimensions**: 10 raw measurements → 13 engineered features

### **Clade Distribution**

| Clade | Specimens | Percentage | Challenge |
|-------|-----------|------------|-----------|
| Hoya | 35 | 54.7% | Majority class |
| Acanthostemma | 24 | 37.5% | Well-represented |
| Pterostelma | 4 | 6.3% | Minority - limited training |
| Centrostemma | 1 | 1.6% | Severe imbalance |

**Note**: Class imbalance (particularly Centrostemma with 1 sample) limits model performance on minority clades.

### **Raw Features Measured**

1. **Pollinia**: length, width
2. **Corpusculum**: length, width
3. **Translator**: arm length, stalk length
4. **Corpusculum anatomy**: shoulder, waist, hips
5. **Extension**: caudicle extension length

---

## 🏗️ Model Architecture

### **Ensemble Components**

```
Soft Voting Ensemble
├── SVM (RBF Kernel)
│   ├── C=1.0
│   ├── gamma='scale'
│   └── class_weight='balanced'
│
├── Gradient Boosting
│   ├── n_estimators=100
│   ├── max_depth=3
│   └── learning_rate=0.1
│
└── Extra Trees
    ├── n_estimators=200
    ├── class_weight='balanced'
    └── random_state=42
```

### **Feature Engineering Pipeline**

```python
Raw Measurements (10)
    ↓
Shape Descriptors
    - Pollinia compactness
    - Corpusculum eccentricity
    ↓
Allometric Features
    - Log-log scaling
    - Biological power laws
    ↓
Mechanical Features
    - Translator leverage
    - Functional morphology
    ↓
Final Feature Set (13)
    ↓
RobustScaler Normalization
    ↓
Ensemble Prediction
```

### **Why Ensemble?**

Individual models (Ridge, XGBoost, GradBoost) all plateau at **71.88% accuracy**, suggesting an algorithmic ceiling. The ensemble breaks through this by:

1. **Error Diversity**: Different models make different mistakes
2. **Probability Averaging**: Smooths uncertainty
3. **Complementary Strengths**: 
   - SVM: Complex decision boundaries
   - GradBoost: Minority class handling (best Kappa)
   - ExtraTrees: Ensemble diversity

---

## 💻 Usage

### **Training a New Model**

```python
# See Hoya_Clade_Classifier_Enhanced.ipynb for full training pipeline

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

# Define ensemble
ensemble = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')),
        ('gradboost', GradientBoostingClassifier(n_estimators=100, max_depth=3)),
        ('extratrees', ExtraTreesClassifier(n_estimators=200, class_weight='balanced'))
    ],
    voting='soft'
)

# Train with LOOCV for small datasets
from sklearn.model_selection import LeaveOneOut
# See notebook for full LOOCV implementation
```

### **Inference**

```python
# Load production model
model_package = pickle.load(open('hoya_clade_classifier_production.pkl', 'rb'))

# Predict
def classify_specimen(measurements):
    X = engineer_features(measurements)
    X_scaled = model_package['scaler'].transform(X)
    
    pred = model_package['model'].predict(X_scaled)[0]
    proba = model_package['model'].predict_proba(X_scaled)[0]
    
    clade = model_package['label_encoder'].inverse_transform([pred])[0]
    confidence = max(proba)
    
    return {
        'clade': clade,
        'confidence': confidence,
        'all_probabilities': dict(zip(model_package['metadata']['classes'], proba))
    }
```

### **Deployment Strategy**

**Confidence-Based Workflow:**

```python
result = classify_specimen(measurements)

if result['confidence'] >= 0.70:
    # High confidence - accept automatically
    print(f"✅ Classification: {result['clade']} ({result['confidence']:.1%})")
    
elif result['confidence'] >= 0.50:
    # Medium confidence - show probabilities, request verification
    print(f"⚠️ Uncertain: {result['clade']} ({result['confidence']:.1%})")
    print("Top 3 predictions:", sorted(result['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True)[:3])
    print("→ Recommend expert review")
    
else:
    # Low confidence - flag for expert
    print(f"❌ Low confidence - Expert review required")
```

---

## 📈 Results

### **Validation Methodology**

- **Cross-Validation**: Leave-One-Out (LOOCV)
  - Rationale: Small dataset (N=64) requires maximum data utilization
  - Each specimen used as test once, trained on remaining 63
  
- **Metrics**:
  - **Accuracy**: Overall correctness
  - **Cohen's Kappa**: Agreement beyond chance (accounts for class imbalance)
  - **Precision/Recall**: Per-clade performance

### **Baseline Comparison**

| Approach | This Study | Related Work |
|----------|------------|--------------|
| **Target** | Pollinaria (clades) | Leaves (species) |
| **Dataset Size** | 64 specimens | 700+ images |
| **Accuracy** | **75.0%** | 76.1% (Fitriyah 2023) |
| **Innovation** | **First pollinaria ML** | Leaf CNN |

**Key Distinction**: This is the **first automated pollinaria classification system** - no prior SOTA exists for comparison.

### **Biological Insights**

1. **Golden Trio Validated**: Feature importance confirms pollinia length/width and corpusculum length are most discriminative

2. **Morphological Convergence**: High-confidence errors reveal genuine biological phenomena (e.g., Tsangii Paradox - Acanthostemma mimicking Hoya)

3. **Class Imbalance Impact**: Performance ceiling (~75%) partly due to:
   - Centrostemma: 1 sample (impossible to learn)
   - Pterostelma: 4 samples (insufficient variance)

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{bongalos2026hoya,
  title={Deployable AI for Rapid Morphometric Classification of Philippine Hoya Clades},
  author={Bongalos, Jerald B.},
  booktitle={National Summit on Botanic Gardens and Arboreta},
  year={2026},
  organization={Philippine Botanic Gardens and Arboreta Association}
}
```

**Conference Presentation**: May 25-29, 2026

---

## 🛠️ Repository Structure

```
HOYA-FLWR-AI/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── Hoya_Clade_Classifier_Enhanced.ipynb        # Training notebook
├── hoya_clade_classifier_production.pkl        # Pre-trained model
├── data/
│   └── (dataset files - contact for access)
├── visualizations/
│   ├── model_evaluation_dashboard.png
│   └── feature_importance.png
├── deployment/
│   ├── app.py                                  # Streamlit web app (coming soon)
│   └── api.py                                  # FastAPI endpoint (coming soon)
└── docs/
    ├── extended_abstract.pdf
    └── presentation_slides.pptx
```

---

## 🔮 Future Work

### **Near-Term (2026)**

1. **Expand Dataset**: Target 100+ specimens
   - Focus on underrepresented clades (Centrostemma, Pterostelma)
   - Add more species per clade

2. **Species-Level Classification**: 
   - Current: Clade-level (4 classes)
   - Goal: Species-level (40+ classes)

3. **Web Deployment**:
   - Streamlit interface for botanic gardens
   - Mobile-friendly responsive design
   - Real-time prediction API

### **Long-Term**

4. **Multimodal Integration**:
   - Combine pollinaria + leaf morphology
   - Add molecular markers (ITS, matK)
   - Geographic origin features

5. **Transfer Learning**:
   - Pre-train on broader Apocynaceae family
   - Fine-tune for rare species

6. **Conservation Applications**:
   - Automated herbarium digitization
   - Field identification app
   - Illegal trade monitoring

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **Data Collection**: Additional specimen measurements
- **Model Improvements**: Novel architectures, hyperparameter tuning
- **Deployment**: Web interfaces, mobile apps
- **Documentation**: Tutorials, use cases

---

## 📧 Contact

**Jerald B. Bongalos**  
PhD Candidate, Data Science  
Asian Institute of Management  
Email: [Your Email]  
GitHub: [@Jbong17](https://github.com/Jbong17)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Advisers**: Asian Institute of Management Faculty
- **Co-advisers**: National University Manila (Computer Engineering theses)
- **Collaborators**: Philippine National Research Institute (PNRI)
- **Specimens**: Philippine botanic gardens and herbaria
- **Conference**: National Summit on Botanic Gardens and Arboreta 2026

---

## 📊 Model Metadata

```json
{
  "model_type": "Soft Voting Ensemble",
  "training_date": "2026-04-29",
  "n_samples": 64,
  "n_features": 13,
  "classes": ["Acanthostemma", "Centrostemma", "Hoya", "Pterostelma"],
  "loocv_accuracy": 0.75,
  "cohens_kappa": 0.531,
  "confidence_threshold": 0.70,
  "ensemble_components": "SVM + Gradient Boosting + Extra Trees"
}
```

---

**⭐ Star this repo if you found it useful!**

**🌺 Preserving Philippine biodiversity through AI**
