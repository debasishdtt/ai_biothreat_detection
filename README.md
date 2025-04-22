# AI-Driven Early Detection of Biological Threats Using Public Health Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for **early detection of biological threats** (e.g., SARS-CoV-2 variants) by fusing heterogeneous public health data streams:

- **Genomic surveillance** (PANGO lineages, variant scores)
- **Epidemiological trends** (cases, hospitalizations, Rₜ)
- **Population mobility** (retail/recreation anomalies)

Developed as part of a Master's thesis at Frankfurt University of Applied Sciences, this system achieves **14-day early warnings** with 97% precision, enabling proactive resource allocation for health agencies.

---

## 🔑 Key Innovations

- **Multimodal Fusion**: Hybrid CNN (genomics) + BiLSTM (mobility) + Dense (epidemiology) with attention gating
- **Class Imbalance Mitigation**: SMOTE-ENN resampling + adaptive focal loss (γ = 2.2)
- **Explainable AI**: SHAP-driven thresholds (e.g., mobility ΔM < −15%)
- **Temporal Alignment**: ISO-week synchronization to resolve genomic reporting lags (Δt = 14 ± 5 days)

---

## 🏆 Performance Highlights

| Metric    | Proposed Model | Best Baseline (XGBoost) |
| --------- | -------------- | ----------------------- |
| Precision | 0.94           | 0.89                    |
| Recall    | 0.96           | 0.71                    |
| F1-Score  | 0.95           | 0.71                    |
| AUC-PR    | 0.97           | 0.89                    |
| MCC       | 0.77           | 0.63                    |

_Validated on 49 weeks of U.S. data (6,305 samples) with 3-fold time-series splits._

---

## ⚙️ Installation

```bash
git clone https://github.com/debasishdtt/ai_biothreat_detection.git
cd ai_biothreat_detection
pip install -r requirements.txt

Required Libraries:

TensorFlow 2.15.0

SHAP 0.44.0

imbalanced-learn 0.11.0

scikit-learn 1.3.2

.
├── data/
│   ├── raw/                  # Raw datasets (CDC, GISAID, Google Mobility)
│   └── processed/            # Preprocessed features (VariantScore, EMA-smoothed mobility)
├── models/                   # Pretrained model weights (HDF5 format)
├── scripts/
│   ├── preprocessing/        # Data pipelines
│   │   ├── preprocess_epidemiological.py
│   │   ├── preprocess_genomic.py
│   │   └── preprocess_mobility.py
│   ├── training/             # Model training logic
│   └── deployment/           # TensorFlow Lite conversion
├── notebooks/                # SHAP analysis notebooks
└── requirements.txt          # Dependency list

🧠 Model Training & Inference
Preprocess Data

# Epidemiological data
python scripts/preprocessing/preprocess_epidemiological.py

# Genomic VariantScores
python scripts/preprocessing/preprocess_genomic.py

# Mobility anomalies
python scripts/preprocessing/preprocess_mobility.py


Train the Model

python scripts/training/train_model.py \
  --epochs 25 \
  --batch_size 32 \
  --focal_gamma 2.2

Generate Predictions

python scripts/inference/predict_threats.py \
  --model_path models/production_model.h5 \
  --data_dir data/processed/


🌐 Data Sources
Genomic: GISAID PANGO lineage data; VariantScore (gradient-based)

Epidemiological: CDC case/hospitalization reports; 7-day forecast label

Mobility: Google Mobility Reports; baseline-corrected anomalies

🔍 Model Interpretability & Limitations
SHAP-Driven Insights:

Key indicators: variant B.1.2 prevalence (>18%), mobility ΔM < −15%

Early detection: alerts precede case spikes by ~10 days (r = −0.67)

python scripts/analysis/explain_model.py --sample_size 50 --output_dir shap_plots/


Limitations:

81% of training data from urban counties

Genomic reporting lag: Δt = 14 ± 5 days

CPU-only training

🚧 Future Enhancements
Temporal Adaptation: Integrate vaccination data from CDC

Global Deployment: Tune thresholds for EU (e.g., ΔM < −10%)

Edge Optimization: TFLite quantization (FP16 → INT8)

🤝 Contributing
Fork this repo

Create a branch: git checkout -b feature/my-feature

Submit a pull request

📄 License
Distributed under the MIT License.

📧 Contact
Debasish Dutta
📫 debasish.dutta@stud.fra-uas.de
```
