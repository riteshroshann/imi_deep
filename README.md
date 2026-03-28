# AI-Driven Property Optimization of Carbon Fiber Reinforced Polymer Composites

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A publication-ready research pipeline for multi-modal deep learning, physics-informed neural networks, and Bayesian optimization applied to CFRP composite structural health monitoring using the NASA PCoE dataset.

## Dataset

**NASA CFRP Composites Dataset (PCoE)**
- **Source:** [NASA Prognostics Data Repository](https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip)
- **Content:** Run-to-failure experiments on CFRP panels under tension-tension fatigue
- **Sensors:** 16 PZT piezoelectric sensors (Lamb wave signals), triaxial strain gages
- **Layups:** 3 distinct CFRP configurations

> **Citation:** Saxena A., Goebel K., Larrosa C.C., Chang F-K., "CFRP Composites Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/riteshroshann/nasa_dl-imi_cw.git
cd nasa_dl-imi_cw

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (downloads data automatically)
python main.py --data_path ./data --mode full_pipeline
```

## Pipeline Modes

| Mode | Description |
|------|-------------|
| `full_pipeline` | Run all stages end-to-end |
| `data_only` | Download + feature extraction only |
| `baselines` | Data + baseline ML models |
| `deep_learning` | Data + baselines + DL models |
| `pinn` | Data through PINN stage |
| `visualization` | Regenerate all figures |

## Project Structure

```
cfrp_project/
├── main.py                      ← Single entry point
├── requirements.txt
├── data/
│   ├── raw/                     ← NASA zip extracted here
│   └── processed/               ← Feature matrices, labels
├── src/
│   ├── data_loader.py           ← NASA data parser + synthetic fallback
│   ├── feature_extraction.py    ← DWT, ToF, DI, cross-correlation
│   ├── visualization.py         ← All 15 publication figures
│   ├── explainability.py        ← SHAP, Grad-CAM, attention analysis
│   ├── uncertainty.py           ← MC Dropout, Conformal Prediction
│   ├── optimization.py          ← Multi-objective Bayesian optimization
│   └── models/
│       ├── cnn1d.py             ← 1D-CNN (Conv1D×4 → GAP → Dense)
│       ├── bilstm.py            ← BiLSTM with attention gate
│       ├── tcn.py               ← Temporal Convolutional Network
│       ├── transformer.py       ← Multi-head sensor attention
│       ├── pinn.py              ← Physics-Informed NN (Paris Law)
│       └── ensemble.py          ← Stacked ensemble (Ridge meta-learner)
├── results/
│   ├── figures/                 ← All 15 plots (300 DPI)
│   └── tables/                  ← CSV metric tables
└── paper/
    └── paper.tex                ← IEEE conference paper
```

## Models Implemented

| # | Model | Architecture | Task |
|---|-------|-------------|------|
| 1 | Linear Regression | Baseline | RUL regression |
| 2 | Random Forest | 200 estimators, depth=15 | RUL regression |
| 3 | XGBoost | 300 trees, lr=0.05, depth=8 | RUL regression |
| 4 | 1D-CNN | Conv1D×4 → BN → MaxPool → GAP → Dense | RUL + Classification |
| 5 | BiLSTM | 2-layer BiLSTM, hidden=128, attention gate | RUL + Classification |
| 6 | TCN | 4 dilated causal conv blocks, residual | RUL + Classification |
| 7 | Transformer | 3-layer encoder, 4 heads, d=128 | RUL + Classification |
| 8 | PINN | Paris Law constraint, CDM coupling | Property degradation |
| 9 | Ensemble | Stacked (CNN+LSTM+Transformer) + Ridge | RUL regression |

## Key Features

- **Multi-modal feature engineering**: ToF, DWT (db4, 5 levels), signal energy, cross-correlation, Damage Index
- **Physics-informed learning**: Paris Law (da/dN = C·ΔK^m) embedded as loss constraint
- **Uncertainty quantification**: MC Dropout + Conformal Prediction with coverage guarantees
- **Explainability**: SHAP, Grad-CAM, Transformer attention maps
- **Bayesian optimization**: Multi-objective (RUL × stiffness × strength) with Pareto front
- **Reproducibility**: seed=42 everywhere, deterministic training

## Reproducibility

All random seeds are fixed to `42`. Set `PYTHONHASHSEED=42` for full determinism:

```bash
PYTHONHASHSEED=42 python main.py --mode full_pipeline
```

## License

MIT License. See [LICENSE](LICENSE) for details.
