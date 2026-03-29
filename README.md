# AI-Driven Property Optimization of CFRP Composites

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Material_Informatics-State_of_the_Art-blue?style=flat-square" alt="SOTA"/>
    <img src="https://img.shields.io/badge/Status-Publication_Ready-success?style=flat-square" alt="Status"/>
  </p>
</div>

### **Authors**
**Ritesh Roshan Sahoo** (DL.AI.U4AID24036), **Arushi Uppal** (DL.AI.U4AID24009), **Arnav Sharma** (DL.AI.U4AID24006), **Rudru Mahima** (DL.AI.U4AID24037)  
*School of AI, Amrita Vishwa Vidyapeetham, Delhi NCR*

---

## **Abstract**
This framework presents a unified paradigm for the **Prognostics and Health Management (PHM)** of Carbon Fiber Reinforced Polymer (CFRP) composites. By distilling 4.6 GB of raw high-frequency Lamb wave waveforms into a 10 MB physics-invariant tensor, we enable real-time fatigue diagnostics and autonomous layup optimization. 

Our core innovation integrates **Physics-Informed Neural Networks (PINNs)** constrained by the *Paris Law* of crack kinetics with **Multi-Head Self-Attention Transformers** for spatial-temporal damage localization.

## **Pipeline Architecture**
The autonomous research pipeline operates through 9 integrated stages:
1.  **Acoustic Distillation**: ToF, DWT (db4), and Damage Index engineering.
2.  **Baseline Benchmarking**: Gradient Boosting (XGB/LGBM) performance envelopes.
3.  **Deep Sequence Modeling**: Evaluative comparison of TCN, BiLSTM, and Transformers.
4.  **Physics Regularization**: PINN loss manifold enforcing monotonic stiffness decay.
5.  **Uncertainty Quantification**: Conformal Prediction establishing 90% survival coverage.
6.  **Explainability (XAI)**: SHAP and Grad-CAM 1D visualization of delamination paths.
7.  **Bayesian Discovery**: Multi-objective Pareto optimization for layup configuration.
8.  **Publication Graphics**: Generation of 15 IEEE-formatted vector visualizations.
9.  **Manuscript Synthesis**: Automated drafting of the research paper results.

## **Key Technical Benchmarks**
- **Precision**: 23% reduction in terminal-epoch RUL RMSE via PINN kinetic constraints.
- **Explainability**: Transformer attention weights successfully map physical $4 \times 4$ sensor grid decoupling.
- **Optimization**: Autonomous identification of quasi-isotropic $[0/45/90/-45]_{2s}$ as the global Pareto-optimal solution.
- **Efficiency**: 4.6 GB raw archive distilled to 11.5 MB dense feature matrix.

## **Quick Start**
```bash
# 1. Setup
git clone https://github.com/riteshroshann/nasa_dl-imi_cw.git
pip install -r requirements.txt

# 2. Execute Full Pipeline
python main.py --mode full_pipeline
```

## **Resources**
- 📄 **[Final Research Paper (PDF)](paper/paper_for_IMI_deep_learning.pdf)**
- 📓 **[Interactive Colab Explorer](notebooks/colab_dataset_explorer.ipynb)**
- ⚛️ **[Physics-Ready Dataset](dataset/features.npz)**

---
*Developed for the Introduction to Material Informatics & Deep Learning Curriculum.*
