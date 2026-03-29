# AI-Driven Property Optimization of CFRP Composites
### *Physics-Informed Deep Learning for High-Performance Materials Discovery*

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Material_Informatics-State_of_the_Art-blue?style=flat-square" alt="SOTA"/>
    <img src="https://img.shields.io/badge/Status-Publication_Ready-success?style=flat-square" alt="Status"/>
  </p>
</div>

---

### **Authors & Institution**
**Ritesh Roshan Sahoo** (DL.AI.U4AID24036), **Arushi Uppal** (DL.AI.U4AID24009), **Arnav Sharma** (DL.AI.U4AID24006), **Rudru Mahima** (DL.AI.U4AID24037)  
*School of AI, Amrita Vishwa Vidyapeetham, Delhi NCR*

---

## **Overview**
This repository introduces a unified Material Informatics pipeline bridging **Active Lamb Wave Propogation** and **Physics-Informed Neural Networks (PINNs)**. By distilling 4.6 GB of raw 16-channel waveforms into an 11.5 MB dense feature matrix, we achieve real-time fatigue diagnostics and autonomous layup discovery for aerospace composites.

<div align="center">
  <table>
    <tr>
      <td><img src="results/figures/fig11_shap_xgboost.png" width="350"/><br/><sub><b>XAI Performance Metrics</b></sub></td>
      <td><img src="results/figures/fig10_attention_heatmap.png" width="350"/><br/><sub><b>Transformer Attention Grid</b></sub></td>
    </tr>
    <tr>
      <td><img src="results/figures/fig09_pinn_degradation.png" width="350"/><br/><sub><b>PINN Monotonic Constraint</b></sub></td>
      <td><img src="results/figures/fig13_pareto_front.png" width="350"/><br/><sub><b>Bayesian Pareto Front</b></sub></td>
    </tr>
  </table>
</div>

## **Research Benchmarks**
- **SOTA Accuracy**: 23% RMSE reduction via Physics-Informed *Paris Law* regularizers.
- **Domain Mapping**: Self-Attention weights successfully trace physical $4 \times 4$ sensor grid decoupling.
- **Inverse Design**: Multi-objective Bayesian Optimization identified $[0/45/90/-45]_{2s}$ as the Pareto-optimal layup.
- **Topological Distillation**: 4.6 GB waveforms distilled to 11.5 MB physics-invariant sensors.

## **Quick Start**
```bash
# Setup
git clone https://github.com/riteshroshann/nasa_dl-imi_cw.git
pip install -r requirements.txt

# Run Pipeline
python main.py --mode full_pipeline
```

## **Resources**
- 📄 **[Final Research Paper (PDF)](paper/paper_for_IMI_deep_learning.pdf)**
- 📓 **[Interactive Colab Explorer](notebooks/colab_dataset_explorer.ipynb)**
- ⚛️ **[Physics-Ready Dataset](dataset/features.npz)**
- 📦 **[Implementation Source](src/models/)**

---
*Developed for the Introduction to Material Informatics & Deep Learning Curriculum.*
