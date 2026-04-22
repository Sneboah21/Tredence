# Tredence
# 🧠 Self-Pruning Neural Network — CIFAR-10

**Tredence Analytics | AI Engineering Internship 2025 Cohort**

A neural network that **learns to prune itself during training** using learnable sigmoid gates, L1 sparsity regularization, and the Straight-Through Estimator (STE).

---

## 🔬 Method

Each weight in the classifier layers is multiplied by a **learnable gate** ∈ (0, 1):

- **Forward pass** — Hard binary mask via STE: `gate ≥ 0.5 → 1, else → 0` (true structural zeros)
- **Backward pass** — Gradients flow through smooth sigmoid (STE trick — maintains differentiability)
- **L1 penalty** on gate values pushes scores toward −∞ → sigmoid → 0 → weight is pruned
  
Total Loss = CrossEntropy(label_smoothing=0.1) + λ × mean(sigmoid(gate_scores))

---

## 📊 Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) | Weights Retained (%) |
|:----------:|:-----------------:|:------------:|:--------------------:|
| 0.001 | **89.53** | 71.03 | 28.97% |
| 0.01 | **89.51** | 77.64 | 22.36% |
| 0.05 | **89.34** | 85.47 | 14.53% |

> 💡 **85.47% of linear layer weights zeroed out** — with only **0.19% accuracy cost** vs baseline.

---

## 🏗️ Architecture
CNN Backbone (not pruned):
Conv2d 3→64 → BN → ReLU ×2 → MaxPool → Dropout2d(0.15)
Conv2d 64→128 → BN → ReLU ×2 → MaxPool → Dropout2d(0.20)
Conv2d 128→256 → BN → ReLU ×2 → AdaptiveAvgPool(4×4) → 4096-dim

Classifier (pruned):
Dropout(0.4) → PrunableLinear(4096→512) → BN1d → ReLU
Dropout(0.3) → PrunableLinear(512→10)

---

## ⚙️ Setup & Usage

### 1. Install dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 2. Run training (all 3 lambda values, 30 epochs each)

```bash
python solution.py
```

> CIFAR-10 will be auto-downloaded to `./data/` on first run.

### 3. Outputs saved to `./output/`
output/
├── training_curves.png # Test accuracy, sparsity & loss over epochs
├── gate_distributions.png # Bimodal gate histograms for each λ
└── results_summary.json # Final metrics for all 3 λ runs

---

## 🔧 Key Hyperparameters

| Parameter | Value | Note |
|:----------|:-----:|:-----|
| Epochs | 30 | All λ runs — fair comparison |
| Batch size | 128 | Standard CIFAR-10 |
| LR (weights) | 1e-3 | AdamW |
| LR (gate_scores) | 5e-3 | 5× higher — gates respond faster to L1 |
| Lambdas (λ) | 0.001, 0.01, 0.05 | Controls sparsity level |
| Prune threshold | 0.5 | sigmoid(gate_score) < 0.5 → masked to 0 |
| Seed | 42 | Fully reproducible |

---

## 💻 Hardware

Trained on **CUDA GPU**. Each λ run takes ~11 minutes (30 epochs × ~23s/epoch).
CPU training is supported but will be significantly slower.

---

*Submitted for Tredence Analytics — AI Engineering Internship 2025 Cohort*
