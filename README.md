# EIM-Score

This repository provides an implementation of **Element Interactive Manifold (EIM)** features for:

* Protein–ligand **binding affinity prediction** (regression)
* Binding site **similarity analysis** (AUC evaluation)

The framework supports **global**, **local**, and **hybrid** feature representations, along with complete machine learning pipelines.

---

## 📂 Repository Structure

```
EIM-Score/
│
├── examples/              # End-to-end pipelines (feature extraction + ML)
│   ├── get_eim_features_global.py
│   ├── get_eim_features_local.py
│   ├── get_eim_features_hybrid.py
│   ├── run_gbt_regression.py
│   ├── compute_similarity.py
│   └── run_auc_evaluation.py
│
├── src/                   # Core EIM implementation
│   ├── eim_combine_score_global_surface.py
│   ├── eim_combine_score_local_surface.py
│   ├── element_interactive_density.py
│   ├── element_interactive_curvature.py
│   └── numba_utils_methods.py
│
├── utils/                 # Dataset metadata
│   ├── PDBbindv2016_RefinedSet.csv
│   ├── CASF_2016_CoreSet.csv
│   └── binding_data.csv
│
├── features/              # Generated feature outputs (.npz / .csv)
│
└── README.md
```

---

## ⚙️ Method Overview

EIM features are constructed using **element-specific interactions** between protein and ligand atoms.

The pipeline includes:

* Atomic density evaluation on 3D grids
* Isosurface-based **surface area** and **volume** computation
* Curvature-based descriptors:

  * Mean curvature (H)
  * Gaussian curvature (K)
  * Principal curvatures

These features encode **geometric, topological, and physicochemical interactions**.

⚠️ Feature extraction is **computationally expensive**. Parallel execution (e.g., HPC/SLURM) is recommended.

---

## 🚀 Feature Extraction

All feature extraction scripts are located in `examples/`.

### Global Features

```
python examples/get_eim_features_global.py
```

### Local Features

```
python examples/get_eim_features_local.py
```

### Hybrid Features (Global + Local)

```
python examples/get_eim_features_hybrid.py
```

---

### 📌 Output

Each script generates:

* `train_features_*.npz`
* `test_features_*.npz`
* optional combined `.csv`

with CASF-2016 split:

* Train: Refined set − Core set (~3772 samples)
* Test: Core set (285 samples)

---

## 🤖 Machine Learning (Regression)

Train and evaluate **Gradient Boosting Trees (GBT)**:

```
python examples/run_gbt_regression.py
```

Select feature type inside the script:

```
MODE = "local"   # or "global" or "hybrid"
```

### Output

* Test predictions
* RMSE, MAE, R², Pearson correlation
* Feature importance
* Results saved in `gbt_results_*`

---

## 🔬 Binding Site Similarity (AUC)

### Step 1: Compute Similarity Matrices

```
python examples/compute_similarity.py
```

Generates:

* Cosine similarity
* Tanimoto similarity
* Dice similarity

---

### Step 2: AUC Evaluation

```
python examples/run_auc_evaluation.py
```

### Output

* ROC curve
* AUC score
* Results saved in `auc_results/`

---

## 📊 Dataset

This implementation uses:

* **PDBbind v2016 Refined Set**
* **CASF-2016 Core Set**

Expected directory structure:

```
pdbbind_v2016_general-set/
└── general-set/
    ├── XXXX/
    │   ├── XXXX_protein.pdb
    │   └── XXXX_ligand.sdf
```

---

## ⚠️ Notes

* Feature extraction includes **checkpointing** for long computations
* Missing or invalid structures are automatically skipped
* NaN/Inf values are handled during processing
* Recommended to run on **cluster environments** for large-scale datasets

---

## 📌 Summary of Capabilities

| Task               | Method                      |
| ------------------ | --------------------------- |
| Feature Extraction | Global / Local / Hybrid EIM |
| Regression         | Gradient Boosting Trees     |
| Similarity         | Cosine / Tanimoto / Dice    |
| Evaluation         | RMSE, Pearson R, AUC        |

---

## 🔧 Future Improvements

* Parallel batch processing (SLURM integration)
* Deep learning models on EIM features
* Feature selection and dimensionality reduction
* Integration with docking pipelines

---



---

## ✍️ Citation

(to be added).

---
