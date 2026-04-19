
# EIM-Score

This repository provides an implementation of Element Interactive Manifold (EIM) features for protein–ligand binding affinity prediction.

---

## 📂 Repository Structure

There are three main folders:

* **src/**
  Contains the core source code for EIM feature extraction.
  This includes:

  * `eim_score_global.py`
  * `eim_score_local.py` 
  * `element_interactive_density.py`
  * `numba_utils_methods.py`

* **examples/**
  Contains scripts to generate features for an entire dataset.

* **utils/**
  Contains dataset CSV files with `PDBID` and `pK` values:

  * `PDBbindv2016_RefinedSet.csv`
  * `CASF_2016_CoreSet.csv`

* **features/**
  Output directory where generated features will be stored.

---

## ⚙️ Description

The EIM method computes geometric and topological features based on element-specific interactions between protein and ligand atoms.

The feature extraction involves:

* evaluation of atomic density functions on 3D grids
* surface area and volume calculations
* curvature-based descriptors

⚠️ These computations are **computationally expensive**, and parallel processing is recommended.

---

## 🚀 Current Implementation

At this stage, the repository includes:

* ✅ Global feature extraction (implemented)
* ⏳ Local feature extraction (to be added)
* ⏳ Hybrid features (to be added)

---

## 🚀 Generate Features

### Global Features

Run:

```bash
python examples/get_eim_features_global.py \
  --dataset_csv_file utils/PDBbindv2016_RefinedSet.csv \
  --data_folder PATH_TO_PDBBIND \
  --out_dir features \
  --kernel_type exponential \
  --kernel_tau 1.0 \
  --kernel_power 3.0 \
  --cutoff 12.0 \
  --pdbid_index 0
```

---

### Local Features (Coming Soon)

```bash
python examples/get_eim_features_local.py ...
```

---

### Hybrid Features (Coming Soon)

```bash
python examples/get_eim_features_hybrid.py ...
```

---

## ⚠️ Notes

* To process the full dataset, run the script for:

  ```
  --pdbid_index = 0 → N-1
  ```
* Checkpointing is included for long runs
* Output includes:

  * `.npz` files (train/test)
  * `.csv` combined dataset

---

## 📌 Future Work

* Add local EIM features
* Add hybrid (local + global) features
* Add parallel and SLURM support

---

