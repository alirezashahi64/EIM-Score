#!/usr/bin/env python
"""
Compute similarity matrices from EIM features
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.expanduser("~/Desktop/Refined_EIM")
FEATURE_FILE = os.path.join(BASE_DIR, "features", "train_features_local.npz")  # change if needed
OUT_DIR = os.path.join(BASE_DIR, "similarities")
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(FEATURE_FILE, allow_pickle=True)
X = data["features"]

# === Cosine ===
cosine = cosine_similarity(X)

# === Tanimoto & Dice ===
dot = X @ X.T
norm2 = np.sum(X**2, axis=1)

eps = 1e-10
tanimoto = dot / (norm2[:,None] + norm2[None,:] - dot + eps)
dice     = 2*dot / (norm2[:,None] + norm2[None,:] + eps)

np.save(os.path.join(OUT_DIR, "cosine.npy"), cosine)
np.save(os.path.join(OUT_DIR, "tanimoto.npy"), tanimoto)
np.save(os.path.join(OUT_DIR, "dice.npy"), dice)

print("✅ Similarities saved")
