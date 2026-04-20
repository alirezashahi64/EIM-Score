#!/usr/bin/env python
"""
AUC evaluation for binding site similarity
"""

import os
import numpy as np
import itertools
import random
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/Desktop/Refined_EIM")

SIM_FILE = os.path.join(BASE_DIR, "similarities", "cosine.npy")
FEATURE_FILE = os.path.join(BASE_DIR, "features", "train_features_local.npz")

INDEX_FILE = os.path.join(
    BASE_DIR,
    "pdbbind_v2016_general-set/general-set/index/INDEX_general_PL.2016"
)

OUT_DIR = os.path.join(BASE_DIR, "auc_results")
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD ===
data = np.load(FEATURE_FILE, allow_pickle=True)
pdb_ids = data["pdbids"]
sim = np.load(SIM_FILE)

id_to_idx = {pid:i for i,pid in enumerate(pdb_ids)}

# === LIGAND MAP ===
lig_map = {}
with open(INDEX_FILE, 'r', encoding='latin-1') as f:
    for line in f:
        if line.startswith("#"): continue
        parts = line.split()
        if not parts: continue
        pid = parts[0]
        if "(" in line:
            lig = line.split("(")[1].split(")")[0]
            lig_map[pid] = lig

# === POSITIVE / NEGATIVE PAIRS ===
lig_groups = {}
for pid, lig in lig_map.items():
    if pid in id_to_idx:
        lig_groups.setdefault(lig, []).append(pid)

pos = []
for pids in lig_groups.values():
    if len(pids) > 1:
        pos += list(itertools.combinations(pids, 2))

neg = []
all_ids = list(id_to_idx.keys())

while len(neg) < len(pos):
    a, b = random.sample(all_ids, 2)
    if lig_map.get(a) != lig_map.get(b):
        neg.append((a, b))

# === SCORES ===
pairs = pos + neg
labels = [1]*len(pos) + [0]*len(neg)

scores = [sim[id_to_idx[a], id_to_idx[b]] for a,b in pairs]

# === AUC ===
auc = roc_auc_score(labels, scores)
print(f"AUC: {auc:.4f}")

# === ROC CURVE ===
fpr, tpr, _ = roc_curve(labels, scores)

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()

plt.savefig(os.path.join(OUT_DIR, "roc.png"))
plt.show()
