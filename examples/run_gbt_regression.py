#!/usr/bin/env python
"""
GBT Regression for EIM Features (local / global / hybrid)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

# === CONFIG ===
BASE_DIR = os.path.expanduser("~/Desktop/Refined_EIM")

MODE = "local"  # change to: global / hybrid

TRAIN_NPZ = os.path.join(BASE_DIR, "features", f"train_features_{MODE}.npz")
TEST_NPZ  = os.path.join(BASE_DIR, "features", f"test_features_{MODE}.npz")

OUT_DIR = os.path.join(BASE_DIR, f"gbt_results_{MODE}")
os.makedirs(OUT_DIR, exist_ok=True)

GBT_PARAMS = dict(
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=7,
    min_samples_split=3,
    subsample=0.3,
    max_features='sqrt',
    random_state=42
)

# === LOAD DATA ===
train = np.load(TRAIN_NPZ, allow_pickle=True)
test  = np.load(TEST_NPZ, allow_pickle=True)

X_train, y_train = train['features'], train['labels']
X_test,  y_test  = test['features'],  test['labels']
feature_names = train['feature_names']

X_train = np.nan_to_num(X_train)
X_test  = np.nan_to_num(X_test)

# === CROSS VALIDATION ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for i, (tr, val) in enumerate(kf.split(X_train), 1):
    model = GradientBoostingRegressor(**GBT_PARAMS)
    model.fit(X_train[tr], y_train[tr])
    
    pred = model.predict(X_train[val])
    rmse = np.sqrt(mean_squared_error(y_train[val], pred))
    r, _ = pearsonr(y_train[val], pred)
    
    print(f"Fold {i}: RMSE={rmse:.4f}, R={r:.4f}")
    cv_results.append((rmse, r))

# === TRAIN FINAL MODEL ===
model = GradientBoostingRegressor(**GBT_PARAMS)
model.fit(X_train, y_train)

# === TEST EVALUATION ===
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae  = mean_absolute_error(y_test, pred)
r2   = r2_score(y_test, pred)
r, _ = pearsonr(y_test, pred)

print("\n=== TEST RESULTS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")
print(f"R   : {r:.4f}")

# === SAVE ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

pd.DataFrame({
    "PDBID": test['pdbids'],
    "true": y_test,
    "pred": pred
}).to_csv(os.path.join(OUT_DIR, f"pred_{MODE}_{timestamp}.csv"), index=False)

pd.DataFrame([{
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2,
    "PearsonR": r
}]).to_csv(os.path.join(OUT_DIR, f"metrics_{MODE}_{timestamp}.csv"), index=False)
