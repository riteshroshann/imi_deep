"""
diagnose.py — understand the real NASA data before fixing training
"""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

# ── 1. Check raw parquet directly ────────────────────────────────────────────
parquet_dir = r"C:\Users\rites\.gemini\antigravity\scratch\parsed_cfrp"
df = pd.read_parquet(os.path.join(parquet_dir, "pzt_waveforms.parquet"))

SCALAR_FEATS = [
    "amplitude_max", "amplitude_min", "amplitude_pp",
    "rms", "energy", "mean", "std", "variance",
    "skewness", "kurtosis", "zero_crossing_rate",
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "envelope_max", "envelope_mean", "toa_index",
]
present = [c for c in SCALAR_FEATS if c in df.columns]

print("=== RAW PARQUET ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Feature cols present: {present}")
print(f"\nSample values (first row):")
print(df[present].head(3).to_string())
print(f"\nFeature statistics:")
print(df[present].describe().to_string())

# ── 2. Check data loader output ───────────────────────────────────────────────
print("\n\n=== DATA LOADER OUTPUT ===")
from src.data_loader import parse_nasa_composites, PARSED_DATA_DIR

ds = parse_nasa_composites(str(PARSED_DATA_DIR), seed=42)
signals = ds['signals']   # (N, 17, 16) hopefully
rul     = ds['rul']

print(f"Signals shape: {signals.shape}")
print(f"Signals dtype: {signals.dtype}")
print(f"Signals global: mean={signals.mean():.6f}  std={signals.std():.6f}")
print(f"Signals range: [{signals.min():.6f}, {signals.max():.6f}]")

# Per-channel stats (17 feature channels across 1319 samples × 16 sensors)
print(f"\nPer-channel (feature) stats (min std, mean std over channels):")
per_ch_std = signals.std(axis=(0, 2))  # std over samples & sensor dim
print(f"  STDs: {per_ch_std}")

print(f"\nRUL: range=[{rul.min():.4f},{rul.max():.4f}]  mean={rul.mean():.4f}  std={rul.std():.4f}")
print(f"RUL variance: {rul.var():.6f}")
print(f"Quantiles(0,25,50,75,100): {np.percentile(rul,[0,25,50,75,100])}")

# ── 3. Are features correlated with RUL? ─────────────────────────────────────
print("\n\n=== FEATURE-RUL CORRELATION ===")
flat = signals.reshape(len(signals), -1)   # (N, 272)
corrs = np.array([np.corrcoef(rul, flat[:, i])[0, 1] for i in range(flat.shape[1])])
corrs = np.nan_to_num(corrs)
print(f"Max |corr| with RUL: {np.abs(corrs).max():.4f}")
print(f"Mean |corr| with RUL: {np.abs(corrs).mean():.4f}")
print(f"Top-5 corr indices: {np.argsort(np.abs(corrs))[-5:]}")

# ── 4. Quick sklearn check: is RUL learnable at all? ─────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(flat)
y = rul
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gb.fit(X_tr, y_tr)
from sklearn.metrics import r2_score, mean_squared_error
y_pred = gb.predict(X_te)
r2  = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred) ** 0.5
print(f"\nGBM quick check  ->  R²={r2:.4f}  RMSE={rmse:.4f}")

print("\nDiagnosis complete.")
