"""
fix_parquet.py  —  understand the raw parquet structure and build the correct loader
"""
import os
import numpy as np
import pandas as pd

parquet_dir = r"C:\Users\rites\.gemini\antigravity\scratch\parsed_cfrp"
df = pd.read_parquet(os.path.join(parquet_dir, "pzt_waveforms.parquet"))

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nSample:")
print(df.head(3).to_string())

SCALAR_FEATS = [
    "amplitude_max", "amplitude_min", "amplitude_pp",
    "rms", "energy", "mean", "std", "variance",
    "skewness", "kurtosis", "zero_crossing_rate",
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "envelope_max", "envelope_mean", "toa_index",
]
present = [c for c in SCALAR_FEATS if c in df.columns]
print("\nPresent feature cols:", present)

# Check how many rows have non-zero features
non_zero = (df[present].abs() > 1e-10).any(axis=1)
print(f"\nRows with ANY non-zero feature: {non_zero.sum()} / {len(df)} ({100*non_zero.mean():.1f}%)")

df_valid = df[non_zero].copy()
print(f"\nValid rows: {len(df_valid)}")
print("Columns available:", [c for c in df_valid.columns if c not in present])

# Check group keys
group_keys = [k for k in ["layup", "specimen", "cycles", "boundary_code", "repetition"] if k in df_valid.columns]
print("\nGroup keys:", group_keys)
print("Unique layups:", df_valid["layup"].unique() if "layup" in df_valid else "N/A")
print("Unique specimens:", df_valid["specimen"].unique() if "specimen" in df_valid else "N/A")
print("Unique boundary_code:", df_valid["boundary_code"].unique() if "boundary_code" in df_valid else "N/A")

# Count groups
grp = df_valid.groupby(group_keys)
print(f"\nUnique (measurement) groups: {len(grp)}")
print(f"Mean paths per group: {grp.size().mean():.1f}")
print(f"Paths per group distribution:\n{grp.size().describe()}")

# --- check what the actual cycles look like
print("\nCycles range:", df_valid["cycles"].min(), "-", df_valid["cycles"].max())
print("Per-specimen cycle range:")
if "specimen" in df_valid.columns:
    for sp, sg in df_valid.groupby("specimen"):
        print(f"  Specimen {sp}: {sg['cycles'].min()} - {sg['cycles'].max()}, n_rows={len(sg)}")

# Check feature variance across valid rows
print("\nFeature variance (valid rows only):")
print(df_valid[present].var().to_string())

# Build a sample pivot for one specimen
sp0 = df_valid["specimen"].iloc[0] if "specimen" in df_valid else None
if sp0 is not None:
    sub = df_valid[df_valid["specimen"] == sp0].copy()
    grp0 = sub.groupby([c for c in group_keys if c != "specimen"])
    print(f"\nSpecimen {sp0}: {len(grp0)} measurement groups")
    # RUL for this specimen
    c_max = sub["cycles"].max()
    c_min = sub["cycles"].min()
    print(f"  cycles: {c_min} - {c_max}")
    # Sample feature matrix
    first_key, first_grp = next(iter(grp0))
    feat_mat = first_grp[present].values
    print(f"  Feature matrix shape for one group: {feat_mat.shape}")
    print(f"  Feature values (first row): {feat_mat[0]}")
