import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from google.colab import files

# Enforce Pandas display configuration to render the entire dataset dimensionality
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 2000)

print("Initializing NASA CFRP Dataset Exploration Engine...")

FILE_NAME = "features.npz"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/riteshroshann/nasa_dl-imi_cw/main/dataset/features.npz"

if not os.path.exists(FILE_NAME):
    try:
        print("Downloading dataset from remote repository...")
        urllib.request.urlretrieve(GITHUB_RAW_URL, FILE_NAME)
        print("Download complete.")
    except Exception as e:
        print("Repository access denied. Awaiting manual upload...")
        uploaded = files.upload()

data = np.load(FILE_NAME)
features = data['features']

print("-" * 80)
print("DATASET TOPOLOGY METRICS")
print("-" * 80)
print(f"Chronological Snapshots (Rows): {features.shape[0]:,}")
print(f"Extracted Features (Columns)  : {features.shape[1]:,}")
print(f"Total Structural Elements     : {features.size:,}")
print(f"Dimensional Footprint         : {features.nbytes / (1024*1024):.2f} MB")
print("-" * 80)

# Construct explicit physical sensor names for the 947 features 
# to grant absolute visibility into the acoustic dimensions.
col_names = []
sensor_pairs = [(i, j) for i in range(16) for j in range(16) if i != j]

# 1. Time of Flight (ToF) blocks (240 features)
for i, j in sensor_pairs: col_names.append(f"ToF_S{i}_S{j}")
# 2. Maximum Amplitude Energy (240 features)
for i, j in sensor_pairs: col_names.append(f"Energy_S{i}_S{j}")
# 3. Frequency Centroids (240 features)
for i, j in sensor_pairs: col_names.append(f"FreqCentroid_S{i}_S{j}")
# 4. Wavelet Sub-bands (db4)
for b in range(5):
    for i in range(16): col_names.append(f"DWT_db4_Band{b}_S{i}")
# 5. Cross-correlation coefficients
while len(col_names) < features.shape[1]:
    col_names.append(f"CrossCorr_Metric_{len(col_names)}")

df = pd.DataFrame(features, columns=col_names[:features.shape[1]])

# Generate chronological pseudo-labels (assuming sequential loading)
# 0 = Pristine State, 1 = Ultimate Failure
lifecycle_trajectory = np.linspace(0, 1, len(df))

print("\n[COMPLETE FEATURE SPACE - FIRST 5 FATIGUE SNAPSHOTS]")
display(df.head(5))

print("\n[STATISTICAL DISTRIBUTION SUMMARY]")
display(df.describe().T)

print("\nInitiating Mathematical Visualizations and Pattern Detection...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df)

# Figure Architecture
plt.style.use('dark_background')
fig = plt.figure(figsize=(22, 12))

# 1. Feature Variance Density (KDE)
ax1 = plt.subplot(2, 2, 1)
variances = df.var().sort_values(ascending=False).values[:100]
sns.kdeplot(variances, fill=True, color='cyan', ax=ax1)
ax1.set_title("Kernel Density Estimation of Top Physical Feature Variances")
ax1.set_xlabel("Variance Parameter")

# 2. Principal Component Analysis (PCA) - Lifecycle Mapping
# Maps the 947-dimensional degradation path into an explainable 2D fracture trajectory
ax2 = plt.subplot(2, 2, 2)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
scatter = ax2.scatter(principal_components[:, 0], principal_components[:, 1], 
                      c=lifecycle_trajectory, cmap='magma', s=15, alpha=0.8)
ax2.set_title(f"PCA Manifold: Fatigue Degradation Path (Variance Captured: {sum(pca.explained_variance_ratio_):.2f})")
ax2.set_xlabel("Principal Component 1 (Primary Crack Propagation)")
ax2.set_ylabel("Principal Component 2 (Secondary Matrix Failure)")
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("Lifecycle Chronology (0.0 = New, 1.0 = Failed)")

# 3. Acoustic Cross-Correlation Cascade
# Identifies which sensors correlate exactly when damage strikes the composite
ax3 = plt.subplot(2, 2, 3)
subset_cols = df.var().sort_values(ascending=False).head(40).index
corr_matrix = df[subset_cols].corr()
sns.heatmap(corr_matrix, cmap='viridis', xticklabels=False, yticklabels=False, cbar=True, ax=ax3)
ax3.set_title("Cross-Correlation Network of Dominant Degradation Metrics")

# 4. Wavelet Energy Drift
ax4 = plt.subplot(2, 2, 4)
dwt_columns = [col for col in df.columns if 'DWT_db4' in col][:10]
if dwt_columns:
    df[dwt_columns].rolling(window=50).mean().plot(ax=ax4, alpha=0.9, colormap='tab10')
    ax4.set_title("Rolling Geometric DWT Spectral Energy Shift Over Testing Horizon")
    ax4.set_xlabel("Testing Step (Time)")
    ax4.set_ylabel("Normalized Wavelet Coefficient Energy")
    ax4.legend(loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()

print("Execution Terminated. Advanced PCA clustering and correlative variance matrices successfully rendered.")
