"""
visualize_compare.py
--------------------
Displays real vs synthetic NASA C-MAPSS visualizations directly
to the screen (no saving, no output folder).

Visualizations:
 - PCA (Real vs Synthetic)
 - Real data correlation heatmap (All sensors + op settings)
 - Synthetic data correlation heatmap
 - Feature mean difference barplot
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.abspath(__file__))

REAL_FILE = os.path.join(BASE, "test_FD001.txt")
SYN_FILE  = os.path.join(BASE, "synthetic_FD001.txt")

# NASA FD001 column names
columns = (
    ["unit", "cycle", "op1", "op2", "op3"] +
    [f"sensor_{i}" for i in range(1,22)]
)

# ------------------------------------------------------
# Load TXT files
# ------------------------------------------------------
def load_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    # keep only numeric columns
    return df.select_dtypes(include=[np.number]).reset_index(drop=True)

real = load_txt(REAL_FILE)
syn  = load_txt(SYN_FILE)

# ------------------------------------------------------
# Align lengths by padding synthetic data
# ------------------------------------------------------
while len(syn) < len(real):
    syn.loc[len(syn)] = syn.iloc[-1]

print("Data aligned:")
print("Real shape:", real.shape)
print("Synthetic shape:", syn.shape)

# ------------------------------------------------------
# PCA Visualization
# ------------------------------------------------------
print("\nRunning PCA...")

Xs = StandardScaler().fit_transform(
    np.vstack([real.values, syn.values])
)

pca = PCA(n_components=2)
Z = pca.fit_transform(Xs)

plt.figure(figsize=(8,6))
plt.scatter(Z[:len(real),0], Z[:len(real),1], s=10, alpha=0.6, label="Real")
plt.scatter(Z[len(real):,0], Z[len(real):,1], s=10, alpha=0.6, label="Synthetic")
plt.title("PCA: Real vs Synthetic")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# Correlation Heatmaps
# ------------------------------------------------------
print("Generating correlation heatmaps...")

plt.figure(figsize=(14,12))
sns.heatmap(real.corr(), cmap="coolwarm", center=0)
plt.title("Real Data Correlation Heatmap (All sensors + op settings)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,12))
sns.heatmap(syn.corr(), cmap="coolwarm", center=0)
plt.title("Synthetic Data Correlation Heatmap (All sensors + op settings)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# Feature Mean Differences Barplot
# ------------------------------------------------------
print("Computing feature differences...")

diffs = abs(real.mean() - syn.mean())
df_diff = pd.DataFrame({"feature": real.columns, "difference": diffs})
df_diff = df_diff.sort_values("difference", ascending=False)

plt.figure(figsize=(12,5))
plt.bar(df_diff["feature"].head(20), df_diff["difference"].head(20))
plt.xticks(rotation=45)
plt.title("Top 20 Feature Mean Differences (Real vs Synthetic)")
plt.ylabel("Absolute Difference")
plt.tight_layout()
plt.show()

print("\nVisualization completed.")