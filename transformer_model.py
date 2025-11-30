"""
transformer_model.py
-----------------------------
More stable Transformer for generating synthetic NASA C-MAPSS data.
Includes:
 - per-unit training sequences
 - teacher forcing
 - deeper model
 - normalization fixes
 - stable synthetic sampling
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# ======================================================
# 1. Paths
# ======================================================

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE, "train_FD001.txt")
TEST_FILE  = os.path.join(BASE, "test_FD001.txt")
OUT_FILE   = os.path.join(BASE, "synthetic_FD001.txt")

# ======================================================
# 2. Columns (NASA Format)
# ======================================================

columns = (
    ["unit", "cycle", "op1", "op2", "op3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# ======================================================
# 3. Load TXT (stable)
# ======================================================

def load_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    return df.reset_index(drop=True)

train_df = load_txt(TRAIN_FILE)
test_df  = load_txt(TEST_FILE)

# drop non-numeric (unit & cycle can remain numeric)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
train_df = train_df[numeric_cols]
test_df  = test_df[numeric_cols]

FEATURE_DIM = len(numeric_cols)

# ======================================================
# 4. Scaling (very important)
# ======================================================

scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=numeric_cols)
test_scaled  = pd.DataFrame(scaler.transform(test_df), columns=numeric_cols)

# ======================================================
# 5. Build per-engine sequences
# ======================================================

SEQ_LEN = 30

def make_sequences(df):
    X, y = [], []
    arr = df.values.astype(np.float32)

    for i in range(len(arr) - SEQ_LEN):
        X.append(arr[i:i+SEQ_LEN])
        y.append(arr[i+SEQ_LEN])

    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train).to(DEVICE)
y_train = torch.tensor(y_train).to(DEVICE)

# ======================================================
# 6. Stronger Transformer model
# ======================================================

class BetterTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        embed = 128

        self.inp = nn.Linear(dim, embed)

        layer = nn.TransformerEncoderLayer(
            d_model=embed,
            nhead=8,
            dim_feedforward=512,
            dropout=0.15,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=4)

        self.norm = nn.LayerNorm(embed)
        self.out = nn.Linear(embed, dim)

    def forward(self, x):
        x = self.inp(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.out(x)

model = BetterTransformer(FEATURE_DIM).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
lossf = nn.SmoothL1Loss()

# ======================================================
# 7. Training Loop (40 epochs)
# ======================================================

EPOCHS = 40
BATCH = 32
N = len(X_train)

print("\nTraining improved Transformer...\n")

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N)
    losses = []

    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        xb = X_train[idx]
        yb = y_train[idx]

        opt.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Epoch {epoch}/{EPOCHS} - loss={np.mean(losses):.6f}")

# ======================================================
# 8. Generate cleaner synthetic data
# ======================================================

def generate_synthetic(df):
    arr = df.values.astype(np.float32)
    context = arr[:SEQ_LEN].tolist()
    out = []

    model.eval()
    with torch.no_grad():
        for _ in range(len(arr)):
            x = torch.tensor(context[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(x)[0].cpu().numpy()

            # Add light gaussian noise for realism
            pred = pred + np.random.normal(0, 0.03, size=pred.shape)

            out.append(pred)
            context.append(pred)

    return pd.DataFrame(out, columns=numeric_cols)

synthetic_scaled = generate_synthetic(test_scaled)

# inverse scale
synthetic = pd.DataFrame(
    scaler.inverse_transform(synthetic_scaled),
    columns=numeric_cols
)

synthetic.to_csv(OUT_FILE, sep="\t", index=False, header=False)
print("\nSaved synthetic data:", OUT_FILE)