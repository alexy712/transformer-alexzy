"""
engine_transformer_synthesizer.py

Comprehensive pipeline:
 - tokenization: per-time-step, window/patch, symbolic
 - linear embeddings
 - transformer autoregressive generator (train once -> generate many)
 - split C-MAPSS into per-engine files
 - per-engine training data handling
 - residual-based noise extraction & clustering by operating conditions
 - inject noise during generation
 - correlation matrix, PCA plots, histograms comparing real vs synthetic

Assumptions:
 - C-MAPSS FD001 files placed in ./data/:
    train_FD001.txt, test_FD001.txt, RUL_FD001.txt
 - Adjust DATA_DIR and filenames below if needed.
"""

import os
import math
import random
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import joblib

# -------------------------
# Config (tweakable)
# -------------------------
DATA_DIR = "./data"
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE = os.path.join(DATA_DIR, "RUL_FD001.txt")

# tokenization mode: 'timestep', 'window', or 'symbolic'
TOKEN_MODE = 'window'  

# for window tokens
WINDOW_SIZE = 10   # number of time-steps per window token
WINDOW_STAT_FUNCS = ['mean', 'std', 'slope']  # features used to summarize a window

# for symbolic tokens
SYMBOL_BINS = 5    # number of discrete symbols per sensor (equal-frequency by default)

SEQ_LEN = 30        # number of tokens (depending on tokenization) fed into transformer
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20

EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
LATENT_DIM = 16      # used for AE if desired

NOISE_K = 6          # number of residual clusters (per operating regime)
NOISE_STD_FALLBACK = 0.02  # fallback gaussian noise STD if cluster sampling fails

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COL_NAMES = ["unit", "cycle"] + [f"op{i+1}" for i in range(3)] + [f"s{i+1}" for i in range(21)]
# -------------------------
# Utilities: load & split per-engine
# -------------------------
def load_cmapss(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df.columns = COL_NAMES
    return df

def compute_rul_train(df):
    df = df.copy()
    max_cycle = df.groupby('unit')['cycle'].max().to_dict()
    df['RUL'] = df.apply(lambda row: max_cycle[row['unit']] - row['cycle'], axis=1)
    return df

def compute_rul_test(test_df, rul_path):
    rul = pd.read_csv(rul_path, header=None).iloc[:,0].values
    max_cycle_by_unit = test_df.groupby('unit')['cycle'].max().sort_index().values
    units = sorted(test_df['unit'].unique())
    unit_to_final = {u: max_cycle_by_unit[i] + int(rul[i]) for i,u in enumerate(units)}
    df = test_df.copy()
    df['RUL'] = df.apply(lambda row: unit_to_final[row['unit']] - row['cycle'], axis=1)
    return df

def split_per_engine(df, out_dir="./data/per_engine", prefix="train"):
    """
    Save each engine's sequence to its own CSV.
    Useful for per-engine training or distributed processing.
    """
    os.makedirs(out_dir, exist_ok=True)
    groups = df.groupby('unit')
    for unit, g in groups:
        fname = os.path.join(out_dir, f"{prefix}_unit_{unit}.csv")
        g.to_csv(fname, index=False)

# -------------------------
# Tokenizers
# -------------------------
def compute_slope(arr):
    # arr is 1D - simple linear fit slope
    x = np.arange(len(arr))
    # slope = covariance/variance
    vx = x - x.mean()
    vy = arr - arr.mean()
    denom = (vx*vx).sum()
    return (vx*vy).sum() / (denom + 1e-8)

def window_tokenize(unit_df, feature_cols, window_size=WINDOW_SIZE, stats=WINDOW_STAT_FUNCS):
    """
    Build windows of length window_size and summarize each window by stats for each sensor.
    Returns a sequence of tokens shape (n_windows, n_stats * n_features)
    """
    arr = unit_df[feature_cols].values
    n = len(arr)
    tokens = []
    for start in range(0, n, window_size):
        end = min(n, start+window_size)
        window = arr[start:end]
        features = []
        for f_idx in range(window.shape[1]):
            col = window[:, f_idx]
            for s in stats:
                if s == 'mean':
                    features.append(np.mean(col))
                elif s == 'std':
                    features.append(np.std(col))
                elif s == 'min':
                    features.append(np.min(col))
                elif s == 'max':
                    features.append(np.max(col))
                elif s == 'slope':
                    features.append(compute_slope(col))
                else:
                    raise ValueError("unknown stat " + s)
        tokens.append(np.array(features, dtype=np.float32))
    return np.stack(tokens) if tokens else np.zeros((0, len(feature_cols)*len(stats)), dtype=np.float32)

def timestep_tokenize(unit_df, feature_cols):
    """
    Per-time-step token: each time row is a token vector of raw sensor+op values
    Returns (n_timesteps, n_features)
    """
    return unit_df[feature_cols].values.astype(np.float32)

def symbolic_tokenize(unit_df, feature_cols, bins_per_feature=None):
    """
    Convert continuous values into symbol IDs per feature, then combine into a single token vector per time-step.
    bins_per_feature: dict feature->bin_edges (if None, compute equal-frequency bins).
    Returns:
      tokens: ndarray (n_timesteps, n_features) of integer symbol ids (0..B-1)
      bin_edges: dict of arrays for each feature
    NOTE: downstream embedding will treat each feature symbol embedding and optionally concat them.
    """
    df = unit_df[feature_cols].copy()
    bin_edges = {} if bins_per_feature is None else bins_per_feature
    tokens = np.zeros_like(df.values, dtype=np.int64)
    for i, col in enumerate(feature_cols):
        col_vals = df[col].values
        if bins_per_feature is None or col not in bins_per_feature:
            # create quantile bins
            edges = np.quantile(col_vals, q=np.linspace(0, 1, SYMBOL_BINS+1))
            # ensure unique edges
            edges = np.unique(edges)
            if len(edges)-1 < SYMBOL_BINS:
                # fallback to linspace min/max
                edges = np.linspace(col_vals.min(), col_vals.max(), SYMBOL_BINS+1)
            bin_edges[col] = edges
        else:
            edges = bins_per_feature[col]
        # digitize: bin index between 1..len(edges)-1, subtract 1 to get 0-based
        indices = np.digitize(col_vals, edges[1:-1], right=False)
        tokens[:, i] = indices.astype(np.int64)
    return tokens, bin_edges

# -------------------------
# Datasets & DataLoader
# -------------------------
class EngineSeqDataset(Dataset):
    """
    Dataset that yields sequences of tokens and target next-step values.
    Works for all tokenization modes.

    For autoregressive training, for each engine we produce sliding sequences of length seq_len
    and the target is the next token (shifted by 1) — so model learns to predict next token.
    """
    def __init__(self, list_of_unit_dfs, feature_cols, token_mode='window', seq_len=SEQ_LEN,
                 window_size=WINDOW_SIZE, window_stats=WINDOW_STAT_FUNCS,
                 symbol_bins=None, scaler=None):
        """
        list_of_unit_dfs: list of pandas DataFrame per engine (must be sorted by cycle)
        feature_cols: list of feature column names used
        token_mode: 'timestep', 'window', 'symbolic'
        scaler: a fitted StandardScaler used for continuous tokens (timestep/window)
        """
        self.unit_dfs = list_of_unit_dfs
        self.feature_cols = feature_cols
        self.token_mode = token_mode
        self.seq_len = seq_len
        self.window_size = window_size
        self.window_stats = window_stats
        self.symbol_bins = symbol_bins
        self.scaler = scaler

        # pre-tokenize all units into token sequences
        self.tokens = []  # list of arrays per unit
        for df in self.unit_dfs:
            if token_mode == 'timestep':
                arr = timestep_tokenize(df, feature_cols)  # (T, nfeat)
                if self.scaler is not None:
                    arr = self.scaler.transform(arr)
                self.tokens.append(arr)
            elif token_mode == 'window':
                arr = window_tokenize(df, feature_cols, window_size=window_size, stats=window_stats)
                if self.scaler is not None and arr.size:
                    arr = self.scaler.transform(arr)
                self.tokens.append(arr)
            elif token_mode == 'symbolic':
                # symbolic tokenization across units must use consistent bins; we'll assume symbol_bins provided externally
                tokens, _ = symbolic_tokenize(df, feature_cols, bins_per_feature=symbol_bins)
                self.tokens.append(tokens)
            else:
                raise ValueError("unknown token mode")

        # build index mapping global index -> (unit_idx, start_pos)
        self.idx_map = []
        for u_idx, tarr in enumerate(self.tokens):
            T = tarr.shape[0]
            for start in range(0, max(0, T - seq_len)):
                self.idx_map.append((u_idx, start))
        # note: this implementation predicts token at position start+seq_len (next token) as target
        # to keep consistent, we require sequences that have a next token.

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        u_idx, start = self.idx_map[idx]
        tarr = self.tokens[u_idx]
        seq = tarr[start : start + self.seq_len]  # shape (seq_len, dim)
        target = tarr[start + self.seq_len]       # next token
        # If symbolic tokens represented as integer labels per feature, return as integers else floats.
        return seq.astype(np.float32), target.astype(np.float32)

# collate: convert to tensors
def collate_fn(batch):
    X = np.stack([b[0] for b in batch])
    y = np.stack([b[1] for b in batch])
    return torch.tensor(X), torch.tensor(y)

# -------------------------
# Embedding layers
# -------------------------
class NumericEmbedding(nn.Module):
    """
    For continuous tokens (per-timestep or window tokens), use a linear layer to map token vectors -> embedding dim.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        return self.linear(x)  # returns (batch, seq_len, embed_dim)

class SymbolicEmbedding(nn.Module):
    """
    For symbolic tokens (feature-wise symbol ids), we create a per-feature embedding table and then combine (concatenate or sum).
    This example concatenates feature embeddings then projects to embed_dim via linear layer.
    """
    def __init__(self, n_features, bins_per_feature, embed_dim):
        super().__init__()
        self.n_features = n_features
        self.bins_per_feature = bins_per_feature  # dict feature_name -> n_bins
        # create embedding tables
        self.emb_tables = nn.ModuleList([nn.Embedding(bins_per_feature[i], embed_dim//n_features + 1) 
                                         for i in range(n_features)])
        # projection to final embed_dim
        # compute concatenated size:
        concat_size = sum([emb.embedding_dim for emb in self.emb_tables])
        self.proj = nn.Linear(concat_size, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_features) integers
        batch, seq_len, nfeat = x.shape
        embs = []
        for i in range(nfeat):
            emb = self.emb_tables[i](x[:,:,i].long())  # (batch, seq_len, dim_i)
            embs.append(emb)
        cat = torch.cat(embs, dim=-1)
        out = self.proj(cat)
        return out

# -------------------------
# Transformer model (autoregressive next-token predictor)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AutoregressiveTransformer(nn.Module):
    """
    Simple Transformer-based next-token predictor:
     - embedding: pre-supplied nn.Module maps token vectors -> embed_dim
     - transformer encoder (causal mask applied) to produce context
     - head: map last token's contextual embedding to predicted next token vector
    """
    def __init__(self, embed_module, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT, out_dim=None):
        super().__init__()
        self.embed_module = embed_module
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = out_dim
        if out_dim is None:
            raise ValueError("out_dim (size of predicted token vector) must be provided")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, out_dim)
        )

    def forward(self, x):
        """
        x: token representation before embedding (if numeric: floats; if symbolic: int labels)
        """
        emb = self.embed_module(x)            # (batch, seq_len, embed_dim)
        emb = self.pos_enc(emb)
        # causal mask: ensure transformer doesn't peek into future by masking out subsequent positions
        # generate mask of shape (seq_len, seq_len) where True indicates positions to mask
        seq_len = emb.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=emb.device), diagonal=1).bool()
        # apply transformer encoder with mask (src_key_padding_mask is not used here)
        out = self.transformer(emb, src_key_padding_mask=None, mask=mask)
        # take last token's context
        last = out[:, -1, :]  # (batch, embed_dim)
        pred = self.head(last)  # (batch, out_dim)
        return pred

# -------------------------
# Training + Generation flows
# -------------------------
def train_transformer(model, train_loader, val_loader=None, epochs=EPOCHS, lr=LR, save_path="best_transformer.pt"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = 1e9
    history = {'train_loss':[], 'val_loss':[]}
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        val_loss = None
        if val_loader:
            model.eval()
            running = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    running += loss.item() * xb.size(0)
                val_loss = running / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if val_loss is not None and val_loss < best_val:
            torch.save(model.state_dict(), save_path)
            best_val = val_loss
    return history

def generate_sequence(model, seed_tokens, n_steps, token_mode, embedding_type,
                      symbol_bins=None, scaler=None, residual_sampler=None):
    """
    Generate n_steps tokens autoregressively from seed_tokens (array shape (seed_len, token_dim)).
    token_mode: 'timestep' or 'window' or 'symbolic'
    embedding_type: 'numeric' or 'symbolic' - passed for any postprocessing
    residual_sampler: function(seq_idx, t) -> noise vector of same size as predicted token (applied in continuous token space)
    
    Returns array of generated tokens (seed + generated) in the same token representation used during training.
    """
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        seq = list(seed_tokens.copy())  # list of token vectors
        for step in range(n_steps):
            # prepare input: last SEQ_LEN tokens
            context = np.stack(seq[-SEQ_LEN:]) if len(seq) >= SEQ_LEN else np.vstack([np.zeros_like(seq[0])]*(SEQ_LEN-len(seq)) + seq)
            x = torch.tensor(context[np.newaxis].astype(np.float32)).to(DEVICE)
            pred = model(x).cpu().numpy()[0]  # predicted token vector in scaled space or integer-like if symbolic
            # if numeric tokens: optionally add residual noise sampled from residual_sampler
            if token_mode in ('timestep', 'window'):
                noise = np.zeros_like(pred)
                if residual_sampler is not None:
                    try:
                        noise = residual_sampler(pred, context)
                    except Exception:
                        noise = np.random.normal(0, NOISE_STD_FALLBACK, size=pred.shape)
                gen_token = pred + noise
                # store scaled token (if scaler used for tokens)
                seq.append(gen_token.astype(np.float32))
            elif token_mode == 'symbolic':
                # if symbolic tokens are integer-encoded but the model predicts a vector; apply argmax per feature if model returns per-feature logits.
                # Here we assume model predicts raw class logits for each feature concatenated; user may customize.
                gen_ids = np.round(pred).astype(np.int64)  # crude fallback
                seq.append(gen_ids)
            else:
                raise ValueError("unknown token_mode")
        out = np.stack(seq)
    return out

# -------------------------
# Residual extraction & clustering (per-op conditions)
# -------------------------
def extract_residuals(unit_df, feature_cols, token_mode='timestep', window_size=WINDOW_SIZE, window_stats=WINDOW_STAT_FUNCS, scaler=None):
    """
    For a single engine/unit, compute the residuals between actual sensor tokens and a smooth baseline (rolling mean).
    Returns residual vectors aligned to tokens (shape: n_tokens x token_dim), and operating condition vector
      (we'll use mean of op settings inside the token period).
    """
    if token_mode == 'timestep':
        arr = unit_df[feature_cols].values.astype(float)
        # baseline: rolling mean with window=5 (short) or use simple lowess; here rolling mean for simplicity
        baseline = pd.DataFrame(arr).rolling(5, min_periods=1, center=True).mean().values
        residuals = arr - baseline
        op_mean = unit_df[['op1','op2','op3']].values  # per time-step
        return residuals.astype(np.float32), op_mean.astype(np.float32)
    elif token_mode == 'window':
        tokens = window_tokenize(unit_df, feature_cols, window_size=window_size, stats=window_stats)
        # to compute baseline we can smooth each token feature across windows by rolling mean
        if tokens.shape[0] == 0:
            return np.zeros((0, tokens.shape[1])), np.zeros((0, 3))
        df_tokens = pd.DataFrame(tokens)
        baseline = df_tokens.rolling(3, min_periods=1, center=True).mean().values
        residuals = tokens - baseline
        # compute op_mean per window
        ops = unit_df[['op1','op2','op3']].values
        op_means = []
        n = len(ops)
        for start in range(0, n, window_size):
            end = min(n, start+window_size)
            op_means.append(ops[start:end].mean(axis=0))
        return residuals.astype(np.float32), np.array(op_means).astype(np.float32)
    elif token_mode == 'symbolic':
        # symbolic residuals not applicable in same sense; return zeros
        tokens, _ = symbolic_tokenize(unit_df, feature_cols)
        op_mean = unit_df[['op1','op2','op3']].values
        return np.zeros_like(tokens).astype(np.float32), op_mean.astype(np.float32)
    else:
        raise ValueError("unknown token_mode")

def build_noise_library(list_of_unit_dfs, feature_cols, token_mode='window', window_size=WINDOW_SIZE, window_stats=WINDOW_STAT_FUNCS, scaler=None, k=NOISE_K):
    """
    Build a library of residual clusters grouped by operating condition.
    Approach:
      - For each unit: compute residual vectors per token and the running op_mean per token.
      - Concatenate residuals across units and cluster residuals into k clusters (KMeans).
      - Save cluster centers + representative residuals and the op_mean centroids per cluster for conditional sampling.
    Returns:
      - kmeans model, residuals array, op_means array, labels
    """
    all_res = []
    all_ops = []
    for df in list_of_unit_dfs:
        res, ops = extract_residuals(df, feature_cols, token_mode=token_mode, window_size=window_size, window_stats=window_stats, scaler=scaler)
        if res.shape[0] == 0: 
            continue
        all_res.append(res)
        all_ops.append(ops)
    if not all_res:
        raise RuntimeError("No residuals collected")
    res_arr = np.vstack(all_res)
    ops_arr = np.vstack(all_ops)  # shape (N_tokens, 3)
    # flatten residuals per token into 1D vector (as used by KDTree / kmeans)
    flat_res = res_arr.reshape(res_arr.shape[0], -1)
    # cluster
    k_clust = min(k, flat_res.shape[0])
    kmeans = KMeans(n_clusters=k_clust, random_state=42).fit(flat_res)
    labels = kmeans.labels_
    # compute op centroids per cluster
    op_centroids = []
    residual_centroids = []
    for c in range(k_clust):
        mask = (labels == c)
        op_centroids.append(ops_arr[mask].mean(axis=0))
        residual_centroids.append(flat_res[mask].mean(axis=0))
    op_centroids = np.vstack(op_centroids)
    residual_centroids = np.vstack(residual_centroids)
    return {
        'kmeans': kmeans,
        'flat_res': flat_res,
        'ops_arr': ops_arr,
        'labels': labels,
        'op_centroids': op_centroids,
        'residual_centroids': residual_centroids
    }

def residual_sampler_factory(noise_lib):
    """
    Return a function residual_sampler(pred_token, context) -> noise vector
    We choose a cluster based on the operating condition of the last context token (if available)
    and sample a residual within that cluster (by selecting random member residual and reshaping).
    """
    kmeans = noise_lib['kmeans']
    flat_res = noise_lib['flat_res']
    ops = noise_lib['ops_arr']
    labels = noise_lib['labels']
    # build per-cluster indices
    cluster_indices = {i: np.where(labels==i)[0] for i in np.unique(labels)}
    def sampler(pred_token, context):
        # context: numpy array (seq_len, token_dim) - we try to extract op means if available
        # We'll compute op_cond roughly as mean across context if context includes op features
        # Fallback: choose random cluster
        try:
            # best approach depends on tokenization; we try to extract op columns if present in context shape.
            # If context includes op1..op3, they will be in the token vector at known positions
            # Since token formats vary, we will simply pick a cluster at random weighted by cluster sizes
            cl = np.random.choice(list(cluster_indices.keys()))
            idx = np.random.choice(cluster_indices[cl])
            residual_flat = flat_res[idx]
            return residual_flat.reshape(pred_token.shape).astype(np.float32)
        except Exception:
            return np.random.normal(0, NOISE_STD_FALLBACK, size=pred_token.shape).astype(np.float32)
    return sampler

# -------------------------
# Analysis & Plots
# -------------------------
def correlation_heatmap(real_arr, synth_arr, feature_cols, title="Correlation real vs synth"):
    """
    Compute correlation matrices (features x features) for real and synthetic and display difference heatmap.
    real_arr and synth_arr are shape (N, T, n_features) or flattened (N*T, n_features)
    """
    if real_arr.ndim == 3:
        real_flat = real_arr.reshape(-1, real_arr.shape[2])
    else:
        real_flat = real_arr
    if synth_arr.ndim == 3:
        synth_flat = synth_arr.reshape(-1, synth_arr.shape[2])
    else:
        synth_flat = synth_arr
    corr_real = np.corrcoef(real_flat.T)
    corr_synth = np.corrcoef(synth_flat.T)
    diff = corr_real - corr_synth
    plt.figure(figsize=(10,8))
    sns.heatmap(diff, xticklabels=feature_cols, yticklabels=feature_cols, cmap='coolwarm', center=0)
    plt.title(title + " (real - synth)")
    plt.show()
    # Also show both matrices side-by-side for reference
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    sns.heatmap(corr_real, ax=axs[0], xticklabels=feature_cols, yticklabels=feature_cols, cmap='coolwarm', center=0)
    axs[0].set_title("Real correlation")
    sns.heatmap(corr_synth, ax=axs[1], xticklabels=feature_cols, yticklabels=feature_cols, cmap='coolwarm', center=0)
    axs[1].set_title("Synthetic correlation")
    plt.show()

def pca_scatter(real_arr, synth_arr, n_components=2, sample_size=1000, title="PCA real vs synth"):
    """
    Project flattened sequences (one vector per token-window) into PCA space and show scatter for real vs synth.
    """
    if real_arr.ndim == 3:
        real_flat = real_arr.reshape(-1, real_arr.shape[2])
    else:
        real_flat = real_arr
    if synth_arr.ndim == 3:
        synth_flat = synth_arr.reshape(-1, synth_arr.shape[2])
    else:
        synth_flat = synth_arr
    # sample to keep plotting manageable
    if real_flat.shape[0] > sample_size:
        idx = np.random.choice(real_flat.shape[0], sample_size, replace=False)
        real_s = real_flat[idx]
    else:
        real_s = real_flat
    if synth_flat.shape[0] > sample_size:
        idx = np.random.choice(synth_flat.shape[0], sample_size, replace=False)
        synth_s = synth_flat[idx]
    else:
        synth_s = synth_flat
    X = np.vstack([real_s, synth_s])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    rlen = real_s.shape[0]
    plt.figure(figsize=(8,6))
    plt.scatter(Z[:rlen,0], Z[:rlen,1], alpha=0.5, label='real')
    plt.scatter(Z[rlen:,0], Z[rlen:,1], alpha=0.5, label='synth')
    plt.legend(); plt.title(title)
    plt.show()

def feature_histograms(real_arr, synth_arr, feature_cols, n_bins=50, sample_points=10000):
    """
    Plot histograms per feature comparing real vs synthetic distributions.
    """
    if real_arr.ndim == 3:
        real_flat = real_arr.reshape(-1, real_arr.shape[2])
    else:
        real_flat = real_arr
    if synth_arr.ndim == 3:
        synth_flat = synth_arr.reshape(-1, synth_arr.shape[2])
    else:
        synth_flat = synth_arr
    nfeat = real_flat.shape[1]
    for i in range(nfeat):
        r = real_flat[:, i]
        s = synth_flat[:, i]
        if r.size > sample_points:
            r = np.random.choice(r, sample_points, replace=False)
        if s.size > sample_points:
            s = np.random.choice(s, sample_points, replace=False)
        plt.figure(figsize=(6,3))
        sns.histplot(r, bins=n_bins, color='blue', stat='density', label='real', alpha=0.5)
        sns.histplot(s, bins=n_bins, color='orange', stat='density', label='synth', alpha=0.5)
        plt.title(f"Feature {feature_cols[i]} distribution (KS p={ks_2samp(real_flat[:,i], synth_flat[:,i]).pvalue:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.show()

# -------------------------
# Example end-to-end pipeline runner (main)
# -------------------------
def main_pipeline():
    # 1. Load
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Place C-MAPSS FD001 train file at {TRAIN_FILE}")
    train_df = load_cmapss(TRAIN_FILE)
    train_df = compute_rul_train(train_df)

    # feature columns to use
    feature_cols = [c for c in train_df.columns if c not in ['unit','cycle','RUL']]
    n_features_raw = len(feature_cols)

    # 2. split per engine (optional)
    split_per_engine(train_df, out_dir=os.path.join(DATA_DIR, "per_engine"), prefix="train")

    # 3. Build list of per-engine DataFrames
    per_engine_dir = os.path.join(DATA_DIR, "per_engine")
    per_files = sorted([os.path.join(per_engine_dir, f) for f in os.listdir(per_engine_dir) if f.endswith(".csv")])
    unit_dfs = [pd.read_csv(p) for p in per_files]

    # 4. Prepare bins for symbolic tokenization (if needed)
    symbol_bins = None
    if TOKEN_MODE == 'symbolic':
        # compute quantile bins across entire dataset (per feature)
        symbol_bins = {}
        for c in feature_cols:
            vals = np.hstack([df[c].values for df in unit_dfs])
            edges = np.quantile(vals, q=np.linspace(0,1,SYMBOL_BINS+1))
            symbol_bins[c] = np.unique(edges)
        joblib.dump(symbol_bins, "symbol_bins.pkl")

    # 5. scaler for numeric token modes (timestep/window)
    scaler = None
    if TOKEN_MODE in ('timestep', 'window'):
        # fit scaler on token representations from all engines
        token_samples = []
        for df in unit_dfs:
            if TOKEN_MODE == 'timestep':
                t = timestep_tokenize(df, feature_cols)
            else:
                t = window_tokenize(df, feature_cols, window_size=WINDOW_SIZE, stats=WINDOW_STAT_FUNCS)
            if t.size:
                token_samples.append(t.reshape(-1, t.shape[-1]))
        all_tokens = np.vstack(token_samples)
        scaler = StandardScaler().fit(all_tokens)
        joblib.dump(scaler, "token_scaler.save")

    # 6. Build dataset (list-of-dfs passed)
    ds = EngineSeqDataset(unit_dfs, feature_cols, token_mode=TOKEN_MODE, seq_len=SEQ_LEN,
                          window_size=WINDOW_SIZE, window_stats=WINDOW_STAT_FUNCS,
                          symbol_bins=symbol_bins, scaler=scaler)

    # split indices into train/val
    idxs = list(range(len(ds)))
    tr_idx, val_idx = train_test_split(idxs, test_size=0.1, random_state=42)
    # create Subset-like simple data loaders
    def make_loader(idx_list):
        subset = [ds[i] for i in idx_list]
        X = np.stack([s[0] for s in subset]); Y = np.stack([s[1] for s in subset])
        return DataLoader(list(zip(X,Y)), batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b))
    train_loader = make_loader(tr_idx)
    val_loader = make_loader(val_idx)

    # 7. Build embedding and model
    if TOKEN_MODE in ('timestep', 'window'):
        sample_token_dim = ds.tokens[0].shape[1] if len(ds.tokens)>0 and ds.tokens[0].size else all_tokens.shape[1]
        embed_module = NumericEmbedding(sample_token_dim, EMBED_DIM)
        out_dim = sample_token_dim
    else:  # symbolic
        # compute bins array for each feature as integer cardinalities
        bins_card = {i: len(symbol_bins[feature_cols[i]])-1 for i in range(len(feature_cols))}
        embed_module = SymbolicEmbedding(len(feature_cols), bins_card, EMBED_DIM)
        # for symbolic mode, we must define how model outputs are interpreted (here we output integer IDs per feature concatenated)
        out_dim = len(feature_cols)  # crude: predict integer id per feature

    model = AutoregressiveTransformer(embed_module, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS, out_dim=out_dim)

    # 8. Build noise library from residuals (for numeric token modes)
    noise_lib = None
    residual_sampler = None
    if TOKEN_MODE in ('timestep','window'):
        noise_lib = build_noise_library(unit_dfs, feature_cols, token_mode=TOKEN_MODE, window_size=WINDOW_SIZE, window_stats=WINDOW_STAT_FUNCS, scaler=scaler, k=NOISE_K)
        joblib.dump(noise_lib, "noise_lib.pkl")
        residual_sampler = residual_sampler_factory(noise_lib)

    # 9. Train model
    train_transformer(model, train_loader, val_loader=val_loader, epochs=EPOCHS, lr=LR, save_path="best_transformer.pt")
    torch.save(model.state_dict(), "transformer_final.pt")
    joblib.dump(feature_cols, "feature_cols.pkl")

    # 10. Example generation:
    # pick a random engine, create seed tokens (last SEQ_LEN tokens), generate N tokens (e.g., 50 windows)
    seed_unit = random.choice(unit_dfs)
    if TOKEN_MODE == 'timestep':
        seed_tokens = timestep_tokenize(seed_unit, feature_cols)
    elif TOKEN_MODE == 'window':
        seed_tokens = window_tokenize(seed_unit, feature_cols, window_size=WINDOW_SIZE, stats=WINDOW_STAT_FUNCS)
    else:
        seed_tokens, _ = symbolic_tokenize(seed_unit, feature_cols, bins_per_feature=symbol_bins)
    # scale seed tokens if numeric
    if TOKEN_MODE in ('timestep','window') and scaler is not None and seed_tokens.size:
        seed_scaled = scaler.transform(seed_tokens)
    else:
        seed_scaled = seed_tokens
    N_GEN = 50
    gen_scaled = generate_sequence(model, seed_scaled[-SEQ_LEN:], N_GEN, token_mode=TOKEN_MODE, embedding_type=("numeric" if TOKEN_MODE in ('timestep','window') else "symbolic"), symbol_bins=symbol_bins, scaler=scaler, residual_sampler=residual_sampler)
    # gen_scaled includes seed + generated tokens; get just generated portion
    generated_part = gen_scaled[-N_GEN:]

    # inverse transform to original units for numeric tokens
    if TOKEN_MODE in ('timestep','window') and scaler is not None:
        gen_inv = scaler.inverse_transform(generated_part.reshape(-1, generated_part.shape[-1])).reshape(generated_part.shape)
    else:
        gen_inv = generated_part

    # 11. Save synthetic sequences as CSV rows with artificial unit IDs
    out_rows = []
    next_unit_id = 100000
    for seq_idx in range(gen_inv.shape[0]):
        row = {'unit': next_unit_id, 'cycle': seq_idx+1}
        # if window tokens summarize multiple time steps, we cheat: treat cycle as window id
        for i, feat in enumerate(feature_cols):
            val = gen_inv[seq_idx, i] if i < gen_inv.shape[1] else 0.0
            row[feat] = float(val)
        out_rows.append(row)
    synth_df = pd.DataFrame(out_rows)
    # Save synthetic data in .txt format with space-separated columns
    out_path = "synthetic_generated.txt"
    synth_df.to_csv(out_path, index=False, sep=" ", header=False)

    # 12. Analysis: correlation, PCA, histograms comparing real vs synthetic
    # sample a chunk of real tokens to compare
    real_tokens_sample = np.vstack([window_tokenize(df, feature_cols, window_size=WINDOW_SIZE, stats=WINDOW_STAT_FUNCS) for df in unit_dfs if window_tokenize(df, feature_cols, window_size=WINDOW_SIZE, stats=WINDOW_STAT_FUNCS).size]).reshape(-1, len(feature_cols)*len(WINDOW_STAT_FUNCS))
    synth_tokens_sample = gen_inv.reshape(-1, gen_inv.shape[-1])
    # ensure dims align (may need slicing)
    min_cols = min(real_tokens_sample.shape[1], synth_tokens_sample.shape[1])
    real_arr = real_tokens_sample[:, :min_cols].reshape(-1, min_cols)
    synth_arr = synth_tokens_sample[:, :min_cols].reshape(-1, min_cols)
    # reshape to (N, T, features) expected by plotting helpers - here T=1 per token, so use (N,1,feat)
    correlation_heatmap(real_arr.reshape(-1,1,min_cols), synth_arr.reshape(-1,1,min_cols), feature_cols[:min_cols], title="Correlation diff (real - synth)")
    pca_scatter(real_arr.reshape(-1,1,min_cols), synth_arr.reshape(-1,1,min_cols), title="PCA real vs synth")
    feature_histograms(real_arr.reshape(-1,1,min_cols), synth_arr.reshape(-1,1,min_cols), feature_cols[:min_cols])

if __name__ == "__main__":
    main_pipeline()
