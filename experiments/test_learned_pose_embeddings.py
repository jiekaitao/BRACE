#!/usr/bin/env python3
"""Test learned pose embeddings for better motion clustering.

Compares pretrained and self-trained embedding approaches against the baseline
spectral distance clustering used in BRACE.

Methods tested:
1. Baseline:     Hand-crafted spectral distance (current production system)
2. MotionBERT:   Pretrained DSTformer (ICCV 2023, 42M params, 512D embeddings)
3. MLP-AE:       Simple autoencoder bottleneck embedding (self-trained)
4. Conv1D-AE:    1D-CNN temporal encoder (self-trained)
5. Transformer:  Small transformer encoder with mean pooling (self-trained)
6. SimCLR:       Contrastive learning with augmented views (self-trained)

Each approach produces fixed-size embeddings per motion segment, then clusters
with agglomerative clustering at various thresholds.

Uses cached SRP features from experiments/.feature_cache/*.npz.
Validates against ground truth from experiments/video_ground_truth.json.
"""

import sys
sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
)

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
MOTIONBERT_WEIGHTS = "/mnt/Data/GitHub/BRACE/experiments/pretrained_models/motionbert_pretrained.bin"
MOTIONBERT_REPO = "/tmp/MotionBERT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLE_LEN = 30   # Fixed temporal length for segments
FEAT_DIM = 28       # 14 joints x 2 coords (2D SRP)
LATENT_DIM = 32     # Embedding dimension for self-trained models
BATCH_SIZE = 32
EPOCHS_AE = 100
EPOCHS_CONTRASTIVE = 150
LR = 1e-3

# Thresholds to sweep for agglomerative clustering
AGGLOM_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

# MediaPipe-33 FEATURE_INDICES -> approximate H36M-17 mapping
# Our 14 feature joints (MP indices): [11,12,13,14,15,16,23,24,25,26,27,28,31,32]
# H36M 17 joints: 0=Hip, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee, 6=LAnkle,
#                  7=Spine, 8=Thorax, 9=Neck, 10=Head,
#                  11=LShoulder, 12=LElbow, 13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist
# Feature idx -> (joint_name, H36M_idx):
# 0:L_shoulder(11)->11, 1:R_shoulder(12)->14, 2:L_elbow(13)->12, 3:R_elbow(14)->15,
# 4:L_wrist(15)->13, 5:R_wrist(16)->16, 6:L_hip(23)->4, 7:R_hip(24)->1,
# 8:L_knee(25)->5, 9:R_knee(26)->2, 10:L_ankle(27)->6, 11:R_ankle(28)->3,
# 12:L_foot(31)->6(duplicate ankle), 13:R_foot(32)->3(duplicate ankle)
FEAT_TO_H36M = {
    0: 11,   # left_shoulder
    1: 14,   # right_shoulder
    2: 12,   # left_elbow
    3: 15,   # right_elbow
    4: 13,   # left_wrist
    5: 16,   # right_wrist
    6: 4,    # left_hip
    7: 1,    # right_hip
    8: 5,    # left_knee
    9: 2,    # right_knee
    10: 6,   # left_ankle
    11: 3,   # right_ankle
}
# Joints 12,13 (feet) map to same as ankles (6,3)


# ── Load Data ────────────────────────────────────────────────────────────────

def load_all_video_data():
    """Load cached features and ground truth for all videos."""
    with open(GT_PATH) as f:
        gt = json.load(f)

    all_data = {}
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".feats.npz"):
            continue
        vname = fname.replace(".feats.npz", "")
        npz = np.load(os.path.join(CACHE_DIR, fname))
        features = npz["features"]
        valid_indices = npz["valid_indices"]
        fps = float(npz["fps"])

        if features.shape[0] < 10:
            continue

        gt_info = gt.get(vname, {})
        if gt_info.get("expected_clusters", -1) == 0:
            continue  # Skip unsuitable videos

        all_data[vname] = {
            "features": features,
            "valid_indices": valid_indices,
            "fps": fps,
            "expected_range": gt_info.get("expected_clusters_range", [1, 10]),
            "expected": gt_info.get("expected_clusters", -1),
            "desc": ", ".join(gt_info.get("distinct_activities", ["unknown"])),
        }

    return all_data


def segment_video(features, valid_indices, fps, min_segment_sec=2.0):
    """Segment features using the production pipeline."""
    segments = segment_motions(features, valid_indices, fps, min_segment_sec=min_segment_sec)
    return segments


def resample_segment(features, target_len=RESAMPLE_LEN):
    """Resample segment to fixed temporal length via linear interpolation."""
    n = features.shape[0]
    if n == target_len:
        return features.astype(np.float32)
    src_x = np.linspace(0, 1, n)
    tgt_x = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, features.shape[1]), dtype=np.float32)
    for d in range(features.shape[1]):
        out[:, d] = np.interp(tgt_x, src_x, features[:, d])
    return out


# ── MotionBERT Integration ──────────────────────────────────────────────────

def srp_features_to_h36m(features_28d):
    """Convert (T, 28) SRP features to (T, 17, 3) approximate H36M format.

    Our SRP features are hip-centered and scale-normalized 2D coordinates.
    We map the 12 unique joints to H36M positions and fill missing joints
    (spine, thorax, neck, head, hip center) with interpolated values.
    The third channel is set to 1.0 (pseudo-confidence).
    """
    T = features_28d.shape[0]
    h36m = np.zeros((T, 17, 3), dtype=np.float32)

    # Reshape (T, 28) -> (T, 14, 2)
    joints_2d = features_28d.reshape(T, 14, 2)

    # Map our feature joints to H36M
    for feat_idx, h36m_idx in FEAT_TO_H36M.items():
        h36m[:, h36m_idx, :2] = joints_2d[:, feat_idx]
        h36m[:, h36m_idx, 2] = 1.0  # confidence

    # Fill missing joints by interpolation
    # Hip center (0) = midpoint of LHip(4) and RHip(1)
    h36m[:, 0, :2] = (h36m[:, 4, :2] + h36m[:, 1, :2]) * 0.5
    h36m[:, 0, 2] = 1.0

    # Spine (7) = midpoint of hip center and thorax
    # Thorax (8) = midpoint of shoulders
    h36m[:, 8, :2] = (h36m[:, 11, :2] + h36m[:, 14, :2]) * 0.5
    h36m[:, 8, 2] = 1.0
    h36m[:, 7, :2] = (h36m[:, 0, :2] + h36m[:, 8, :2]) * 0.5
    h36m[:, 7, 2] = 1.0

    # Neck (9) = slightly above thorax
    h36m[:, 9, :2] = h36m[:, 8, :2] + (h36m[:, 8, :2] - h36m[:, 0, :2]) * 0.3
    h36m[:, 9, 2] = 1.0

    # Head (10) = above neck
    h36m[:, 10, :2] = h36m[:, 9, :2] + (h36m[:, 9, :2] - h36m[:, 8, :2]) * 0.8
    h36m[:, 10, 2] = 1.0

    return h36m


def load_motionbert():
    """Load pretrained MotionBERT DSTformer model."""
    if not os.path.exists(MOTIONBERT_WEIGHTS):
        print("  WARNING: MotionBERT weights not found, skipping.")
        return None

    sys.path.insert(0, MOTIONBERT_REPO)
    try:
        from lib.model.DSTformer import DSTformer
    except ImportError:
        print("  WARNING: MotionBERT repo not found at /tmp/MotionBERT, skipping.")
        return None

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=2,
        num_joints=17, maxlen=243,
    )

    ckpt = torch.load(MOTIONBERT_WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt["model_pos"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def motionbert_embed_segments(model, segments_resampled):
    """Extract MotionBERT embeddings for a list of resampled segments.

    Each segment is (RESAMPLE_LEN, 28) SRP features.
    Returns (N_segments, 512) embeddings.
    """
    embeddings = []
    for seg_28d in segments_resampled:
        # Convert to H36M format: (T, 17, 3)
        h36m = srp_features_to_h36m(seg_28d)
        # MotionBERT expects (B, F, J, C) = (1, T, 17, 3)
        x = torch.tensor(h36m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        # Get representation: (1, T, 17, 512)
        rep = model.get_representation(x)
        # Pool over joints and frames: (1, 512)
        embed = rep.mean(dim=(1, 2)).cpu().numpy()[0]
        embeddings.append(embed)
    return np.stack(embeddings)


# ── Self-Trained Models ──────────────────────────────────────────────────────

class SegmentDataset(Dataset):
    def __init__(self, segments):
        self.segments = [torch.tensor(s, dtype=torch.float32) for s in segments]
    def __len__(self):
        return len(self.segments)
    def __getitem__(self, idx):
        return self.segments[idx]


class ContrastiveDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments
    def __len__(self):
        return len(self.segments)
    def __getitem__(self, idx):
        seg = self.segments[idx]
        return (
            torch.tensor(augment_segment(seg), dtype=torch.float32),
            torch.tensor(augment_segment(seg), dtype=torch.float32),
        )


def augment_segment(seg):
    """Apply random augmentation for contrastive learning."""
    aug = seg.copy()
    r = np.random.random()
    if r < 0.25:
        # Time stretch
        stretch = np.random.uniform(0.7, 1.3)
        temp_len = max(5, int(seg.shape[0] * stretch))
        src_x = np.linspace(0, 1, seg.shape[0])
        mid_x = np.linspace(0, 1, temp_len)
        temp = np.zeros((temp_len, seg.shape[1]), dtype=np.float32)
        for d in range(seg.shape[1]):
            temp[:, d] = np.interp(mid_x, src_x, seg[:, d])
        tgt_x = np.linspace(0, 1, seg.shape[0])
        for d in range(seg.shape[1]):
            aug[:, d] = np.interp(tgt_x, np.linspace(0, 1, temp_len), temp[:, d])
    elif r < 0.5:
        aug = seg + np.random.randn(*seg.shape).astype(np.float32) * 0.02
    elif r < 0.75:
        aug = seg[::-1].copy()
    else:
        crop_frac = np.random.uniform(0.7, 0.95)
        crop_len = max(5, int(seg.shape[0] * crop_frac))
        start = np.random.randint(0, seg.shape[0] - crop_len + 1)
        aug = resample_segment(seg[start:start + crop_len], seg.shape[0])
    return aug


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim=RESAMPLE_LEN * FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )
    def encode(self, x):
        return self.encoder(x.reshape(x.shape[0], -1))
    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        z = self.encoder(flat)
        return self.decoder(z), z


class Conv1DEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_enc = nn.Linear(128, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.ConvTranspose1d(64, feat_dim, kernel_size=5, padding=2),
        )
    def encode(self, x):
        h = self.encoder(x.permute(0, 2, 1)).squeeze(-1)
        return self.fc_enc(h)
    def forward(self, x):
        B, T, D = x.shape
        z = self.encode(x)
        h = self.fc_dec(z).unsqueeze(-1).expand(-1, -1, T)
        return self.decoder(h).permute(0, 2, 1), z


class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM, nhead=4, num_layers=2):
        super().__init__()
        d_model = 64
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, RESAMPLE_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_enc = nn.Linear(d_model, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, feat_dim)
    def encode(self, x):
        h = self.input_proj(x) + self.pos_embed[:, :x.shape[1], :]
        h = self.transformer(h).mean(dim=1)
        return self.fc_enc(h)
    def forward(self, x):
        B, T, D = x.shape
        z = self.encode(x)
        h = self.fc_dec(z).unsqueeze(1).expand(-1, T, -1) + self.pos_embed[:, :T, :]
        h = self.decoder_transformer(h)
        return self.output_proj(h), z


class ContrastiveEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, latent_dim)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
    def encode(self, x):
        h = self.backbone(x.permute(0, 2, 1)).squeeze(-1)
        return self.fc(h)
    def forward(self, x):
        z = self.encode(x)
        p = self.projection(z)
        return F.normalize(p, dim=1), z


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def train_autoencoder(model, segments, epochs=EPOCHS_AE, name="AE"):
    model = model.to(DEVICE)
    loader = DataLoader(SegmentDataset(segments), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, z = model(batch)
            loss = F.mse_loss(recon.reshape(batch.shape[0], -1), batch.reshape(batch.shape[0], -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"    [{name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    model.eval()
    return model


def train_contrastive(model, segments, epochs=EPOCHS_CONTRASTIVE, name="SimCLR"):
    model = model.to(DEVICE)
    loader = DataLoader(ContrastiveDataset(segments), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for v1, v2 in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            p1, _ = model(v1)
            p2, _ = model(v2)
            loss = nt_xent_loss(p1, p2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if n > 0 and (epoch + 1) % 50 == 0:
            print(f"    [{name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def get_embeddings(model, segments):
    model.eval()
    embeddings = []
    for seg in segments:
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z = model.encode(x)
        embeddings.append(z.cpu().numpy()[0])
    return np.stack(embeddings)


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_agglom(embeddings, threshold):
    """Agglomerative clustering on embeddings."""
    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=int), 1
    dists = pdist(embeddings, metric="euclidean")
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance") - 1
    return labels, len(set(labels))


def find_best_threshold(embeddings, expected_range, thresholds=AGGLOM_THRESHOLDS):
    """Find the threshold that produces cluster count closest to expected range."""
    best_thr = thresholds[0]
    best_k = 1
    best_score = -1e9

    for thr in thresholds:
        labels, k = cluster_agglom(embeddings, thr)
        in_range = expected_range[0] <= k <= expected_range[1]
        # Score: prefer in-range, then closer to middle of range
        if in_range:
            mid = (expected_range[0] + expected_range[1]) / 2
            score = 100 - abs(k - mid)
        else:
            score = -abs(k - expected_range[0]) - abs(k - expected_range[1])
        if score > best_score:
            best_score = score
            best_thr = thr
            best_k = k

    return best_thr, best_k


def evaluate_clustering(embeddings, expected_range, thresholds=AGGLOM_THRESHOLDS):
    """Evaluate clustering at multiple thresholds, return summary."""
    results = {}
    for thr in thresholds:
        labels, k = cluster_agglom(embeddings, thr)
        sil = 0.0
        if k >= 2 and len(embeddings) >= 3 and len(set(labels)) >= 2:
            try:
                sil = silhouette_score(embeddings, labels)
            except ValueError:
                pass
        in_range = expected_range[0] <= k <= expected_range[1]
        results[thr] = {"k": k, "silhouette": sil, "in_range": in_range}

    # Find best threshold (in-range with highest silhouette)
    best_thr = None
    best_sil = -1
    for thr, r in results.items():
        if r["in_range"] and r["silhouette"] > best_sil:
            best_sil = r["silhouette"]
            best_thr = thr
    if best_thr is None:
        # No in-range result; pick the one closest to expected
        best_thr, _ = find_best_threshold(embeddings, expected_range, thresholds)

    return results, best_thr


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 90)
    print("LEARNED POSE EMBEDDINGS FOR MOTION CLUSTERING")
    print(f"Device: {DEVICE}")
    print("=" * 90)

    # Step 1: Load all video data
    print("\n[1/6] Loading cached features...")
    all_data = load_all_video_data()
    print(f"  Loaded {len(all_data)} videos")

    # Step 2: Segment all videos and collect resampled segments for training
    print("\n[2/6] Segmenting videos...")
    all_segments_resampled = []
    for vname, vdata in sorted(all_data.items()):
        segments = segment_video(vdata["features"], vdata["valid_indices"], vdata["fps"])
        resampled = [resample_segment(s["features"]) for s in segments]
        vdata["segments"] = segments
        vdata["resampled"] = resampled
        all_segments_resampled.extend(resampled)
        print(f"  {vname}: {len(segments)} segments")

    n_total = len(all_segments_resampled)
    print(f"\n  Total segments across all videos: {n_total}")
    print(f"  Segment shape: ({RESAMPLE_LEN}, {FEAT_DIM})")

    if n_total < 4:
        print("ERROR: Too few segments. Need at least 4.")
        return

    # Step 3: Load/train all embedding models
    print("\n[3/6] Loading/training embedding models...")
    models = {}
    train_times = {}

    # 3a) MotionBERT (pretrained)
    print("\n  --- MotionBERT (pretrained, 42M params) ---")
    t0 = time.time()
    mb_model = load_motionbert()
    if mb_model is not None:
        train_times["MotionBERT"] = time.time() - t0
        models["MotionBERT"] = mb_model
        print(f"    Loaded in {train_times['MotionBERT']:.1f}s")
    else:
        print("    SKIPPED (weights not available)")

    # 3b) MLP Autoencoder
    print("\n  --- MLP Autoencoder ---")
    t0 = time.time()
    mlp = train_autoencoder(MLPAutoencoder(), all_segments_resampled, name="MLP-AE")
    train_times["MLP-AE"] = time.time() - t0
    models["MLP-AE"] = mlp

    # 3c) Conv1D Encoder
    print("\n  --- Conv1D Temporal Encoder ---")
    t0 = time.time()
    conv = train_autoencoder(Conv1DEncoder(), all_segments_resampled, name="Conv1D")
    train_times["Conv1D"] = time.time() - t0
    models["Conv1D"] = conv

    # 3d) Transformer Encoder
    print("\n  --- Transformer Encoder ---")
    t0 = time.time()
    tf = train_autoencoder(TransformerEncoder(), all_segments_resampled, name="Transformer")
    train_times["Transformer"] = time.time() - t0
    models["Transformer"] = tf

    # 3e) Contrastive (SimCLR)
    print("\n  --- Contrastive (SimCLR) ---")
    t0 = time.time()
    simclr = train_contrastive(ContrastiveEncoder(), all_segments_resampled, name="SimCLR")
    train_times["SimCLR"] = time.time() - t0
    models["SimCLR"] = simclr

    print("\n  Training/loading times:")
    for name, t in train_times.items():
        n_params = sum(p.numel() for p in models[name].parameters())
        print(f"    {name:>15}: {t:.1f}s, {n_params:,} params")

    # Step 4: Extract embeddings and evaluate clustering per video
    print("\n[4/6] Extracting embeddings and evaluating clustering...")

    # Store per-method, per-video results
    all_results = {}
    method_names = ["Baseline"] + list(models.keys())

    for vname, vdata in sorted(all_data.items()):
        segments = vdata["segments"]
        resampled = vdata["resampled"]
        expected = vdata["expected_range"]
        n_seg = len(segments)

        if n_seg < 2:
            continue

        print(f"\n  === {vname} ({n_seg} segments, expect {expected[0]}-{expected[1]} clusters) ===")
        print(f"      Activities: {vdata['desc']}")

        video_results = {}

        # Baseline: production spectral distance clustering
        t0 = time.time()
        clustered = cluster_segments(segments, distance_threshold=2.0)
        k_base = len(set(s["cluster"] for s in clustered))
        base_time = time.time() - t0
        in_range_base = expected[0] <= k_base <= expected[1]
        video_results["Baseline"] = {
            "k": k_base, "in_range": in_range_base,
            "best_threshold": 2.0, "time_ms": base_time * 1000,
            "embed_dim": FEAT_DIM,
        }

        # Learned embedding methods
        for model_name, model in models.items():
            t0 = time.time()
            if model_name == "MotionBERT":
                embeddings = motionbert_embed_segments(model, resampled)
            else:
                embeddings = get_embeddings(model, resampled)
            embed_time = time.time() - t0

            results_by_thr, best_thr = evaluate_clustering(embeddings, expected)
            best_result = results_by_thr[best_thr]

            video_results[model_name] = {
                "k": best_result["k"],
                "in_range": best_result["in_range"],
                "silhouette": best_result["silhouette"],
                "best_threshold": best_thr,
                "time_ms": embed_time * 1000,
                "embed_dim": embeddings.shape[1],
                "all_thresholds": {
                    t: {"k": r["k"], "in_range": r["in_range"]}
                    for t, r in results_by_thr.items()
                },
            }

        all_results[vname] = video_results

        # Print comparison
        print(f"      {'Method':>15} | {'K':>3} | {'In Range':>8} | {'Thr':>5} | {'Dim':>4} | {'Time':>8}")
        print(f"      {'-'*15}-+-{'-'*3}-+-{'-'*8}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}")
        for method in method_names:
            if method not in video_results:
                continue
            r = video_results[method]
            mark = "YES" if r["in_range"] else "NO"
            thr_str = f"{r['best_threshold']:.1f}" if isinstance(r["best_threshold"], float) else str(r["best_threshold"])
            print(f"      {method:>15} | {r['k']:>3} | {mark:>8} | {thr_str:>5} | {r['embed_dim']:>4} | {r['time_ms']:>6.1f}ms")

    # Step 5: Inference speed benchmark
    print("\n[5/6] Inference speed benchmark (single segment)...")
    dummy = all_segments_resampled[0]

    for model_name, model in models.items():
        # Warm up
        for _ in range(5):
            if model_name == "MotionBERT":
                motionbert_embed_segments(model, [dummy])
            else:
                get_embeddings(model, [dummy])
        # Benchmark
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            if model_name == "MotionBERT":
                motionbert_embed_segments(model, [dummy])
            else:
                get_embeddings(model, [dummy])
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"    {model_name:>15}: {mean_ms:.3f} +/- {std_ms:.3f} ms/segment")

    # Step 6: Summary
    print("\n[6/6] SUMMARY")
    print("=" * 90)

    # Count passes per method
    pass_counts = {m: 0 for m in method_names}
    total_videos = 0

    for vname, vresults in all_results.items():
        total_videos += 1
        for method in method_names:
            if method in vresults and vresults[method]["in_range"]:
                pass_counts[method] += 1

    print(f"\n  Videos with cluster count in expected range ({total_videos} total):")
    print(f"  {'Method':>15} | {'Pass':>4} | {'Rate':>6}")
    print(f"  {'-'*15}-+-{'-'*4}-+-{'-'*6}")
    for method in method_names:
        if method not in pass_counts:
            continue
        p = pass_counts[method]
        rate = p / total_videos * 100 if total_videos > 0 else 0
        marker = " <-- BEST" if p == max(pass_counts.values()) else ""
        print(f"  {method:>15} | {p:>4} | {rate:>5.1f}%{marker}")

    # Detailed per-video comparison
    print(f"\n  Per-video cluster counts (expected range in brackets):")
    header = f"  {'Video':>30}"
    for method in method_names:
        header += f" | {method:>12}"
    print(header)
    print(f"  {'-'*30}" + "".join(f"-+-{'-'*12}" for _ in method_names))

    for vname in sorted(all_results.keys()):
        exp = all_data[vname]["expected_range"]
        row = f"  {vname:>30}"
        for method in method_names:
            if method in all_results[vname]:
                r = all_results[vname][method]
                mark = "*" if r["in_range"] else " "
                row += f" | {r['k']:>3}{mark}[{exp[0]}-{exp[1]}]"
            else:
                row += f" | {'---':>12}"
        print(row)

    print("\n  * = cluster count within expected range")

    # Practicality assessment
    print("\n  Practicality for real-time streaming (target: <5ms/segment):")
    for model_name in models:
        n_params = sum(p.numel() for p in models[model_name].parameters())
        if model_name == "MotionBERT":
            print(f"    {model_name:>15}: {n_params:,} params - LARGE, requires GPU, needs skeleton format conversion")
        elif n_params < 100_000:
            print(f"    {model_name:>15}: {n_params:,} params - VERY SMALL, suitable for real-time")
        elif n_params < 500_000:
            print(f"    {model_name:>15}: {n_params:,} params - SMALL, likely suitable for real-time")
        else:
            print(f"    {model_name:>15}: {n_params:,} params - MEDIUM, may need optimization")

    # Key insights
    print("\n  KEY INSIGHTS:")
    print("  - MotionBERT is pretrained on AMASS/H36M for 3D pose lifting, not action recognition")
    print("  - Its embeddings capture motion structure but may not separate activities well")
    print("  - Self-trained models adapt to our specific SRP feature distribution")
    print("  - The baseline spectral distance already captures phase-invariant motion patterns")
    print("  - Learned embeddings must beat baseline AND be fast enough for real-time use")

    # Save results
    output = {
        "config": {
            "resample_len": RESAMPLE_LEN, "feat_dim": FEAT_DIM,
            "latent_dim": LATENT_DIM, "device": str(DEVICE),
        },
        "pass_counts": pass_counts,
        "total_videos": total_videos,
        "per_video": {},
    }
    for vname, vresults in all_results.items():
        output["per_video"][vname] = {}
        for method, r in vresults.items():
            # Remove non-serializable items
            clean = {k: v for k, v in r.items() if k != "all_thresholds"}
            output["per_video"][vname][method] = clean

    out_path = "/mnt/Data/GitHub/BRACE/experiments/learned_embeddings_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
