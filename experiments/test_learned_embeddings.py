#!/usr/bin/env python3
"""Test learned motion embeddings for better clustering.

Compares:
1. Baseline: hand-crafted spectral distance (current system)
2. MLP Autoencoder: simple bottleneck embedding
3. 1D-CNN Temporal Encoder: convolutions over time
4. Transformer Encoder: self-attention with mean pooling
5. Contrastive Learning (SimCLR-style): augmentation-based

Each approach:
- Extracts pose features from videos using YOLO-pose
- Trains an encoder on the extracted segments
- Produces fixed-size embeddings per segment
- Clusters with KMeans in latent space
- Reports cluster counts, silhouette scores, and runtime
"""

import sys
sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import os
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from ultralytics import YOLO
from brace.core.motion_segments import (
    normalize_frame,
    segment_motions,
    cluster_segments,
    _segment_distance,
    detect_motion_boundaries,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
VIDEOS = [
    "basketball_solo.mp4",
    "exercise.mp4",
    "gym_crossfit.mp4",
    "mma_spar.mp4",
    "soccer_match2.mp4",
]

# Expected cluster behavior (approximate ground truth)
EXPECTED = {
    "basketball_solo.mp4": {"min": 2, "max": 3, "desc": "dribble + dunk"},
    "exercise.mp4": {"min": 1, "max": 3, "desc": "single exercise type"},
    "gym_crossfit.mp4": {"min": 3, "max": 7, "desc": "multiple exercises"},
    "mma_spar.mp4": {"min": 2, "max": 5, "desc": "different fight sequences"},
    "soccer_match2.mp4": {"min": 2, "max": 5, "desc": "run/kick/idle"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLE_LEN = 30  # Fixed temporal length for segments
FEAT_DIM = 28      # 14 joints x 2 coords
LATENT_DIM = 16    # Embedding dimension
BATCH_SIZE = 32
EPOCHS_AE = 100
EPOCHS_CONTRASTIVE = 150
LR = 1e-3


# ── Feature Extraction ─────────────────────────────────────────────────────

def extract_features_from_video(video_path: str) -> tuple[np.ndarray, list[int], float]:
    """Extract SRP-normalized features using YOLO-pose."""
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    features = []
    valid_indices = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            xy = kp.xy.cpu().numpy()[0]
            conf = kp.conf.cpu().numpy()[0]
            kp_with_conf = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_with_conf, img_w, img_h)
            feat = normalize_frame(mp33)
            if feat is not None:
                feat_vec = feat[FEATURE_INDICES, :2].flatten()
                if not (np.any(np.isnan(feat_vec)) or np.any(np.isinf(feat_vec))):
                    features.append(feat_vec)
                    valid_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    if not features:
        return np.zeros((0, FEAT_DIM)), [], fps
    return np.stack(features), valid_indices, fps


def resample_segment(features: np.ndarray, target_len: int = RESAMPLE_LEN) -> np.ndarray:
    """Resample segment to fixed temporal length."""
    if features.shape[0] == target_len:
        return features.astype(np.float32)
    src_x = np.linspace(0, 1, features.shape[0])
    tgt_x = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, features.shape[1]), dtype=np.float32)
    for d in range(features.shape[1]):
        out[:, d] = np.interp(tgt_x, src_x, features[:, d])
    return out


# ── Data Augmentation for Contrastive Learning ─────────────────────────────

def augment_segment(seg: np.ndarray) -> np.ndarray:
    """Apply random augmentation to a motion segment."""
    aug = seg.copy()
    r = np.random.random()

    if r < 0.25:
        # Time stretch: resample to random length then back
        stretch = np.random.uniform(0.7, 1.3)
        temp_len = max(5, int(seg.shape[0] * stretch))
        src_x = np.linspace(0, 1, seg.shape[0])
        mid_x = np.linspace(0, 1, temp_len)
        tgt_x = np.linspace(0, 1, seg.shape[0])
        temp = np.zeros((temp_len, seg.shape[1]), dtype=np.float32)
        for d in range(seg.shape[1]):
            temp[:, d] = np.interp(mid_x, src_x, seg[:, d])
        for d in range(seg.shape[1]):
            aug[:, d] = np.interp(tgt_x, np.linspace(0, 1, temp_len), temp[:, d])
    elif r < 0.5:
        # Add Gaussian noise
        noise = np.random.randn(*seg.shape).astype(np.float32) * 0.02
        aug = seg + noise
    elif r < 0.75:
        # Temporal reverse
        aug = seg[::-1].copy()
    else:
        # Random crop and pad back
        crop_frac = np.random.uniform(0.7, 0.95)
        crop_len = max(5, int(seg.shape[0] * crop_frac))
        start = np.random.randint(0, seg.shape[0] - crop_len + 1)
        cropped = seg[start:start + crop_len]
        aug = resample_segment(cropped, seg.shape[0])

    return aug


# ── Datasets ───────────────────────────────────────────────────────────────

class SegmentDataset(Dataset):
    """Dataset of resampled motion segments for autoencoder training."""
    def __init__(self, segments: list[np.ndarray]):
        self.segments = [torch.tensor(s, dtype=torch.float32) for s in segments]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]  # (T, D)


class ContrastiveDataset(Dataset):
    """Dataset that returns pairs of augmented views for contrastive learning."""
    def __init__(self, segments: list[np.ndarray]):
        self.segments = segments  # list of (T, D) np arrays

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        view1 = torch.tensor(augment_segment(seg), dtype=torch.float32)
        view2 = torch.tensor(augment_segment(seg), dtype=torch.float32)
        return view1, view2


# ── Models ─────────────────────────────────────────────────────────────────

class MLPAutoencoder(nn.Module):
    """Simple MLP autoencoder. Flattens (T, D) -> T*D, encodes to latent."""
    def __init__(self, input_dim=RESAMPLE_LEN * FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        # x: (B, T, D) -> (B, T*D)
        B = x.shape[0]
        return self.encoder(x.reshape(B, -1))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        B = x.shape[0]
        flat = x.reshape(B, -1)
        z = self.encoder(flat)
        recon = self.decoder(z)
        return recon, z


class Conv1DEncoder(nn.Module):
    """1D-CNN temporal encoder. Processes (B, D, T) with temporal convolutions."""
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)
        )
        self.fc_enc = nn.Linear(128, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, feat_dim, kernel_size=5, padding=2),
        )

    def encode(self, x):
        # x: (B, T, D) -> (B, D, T)
        h = self.encoder(x.permute(0, 2, 1))  # (B, 128, 1)
        return self.fc_enc(h.squeeze(-1))  # (B, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        z = self.encode(x)
        h = self.fc_dec(z).unsqueeze(-1).expand(-1, -1, T)  # (B, 128, T)
        recon = self.decoder(h)  # (B, D, T)
        return recon.permute(0, 2, 1), z  # (B, T, D), (B, latent)


class TransformerEncoder(nn.Module):
    """Small Transformer encoder with mean pooling for fixed-size embedding."""
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM, nhead=4, num_layers=2):
        super().__init__()
        d_model = 64
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, RESAMPLE_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_enc = nn.Linear(d_model, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, feat_dim)

    def encode(self, x):
        # x: (B, T, D)
        h = self.input_proj(x) + self.pos_embed[:, :x.shape[1], :]
        h = self.transformer(h)  # (B, T, d_model)
        pooled = h.mean(dim=1)  # (B, d_model)
        return self.fc_enc(pooled)  # (B, latent_dim)

    def forward(self, x):
        B, T, D = x.shape
        z = self.encode(x)
        h = self.fc_dec(z).unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)
        h = h + self.pos_embed[:, :T, :]
        h = self.decoder_transformer(h)
        recon = self.output_proj(h)  # (B, T, D)
        return recon, z


class ContrastiveEncoder(nn.Module):
    """SimCLR-style encoder. Same backbone as Conv1D but with projection head."""
    def __init__(self, feat_dim=FEAT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, latent_dim)
        # Projection head (only used during training)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode(self, x):
        # x: (B, T, D)
        h = self.backbone(x.permute(0, 2, 1)).squeeze(-1)
        return self.fc(h)

    def forward(self, x):
        z = self.encode(x)
        p = self.projection(z)
        return F.normalize(p, dim=1), z


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ── Training ───────────────────────────────────────────────────────────────

def train_autoencoder(model, segments, epochs=EPOCHS_AE, name="AE"):
    """Train an autoencoder model and return it."""
    model = model.to(DEVICE)
    dataset = SegmentDataset(segments)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
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
        if (epoch + 1) % 25 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    model.eval()
    return model


def train_contrastive(model, segments, epochs=EPOCHS_CONTRASTIVE, name="Contrastive"):
    """Train contrastive model."""
    model = model.to(DEVICE)
    dataset = ContrastiveDataset(segments)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for v1, v2 in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            p1, _ = model(v1)
            p2, _ = model(v2)
            loss = nt_xent_loss(p1, p2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if n_batches > 0 and (epoch + 1) % 25 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    model.eval()
    return model


# ── Embedding & Clustering ─────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(model, segments):
    """Get embeddings for all segments."""
    model.eval()
    embeddings = []
    for seg in segments:
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z = model.encode(x)
        embeddings.append(z.cpu().numpy()[0])
    return np.stack(embeddings)


def cluster_embeddings_kmeans(embeddings, n_clusters_range=(2, 8)):
    """Find best K via silhouette score."""
    if len(embeddings) < 3:
        return np.zeros(len(embeddings), dtype=int), 1, 0.0

    best_score = -1
    best_labels = None
    best_k = 2

    max_k = min(n_clusters_range[1], len(embeddings) - 1)
    for k in range(n_clusters_range[0], max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        return np.zeros(len(embeddings), dtype=int), 1, 0.0
    return best_labels, best_k, best_score


def cluster_embeddings_agglom(embeddings, threshold=3.5):
    """Agglomerative clustering in embedding space (like the baseline but on learned embeddings)."""
    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=int), 1

    dists = pdist(embeddings, metric='euclidean')
    Z = linkage(dists, method='single')
    labels = fcluster(Z, t=threshold, criterion='distance') - 1  # 0-indexed
    n_clusters = len(set(labels))
    return labels, n_clusters


# ── Baseline: Current System ───────────────────────────────────────────────

def baseline_clustering(features, valid_indices, fps, threshold=3.5):
    """Run the current hand-crafted spectral distance clustering."""
    segments = segment_motions(features, valid_indices, fps, min_segment_sec=1.0)
    if len(segments) < 2:
        return len(segments), 0.0, segments

    segments = cluster_segments(segments, distance_threshold=threshold)
    n_clusters = len(set(s["cluster"] for s in segments))

    # Compute silhouette if possible
    if n_clusters >= 2 and len(segments) >= 3:
        resampled = np.stack([resample_segment(s["features"]).flatten() for s in segments])
        labels = [s["cluster"] for s in segments]
        sil = silhouette_score(resampled, labels)
    else:
        sil = 0.0

    return n_clusters, sil, segments


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 80)
    print("LEARNED MOTION EMBEDDINGS FOR CLUSTERING")
    print(f"Device: {DEVICE}, Latent dim: {LATENT_DIM}, Resample len: {RESAMPLE_LEN}")
    print("=" * 80)

    # Step 1: Extract features from all videos
    print("\n[1/5] Extracting pose features from videos...")
    all_video_data = {}
    all_segments_resampled = []  # For training

    for vname in VIDEOS:
        vpath = os.path.join(VIDEO_DIR, vname)
        if not os.path.exists(vpath):
            print(f"  SKIP: {vname} not found")
            continue

        print(f"  Processing {vname}...")
        t0 = time.time()
        features, valid_indices, fps = extract_features_from_video(vpath)
        extract_time = time.time() - t0
        print(f"    {features.shape[0]} frames, {fps:.1f} fps, extracted in {extract_time:.1f}s")

        # Segment
        segments = segment_motions(features, valid_indices, fps, min_segment_sec=1.0)
        print(f"    {len(segments)} segments")

        # Resample segments for training
        resampled = [resample_segment(s["features"]) for s in segments]
        all_segments_resampled.extend(resampled)

        all_video_data[vname] = {
            "features": features,
            "valid_indices": valid_indices,
            "fps": fps,
            "segments": segments,
            "resampled": resampled,
        }

    n_total = len(all_segments_resampled)
    print(f"\nTotal segments across all videos: {n_total}")
    print(f"Segment shape: ({RESAMPLE_LEN}, {FEAT_DIM})")

    if n_total < 4:
        print("ERROR: Too few segments to train models. Need at least 4.")
        return

    # Step 2: Train models on ALL segments pooled
    print("\n[2/5] Training models on pooled segments...")

    models = {}
    train_times = {}

    # 2a) MLP Autoencoder
    print("\n--- MLP Autoencoder ---")
    t0 = time.time()
    mlp_ae = MLPAutoencoder()
    mlp_ae = train_autoencoder(mlp_ae, all_segments_resampled, name="MLP-AE")
    train_times["MLP-AE"] = time.time() - t0
    models["MLP-AE"] = mlp_ae

    # 2b) Conv1D Encoder
    print("\n--- Conv1D Temporal Encoder ---")
    t0 = time.time()
    conv_ae = Conv1DEncoder()
    conv_ae = train_autoencoder(conv_ae, all_segments_resampled, name="Conv1D")
    train_times["Conv1D"] = time.time() - t0
    models["Conv1D"] = conv_ae

    # 2c) Transformer Encoder
    print("\n--- Transformer Encoder ---")
    t0 = time.time()
    tf_ae = TransformerEncoder()
    tf_ae = train_autoencoder(tf_ae, all_segments_resampled, name="Transformer")
    train_times["Transformer"] = time.time() - t0
    models["Transformer"] = tf_ae

    # 2d) Contrastive (SimCLR)
    print("\n--- Contrastive (SimCLR) ---")
    t0 = time.time()
    contrastive = ContrastiveEncoder()
    contrastive = train_contrastive(contrastive, all_segments_resampled, name="SimCLR")
    train_times["SimCLR"] = time.time() - t0
    models["SimCLR"] = contrastive

    print("\nTraining times:")
    for name, t in train_times.items():
        print(f"  {name}: {t:.1f}s")

    # Step 3: Evaluate each approach on each video
    print("\n[3/5] Evaluating clustering per video...")

    results = {}

    for vname, vdata in all_video_data.items():
        segments = vdata["segments"]
        resampled = vdata["resampled"]
        n_seg = len(segments)

        if n_seg < 2:
            print(f"\n  {vname}: Only {n_seg} segment(s), skipping")
            continue

        print(f"\n  === {vname} ({n_seg} segments) ===")
        expected = EXPECTED.get(vname, {"min": 1, "max": 10, "desc": "unknown"})
        print(f"  Expected: {expected['desc']} ({expected['min']}-{expected['max']} clusters)")

        video_results = {}

        # Baseline
        t0 = time.time()
        n_clust_base, sil_base, _ = baseline_clustering(
            vdata["features"], vdata["valid_indices"], vdata["fps"]
        )
        base_time = time.time() - t0
        video_results["Baseline"] = {
            "n_clusters": n_clust_base,
            "silhouette": sil_base,
            "time_ms": base_time * 1000,
        }

        # Learned embedding methods
        for model_name, model in models.items():
            t0 = time.time()
            embeddings = get_embeddings(model, resampled)

            # KMeans clustering
            labels_km, k_km, sil_km = cluster_embeddings_kmeans(embeddings)

            # Also try agglomerative with various thresholds to find best
            best_agglom_k = 1
            best_agglom_sil = -1
            for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
                labels_ag, k_ag = cluster_embeddings_agglom(embeddings, threshold=thr)
                if k_ag >= 2 and len(set(labels_ag)) >= 2 and n_seg >= 3:
                    try:
                        sil_ag = silhouette_score(embeddings, labels_ag)
                        if sil_ag > best_agglom_sil:
                            best_agglom_sil = sil_ag
                            best_agglom_k = k_ag
                    except:
                        pass

            embed_time = time.time() - t0

            video_results[model_name] = {
                "n_clusters_kmeans": k_km,
                "silhouette_kmeans": sil_km,
                "n_clusters_agglom": best_agglom_k,
                "silhouette_agglom": best_agglom_sil,
                "time_ms": embed_time * 1000,
                "embedding_shape": embeddings.shape,
            }

        results[vname] = video_results

        # Print comparison table
        print(f"  {'Method':>20} | {'K (KM)':>6} | {'Sil (KM)':>8} | {'K (Agg)':>7} | {'Sil (Agg)':>9} | {'Time':>8}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}")

        br = video_results["Baseline"]
        print(f"  {'Baseline':>20} | {br['n_clusters']:>6} | {br['silhouette']:>8.3f} | {'':>7} | {'':>9} | {br['time_ms']:>6.1f}ms")

        for model_name in models:
            mr = video_results[model_name]
            print(f"  {model_name:>20} | {mr['n_clusters_kmeans']:>6} | {mr['silhouette_kmeans']:>8.3f} | {mr['n_clusters_agglom']:>7} | {mr['silhouette_agglom']:>9.3f} | {mr['time_ms']:>6.1f}ms")

    # Step 4: Inference speed benchmark
    print("\n[4/5] Inference speed benchmark (single segment)...")
    dummy_seg = all_segments_resampled[0]

    for model_name, model in models.items():
        # Warm up
        for _ in range(10):
            get_embeddings(model, [dummy_seg])

        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            get_embeddings(model, [dummy_seg])
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"  {model_name:>20}: {mean_ms:.3f} +/- {std_ms:.3f} ms/segment")

    # Step 5: Summary
    print("\n[5/5] Summary & Recommendations")
    print("=" * 80)

    print("\nTraining times:")
    for name, t in train_times.items():
        n_params = sum(p.numel() for p in models[name].parameters())
        print(f"  {name:>20}: {t:.1f}s training, {n_params:,} params")

    print("\nCluster quality across videos:")
    for vname in results:
        expected = EXPECTED.get(vname, {"min": 1, "max": 10})
        print(f"\n  {vname}:")
        for method, mr in results[vname].items():
            if method == "Baseline":
                k = mr["n_clusters"]
                sil = mr["silhouette"]
                in_range = expected["min"] <= k <= expected["max"]
                mark = "OK" if in_range else "MISS"
                print(f"    {method:>20}: K={k} sil={sil:.3f} [{mark}]")
            else:
                k_km = mr["n_clusters_kmeans"]
                sil_km = mr["silhouette_kmeans"]
                in_range_km = expected["min"] <= k_km <= expected["max"]
                mark_km = "OK" if in_range_km else "MISS"
                print(f"    {method:>20}: K={k_km}(KM) sil={sil_km:.3f} [{mark_km}]")

    # Practicality assessment
    print("\nPracticality for real-time streaming:")
    print("  The streaming system processes ~30fps. Clustering runs periodically (every ~1s).")
    print("  A method is practical if single-segment inference < 5ms.")
    for model_name in models:
        n_params = sum(p.numel() for p in models[model_name].parameters())
        print(f"  {model_name:>20}: {n_params:,} params - ", end="")
        if n_params < 100_000:
            print("VERY SMALL - suitable for real-time")
        elif n_params < 500_000:
            print("SMALL - likely suitable for real-time")
        else:
            print("MEDIUM - may need optimization for real-time")


if __name__ == "__main__":
    run_experiment()
