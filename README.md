# BRACE

SRP-based gait anomaly detection system. Uses Scale/Rotation/Position-invariant skeleton normalization to detect pathological gait patterns from a single baseline of normal walking.

## Core Idea

Train on a person's normal gait → detect when their movement changes (injury, pain, neurological condition). No labeled pathology data needed for training — just compare against the person's own baseline.

## Results (on kooksung pathological gait dataset)

| Metric | Value |
|--------|-------|
| Normal gait mean anomaly score | 0.886 |
| Pathological gait mean anomaly score | 1.398 |
| Separation ratio | 1.6x |
| K-Means clustering ARI | 0.029 |

Every subject's normal gait scores below all their pathological gaits. Lurch gait consistently produces the highest anomaly scores (most biomechanically distinct from normal walking).

## Quick Start

```bash
# Clone and install
git clone <this-repo>
cd BRACE
pip install -e .

# Download data
bash scripts/01_download_data.sh

# Run full demo
python scripts/05_full_demo.py
```

## Project Structure

```
brace/
├── core/
│   ├── srp.py          # SRP normalization (3D body-frame transform)
│   ├── features.py     # Feature extraction + z-score scaling
│   ├── gait_cycle.py   # Gait cycle detection via ankle oscillation
│   ├── baseline.py     # Build motion baseline from normal gait
│   ├── anomaly.py      # Anomaly scoring against baseline
│   └── clustering.py   # Unsupervised gait type clustering
├── data/
│   ├── kinect_loader.py  # Parse Kinect v2 skeleton CSV files
│   └── joint_map.py      # Kinect joint constants
└── viz/
    └── plots.py          # Matplotlib visualizations
```

## Math (ported from EXPERIMENT_PT_coach)

- **SRP normalization**: pelvis-centered, hip-width-scaled, Gram-Schmidt body frame
- **Procrustes alignment**: SVD-based optimal rotation + scale (extended to 3D)
- **Gait cycle detection**: ankle y-oscillation → Butterworth low-pass → peak detection
- **Anomaly scoring**: per-frame RMS deviation from mean normal trajectory in std units
- **Clustering**: K-Means on SRP-normalized cycle feature vectors + t-SNE visualization
