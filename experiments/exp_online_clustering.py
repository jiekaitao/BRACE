"""Experiment: Online/Incremental Clustering for Streaming Video Analysis.

Compares four clustering approaches to find one that STABILIZES cluster count
over time as segments accumulate from streaming video.

CORE PROBLEM: StreamingAnalyzer runs full agglomerative clustering on ALL
accumulated segments each analysis pass. On looped video, segments accumulate
(900+ after 3 loops) and cluster count grows to 81+.

ROOT CAUSE ANALYSIS: In real video processing, the boundary detector finds
slightly different boundary positions each time new data arrives, because the
velocity curve changes shape as context expands. This means the SAME motion
gets segmented into slightly different chunks each pass, and when all
these overlapping segments are thrown into agglomerative clustering, many end
up as separate clusters because their trajectories don't perfectly align.

SIMULATION: We generate segments from the same motion with varying lengths
(simulating boundary instability) and controlled noise to reproduce the
explosion. The threshold is set tight enough that length variation + noise
causes some same-class segments to exceed it.

Approaches:
  A) Centroid Matching - freeze prototypes after initial clustering
  B) Sliding Window - only cluster last N segments
  C) Incremental with Merge - running prototypes with periodic merge
  D) Current (full re-clustering) - baseline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from copy import deepcopy
from collections import defaultdict

from brace.core.motion_segments import (
    cluster_segments,
    _resample_segment,
    _segment_distance,
)
from brace.core.pose import FEATURE_INDICES

FEAT_DIM = len(FEATURE_INDICES) * 2  # 28D
RESAMPLE_LEN = 30


# ---------------------------------------------------------------------------
# Segment generation with realistic boundary instability
# ---------------------------------------------------------------------------

def make_base_pattern(pattern_id, seg_len=30):
    """Distinct motion trajectory."""
    t = np.linspace(0, 2 * np.pi, seg_len)
    p = np.zeros((seg_len, FEAT_DIM), dtype=np.float32)
    if pattern_id == 0:
        for d in range(FEAT_DIM):
            p[:, d] = 0.5 * np.sin(t + d * 0.1)
    elif pattern_id == 1:
        for d in range(FEAT_DIM):
            p[:, d] = 0.8 * np.sin(2.5 * t + d * 0.4) + 0.3 * np.cos(3.2 * t + d * 0.2)
    elif pattern_id == 2:
        for d in range(FEAT_DIM):
            p[:, d] = 0.03 * np.sin(0.5 * t + d * 0.05)
    return p


def make_segment_variable_length(rng, base_pattern, noise_std, len_variation):
    """Create a segment with variable length (simulating boundary instability).

    The key insight: resampling a 22-frame segment vs a 30-frame segment of the
    same motion to the same RESAMPLE_LEN produces different trajectories because
    the phase alignment differs. This is what causes cluster splitting.
    """
    base_len = base_pattern.shape[0]
    # Vary length by +/- len_variation frames
    actual_len = base_len + rng.randint(-len_variation, len_variation + 1)
    actual_len = max(5, actual_len)

    # Resample base pattern to actual_len, then add noise
    src_x = np.linspace(0, 1, base_len)
    tgt_x = np.linspace(0, 1, actual_len)
    feat = np.zeros((actual_len, FEAT_DIM), dtype=np.float32)
    for d in range(FEAT_DIM):
        feat[:, d] = np.interp(tgt_x, src_x, base_pattern[:, d])

    # Add noise
    feat += rng.randn(actual_len, FEAT_DIM).astype(np.float32) * noise_std

    return {
        "features": feat,
        "mean_feature": feat.mean(axis=0),
        "start_frame": 0, "end_frame": actual_len,
        "start_valid": 0, "end_valid": actual_len,
    }


def generate_looped(n_per_loop=12, n_loops=3, noise_std=0.06, len_var=8):
    """Looped video: same exercise across loops, with boundary instability."""
    rng = np.random.RandomState(42)
    pattern = make_base_pattern(0, seg_len=30)
    segs = []
    for loop in range(n_loops):
        for i in range(n_per_loop):
            s = make_segment_variable_length(rng, pattern, noise_std, len_var)
            s["start_frame"] = (loop * n_per_loop + i) * 30
            segs.append(s)
    return segs


def generate_multi(n_per=10, noise_std=0.06, len_var=5):
    """3 distinct activities."""
    rng = np.random.RandomState(200)
    segs = []
    for aid in range(3):
        pat = make_base_pattern(aid, seg_len=30)
        for i in range(n_per):
            s = make_segment_variable_length(rng, pat, noise_std, len_var)
            s["start_frame"] = (aid * n_per + i) * 30
            segs.append(s)
    return segs


def generate_long(n_segments=120, noise_std=0.06, len_var=8):
    """Long session of same motion."""
    rng = np.random.RandomState(300)
    pattern = make_base_pattern(0, seg_len=30)
    segs = []
    for i in range(n_segments):
        s = make_segment_variable_length(rng, pattern, noise_std, len_var)
        s["start_frame"] = i * 30
        segs.append(s)
    return segs


# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

class CentroidMatcher:
    def __init__(self, thr, init_size=5):
        self.thr = thr; self.init = init_size
        self.centroids = []; self.nid = 0; self.ok = False; self.np = 0
    def _c(self, segs):
        return np.mean([_resample_segment(s["features"], RESAMPLE_LEN) for s in segs], axis=0)
    def _d(self, seg, c):
        rs = _resample_segment(seg["features"], RESAMPLE_LEN)
        return float(np.mean(np.linalg.norm(rs - c, axis=1)))
    def process(self, segs):
        if len(segs) <= self.np: return len(self.centroids)
        if not self.ok and len(segs) >= self.init:
            sc = deepcopy(segs)
            cluster_segments(sc, distance_threshold=self.thr)
            cl = defaultdict(list)
            for s in sc: cl[s["cluster"]].append(s)
            self.centroids = [(k, self._c(v)) for k, v in sorted(cl.items())]
            self.nid = max(k for k, _ in self.centroids) + 1
            self.ok = True; self.np = len(segs)
            return len(self.centroids)
        if not self.ok: self.np = len(segs); return 0
        for s in segs[self.np:]:
            ds = [(k, self._d(s, c)) for k, c in self.centroids]
            bk, bd = min(ds, key=lambda x: x[1])
            if bd < self.thr: s["cluster"] = bk
            else:
                self.centroids.append((self.nid, _resample_segment(s["features"], RESAMPLE_LEN)))
                self.nid += 1
        self.np = len(segs)
        return len(self.centroids)


class SlidingWindow:
    def __init__(self, ws=30, thr=2.0):
        self.ws = ws; self.thr = thr
    def process(self, segs):
        if len(segs) < 2: return max(1, len(segs))
        sc = deepcopy(segs[-self.ws:])
        cluster_segments(sc, distance_threshold=self.thr)
        return len(set(s["cluster"] for s in sc))


class IncrementalMerge:
    def __init__(self, thr=2.0, mi=10):
        self.thr = thr; self.mi = mi; self.p = {}; self.nid = 0; self.ns = 0; self.np = 0
    def _d(self, a, b): return float(np.mean(np.linalg.norm(a - b, axis=1)))
    def _merge(self):
        ks = list(self.p.keys()); m = True
        while m:
            m = False
            for i in range(len(ks)):
                for j in range(i+1, len(ks)):
                    ci, cj = ks[i], ks[j]
                    if ci not in self.p or cj not in self.p: continue
                    ca, na = self.p[ci]; cb, nb = self.p[cj]
                    if self._d(ca, cb) < self.thr * 0.7:
                        self.p[ci] = ((ca*na + cb*nb)/(na+nb), na+nb)
                        del self.p[cj]; m = True; break
                if m: ks = list(self.p.keys()); break
    def process(self, segs):
        if len(segs) <= self.np: return len(self.p)
        for s in segs[self.np:]:
            rs = _resample_segment(s["features"], RESAMPLE_LEN)
            if not self.p:
                self.p[self.nid] = (rs, 1); self.nid += 1
            else:
                ds = [(k, self._d(rs, c)) for k, (c, _) in self.p.items()]
                bk, bd = min(ds, key=lambda x: x[1])
                if bd < self.thr:
                    oc, on = self.p[bk]; self.p[bk] = ((oc*on+rs)/(on+1), on+1)
                else:
                    self.p[self.nid] = (rs, 1); self.nid += 1
            self.ns += 1
            if self.ns >= self.mi: self._merge(); self.ns = 0
        self.np = len(segs)
        return len(self.p)


class FullRecluster:
    def __init__(self, thr=2.0): self.thr = thr
    def process(self, segs):
        if len(segs) < 2: return max(1, len(segs))
        sc = deepcopy(segs)
        cluster_segments(sc, distance_threshold=self.thr)
        return len(set(s["cluster"] for s in sc))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(all_segs, chunk=3, thr=2.0):
    a = CentroidMatcher(thr); b = SlidingWindow(30, thr)
    c = IncrementalMerge(thr); d = FullRecluster(thr)
    R = {"A_centroid": [], "B_sliding": [], "C_incremental": [], "D_full_recluster": []}
    n = len(all_segs)
    for end in range(chunk, n+1, chunk):
        cur = all_segs[:end]
        R["A_centroid"].append((end, a.process(cur)))
        R["B_sliding"].append((end, b.process(cur)))
        R["C_incremental"].append((end, c.process(cur)))
        R["D_full_recluster"].append((end, d.process(cur)))
    if n % chunk:
        R["A_centroid"].append((n, a.process(all_segs)))
        R["B_sliding"].append((n, b.process(all_segs)))
        R["C_incremental"].append((n, c.process(all_segs)))
        R["D_full_recluster"].append((n, d.process(all_segs)))
    return R


def print_timeline(R, title, max_w=50):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    mx = max(c for d in R.values() for _, c in d) or 1
    for ap, data in R.items():
        cts = [c for _, c in data]; fin = cts[-1]; pk = max(cts)
        print(f"\n  {ap} (final={fin}, peak={pk}):")
        step = max(1, len(data)//20); sc = max_w/max(mx,1)
        for i in range(0, len(data), step):
            ns, cnt = data[i]
            print(f"    seg{ns:4d} | {'#'*int(cnt*sc)} {cnt}")
    print()


def print_summary(R, title):
    print(f"\n  {title}:")
    print(f"  {'Approach':<25} {'Final':>6} {'Peak':>6} {'Stable?':>8}")
    print(f"  {'-'*50}")
    for ap, data in R.items():
        cts = [c for _, c in data]; fin = cts[-1]; pk = max(cts)
        last = cts[-5:] if len(cts) >= 5 else cts
        st = "YES" if max(last)-min(last) <= 1 else "NO"
        print(f"  {ap:<25} {fin:>6} {pk:>6} {st:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*72)
    print("  EXPERIMENT: Online/Incremental Clustering for Streaming")
    print("  Tests with variable-length segments (boundary instability)")
    print("="*72)

    # Find a threshold that creates the explosion problem
    # Generate same-class segments and measure pairwise distances
    print("\n--- Distance analysis ---")
    test_segs = generate_looped(n_per_loop=30, n_loops=1, noise_std=0.06, len_var=8)
    rng = np.random.RandomState(999)
    dists = []
    for _ in range(500):
        i, j = rng.choice(len(test_segs), 2, replace=False)
        dists.append(_segment_distance(test_segs[i], test_segs[j]))
    dists = np.array(dists)
    print(f"  Within-class: mean={dists.mean():.4f}, std={dists.std():.4f}, "
          f"p50={np.percentile(dists,50):.4f}, p95={np.percentile(dists,95):.4f}, "
          f"max={dists.max():.4f}")

    # Check inter-class
    rng2 = np.random.RandomState(201)
    pat_b = make_base_pattern(1, 30)
    segs_b = [make_segment_variable_length(rng2, pat_b, 0.06, 5) for _ in range(10)]
    inter = [_segment_distance(test_segs[i], segs_b[j])
             for i in range(min(10, len(test_segs))) for j in range(len(segs_b))]
    inter = np.array(inter)
    print(f"  Inter-class:  mean={inter.mean():.4f}, min={inter.min():.4f}")

    # Set threshold at ~80th percentile of within-class distances
    # This means ~20% of same-class pairs exceed threshold -> cluster splitting
    THR = float(np.percentile(dists, 80))
    print(f"  Using threshold at 80th pct of within-class: {THR:.4f}")
    print(f"  Fraction of same-class pairs above threshold: "
          f"{np.mean(dists > THR):.1%}")
    print(f"  Fraction of inter-class pairs above threshold: "
          f"{np.mean(inter > THR):.1%}")

    # ------------------------------------------------------------------
    # Scenario 1: LOOPED VIDEO
    # ------------------------------------------------------------------
    print(f"\n\n>>> SCENARIO 1: LOOPED VIDEO (36 segs, 3 loops)")
    print(f"    Variable segment lengths, noise_std=0.06, len_var=8")
    print(f"    Ideal: 1 cluster, STABLE")
    segs = generate_looped(noise_std=0.06, len_var=8)
    lens = [s["features"].shape[0] for s in segs]
    print(f"    Segment lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    r1 = simulate(segs, chunk=2, thr=THR)
    print_timeline(r1, "LOOPED VIDEO: Cluster Count Over Time")
    print_summary(r1, "LOOPED VIDEO")

    # ------------------------------------------------------------------
    # Scenario 2: MULTI-ACTIVITY
    # ------------------------------------------------------------------
    print(f"\n\n>>> SCENARIO 2: MULTI-ACTIVITY (30 segs, 3 activities)")
    print(f"    Ideal: 3 clusters")
    segs = generate_multi(noise_std=0.06, len_var=5)

    r2 = simulate(segs, chunk=2, thr=THR)
    print_timeline(r2, "MULTI-ACTIVITY: Cluster Count Over Time")
    print_summary(r2, "MULTI-ACTIVITY")

    # ------------------------------------------------------------------
    # Scenario 3: LONG SESSION
    # ------------------------------------------------------------------
    print(f"\n\n>>> SCENARIO 3: LONG SESSION (120 segs, same motion)")
    print(f"    Ideal: 1-2 clusters, should NOT keep growing")
    segs = generate_long(n_segments=120, noise_std=0.06, len_var=8)

    r3 = simulate(segs, chunk=5, thr=THR)
    print_timeline(r3, "LONG SESSION: Cluster Count Over Time")
    print_summary(r3, "LONG SESSION")

    # ------------------------------------------------------------------
    # Conclusions
    # ------------------------------------------------------------------
    print("\n" + "="*72)
    print("  SCORING & CONCLUSIONS")
    print(f"  Threshold: {THR:.4f}")
    print("="*72)

    print(f"\n  {'Approach':<25} {'Loop':>5} {'Multi':>6} {'Long':>5} "
          f"{'LPeak':>6} {'Score':>7}")
    print(f"  {'-'*58}")

    best, bs = None, float("inf")
    for ap in r1:
        lf = r1[ap][-1][1]; mf = r2[ap][-1][1]; sf = r3[ap][-1][1]
        sp = max(c for _, c in r3[ap])
        sl = abs(lf-1)*3; sm = abs(mf-3)*2; ss = abs(sf-1) + max(0,sp-3)
        t = sl + sm + ss
        print(f"  {ap:<25} {lf:>5} {mf:>6} {sf:>5} {sp:>6} {t:>7.1f}")
        if t < bs: bs = t; best = ap

    print(f"\n  >>> WINNER: {best} (score={bs:.1f})")

    print("\n  LONG SESSION stability:")
    for ap in r3:
        cts = [c for _, c in r3[ap]]
        lq = cts[3*len(cts)//4:]
        print(f"    {ap}: final={cts[-1]}, peak={max(cts)}, "
              f"last-quarter=[{min(lq)}-{max(lq)}], var={np.var(lq):.2f}")

    print("\n  KEY FINDINGS:")
    d_final = r3["D_full_recluster"][-1][1]
    b_final = r3["B_sliding"][-1][1]
    a_final = r3["A_centroid"][-1][1]
    c_final = r3["C_incremental"][-1][1]
    print(f"    D (full recluster) final clusters: {d_final}")
    print(f"    A (centroid match)  final clusters: {a_final}")
    print(f"    B (sliding window)  final clusters: {b_final}")
    print(f"    C (incr. merge)     final clusters: {c_final}")

    if d_final > 5:
        print(f"\n    CLUSTER EXPLOSION REPRODUCED: full recluster hit {d_final} "
              f"clusters on single-motion data!")
    if b_final < d_final:
        print(f"    Sliding window caps growth at {b_final} (window_size=30)")
    if a_final < d_final:
        print(f"    Centroid matching stabilized at {a_final}")
    if c_final < d_final:
        print(f"    Incremental merge stabilized at {c_final}")

    print()


if __name__ == "__main__":
    main()
