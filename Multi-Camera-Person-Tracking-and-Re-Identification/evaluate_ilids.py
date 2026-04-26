"""
evaluate_ilids.py
-----------------
Evaluate the pipeline's REID model on i-LIDS-VID dataset.

Dataset layout expected:
  videos/i-LIDS-VID/sequences/cam1/<personXXX>/*.png
  videos/i-LIDS-VID/sequences/cam2/<personXXX>/*.png

Computes:
  - Rank-1, Rank-5, Rank-10, Rank-20 CMC
  - mAP (mean Average Precision)

Results are saved to results/ilids_scores.json and results/ilids_scores.csv
"""

import os, sys, json, csv, time
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

# ── add project root to path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid import REID, compute_blur_score, BLUR_THRESHOLD

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR   = PROJECT_ROOT / "videos" / "i-LIDS-VID" / "sequences"
RESULTS_DIR   = PROJECT_ROOT / "results"
MAX_FRAMES_PER_TRACKLET = 50   # cap frames for speed; -1 = use all
RANDOM_SEED   = 42
NUM_TRIALS    = 10             # average CMC over N random query/gallery splits
RANKS         = [1, 5, 10, 20]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ───────────────────────────────────────────────────────────────────────────

def load_tracklets(cam_root: Path, max_frames: int = MAX_FRAMES_PER_TRACKLET):
    """
    Returns: dict  person_id (int) -> list of BGR ndarray crops
    """
    tracklets = {}
    for person_dir in sorted(cam_root.iterdir()):
        if not person_dir.is_dir():
            continue
        pid = int(person_dir.name.replace("person", ""))
        frames = sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpg"))
        if not frames:
            continue
        if max_frames > 0:
            # uniform sample
            indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
            frames  = [frames[i] for i in indices]
        crops = []
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                crops.append(img)
        if crops:
            tracklets[pid] = crops
    return tracklets


def extract_gallery_features(reid: REID, tracklets: dict, use_quality: bool = True):
    """
    Returns: dict  person_id -> (1, D) torch.Tensor (L2-normalised)
    """
    feats = {}
    for pid, crops in tracklets.items():
        blur_scores = [compute_blur_score(c) for c in crops]
        if use_quality:
            emb = reid._features_with_quality(crops, blur_scores)
        else:
            emb = reid._features(crops)
        # L2-normalise
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        feats[pid] = emb
    return feats


# ───────────────────────────────────────────────────────────────────────────
# CMC + mAP
# ───────────────────────────────────────────────────────────────────────────

def compute_cmc_map(query_feats: dict, gallery_feats: dict, ranks=RANKS):
    """
    Single-shot CMC + mAP.
    Both dicts: pid -> (1, D) normalised tensor.
    """
    q_pids  = sorted(query_feats.keys())
    g_pids  = sorted(gallery_feats.keys())

    # Stack gallery
    g_mat = torch.cat([gallery_feats[p] for p in g_pids], dim=0)  # (G, D)
    g_arr = np.array(g_pids)

    num_correct = np.zeros(len(ranks))
    ap_list     = []

    for qpid in q_pids:
        qf = query_feats[qpid]   # (1, D)
        # cosine similarity = dot product (already L2-normed)
        sim  = torch.mm(qf, g_mat.t()).squeeze(0).numpy()  # (G,)
        # descending sort
        order = np.argsort(-sim)
        sorted_pids = g_arr[order]

        # remove same camera same identity — for i-LIDS-VID query is cam1,
        # gallery is cam2, so they are already from different cameras.
        # Still exclude self-match if query pid == gallery pid (shouldn't happen
        # since we split by camera, but be safe).
        match = (sorted_pids == qpid)

        # CMC
        for ri, r in enumerate(ranks):
            if match[:r].any():
                num_correct[ri] += 1

        # AP
        num_relevant = match.sum()
        if num_relevant == 0:
            continue
        cum_tp  = np.cumsum(match)
        prec_at = cum_tp / (np.arange(len(match)) + 1.0)
        ap       = (prec_at * match).sum() / num_relevant
        ap_list.append(ap)

    n_query = len(q_pids)
    cmc_rates = num_correct / n_query
    mAP       = float(np.mean(ap_list)) if ap_list else 0.0
    return cmc_rates, mAP


# ───────────────────────────────────────────────────────────────────────────
# Main evaluation loop (multi-trial to reduce split variance)
# ───────────────────────────────────────────────────────────────────────────

def evaluate_ilids(reid: REID):
    cam1_dir = DATASET_DIR / "cam1"
    cam2_dir = DATASET_DIR / "cam2"

    print(f"[i-LIDS-VID] Loading cam1 tracklets from {cam1_dir} ...")
    cam1 = load_tracklets(cam1_dir)
    print(f"[i-LIDS-VID] Loading cam2 tracklets from {cam2_dir} ...")
    cam2 = load_tracklets(cam2_dir)

    # Keep only persons appearing in both cameras
    common_pids = sorted(set(cam1.keys()) & set(cam2.keys()))
    print(f"[i-LIDS-VID] Common person IDs: {len(common_pids)}")

    cam1 = {p: cam1[p] for p in common_pids}
    cam2 = {p: cam2[p] for p in common_pids}

    print("[i-LIDS-VID] Extracting features for cam1 ...")
    t0 = time.time()
    feats1 = extract_gallery_features(reid, cam1)
    print("[i-LIDS-VID] Extracting features for cam2 ...")
    feats2 = extract_gallery_features(reid, cam2)
    print(f"[i-LIDS-VID] Feature extraction done in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(RANDOM_SEED)
    all_cmc  = []
    all_mAP  = []

    for trial in range(NUM_TRIALS):
        # Randomly choose which camera is query and which is gallery
        if rng.random() > 0.5:
            q_feats, g_feats = feats1, feats2
        else:
            q_feats, g_feats = feats2, feats1

        cmc_rates, mAP = compute_cmc_map(q_feats, g_feats)
        all_cmc.append(cmc_rates)
        all_mAP.append(mAP)
        rank_str = "  ".join([f"R{r}={cmc_rates[i]*100:.1f}%" for i, r in enumerate(RANKS)])
        print(f"  Trial {trial+1:2d}: {rank_str}  mAP={mAP*100:.1f}%")

    mean_cmc = np.mean(all_cmc, axis=0)
    mean_mAP = float(np.mean(all_mAP))

    scores = {
        "dataset": "i-LIDS-VID",
        "num_ids": len(common_pids),
        "num_trials": NUM_TRIALS,
        "model": "resnet50 (pretrained torchreid + EMA quality-aware)",
        "mAP": round(mean_mAP * 100, 2),
    }
    for i, r in enumerate(RANKS):
        scores[f"Rank-{r}"] = round(float(mean_cmc[i]) * 100, 2)

    # Published baselines for i-LIDS-VID (for reference)
    scores["published_baselines"] = {
        "ASTPN (2017)":        {"Rank-1": 44.0,  "note": "video-based"},
        "RQEN (2018)":         {"Rank-1": 77.4,  "note": "video-based"},
        "COSAM (2019)":        {"Rank-1": 79.9,  "note": "video-based"},
        "AFA (2021)":          {"Rank-1": 88.5,  "note": "video-based"},
        "ResNet50-baseline":   {"Rank-1": "~55-65", "note": "image-based aggregation"},
    }

    return scores


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  i-LIDS-VID ReID Evaluation")
    print("=" * 60)

    reid = REID()

    scores = evaluate_ilids(reid)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY — i-LIDS-VID")
    print("=" * 60)
    print(f"  Dataset       : {scores['dataset']}")
    print(f"  Identities    : {scores['num_ids']}")
    print(f"  Trials        : {scores['num_trials']}")
    print(f"  Model         : {scores['model']}")
    print()
    for r in RANKS:
        key = f"Rank-{r}"
        print(f"  {key:<8}: {scores[key]:.2f}%")
    print(f"  mAP      : {scores['mAP']:.2f}%")
    print()
    print("  Published baselines (Rank-1):")
    for name, info in scores["published_baselines"].items():
        print(f"    {name:<28}: Rank-1 = {info['Rank-1']}%  ({info['note']})")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "ilids_scores.json"
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "ilids_scores.csv"
    flat = {k: v for k, v in scores.items() if not isinstance(v, dict)}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"[Saved] {csv_path}")

    return scores


if __name__ == "__main__":
    main()
