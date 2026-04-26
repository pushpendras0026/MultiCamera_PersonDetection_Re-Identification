"""
evaluate_market1501.py
----------------------
Evaluate the pipeline's REID model on Market-1501 dataset.

Expected dataset layout (standard Market-1501):
  <market_root>/
    bounding_box_train/   <- not used in eval
    bounding_box_test/    <- gallery images
    query/                <- query images

Image naming: <pid>_c<cam>s<seq>_<frame>_<det>.jpg
  pid = 4-digit person ID (0000 = junk, -1 = distractor)
  e.g. 0002_c1s1_000451_03.jpg

Computes:
  - Rank-1, Rank-5, Rank-10, Rank-20 CMC
  - mAP

Results saved to results/market1501_scores.json and results/market1501_scores.csv
"""

import os, sys, json, csv, time, glob
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid import REID, compute_blur_score, BLUR_THRESHOLD

# ── Config ──────────────────────────────────────────────────────────────────
# Adjust this to wherever the user extracted the Market-1501 zip
MARKET_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data" / "Market_1501_data" / "Market-1501-v15.09.15",
    PROJECT_ROOT / "data" / "Market-1501-v15.09.15",
    PROJECT_ROOT / "data" / "Market_1501_data",
    PROJECT_ROOT / "data" / "Market1501",
    PROJECT_ROOT / "data" / "market1501",
    Path("C:/Users/Lenovo/Downloads/Market-1501-v15.09.15"),
]

RESULTS_DIR = PROJECT_ROOT / "results"
RANKS       = [1, 5, 10, 20]
# Subsample gallery for speed: -1 = all, N = random N images per ID
MAX_GALLERY_PER_ID = -1
# Subsample query: -1 = all
MAX_QUERY = 500   # 500 queries gives stable CMC/mAP and runs fast

os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def find_market_root():
    for p in MARKET_ROOT_CANDIDATES:
        if p.exists() and (p / "query").exists():
            return p
    return None


def parse_market_filename(fname: str):
    """Returns (pid, camid) from Market-1501 filename."""
    name = Path(fname).stem
    parts = name.split("_")
    pid   = int(parts[0])
    camid = int(parts[1][1]) - 1   # 'c1' -> 0
    return pid, camid


def load_image_dir(directory: Path, skip_pids=(-1, 0)):
    """
    Returns list of (filepath, pid, camid) for all valid images.
    skip_pids: ignore distractor (-1) and junk (0) person IDs.
    """
    records = []
    for fp in sorted(directory.glob("*.jpg")):
        try:
            pid, camid = parse_market_filename(fp.name)
        except Exception:
            continue
        if pid in skip_pids:
            continue
        records.append((fp, pid, camid))
    return records


def group_by_pid(records):
    """Group list of (fp, pid, camid) by pid."""
    grouped = defaultdict(list)
    for fp, pid, camid in records:
        grouped[pid].append((fp, camid))
    return grouped


# ───────────────────────────────────────────────────────────────────────────
# Feature extraction
# ───────────────────────────────────────────────────────────────────────────

def extract_single_image_feature(reid: REID, img_path: Path):
    """Extract (1, D) feature from a single image file."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    blur = compute_blur_score(img)
    emb  = reid._embed_crop(img)                          # (1, D)
    emb  = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)  # L2-norm
    return emb


def extract_features_list(reid: REID, records, desc=""):
    """
    records: list of (filepath, pid, camid)
    Returns: (feature_matrix np.ndarray [N,D], pids list, camids list)
    """
    feats  = []
    pids   = []
    camids = []
    total  = len(records)
    for i, (fp, pid, camid) in enumerate(records):
        if i % 200 == 0:
            print(f"  {desc} {i}/{total} ...")
        emb = extract_single_image_feature(reid, fp)
        if emb is None:
            continue
        feats.append(emb.numpy())
        pids.append(pid)
        camids.append(camid)
    feat_mat = np.vstack(feats) if feats else np.zeros((0, 2048))
    return feat_mat, np.array(pids), np.array(camids)


# ───────────────────────────────────────────────────────────────────────────
# Standard Market-1501 CMC + mAP
# (follows the DukeMTMC/Market-1501 evaluation protocol)
# ───────────────────────────────────────────────────────────────────────────

def eval_market_protocol(q_feat, q_pids, q_camids,
                          g_feat, g_pids, g_camids,
                          ranks=RANKS):
    """
    Standard single-query evaluation protocol.
    Removes same-camera same-pid gallery entries from consideration
    (junk entries) when computing CMC / AP.
    """
    num_q   = q_feat.shape[0]
    num_r   = len(ranks)
    all_cmc = []
    all_ap  = []

    # cosine distance matrix: (Q, G)
    # features are already L2-normalised → cosine_sim = dot product
    dist_mat = 1.0 - (q_feat @ g_feat.T)   # (Q, G) distance

    for qi in range(num_q):
        qpid   = q_pids[qi]
        qcamid = q_camids[qi]
        dist   = dist_mat[qi]

        order = np.argsort(dist)
        g_pids_sorted   = g_pids[order]
        g_camids_sorted = g_camids[order]

        # Junk: same pid & same camera (exclude from denominator)
        junk_idx = np.where(
            (g_pids_sorted == qpid) & (g_camids_sorted == qcamid)
        )[0]
        # Good: same pid, different camera
        good_idx = np.where(
            (g_pids_sorted == qpid) & (g_camids_sorted != qcamid)
        )[0]

        if len(good_idx) == 0:
            continue

        # Remove junk from ordered list
        mask_keep = np.ones(len(g_pids_sorted), dtype=bool)
        mask_keep[junk_idx] = False
        g_pids_clean   = g_pids_sorted[mask_keep]

        match = (g_pids_clean == qpid)

        # CMC
        cmc = np.zeros(num_r)
        for ri, r in enumerate(ranks):
            if match[:r].any():
                cmc[ri] = 1.0
        all_cmc.append(cmc)

        # AP
        num_rel = match.sum()
        cum_tp  = np.cumsum(match)
        prec    = cum_tp / (np.arange(len(match)) + 1.0)
        ap      = (prec * match).sum() / num_rel
        all_ap.append(ap)

    if not all_cmc:
        return np.zeros(num_r), 0.0

    mean_cmc = np.mean(all_cmc, axis=0)
    mAP      = float(np.mean(all_ap))
    return mean_cmc, mAP


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def evaluate_market1501(reid: REID):
    market_root = find_market_root()
    if market_root is None:
        print("\n[WARN] Market-1501 dataset not found.")
        print("  Searched:")
        for p in MARKET_ROOT_CANDIDATES:
            print(f"    {p}")
        print("  Please extract the zip to one of the above locations and re-run.")
        return None

    print(f"[Market-1501] Found dataset at: {market_root}")

    query_dir   = market_root / "query"
    gallery_dir = market_root / "bounding_box_test"

    print("[Market-1501] Loading query records ...")
    query_records = load_image_dir(query_dir)
    print(f"[Market-1501] Query images: {len(query_records)}")

    print("[Market-1501] Loading gallery records ...")
    gallery_records = load_image_dir(gallery_dir)
    print(f"[Market-1501] Gallery images: {len(gallery_records)}")

    # Optional: subsample query for faster evaluation
    rng = np.random.default_rng(42)
    if MAX_QUERY > 0 and len(query_records) > MAX_QUERY:
        idx = rng.choice(len(query_records), MAX_QUERY, replace=False)
        query_records = [query_records[i] for i in idx]
        print(f"[Market-1501] Subsampled query to {len(query_records)} images")

    print("[Market-1501] Extracting query features ...")
    t0 = time.time()
    q_feat, q_pids, q_camids = extract_features_list(reid, query_records, "Query")

    print("[Market-1501] Extracting gallery features ...")
    g_feat, g_pids, g_camids = extract_features_list(reid, gallery_records, "Gallery")
    print(f"[Market-1501] Feature extraction done in {time.time()-t0:.1f}s")

    print("[Market-1501] Computing CMC + mAP ...")
    mean_cmc, mAP = eval_market_protocol(q_feat, q_pids, q_camids,
                                          g_feat, g_pids, g_camids)

    scores = {
        "dataset":       "Market-1501",
        "query_images":  int(q_feat.shape[0]),
        "gallery_images": int(g_feat.shape[0]),
        "model":         "resnet50 (pretrained torchreid)",
        "mAP":           round(mAP * 100, 2),
    }
    for i, r in enumerate(RANKS):
        scores[f"Rank-{r}"] = round(float(mean_cmc[i]) * 100, 2)

    # Published baselines
    scores["published_baselines"] = {
        "BoT (2019) ResNet50":        {"Rank-1": 94.5,  "mAP": 85.9},
        "MGN (2018)":                 {"Rank-1": 95.7,  "mAP": 86.9},
        "PCB+RPP (2018)":             {"Rank-1": 93.8,  "mAP": 81.6},
        "ResNet50 ID loss baseline":  {"Rank-1": "~88", "mAP": "~72"},
        "OSNet (2019)":               {"Rank-1": 94.8,  "mAP": 84.9},
    }

    return scores


def main():
    print("=" * 60)
    print("  Market-1501 ReID Evaluation")
    print("=" * 60)

    reid = REID()
    scores = evaluate_market1501(reid)

    if scores is None:
        return

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY — Market-1501")
    print("=" * 60)
    print(f"  Dataset       : {scores['dataset']}")
    print(f"  Query images  : {scores['query_images']}")
    print(f"  Gallery images: {scores['gallery_images']}")
    print(f"  Model         : {scores['model']}")
    print()
    for r in RANKS:
        key = f"Rank-{r}"
        print(f"  {key:<8}: {scores[key]:.2f}%")
    print(f"  mAP      : {scores['mAP']:.2f}%")
    print()
    print("  Published baselines:")
    for name, info in scores["published_baselines"].items():
        r1  = info.get('Rank-1', 'N/A')
        map_ = info.get('mAP', 'N/A')
        print(f"    {name:<35}: Rank-1={r1}%  mAP={map_}%")
    print("=" * 60)

    # Save JSON
    json_path = RESULTS_DIR / "market1501_scores.json"
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # Save CSV
    csv_path = RESULTS_DIR / "market1501_scores.csv"
    flat = {k: v for k, v in scores.items() if not isinstance(v, dict)}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"[Saved] {csv_path}")

    return scores


if __name__ == "__main__":
    main()
