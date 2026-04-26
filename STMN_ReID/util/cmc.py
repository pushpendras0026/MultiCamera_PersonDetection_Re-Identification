"""
util/cmc.py – CMC and mAP computation for video-based person Re-ID.
Fixes applied vs. original:
  - Guard `progressbar` import (use tqdm fallback) for Python 3.13 compat.
  - Minor style cleanup only.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import math

try:
    from progressbar import ProgressBar, AnimatedMarker, Percentage
    HAS_PB = True
except ImportError:
    HAS_PB = False

from tqdm import trange


# ──────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────────

def Video_Cmc(features, ids, cams, query_idx, rank_size):
    """
    Compute CMC curve and mAP for video-based Re-ID.

    Args:
        features  (np.ndarray): shape (N, D)
        ids       (np.ndarray): shape (N,)
        cams      (np.ndarray): shape (N,)
        query_idx (np.ndarray): indices of query tracklets
        rank_size (int): maximum rank to compute

    Returns:
        CMC (np.ndarray): cumulative matching curve, shape (rank_size,)
        mAP (float): mean average precision
    """
    data = {'feature': features, 'id': ids, 'cam': cams}

    q_idx = query_idx
    g_idx = [i for i in range(len(ids)) if data['id'][i] != -1]
    q_data = {k: v[q_idx] for k, v in data.items()}
    g_data = {k: v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size:
        rank_size = len(g_idx)

    CMC, mAP = _Cmc(q_data, g_data, rank_size)
    return CMC, mAP


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _Cmc(q_data, g_data, rank_size):
    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    distmat = _np_cdist(q_data['feature'], g_data['feature'])   # (n_query, n_gallery)
    index = np.argsort(distmat, axis=1)                          # small → large

    num_no_gt = 0
    CMC = np.zeros(n_gallery)
    AP = 0.0

    for i in range(n_query):
        query_index = np.argwhere(g_data['id'] == q_data['id'][i])
        camera_index = np.argwhere(g_data['cam'] == q_data['cam'][i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        junk_index = np.intersect1d(query_index, camera_index)
        ap_tmp, CMC_tmp = _Compute_AP(good_index, junk_index, index[i])
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print(f"{num_no_gt} query tracklets have no ground truth in gallery.")

    valid = n_query - num_no_gt
    CMC = CMC / valid
    mAP = AP / valid
    return CMC, mAP


def _Compute_AP(good_index, junk_index, index):
    ap = 0.0
    cmc = np.zeros(len(index))

    # Remove junk indices
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) / (rows_good[i] + 1)
        ap += d_recall * precision

    return ap, cmc


def _np_cdist(feat1, feat2):
    """Cosine distance (returned as negative dot-product of L2-normed vectors)."""
    feat1_u = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-12)
    feat2_u = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-12)
    return -1.0 * np.dot(feat1_u, feat2_u.T)


def np_cdist(feat1, feat2):
    """Public alias for cosine distance."""
    return _np_cdist(feat1, feat2)


def sqdist(feat1, feat2, M=None):
    """Mahalanobis / Euclidean squared distance."""
    if M is None:
        M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1, 1) + feat2_sq.reshape(1, -1) - 2 * np.dot(feat1_M, feat2.T)
