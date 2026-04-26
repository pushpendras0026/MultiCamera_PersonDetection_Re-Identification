"""
model/memory.py – Spatial Memory Module (SMM) and Temporal Memory Module (TMM).

Architecture (from paper Section 3):
  SMM:
    - Learnable memory key K ∈ R^{M × D} and value V ∈ R^{M × D}
    - Query comes from layer4_key_s (spatial feature map)
    - Attention-weighted read → subtracted from val (motion removal)
    - Output: per-frame background-suppressed feature [B×S, D]

  TMM:
    - Learnable memory key K ∈ R^{M × D} and weight W ∈ R^{M × S}
    - Query produced by LSTM over frame sequence → temporal attention weights
    - Weighted aggregation of S frames → sequence-level feature [B, D]

Fixes applied vs. original:
  - Removed hardcoded `.cuda()` from `TemporalMemory.__init__` (self.val parameter).
    The parameter moves to the correct device automatically via `.to(device)` /
    `nn.DataParallel`. Hardcoding `.cuda()` breaks CPU-only runs / sanity checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import sys as _sys
import os as _os
# Add STMN_ReID/ to sys.path so smem_tmem.model resolves correctly
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

# weights_init_kaiming is not actually used in memory.py itself —
# it is used by network.py. Keep the import here for compatibility.
try:
    from smem_tmem.model import weights_init_kaiming
except ImportError:
    pass



# ──────────────────────────────────────────────────────────────────────────────
# Spatial Memory Module (SMM)
# ──────────────────────────────────────────────────────────────────────────────

class SpatialMemory(nn.Module):
    """
    Reads background motion patterns from a spatial memory bank and subtracts
    them from per-frame CNN features (motion removal).

    Args:
        feat_dim (int): feature dimension D (default 2048)
        mem_size (int): number of memory slots M
        margin   (float): margin for memory diversity loss
    """

    def __init__(self, feat_dim=2048, mem_size=10, margin=0.3):
        super().__init__()
        self.key    = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.val    = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.bn     = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias,   0)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim
        self.margin   = margin

    def forward(self, query, val):
        """
        Args:
            query: [B*S, D, H, W]  spatial key features from backbone
            val:   [B*S, D, H, W]  value features from backbone

        Returns:
            dict with keys:
              'out'  : [B*S, D]  background-suppressed pooled feature
              'loss' : dict containing 'mem_trip' diversity loss scalar
        """
        BS, _, H, W = query.shape
        query_rs = query.reshape(-1, self.feat_dim)   # [B*S*H*W, D]

        # Cosine similarity → soft attention over memory slots
        similarity = torch.matmul(
            F.normalize(query_rs, dim=1),
            F.normalize(self.key.t(), dim=0))          # [B*S*H*W, M]
        r_att = F.softmax(similarity, dim=1)            # [B*S*H*W, M]

        # Read memory value
        read   = torch.matmul(r_att, self.val)          # [B*S*H*W, D]
        read_  = self.bn(read)                           # BN over D

        # Subtract (motion removal) and pool
        out = val - read_.reshape(BS, self.feat_dim, H, W)
        out = self.avgpool(out).squeeze(-1).squeeze(-1)  # [B*S, D]

        # Aggregate attention for diversity loss (use per-location mean)
        r_att_mean = r_att.reshape(BS, H * W, -1).mean(1)  # [B*S, M]
        return {'out': out, 'loss': self._loss(r_att_mean, self.margin)}

    def _loss(self, r_att, margin):
        """Memory diversity loss: penalises when top-1 ≈ bottom-1 attention."""
        topk = r_att.topk(r_att.shape[0], dim=0)[0]   # [B*S, M] sorted desc
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.clamp(distance, min=0.0))
        return {'mem_trip': mem_trip}


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Memory Module (TMM)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalMemory(nn.Module):
    """
    Selects the most informative temporal pattern from a memory bank to
    aggregate frame-level features into a sequence-level representation.

    Args:
        feat_dim (int): feature dimension D
        mem_size (int): number of memory slots M
        margin   (float): margin for memory diversity loss
        seq_len  (int): number of frames per tracklet S
    """

    def __init__(self, feat_dim=2048, mem_size=5, margin=0.3, seq_len=6):
        super().__init__()
        self.key  = nn.Parameter(torch.randn(mem_size, feat_dim))
        # ← FIX: removed .cuda() – device assignment happens via DataParallel / .to()
        self.val  = nn.Parameter(torch.empty(mem_size, seq_len).uniform_())
        self.lstm = nn.LSTM(feat_dim, feat_dim, num_layers=1, batch_first=False)
        self.margin = margin
        self.S      = seq_len

    def forward(self, query, val):
        """
        Args:
            query: [B*S, D]  temporal key features (GAP'd from backbone)
            val:   [B*S, D]  background-suppressed features from SMM output

        Returns:
            dict with keys:
              'out'  : [B, D]  sequence-level aggregated feature
              'loss' : dict containing 'mem_trip' diversity loss scalar
        """
        B = query.shape[0] // self.S

        # Reshape to [S, B, D] for LSTM (seq-first)
        query_seq = query.reshape(B, self.S, -1).permute(1, 0, 2)  # [S, B, D]
        h0 = torch.zeros(1, B, query_seq.shape[2], device=query.device)
        c0 = torch.zeros(1, B, query_seq.shape[2], device=query.device)
        if self.training:
            self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(query_seq, (h0, c0))
        query_lstm  = lstm_out[-1]   # [B, D]  – last hidden state

        # Attention over memory keys
        similarity = torch.matmul(
            F.normalize(query_lstm, dim=1),
            F.normalize(self.key.t(), dim=0))           # [B, M]
        r_att = F.softmax(similarity, dim=1)             # [B, M]

        # Read temporal weights W from memory → softmax over S frames
        read = F.softmax(torch.matmul(r_att, self.val), dim=1)  # [B, S]

        # Weighted aggregation of frame features
        val_seq = val.reshape(B, self.S, -1)             # [B, S, D]
        out = torch.bmm(read.unsqueeze(1), val_seq).squeeze(1)  # [B, D]

        return {'out': out, 'loss': self._loss(r_att, self.margin)}

    def _loss(self, r_att, margin):
        """Memory diversity loss."""
        topk     = r_att.topk(r_att.shape[0], dim=0)[0]
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.clamp(distance, min=0.0))
        return {'mem_trip': mem_trip}
