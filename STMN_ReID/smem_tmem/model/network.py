"""
model/network.py – STMN (Spatial-Temporal Memory Network) top-level module.

Architecture (paper Fig. 2):
  Input: [B, S, C, H, W] video tracklet batch
  → ResNet-50 backbone (3 branches: val, key_s, key_t)
  → SpatialMemory (SMM): per-frame background removal, outputs [B*S, D]
  → TemporalMemory (TMM): temporal aggregation, outputs [B, D]
  → BN + classifier head (training only)

Fixes applied vs. original:
  - Removed top-level `import parser; args = parser.parse_args()` – this broke
    eval-only mode and any import from a different working directory.
  - All memory hyper-params are now passed as explicit constructor arguments.
  - Minor: `squeeze()` replaced with `squeeze(-1).squeeze(-1)` (safe for batch=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import smem_tmem.model as model_init
import smem_tmem.model.resnet as res
import smem_tmem.model.memory as mem


class STMN(nn.Module):
    """
    Spatial-Temporal Memory Network for video-based person Re-ID.

    Args:
        feat_dim    (int)   : feature dimension (2048 for ResNet-50)
        num_class   (int)   : number of training identities
        stride      (int)   : last-layer stride of ResNet backbone (1 or 2)
        smem_size   (int)   : number of slots in Spatial Memory
        smem_margin (float) : diversity margin for SMM loss
        tmem_size   (int)   : number of slots in Temporal Memory
        tmem_margin (float) : diversity margin for TMM loss
        seq_len     (int)   : frames per tracklet (S)
    """

    def __init__(self, feat_dim=2048, num_class=710, stride=1,
                 smem_size=10, smem_margin=0.3,
                 tmem_size=5,  tmem_margin=0.3,
                 seq_len=6):
        super().__init__()

        self.seq_len  = seq_len
        self.feat_dim = feat_dim

        # ── Backbone ──────────────────────────────────────────────────────────
        self.features = res.Resnet50(stride=stride)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        # ── Memory modules ────────────────────────────────────────────────────
        self.smem = mem.SpatialMemory(feat_dim=feat_dim,
                                      mem_size=smem_size,
                                      margin=smem_margin)
        self.tmem = mem.TemporalMemory(feat_dim=feat_dim,
                                       mem_size=tmem_size,
                                       margin=tmem_margin,
                                       seq_len=seq_len)

        # ── BN + classifier (spatial branch) ─────────────────────────────────
        self.bn_s = nn.BatchNorm1d(feat_dim)
        self.bn_s.bias.requires_grad_(False)
        self.bn_s.apply(model_init.weights_init_kaiming)
        self.cls_s = nn.Linear(feat_dim, num_class, bias=False)
        self.cls_s.apply(model_init.weights_init_classifier)

        # ── BN + classifier (temporal branch) ────────────────────────────────
        self.bn_t = nn.BatchNorm1d(feat_dim)
        self.bn_t.bias.requires_grad_(False)
        self.bn_t.apply(model_init.weights_init_kaiming)
        self.cls_t = nn.Linear(feat_dim, num_class, bias=False)
        self.cls_t.apply(model_init.weights_init_classifier)

    def forward(self, x):
        """
        Args:
            x: [B, S, C, H, W]  – batch of tracklets

        Returns (training):
            dict: val_s, val_s_cls, smem, val_t, val_t_cls, tmem

        Returns (eval):
            dict: val_bn (used as the retrieval descriptor), smem, tmem, val_t
        """
        B, S, C, H, W = x.size()
        x_flat = x.reshape(B * S, C, H, W)            # [B*S, C, H, W]

        # Backbone: three feature branches
        val, key_s, key_t = self.features(x_flat)
        # val   : [B*S, D, h, w]   spatial value map
        # key_s : [B*S, D, h, w]   spatial key map
        # key_t : [B*S, D]         temporal key (already GAP'd in Resnet50)

        # ── Spatial Memory Module ─────────────────────────────────────────────
        smem_out = self.smem(key_s, val)
        val_s    = smem_out['out']                     # [B*S, D]

        val_s_bn = self.bn_s(val_s)                    # [B*S, D]

        # ── Temporal Memory Module ────────────────────────────────────────────
        tmem_out = self.tmem(key_t, val_s)
        val_t    = tmem_out['out']                     # [B, D]

        val_t_bn = self.bn_t(val_t)                    # [B, D]

        if self.training:
            val_s_cls = self.cls_s(val_s_bn)           # [B*S, num_class]
            val_t_cls = self.cls_t(val_t_bn)           # [B,   num_class]
            return {
                'val_s':     val_s,
                'val_s_cls': val_s_cls,
                'smem':      smem_out,
                'val_t':     val_t,
                'val_t_cls': val_t_cls,
                'tmem':      tmem_out,
            }
        else:
            return {
                'val_bn': val_t_bn,   # ← descriptor used for retrieval
                'smem':   smem_out,
                'tmem':   tmem_out,
                'val_t':  val_t,
            }
