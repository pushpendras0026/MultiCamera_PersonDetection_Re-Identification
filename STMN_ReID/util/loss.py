"""
util/loss.py – Triplet loss and auxiliary losses for video-based person Re-ID.
Fixes applied vs. original:
  - Replaced deprecated `distmat.addmm_(1, -2, x, centers.t())` with explicit arithmetic.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# ──────────────────────────────────────────────────────────────────────────────
# Triplet Loss (soft-margin, batch-hard)
# ──────────────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet loss with optional batch-hard mining.

    Args:
        margin (float | 'soft'): fixed margin or soft-plus margin
        batch_hard (bool): if True, use hardest positive/negative per sample
    """

    def __init__(self, margin=0, batch_hard=False, dim=2048):
        super(TripletLoss, self).__init__()
        self.batch_hard = batch_hard
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(f"Unrecognised margin: {margin}")

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None,
                mode='id', dis_func='eu', n_dis=0):

        if dis_func == 'cdist':
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            dist = self._cdist(feat, feat)
        elif dis_func == 'eu':
            dist = self._cdist(feat, feat)
        else:
            raise ValueError(f"Unknown dis_func: {dis_func}")

        if mode == 'id':
            if id is None:
                raise RuntimeError("mode='id' requires `id` argument.")
            identity_mask = torch.eye(feat.size(0), dtype=torch.bool,
                                      device=feat.device)
            same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
            negative_mask = ~same_id_mask
            positive_mask = same_id_mask & ~identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                raise RuntimeError("mode='mask' requires pos_mask and neg_mask.")
            positive_mask = pos_mask
            same_id_mask = ~neg_mask
            negative_mask = neg_mask
        else:
            raise ValueError(f"Unrecognised mode: {mode}")

        if self.batch_hard:
            if n_dis != 0:
                img_dist = dist[:-n_dis, :-n_dis]
                max_positive = (img_dist * positive_mask[:-n_dis, :-n_dis].float()).max(1)[0]
                min_negative = (img_dist + 1e5 * same_id_mask[:-n_dis, :-n_dis].float()).min(1)[0]
                dis_min_negative = dist[:-n_dis, -n_dis:].min(1)[0]
                z_origin = max_positive - min_negative
            else:
                max_positive = (dist * positive_mask.float()).max(1)[0]
                min_negative = (dist + 1e5 * same_id_mask.float()).min(1)[0]
                z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1, 1)
            positive = torch.gather(dist, dim=1, index=pos)
            neg = negative_mask.topk(k=1, dim=1)[1].view(-1, 1)
            negative = torch.gather(dist, dim=1, index=neg)
            z = positive - negative

        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            if n_dis != 0:
                b_loss = torch.log(1 + torch.exp(z_origin)) - 0.5 * dis_min_negative
            else:
                b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError

        return torch.mean(b_loss)

    @staticmethod
    def _cdist(a, b):
        """Euclidean distance matrix between rows of a and rows of b."""
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff ** 2).sum(2) + 1e-12).sqrt()


# ──────────────────────────────────────────────────────────────────────────────
# Center Loss (auxiliary)
# ──────────────────────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    def __init__(self, num_class=625):
        super(CenterLoss, self).__init__()
        self.num_class = num_class

    def forward(self, x, labels, centers, classes):
        batch_size = x.size(0)
        # ← FIX: replaced deprecated addmm_(1, -2, ...) with explicit ops
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class)
            + torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
            - 2 * torch.mm(x, centers.t())
        )
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))
        dist = distmat * mask.float()
        return dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
