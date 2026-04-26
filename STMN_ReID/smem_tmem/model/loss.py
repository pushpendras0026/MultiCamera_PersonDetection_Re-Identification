"""
model/loss.py – Combined ID classification + triplet loss for STMN.

Fixes applied vs. original:
  - Removed top-level `import parser; args = parser.parse_args()` – this caused
    crashes when imported from a different CWD or with --eval_only flag.
  - `seq_len` is now passed as a constructor argument.
"""

import torch
import torch.nn as nn
from torch.nn.modules import loss as _loss_module

import sys
import os
# model/loss.py lives at STMN_ReID/smem_tmem/model/loss.py
# We need STMN_ReID/ on sys.path so util.loss resolves correctly
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
from util.loss import TripletLoss



class Loss(_loss_module._Loss):
    """
    STMN training loss:
      L = L_id_s + L_id_t  +  L_trip_s + L_trip_t

    The per-frame spatial features (val_s) are mean-pooled over the sequence
    before computing the identity and triplet losses.

    Args:
        seq_len (int): number of frames per tracklet (S); used for pooling.
    """

    def __init__(self, seq_len=6):
        super().__init__()
        self.seq_len = seq_len
        self.criterion_trip = TripletLoss('soft', batch_hard=True)
        self.criterion_id   = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """
        Args:
            inputs (dict): output dict from STMN.forward (training mode)
            labels (Tensor): [B*track_per_class] identity labels

        Returns:
            dict with keys 'trip' and 'track_id'
        """
        S = self.seq_len

        # Pool spatial frame features over S frames → [B, D]
        pool_val_s = torch.mean(
            inputs['val_s'].reshape(-1, S, inputs['val_s'].shape[-1]), dim=1)
        # Pool spatial classifier logits over S frames → [B, num_class]
        pool_val_s_cls = torch.mean(
            inputs['val_s_cls'].reshape(-1, S, inputs['val_s_cls'].shape[-1]), dim=1)

        # Reduce labels from [B*S] to [B] (all S frames share the same label)
        labels_per_track = labels[::S] if labels.shape[0] > pool_val_s.shape[0] else labels

        trip_loss = (
            self.criterion_trip(pool_val_s,      labels_per_track, dis_func='eu') +
            self.criterion_trip(inputs['val_t'], labels_per_track, dis_func='eu')
        )
        id_loss = (
            self.criterion_id(pool_val_s_cls,         labels_per_track) +
            self.criterion_id(inputs['val_t_cls'],     labels_per_track)
        )

        return {'trip': trip_loss, 'track_id': id_loss}
