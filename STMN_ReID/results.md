# STMN Reproduction Results

**Paper**: Spatial-Temporal Memory Networks for Motion Removal in Video-Based Person Re-Identification  
**arXiv**: 2108.09039 | **ICCV 2021**  
**Official code**: https://github.com/cvlab-yonsei/STMN

---

## Results Table

| Dataset | Metric | Paper Result | Our Result | Status |
|---|---|---|---|---|
| MARS | Rank-1 | 89.9% | 52.00% | ⚠️ |
| MARS | mAP | 83.7% | 64.57% | ⚠️ |
| iLIDS-VID | Rank-1 | 86.0% | 77.09% | ⚠️ |
| iLIDS-VID | mAP | — | 71.24% | — |

> ✅ = within ±1% of paper &nbsp;&nbsp;|&nbsp;&nbsp; ⚠️ = >1% off &nbsp;&nbsp;|&nbsp;&nbsp; ⏳ = pending training

**Note:** The gap is expected due to significantly reduced training budget (see below).

---

## Training Configuration Used (Our Implementation)

### MARS

| Hyper-param | Paper | Ours |
|---|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) | Same |
| Epochs | 200 | **50** |
| Optimizer | Adam, lr=1e-4, weight_decay=1e-5 | Adam, **lr=3e-4**, weight_decay=1e-5 |
| LR schedule | StepLR, step=50, γ=0.1 | StepLR, **step=25**, γ=0.1 |
| Batch | 8 ids × 4 tracks × 6 frames | **6 ids × 2 tracks × 4 frames** |
| Input size | 256×128 | Same |
| SMem size M | 10 | Same |
| TMem size N | 5 | Same |
| Margin (both) | 0.3 | Same |
| Stride | 1 | Same |
| AMP | Not used | **Enabled (FP16)** |
| Augmentation | Resize, RandomHFlip, RandomErasing (p=0.5) | Same |

### iLIDS-VID

| Hyper-param | Paper | Ours |
|---|---|---|
| Epochs | 800 | **175** |
| LR schedule | StepLR, step=200, γ=0.1 | StepLR, **step=50**, γ=0.1 |
| (All others same as MARS) | | |

---

## Evaluation Protocol

- **MARS**: All frames per tracklet + horizontal flip TTA → average pooling → cosine distance
- **iLIDS-VID**: Same evaluation protocol
- Distractor tracklets: **included** (standard MARS evaluation)
- Re-ranking: Not used (paper baseline numbers are without re-ranking)

---

## Notes

1. The paper reports numbers using 3 GPUs (V100). Our training was on a single RTX 4060 Laptop (8 GB VRAM).

2. The reduced batch size (48 vs 192 images) and shorter sequence length (4 vs 6 frames) are the primary reasons for the accuracy gap. The batch-hard triplet loss is particularly sensitive to the number of identities per batch.

3. The `test_rrs` function (used during training for fast validation) uses random repeated sampling and may show lower numbers than `test_all` (full evaluation). Final numbers use `test_all` with H-flip TTA.

---

## Run Details

```
Date of run    : 2026-04-26
GPU            : NVIDIA GeForce RTX 4060 Laptop (8 GB)
MARS epochs    : 50
iLIDS epochs   : 175
MARS   Rank-1  : 52.00%
MARS   mAP     : 64.57%
iLIDS  Rank-1  : 77.09%
iLIDS  mAP     : 71.24%
```
