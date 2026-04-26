# STMN_ReID – Setup Guide

> **Paper**: Spatial-Temporal Memory Networks for Motion Removal in Video-Based Person Re-Identification  
> **arXiv**: 2108.09039 | **Venue**: ICCV 2021  
> **Official repo**: https://github.com/cvlab-yonsei/STMN

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 (tested on 3.13) |
| PyTorch | ≥ 2.0 (tested on 2.7 + CUDA 12.8) |
| CUDA | ≥ 11.3 (for GPU training) |
| RAM | ≥ 16 GB |
| GPU VRAM | ≥ 12 GB (e.g. RTX 3080) |
| Disk | ≥ 50 GB for MARS dataset |

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with `progressbar2`:
```bash
pip install progressbar2
```

---

## 3. Verify the Codebase (No Dataset Needed)

```bash
# From STMN_ReID root:
python sanity_check.py
```

Expected output: `ALL SANITY CHECKS PASSED ✅`

---

## 4. Dataset Download

### MARS (Primary dataset)

1. Request access at: https://zheng-lab.cec.wustl.edu/Project/project_mars.html  
2. Download `MARS-v160809.zip` and `MARS-evaluation.zip`  
3. Extract both archives

Structure expected:
```
MARS/
    bbox_train/
        0001/
            0001C1T0001F001.jpg ...
    bbox_test/
        0001/
            ...
MARS-evaluation/
    info/
        tracks_train_info.mat
        tracks_test_info.mat
        query_IDX.mat
```

### iLIDS-VID

1. Download from: https://www.eecs.qmul.ac.uk/~sgg/i-LIDS-VID/  
2. Extract archive

Structure expected:
```
i-LIDS-VID/
    sequences/
        cam1/
            person_001/ person_002/ ...
        cam2/
            person_001/ ...
```

---

## 5. Create Database Files

### MARS

```bash
cd database
python create_MARS_database.py \
    --data_dir  /path/to/MARS/ \
    --info_dir  /path/to/MARS-evaluation/info/ \
    --output_dir ./MARS_database/
cd ..
```

Outputs: `train_path.txt`, `train_info.npy`, `test_path.txt`, `test_info.npy`, `query_IDX.npy`

### iLIDS-VID

```bash
cd database
python create_iLIDS_database.py \
    --data_dir  /path/to/i-LIDS-VID/ \
    --output_dir ./iLIDS_database/
cd ..
```

---

## 6. Training

### Windows PowerShell

```powershell
cd smem_tmem
# MARS:
.\train_mars.ps1

# iLIDS-VID:
.\train_ilids.ps1
```

### Windows CMD

```bat
cd smem_tmem
train_mars.bat
```

### Linux / WSL (bash)

```bash
cd smem_tmem
bash train_mars.sh   # (create this from the .ps1 by replacing backtick continuations with \)
```

Training hyper-parameters (from paper Table):

| Hyper-param | MARS | iLIDS-VID |
|---|---|---|
| Epochs | 200 | 400 |
| Optimizer | Adam | Adam |
| LR | 1e-4 | 1e-4 |
| LR step size | 50 | 100 |
| P (ids/batch) | 8 | 8 |
| K (tracks/id) | 4 | 4 |
| S (frames/track) | 6 | 6 |
| SMem size | 10 | 10 |
| TMem size | 5 | 5 |
| Stride | 1 | 1 |

---

## 7. Evaluation

```powershell
cd smem_tmem
.\evaluate_mars.ps1   # MARS
.\evaluate_ilids.ps1  # iLIDS-VID
```

Best checkpoint is saved as `checkpoints/<dataset>/ckpt_best.pth`.

---

## 8. Expected Results

| Dataset | Metric | Paper | Notes |
|---|---|---|---|
| MARS | Rank-1 | 86.5% | Full evaluation (all frames + HFlip TTA) |
| MARS | mAP | 82.4% | |
| iLIDS-VID | Rank-1 | 86.0% | |

Results within ±1% = ✅ reproduced.

---

## 9. Project Structure

```
STMN_ReID/
├── sanity_check.py           ← Run first (no dataset needed)
├── requirements.txt
├── SETUP.md                  ← This file
├── results.md                ← Filled in after training
├── util/
│   ├── utils.py              ← Dataloaders (FIXED: Python 3.13 + NumPy 2.x)
│   ├── cmc.py                ← CMC / mAP evaluation (FIXED: progressbar)
│   └── loss.py               ← TripletLoss (FIXED: deprecated API)
├── smem_tmem/
│   ├── main.py               ← Training + evaluation entry-point (FIXED)
│   ├── parser.py             ← CLI arguments
│   ├── train_mars.ps1/.bat   ← Windows training scripts
│   ├── train_ilids.ps1       ← Windows training scripts
│   ├── evaluate_mars.ps1     ← Windows evaluation scripts
│   ├── evaluate_ilids.ps1    ← Windows evaluation scripts
│   └── model/
│       ├── __init__.py       ← Weight init helpers
│       ├── resnet.py         ← ResNet-50 backbone (FIXED: weights API)
│       ├── memory.py         ← SMM + TMM modules (FIXED: no .cuda() in init)
│       ├── network.py        ← STMN top-level module (FIXED: no global parser)
│       └── loss.py           ← Combined loss (FIXED: no global parser)
├── database/
│   ├── create_MARS_database.py
│   └── create_iLIDS_database.py  ← NEW
├── checkpoints/              ← Saved model weights
│   ├── mars/
│   └── ilids/
└── logs/                     ← Training logs
```

---

## 10. Troubleshooting

| Error | Fix |
|---|---|
| `AttributeError: module 'collections' has no attribute 'Mapping'` | Use Python 3.10+; already fixed in `util/utils.py` |
| `AttributeError: module 'numpy' has no attribute 'long'` | NumPy 2.x issue; already fixed |
| `CUDA out of memory` | Reduce `--class_per_batch` or `--test_batch` |
| `FileNotFoundError` on database files | Run `create_MARS_database.py` first |
| `RuntimeError: Expected all tensors on same device` | Remove any remaining `.cuda()` in model init |
| Rank-1 is 0% | Check query_IDX.npy indices; ensure cam IDs are correct (1-indexed in .mat → 0-indexed after -1) |
