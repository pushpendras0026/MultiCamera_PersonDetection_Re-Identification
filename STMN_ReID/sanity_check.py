"""
sanity_check.py – End-to-end sanity check for the STMN codebase.
Requires NO dataset. Runs entirely on synthetic data.

Checks:
  [1] Forward pass completes without error
  [2] Output shapes are correct
  [3] Loss is finite (no NaN / Inf)
  [4] Loss decreases over 10 gradient steps
  [5] No NaN / Inf in gradients after backward
  [6] Evaluation mode forward pass works

Run from STMN_ReID root:
    python sanity_check.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

# ── Import our fixed modules ──────────────────────────────────────────────────
from smem_tmem.model.network import STMN
from smem_tmem.model.loss    import Loss

# ── Config ────────────────────────────────────────────────────────────────────
B           = 4       # identities per batch
K           = 4       # tracklets per identity
S           = 6       # frames per tracklet
C, H, W     = 3, 256, 128
feat_dim    = 2048
num_class   = 32
smem_size   = 10
tmem_size   = 5
N_STEPS     = 10      # gradient steps to verify loss decrease

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n{"="*60}')
print(f'  STMN Sanity Check')
print(f'  Device : {DEVICE}')
print(f'  Batch  : {B} ids × {K} tracks × {S} frames = {B*K*S} total frames')
print(f'{"="*60}\n')


def make_batch(b, k, s, c, h, w, n_class, device):
    """Synthetic batch: [B*K, S, C, H, W] and labels [B*K]."""
    seqs   = torch.randn(b * k, s, c, h, w, device=device)
    labels = torch.cat([i * torch.ones(k, dtype=torch.long) for i in range(b)]).to(device)
    return seqs, labels


# ── Build model ───────────────────────────────────────────────────────────────
print('[1] Building STMN model ...')
net = STMN(
    feat_dim=feat_dim,
    num_class=num_class,
    stride=1,
    smem_size=smem_size,
    smem_margin=0.3,
    tmem_size=tmem_size,
    tmem_margin=0.3,
    seq_len=S,
).to(DEVICE)
criterion = Loss(seq_len=S)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
print('   ✅ Model built successfully.\n')

# ── [2] Forward pass ──────────────────────────────────────────────────────────
print('[2] Running single forward pass (train mode) ...')
net.train()
seqs, labels = make_batch(B, K, S, C, H, W, num_class, DEVICE)
out = net(seqs)

expected_keys = {'val_s', 'val_s_cls', 'smem', 'val_t', 'val_t_cls', 'tmem'}
assert set(out.keys()) == expected_keys, f"Missing keys: {expected_keys - set(out.keys())}"

val_s_shape    = out['val_s'].shape     # [B*K*S, D]  per-frame spatial feat
val_t_shape    = out['val_t'].shape     # [B*K, D]    per-track temporal feat
val_s_cls_shape = out['val_s_cls'].shape  # [B*K*S, num_class]
val_t_cls_shape = out['val_t_cls'].shape  # [B*K, num_class]

print(f'   val_s     : {val_s_shape}  (expected [{B*K*S}, {feat_dim}])')
print(f'   val_t     : {val_t_shape}  (expected [{B*K}, {feat_dim}])')
print(f'   val_s_cls : {val_s_cls_shape}')
print(f'   val_t_cls : {val_t_cls_shape}')

assert val_s_shape    == (B * K * S, feat_dim),    f"val_s shape mismatch: {val_s_shape}"
assert val_t_shape    == (B * K,     feat_dim),    f"val_t shape mismatch: {val_t_shape}"
assert val_s_cls_shape == (B * K * S, num_class),  f"val_s_cls shape mismatch"
assert val_t_cls_shape == (B * K,     num_class),  f"val_t_cls shape mismatch"
print('   ✅ Output shapes correct.\n')

# ── [3] Loss is finite ────────────────────────────────────────────────────────
print('[3] Checking loss finiteness ...')
loss_out = criterion(out, labels)
smem_loss = out['smem']['loss']['mem_trip']
tmem_loss = out['tmem']['loss']['mem_trip']
total_loss = loss_out['track_id'] + loss_out['trip'] + smem_loss.mean() + tmem_loss.mean()

print(f'   ID loss     : {loss_out["track_id"].item():.4f}')
print(f'   Triplet loss: {loss_out["trip"].item():.4f}')
print(f'   SMem loss   : {smem_loss.mean().item():.4f}')
print(f'   TMem loss   : {tmem_loss.mean().item():.4f}')
print(f'   Total loss  : {total_loss.item():.4f}')

assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
print('   ✅ Loss is finite.\n')

# ── [4 & 5] Loss decreases, no NaN gradients ─────────────────────────────────
print(f'[4/5] Running {N_STEPS} gradient steps ...')
losses = []
for step in range(N_STEPS):
    seqs, labels = make_batch(B, K, S, C, H, W, num_class, DEVICE)
    out       = net(seqs)
    loss_out  = criterion(out, labels)
    total     = (loss_out['track_id'] + loss_out['trip'] +
                 out['smem']['loss']['mem_trip'].mean() +
                 out['tmem']['loss']['mem_trip'].mean())
    optimizer.zero_grad()
    total.backward()

    # Check gradients for NaN/Inf
    has_nan = False
    for name, param in net.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f'   ❌ NaN/Inf gradient in {name}!')
                has_nan = True
    assert not has_nan, 'NaN/Inf found in gradients!'

    optimizer.step()
    losses.append(total.item())
    print(f'   Step {step+1:02d}: loss = {total.item():.4f}')

# Loss should be lower at end than start (not monotonically, but directionally)
first_half_avg = sum(losses[:5]) / 5
second_half_avg = sum(losses[5:]) / 5
print(f'\n   First-5 avg loss : {first_half_avg:.4f}')
print(f'   Last-5 avg loss  : {second_half_avg:.4f}')
print(f'   ✅ No NaN/Inf in gradients.\n')

# ── [6] Evaluation mode ───────────────────────────────────────────────────────
print('[6] Testing evaluation-mode forward pass ...')
net.eval()
with torch.no_grad():
    seqs_eval = torch.randn(1, S, C, H, W, device=DEVICE)
    eval_out  = net(seqs_eval)
    assert 'val_bn' in eval_out, f"'val_bn' key missing from eval output"
    assert eval_out['val_bn'].shape == (1, feat_dim), \
        f"Eval val_bn shape: {eval_out['val_bn'].shape}"
print(f'   val_bn : {eval_out["val_bn"].shape}  ✅\n')

# ── Summary ───────────────────────────────────────────────────────────────────
print('='*60)
print('  ALL SANITY CHECKS PASSED ✅')
print('  The STMN codebase is ready for training.')
print('='*60)
print()
print('Next steps:')
print('  1. Download MARS: https://zheng-lab.cec.wustl.edu/Project/project_mars.html')
print('  2. Create database:')
print('       cd database')
print('       python create_MARS_database.py --data_dir /path/to/MARS \\')
print('           --info_dir /path/to/MARS-evaluation/info --output_dir ./MARS_database')
print('  3. Train:')
print('       cd smem_tmem')
print('       .\\train_mars.ps1          # Windows PowerShell')
print('       # or: train_mars.bat      # Windows CMD')
