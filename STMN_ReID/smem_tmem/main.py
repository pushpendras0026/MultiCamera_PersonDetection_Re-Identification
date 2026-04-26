"""
smem_tmem/main.py – Training and evaluation entry-point for STMN.

Windows-safe: parse_args() is inside if __name__=='__main__' so DataLoader
spawn-workers never re-execute it.

OPTIMIZED: Uses AMP (Automatic Mixed Precision) for 2-3× GPU speedup.
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import random
from collections import OrderedDict

# ── sys.path: add STMN_ReID/ before any local imports ───────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _here not in sys.path:
    sys.path.insert(0, _here)

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

torch.multiprocessing.set_sharing_strategy('file_system')

# ── Local imports ─────────────────────────────────────────────────────────────
from util.utils import (Get_Video_train_DataLoader,
                        Get_Video_test_rrs_DataLoader,
                        Get_Video_test_all_DataLoader,
                        RandomErasing)
from util.cmc import Video_Cmc
from smem_tmem.model.loss    import Loss
from smem_tmem.model.network import STMN


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_rrs(net, dataloader, seq_len):
    """Quick in-training eval: random-repeated-sampling clips."""
    net.eval()
    gallery_features, gallery_labels, gallery_cams = [], [], []
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Eval(rrs)', ncols=90, leave=False):
            seqs  = data[0].reshape(
                (data[0].shape[0] // seq_len, seq_len) + data[0].shape[1:])
            if torch.cuda.is_available():
                seqs = seqs.cuda(non_blocking=True)
            with autocast('cuda'):
                feat = net(seqs)['val_bn']
            gallery_features.append(feat.float().cpu())
            gallery_labels.append(data[1])
            gallery_cams.append(data[2])

    gallery_features = torch.cat(gallery_features, 0).numpy()
    gallery_labels   = torch.cat(gallery_labels,   0).numpy()
    gallery_cams     = torch.cat(gallery_cams,     0).numpy()
    Cmc, mAP = Video_Cmc(gallery_features, gallery_labels, gallery_cams,
                          dataloader.dataset.query_idx, 10000)
    net.train()
    return float(Cmc[0]), float(mAP)


def test_all(net, dataloader, seq_len):
    """Full eval: all frames per tracklet + horizontal-flip TTA."""
    net.eval()
    gallery_features, gallery_labels, gallery_cams = [], [], []
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Eval(all)', ncols=90, leave=False):
            label = data[1]
            cams  = data[2]
            seqs  = data[0].unsqueeze(0)
            if torch.cuda.is_available():
                seqs = seqs.cuda(non_blocking=True)

            while seqs.size(1) < seq_len:
                seqs = torch.cat([seqs, seqs[:, -2:-1]], dim=1)

            n, f, c, h, w = seqs.size()
            flipped = torch.flip(seqs, [4])
            div     = max(1, f // seq_len)

            clips = []
            for i in range(div):
                clip_orig = torch.cat(
                    [seqs[:,    j * div + i: j * div + i + 1] for j in range(seq_len)], 1)
                clip_flip = torch.cat(
                    [flipped[:, j * div + i: j * div + i + 1] for j in range(seq_len)], 1)
                with autocast('cuda'):
                    out = (net(clip_orig)['val_bn'] + net(clip_flip)['val_bn']) / 2.0
                clips.append(out.float())

            feat_mean = torch.stack(clips, 0).mean(0).cpu()
            gallery_features.append(feat_mean)
            gallery_labels.append(label)
            gallery_cams.append(cams)

    gallery_features = torch.cat(gallery_features, 0).numpy()
    gallery_labels   = torch.cat(gallery_labels,   0).numpy()
    gallery_cams     = torch.cat(gallery_cams,     0).numpy()
    Cmc, mAP = Video_Cmc(gallery_features, gallery_labels, gallery_cams,
                          dataloader.dataset.query_idx, 10000)
    net.train()
    return float(Cmc[0]), float(mAP)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    from smem_tmem.parser import parse_args
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True
    # Enable TF32 for extra speed on Ampere+ GPUs (RTX 30xx/40xx)
    torch.set_float32_matmul_precision('medium')

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()

    print(f'\n{"="*60}')
    print(f'  STMN Training / Evaluation  (AMP enabled)')
    print(f'  GPU    : {"CUDA "+args.gpu_id if use_cuda else "CPU"}')
    print(f'  Seed   : {args.seed}')
    print(f'{"="*60}\n')

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = Compose([
        Resize((256, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]),
    ])
    test_tf = Compose([
        Resize((256, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Data loaders ──────────────────────────────────────────────────────────
    print('Loading datasets ...')
    train_dl = Get_Video_train_DataLoader(
        args.train_txt, args.train_info, train_tf,
        shuffle=True,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        track_per_class=args.track_per_class,
        class_per_batch=args.class_per_batch)

    test_rrs_dl = Get_Video_test_rrs_DataLoader(
        args.test_txt, args.test_info, args.query_info, test_tf,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.num_workers,
        seq_len=args.seq_len, distractor=True)

    test_all_dl = Get_Video_test_all_DataLoader(
        args.test_txt, args.test_info, args.query_info, test_tf,
        batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        seq_len=args.seq_len, distractor=True)

    num_class = train_dl.dataset.n_id
    print(f'  Train IDs   : {num_class}')
    print(f'  Seq length  : {args.seq_len}')
    print(f'  Batch       : {args.class_per_batch} IDs x {args.track_per_class} tracks x {args.seq_len} frames')
    print(f'  Num workers : {args.num_workers}')
    print()

    # ── Model ────────────────────────────────────────────────────────────────
    net = STMN(
        feat_dim=args.feat_dim, num_class=num_class,
        stride=args.stride,
        smem_size=args.smem_size, smem_margin=args.smem_margin,
        tmem_size=args.tmem_size, tmem_margin=args.tmem_margin,
        seq_len=args.seq_len)

    if use_cuda:
        net = nn.DataParallel(net).cuda()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    start_epoch = 1
    if args.load_ckpt:
        raw = torch.load(args.load_ckpt, map_location='cuda' if use_cuda else 'cpu')
        ckpt = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v) for k, v in raw.items())
        net.module.load_state_dict(ckpt) if use_cuda else net.load_state_dict(ckpt)
        print(f'Loaded checkpoint: {args.load_ckpt}')

    # ── Eval-only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        assert args.load_ckpt, '--load_ckpt required for --eval_only'
        print('\n-- Full evaluation (all frames + H-flip TTA) --')
        cmc, mAP = test_all(net, test_all_dl, args.seq_len)
        print(f'\nRank-1 : {cmc * 100:.2f}%')
        print(f'mAP    : {mAP * 100:.2f}%')
        sys.exit(0)

    # ── Training setup ────────────────────────────────────────────────────────
    os.makedirs(args.ckpt, exist_ok=True)
    log_path = os.path.join(args.ckpt, args.log_path)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = (optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, gamma=0.1)
                 if args.lr_step_size > 0 else None)

    criterion = Loss(seq_len=args.seq_len)
    best_cmc  = 0.0

    # ── AMP scaler for mixed precision ────────────────────────────────────────
    scaler = GradScaler('cuda')

    with open(log_path, 'a') as f:
        f.write(f'\n{"="*60}\n')
        f.write(f'train_txt   : {args.train_txt}\n')
        f.write(f'n_class     : {num_class}\n')
        f.write(f'epochs      : {args.n_epochs}\n')
        f.write(f'lr          : {args.lr}  step={args.lr_step_size}\n')
        f.write(f'smem_size   : {args.smem_size}   tmem_size: {args.tmem_size}\n')
        f.write(f'AMP         : ENABLED\n')
        f.write(f'{"="*60}\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):

        # ── Periodic validation ──────────────────────────────────────────────
        if epoch % args.eval_freq == 0:
            cmc, mAP = test_rrs(net, test_rrs_dl, args.seq_len)
            msg = f'[Epoch {epoch:03d}]  R1={cmc*100:.1f}%  mAP={mAP*100:.1f}%'
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')
            if cmc >= best_cmc:
                torch.save(net.state_dict(),
                           os.path.join(args.ckpt, 'ckpt_best.pth'))
                best_cmc = cmc
                print(f'  ** New best R1={best_cmc*100:.1f}% saved')
            torch.save(net.state_dict(),
                       os.path.join(args.ckpt, 'ckpt_latest.pth'))

        # ── One epoch of training ─────────────────────────────────────────────
        net.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, total=len(train_dl), ncols=100,
                    desc=f'Epoch {epoch:03d}/{args.n_epochs}', leave=True)

        for seqs, labels in pbar:
            if use_cuda:
                seqs   = seqs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ── AMP forward pass ──────────────────────────────────────────────
            with autocast('cuda'):
                out      = net(seqs)
                loss_out = criterion(out, labels)
                total_loss = (loss_out['track_id']
                              + loss_out['trip']
                              + out['smem']['loss']['mem_trip'].mean()
                              + out['tmem']['loss']['mem_trip'].mean())

            # ── AMP backward + step ───────────────────────────────────────────
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f'{total_loss.item():.3f}')

        avg = epoch_loss / len(train_dl)
        with open(log_path, 'a') as f:
            f.write(f'[Epoch {epoch:03d}]  avg_loss={avg:.4f}\n')

        if scheduler:
            scheduler.step()

    # ── Training done → full evaluation ──────────────────────────────────────
    print('\n-- Final full evaluation (all frames + TTA) --')
    best_ckpt = os.path.join(args.ckpt, 'ckpt_best.pth')
    raw = torch.load(best_ckpt, map_location='cuda' if use_cuda else 'cpu')
    ckpt = OrderedDict(
        (k[7:] if k.startswith('module.') else k, v) for k, v in raw.items())
    if use_cuda:
        net.module.load_state_dict(ckpt)
    else:
        net.load_state_dict(ckpt)

    cmc, mAP = test_all(net, test_all_dl, args.seq_len)
    result = f'\n[FINAL]  Rank-1={cmc*100:.2f}%   mAP={mAP*100:.2f}%\n'
    print(result)
    with open(log_path, 'a') as f:
        f.write(result)
