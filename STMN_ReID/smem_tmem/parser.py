"""
smem_tmem/parser.py – Command-line argument parser for STMN training/evaluation.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='STMN: Spatial-Temporal Memory Networks for Video-Based Person Re-ID',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Logging / checkpointing ───────────────────────────────────────────────
    parser.add_argument('--log_path',   type=str,  default='loss.txt',
                        help='filename for the loss log inside --ckpt dir')
    parser.add_argument('--ckpt',       type=str,  default='./log',
                        help='directory to save checkpoints and logs')
    parser.add_argument('--load_ckpt',  type=str,  default=None,
                        help='path to checkpoint to load (for eval or resume)')
    parser.add_argument('--resume_validation', type=bool, default=False)

    # ── Dataset paths ─────────────────────────────────────────────────────────
    parser.add_argument('--train_txt',  help='path to train_path.txt')
    parser.add_argument('--train_info', help='path to train_info.npy')
    parser.add_argument('--test_txt',   help='path to test_path.txt')
    parser.add_argument('--test_info',  help='path to test_info.npy')
    parser.add_argument('--query_info', help='path to query_IDX.npy')

    # ── Optimization ──────────────────────────────────────────────────────────
    parser.add_argument('--n_epochs',      type=int,   default=200)
    parser.add_argument('--optimizer',     type=str,   default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--lr',            type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--lr_step_size',  type=int,   default=50,
                        help='StepLR step size (0 = no scheduler)')

    # ── Batch / sampling ──────────────────────────────────────────────────────
    parser.add_argument('--class_per_batch',  type=int, default=8,
                        help='P: identities per mini-batch')
    parser.add_argument('--track_per_class',  type=int, default=4,
                        help='K: tracklets per identity per mini-batch')
    parser.add_argument('--seq_len',          type=int, default=6,
                        help='frames per tracklet (S)')
    parser.add_argument('--test_batch',       type=int, default=64)
    parser.add_argument('--num_workers',      type=int, default=4)

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument('--feat_dim', type=int,   default=2048)
    parser.add_argument('--stride',   type=int,   default=1,
                        help='last stride of ResNet-50 (1 or 2)')

    # ── Memory hyper-params ───────────────────────────────────────────────────
    parser.add_argument('--smem_size',   type=int,   default=10)
    parser.add_argument('--smem_margin', type=float, default=0.3)
    parser.add_argument('--tmem_size',   type=int,   default=5)
    parser.add_argument('--tmem_margin', type=float, default=0.3)

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--gpu_id',    type=str,  default='0',
                        help='comma-separated GPU IDs for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--eval_freq', type=int,  default=10,
                        help='run validation every N epochs')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='skip training; load checkpoint and evaluate')
    parser.add_argument('--seed',      type=int,  default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()
