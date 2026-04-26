$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING="utf-8"

# STMN MARS Training Script

Write-Host "========================================="
Write-Host "  Starting STMN Training on MARS"
Write-Host "========================================="

# Create checkpoint and log directories
New-Item -ItemType Directory -Force -Path "..\checkpoints\mars" | Out-Null

python main.py `
    --train_txt   ..\database\MARS_database\train_path.txt `
    --train_info  ..\database\MARS_database\train_info.npy `
    --test_txt    ..\database\MARS_database\test_path.txt `
    --test_info   ..\database\MARS_database\test_info.npy `
    --query_info  ..\database\MARS_database\query_IDX.npy `
    --ckpt        ..\checkpoints\mars `
    --log_path    mars_loss.txt `
    --n_epochs    200 `
    --optimizer   adam `
    --lr          0.0001 `
    --lr_step_size 50 `
    --class_per_batch 8 `
    --track_per_class 4 `
    --seq_len     6 `
    --test_batch  64 `
    --num_workers 0 `
    --eval_freq   10 `
    --gpu_id      0 `
    --seed        42
