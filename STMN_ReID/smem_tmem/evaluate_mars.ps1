$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING="utf-8"

# STMN MARS Evaluation Script

Write-Host "========================================="
Write-Host "  Starting STMN Evaluation on MARS"
Write-Host "========================================="

python main.py `
    --eval_only `
    --load_ckpt   ..\checkpoints\mars\ckpt_best.pth `
    --test_txt    ..\database\MARS_database\test_path.txt `
    --test_info   ..\database\MARS_database\test_info.npy `
    --query_info  ..\database\MARS_database\query_IDX.npy `
    --seq_len     6 `
    --test_batch  64 `
    --num_workers 0 `
    --gpu_id      0
