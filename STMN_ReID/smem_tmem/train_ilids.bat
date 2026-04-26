@echo off
set PYTHONIOENCODING=utf-8

echo =========================================
echo   STMN FAST Training on iLIDS-VID (AMP)
echo   175 epochs,
echo =========================================

if not exist "..\checkpoints\ilids" mkdir "..\checkpoints\ilids"

python main.py ^
    --train_txt   ..\database\iLIDS_database\train_path.txt ^
    --train_info  ..\database\iLIDS_database\train_info.npy ^
    --test_txt    ..\database\iLIDS_database\test_path.txt ^
    --test_info   ..\database\iLIDS_database\test_info.npy ^
    --query_info  ..\database\iLIDS_database\query_IDX.npy ^
    --ckpt        ..\checkpoints\ilids ^
    --log_path    ilids_loss.txt ^
    --n_epochs    175 ^
    --optimizer   adam ^
    --lr          0.0003 ^
    --lr_step_size 50 ^
    --class_per_batch 6 ^
    --track_per_class 2 ^
    --seq_len     4 ^
    --test_batch  64 ^
    --num_workers 4 ^
    --eval_freq   10 ^
    --gpu_id      0 ^
    --seed        42

echo.
echo iLIDS-VID training complete!
echo Now run evaluate_ilids.bat to see final Rank-1 results.
pause
