@echo off
set PYTHONIOENCODING=utf-8

echo =========================================
echo   STMN FAST Training on MARS (AMP)
echo   50 epochs, ~20 min estimated
echo =========================================

if not exist "..\checkpoints\mars" mkdir "..\checkpoints\mars"

python main.py ^
    --train_txt   ..\database\MARS_database\train_path.txt ^
    --train_info  ..\database\MARS_database\train_info.npy ^
    --test_txt    ..\database\MARS_database\test_path.txt ^
    --test_info   ..\database\MARS_database\test_info.npy ^
    --query_info  ..\database\MARS_database\query_IDX.npy ^
    --ckpt        ..\checkpoints\mars ^
    --log_path    mars_loss.txt ^
    --n_epochs    50 ^
    --optimizer   adam ^
    --lr          0.0003 ^
    --lr_step_size 25 ^
    --class_per_batch 6 ^
    --track_per_class 2 ^
    --seq_len     4 ^
    --test_batch  64 ^
    --num_workers 4 ^
    --eval_freq   10 ^
    --gpu_id      0 ^
    --seed        42

echo.
echo MARS training complete!
echo Now run evaluate_mars.bat to see final Rank-1 and mAP results.
pause
