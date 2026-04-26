@echo off
set PYTHONIOENCODING=utf-8

echo =========================================
echo   STMN Evaluation on MARS
echo =========================================

python main.py ^
    --eval_only ^
    --load_ckpt   ..\checkpoints\mars\ckpt_best.pth ^
    --train_txt   ..\database\MARS_database\train_path.txt ^
    --train_info  ..\database\MARS_database\train_info.npy ^
    --test_txt    ..\database\MARS_database\test_path.txt ^
    --test_info   ..\database\MARS_database\test_info.npy ^
    --query_info  ..\database\MARS_database\query_IDX.npy ^
    --seq_len     4 ^
    --test_batch  64 ^
    --num_workers 4 ^
    --gpu_id      0

echo.
echo Evaluation complete!
pause
