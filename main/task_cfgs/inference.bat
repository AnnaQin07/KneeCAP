@echo off

if not exist "log\" md "log\"

set exp_name=%1
set inf_mode=%2
set device=%3
set model_path=%4

python main.py --exp_name %exp_name% ^
               --task "inf" ^
               --device %device% ^
               --inf_mode %inf_mode% ^
               --config "configs/inference_v2.yaml" 2>&1 ^
               --checkpoint_path %model_path% ^
               | tee "log\%exp_name%.txt"