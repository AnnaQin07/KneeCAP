@echo off

if not exist "log\" md "log\"

set model_path=%1

python main.py --task "gui" ^
               --config "configs/inference.yaml" 2>&1 ^
               --checkpoint_path %model_path% ^