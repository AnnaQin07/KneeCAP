#!/usr/bin/env bash

mkdir -p log

exp_name=%1
cfg_file=%2

python main.py --exp_name ${exp_name} \
               --task test \
               --config "configs/${cfg_file}" 2>&1 \
               | tee "log/${exp_name}.txt"