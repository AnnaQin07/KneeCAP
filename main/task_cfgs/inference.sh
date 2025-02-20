#!/usr/bin/env bash
###
 # @Author: YunxiangLiu u7191378@anu.edu.au
 # @Date: 2024-02-21 18:12:12
 # @LastEditors: YunxiangLiu u7191378@anu.edu.au
 # @LastEditTime: 2024-02-22 22:15:24
 # @FilePath: \MA_detections\task_cfgs\inference.sh
 # @Description: inference
### 

mkdir -p log

exp_name=%1
inf_mode=%2
device=%3
model_path=%4

python main.py --exp_name ${exp_name} \
               --task "inf" \
               --device ${device} \
               --inf_mode ${inf_mode} \
               --config "configs/inference.yaml" 2>&1 \
               --model_path ${model_path} \
               | tee "log/${exp_name}.txt"