#!/usr/bin/env bash
###
 # @Author: YunxiangLiu u7191378@anu.edu.au
 # @Date: 2023-09-26 16:32:37
 # @LastEditors: YunxiangLiu u7191378@anu.edu.au
 # @LastEditTime: 2023-11-23 14:24:29
 # @FilePath: \MA_detections\task_cfgs\train.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

mkdir -p log

exp_name=%1
cfg_file=%2

python main.py --exp_name ${exp_name} \
               --task train \
               --config "configs/${cfg_file}" 2>&1 \
               | tee "log/${exp_name}.txt" 