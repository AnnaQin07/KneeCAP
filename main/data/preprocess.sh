#!/bin/bash
###
 # @Author: YunxiangLiu u7191378@anu.edu.au
 # @Date: 2023-07-29 23:05:56
 # @LastEditors: YunxiangLiu u7191378@anu.edu.au
 # @LastEditTime: 2023-11-02 14:24:34
 # @FilePath: \MA_detections\data\preprocess.sh
 # @Description: configs for preprocessing
### 

dataset_root=$1

python preprocess.py --img_dir ${dataset_root}/images \
                     --train_mask_dir ${dataset_root}/masks \
                     --test_mask_dir  ${dataset_root}/mask_test \
                     --train_save_dir "split/train" \
                     --test_save_dir "split/test" \
                     --with_mask $2 \
                     --sdm_version $3