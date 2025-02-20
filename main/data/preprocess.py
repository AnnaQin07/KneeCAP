'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2023-07-29 01:14:31
LastEditors: YunxiangLiu u7191378@anu.edu.au
LastEditTime: 2023-11-02 14:23:22
FilePath: \MA_detections\data\preprocess.py
Description: preprocess
'''
import os 
import argparse

from split_images import split_imgs
from generate_sdm import generate_sdm


def parse_args():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--img_dir', type=str, default='minimc/images') # exp_1_128x1024_dc_b=8 
    parser.add_argument('--train_mask_dir', type=str, default='minimc/masks')
    parser.add_argument('--test_mask_dir', type=str, default='minimc/mask_test')
    parser.add_argument('--train_save_dir', type=str, default='split/train')
    parser.add_argument('--test_save_dir', type=str, default='split/test')
    parser.add_argument('--with_mask', type=str, default=True)
    parser.add_argument('--sdm_version', type=int, default=0)
    
    return parser.parse_args()

 

def preprocess(img_dir, mask_dir, save_root, with_mask, sdm_version):
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    coordi_save_dir = os.path.join(save_root, 'metas')
    if not os.path.exists(coordi_save_dir):
        os.makedirs(coordi_save_dir)
    
    split_imgs(img_dir, mask_dir, save_root, coordi_save_dir=coordi_save_dir, with_mask=with_mask)
    if with_mask:
        generate_sdm(save_root, 'ld_masks', 'sdm', sdm_version)
        

        

if __name__ == "__main__":
    args = parse_args()
    # preprocess training data 
    if args.with_mask:
        preprocess(args.img_dir, args.train_mask_dir, args.train_save_dir, args.with_mask, args.sdm_version)
    # preprocess test data 
        preprocess(args.img_dir, args.test_mask_dir, args.test_save_dir, args.with_mask, args.sdm_version)
    else:
        preprocess(args.img_dir, None, 'inf', args.with_mask, args.sdm_version)
    
    
    
    
    
    
    