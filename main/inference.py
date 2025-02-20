
import os
import cv2
import math
import json
import torch
import pandas as pd
import numpy as np 
import pingouin as pg
import imageio.v2 as imageio

from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from collections import defaultdict
from easydict import EasyDict as edict
from datasets import Build_dataset, Check_dataset, Split_dataset

from utils import pred2mask
from postprocess.evaluation import evaluation
from postprocess.visualisation import Merge_and_Vis
from postprocess.postprocess import get_points_from_seg, keypoints


def inference_seg(args, dataset, model):
    save_path_seg = os.path.join('experiments', args.exp_name, 'seg_mask')
    if not os.path.exists(save_path_seg):
        os.makedirs(save_path_seg)
    N = len(dataset)
    for i in tqdm(range(N)):
        data = dataset[i]
        img = data['img'].unsqueeze(0).to(args.device)
        mask = model(img)['fine_pred']
        mask = pred2mask(mask).cpu().numpy()
        mask = (255 * mask).astype('uint8')
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path_seg, data['img_name']), mask)
    


def inference(args, dataset, model):
    
    model.eval()
    model.render_head.mode = 'inference'
    if args.inf_mode == 'seg_only':
        inference_seg(args, dataset, model)
            
    elif args.inf_mode == 'postprocess_only':
        save_path_seg = os.path.join('experiments', args.exp_name, 'seg_mask')
        save_dir = os.path.join('experiments', args.exp_name)
        coord_transform_dir = args.datasets.test.coordi_save_dir
        pstprcss_args = args.postprocess
        get_points_from_seg(save_path_seg, save_dir, coord_transform_dir, 
                            rsz_witdth=pstprcss_args.rsz_width, 
                            plateau_peak_neigbor=pstprcss_args.plateau_peak_neigbor, 
                            ankle_neigbor=pstprcss_args.ankle_neigbor)
        
    elif args.inf_mode == 'all':
        inference_seg(args, dataset, model)
        save_path_seg = os.path.join('experiments', args.exp_name, 'seg_mask')
        save_dir = os.path.join('experiments', args.exp_name)
        coord_transform_dir = args.datasets.test.coordi_save_dir
        pstprcss_args = args.postprocess
        get_points_from_seg(save_path_seg, save_dir, coord_transform_dir, 
                            rsz_witdth=pstprcss_args.rsz_width, 
                            plateau_peak_neigbor=pstprcss_args.plateau_peak_neigbor, 
                            ankle_neigbor=pstprcss_args.ankle_neigbor)
    
    if args.get('visualisation', None) is not None:
        visarg = args.visualisation
        img_dir = visarg.raw_img_dir
        mask_dir = os.path.join('experiments', args.exp_name, 'seg_mask')
        coord_change_dir = args.datasets.test.coordi_save_dir
        save_dir = os.path.join('experiments', args.exp_name, 'key_points')
        type_label_path = visarg.get('type_label_path', None)
        gt_hka_path = visarg.get('gt_hka_path', None)
        gt_ahka_path = visarg.get('gt_ahka_path', None)
        seg_only = visarg.get('seg_only', False)
        res_path = visarg.get('seg_path', None)
        if res_path is None:
            possible_path = os.path.join('experiments', args.exp_name, 'measurements.csv')
            if os.path.exists(possible_path):
                res_path = possible_path
            else:
                raise ValueError("No measurement results detected")
        Merge_and_Vis(img_dir, mask_dir, res_path, coord_change_dir, save_dir, rszd_w=256, dpi=100, 
                      seg_only=seg_only, type_label_path=type_label_path, gt_hka_path=gt_hka_path, gt_ahka_path=gt_ahka_path)
    
    if args.get('evaluation') is not None:
        evaluation(args)
        
        
        
        
        
        
            
            
        
        
        
        
        
        
        




if __name__ == "__main__":
    configs = edict({'type': 'dcm', 'img_dir': "data/minimc/images"})
    
    dataset = Build_dataset(configs)
    _ = dataset[0]
    