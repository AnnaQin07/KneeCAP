'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2023-11-27 16:38:58
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2024-03-02 16:29:52
FilePath: \MA_detections\maintaince.py
Description: This scripts is for you to test functions and try new things, feel free to use it
'''


import numpy as np 
import imageio.v2 as imageio

from postprocess.postprocess import keypoints
from matplotlib import pyplot as plt



if __name__ == "__main__":
    
    mask_path = r"experiments/assess_canberra/seg_mask/Vh-IM-0005-0005-0001_r.png"
    mask = imageio.imread(mask_path)
    points = keypoints(mask)
    print(points)
    
    




