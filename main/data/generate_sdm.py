'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2023-07-29 00:48:06
LastEditors: YunxiangLiu u7191378@anu.edu.au
LastEditTime: 2023-11-02 14:20:23
FilePath: \MA_detections\data\generate_sdm.py
Description: tools for preprocessing
'''
import os
import cv2
import numpy as np 
from tqdm import tqdm



def generate_sdm(root, mask_dir, save_dir, version=0):
    
    save_dir = os.path.join(root, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # load path of all masks
    mask_names = [i for i in os.listdir(os.path.join(root, mask_dir)) if i.endswith(".png")]
    mask_paths = list(map(lambda x: os.path.join(root, mask_dir, x), mask_names))   
    for mask_name, mask_path in tqdm(zip(mask_names, mask_paths)):
        mask = cv2.imread(mask_path)
        _, femur, tibia = cv2.split(mask)
        # usdf
        if version == 0:
            unsign_distmap_tibia = usdf_v0(tibia)
            unsign_distmap_femur = usdf_v0(femur)
        else:
            unsign_distmap_tibia = usdf_v1(tibia)
            unsign_distmap_femur = usdf_v1(femur)
        # assign sign for usdf (usdf -> sdf)
        sdm_femur = np.where(femur > 0, -1, 1) * unsign_distmap_femur
        sdm_tibia = np.where(tibia > 0, -1, 1) * unsign_distmap_tibia
        sdm = np.concatenate([sdm_femur[..., None], sdm_tibia[..., None]], axis=-1)
        np.save(os.path.join(save_dir, f"{os.path.splitext(mask_name)[0]}.npy"), sdm)


def usdf_v0(mask):
    """Unsigned distance transform v0
    customized usdf function. slow
    Args:
        mask (np.array([w, h])): the mask of femur/tibia

    Returns:
        usdf map: the usdf map of the input masks
    """
    edge = cv2.Canny(mask, 100, 200)
    h, w = edge.shape 
    raster = generate_raster(w, h)
    edge_ids = np.argwhere(edge > 0)
    l2_distances = np.linalg.norm(raster[..., None, :] - edge_ids.reshape(1, 1, -1, 2), axis=-1)
    return l2_distances.min(axis=-1)


def usdf_v1(mask):
    """Unsigned distance transform

    Args:
        mask (np.array([w, h])): the mask of femur/tibia

    Returns:
        usdf map: the usdf map of the input masks
    """
    edge = cv2.Canny(mask, 100, 200)
    return cv2.distanceTransform(inverse_binary(edge), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

      
def inverse_binary(img):
    img = img * -1
    img[img == 0] = 255
    img[img < 0] = 0
    return img.astype('uint8')



def generate_raster(w, h):
    xs = np.linspace(0, w-1, w)
    ys = np.linspace(0, h-1, h)
    yr, xr = np.meshgrid(ys, xs, indexing='ij')
    return np.concatenate([yr[..., None], xr[..., None]], axis=-1)



        