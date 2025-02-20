import os
import cv2 
import json
import numpy as np
import pandas as pd 
import nibabel as nib
import imageio.v2 as iio

from tqdm import tqdm
from matplotlib import pyplot as plt



def Merge_and_Vis(img_dir, mask_dir, res_path, coord_change_dir, save_dir, rszd_w=256, dpi=100, seg_only=False, 
                    type_label_path=None, gt_hka_path=None, gt_ahka_path=None):
    
    """
    Merge the single leg mask, visualize results and output annotations from model in .nii format
    
    args:
        img_dir: dir that storage the full_limb X-ray in *.jpg format
        mask_dir: dir that storage the single-leg masks produced by model  
        res_path: angle measurement path PATH/TO/*.csv
        coord_change_dir: dir that storage the image splitting info in *.json file
        save_dir: dir for saving the output annotations (free to choose one)
        rszd_w: parameter relate to the merged mask size choice: [128, 256], if the width of model-produced mask is 256, then choose 256
        dpi: relate to the quality of visualized images, larger is better but slower to generate
        seg_only: display segmentation mask only? if yes, the function doesn't show measured landmarks & angles produced by postprocessing algorithm
        type_label_path: path to the file with record X-ray type [digital/analog], it's ok if you don't input it
        gt_hka_path: path to the file with record ground truth (gt) hka angles in *.json format, it's ok if you don't input it and once you input, gt angles shows on images as well
        gt_ahka_path: path to the file with record ground truth (gt) ahka angles in *.json format, it's ok if you don't input it and once you input, gt angles shows on images as well
    """
    # Load files that we want
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f"{save_dir}/niis"):
        os.makedirs(f"{save_dir}/niis")
    frame = pd.read_csv(res_path)
    x_keys = ['femur_head_x', 'condylar_midway_x', 'condylar_leftpeak_x', 'condylar_rightpeak_x', 
              'plateau_center_x', 'plateau_left_x', 'plateau_right_x', 'plafond_center_x']
    y_keys = ['femur_head_y', 'condylar_midway_y', 'condylar_leftpeak_y', 'condylar_rightpeak_y', 
              'plateau_center_y', 'plateau_left_y', 'plateau_right_y', 'plafond_center_y']
    landmark_keys = [x_key[:-2] for x_key in x_keys]
    img_names = frame['img_name'].unique()
    
    if type_label_path is not None:
        with open(type_label_path) as json_file:
            type_labels = json.load(json_file)
    else:
        type_labels = None
    
    if gt_hka_path is not None:
        with open(gt_hka_path) as json_file:
            gt_hkas = json.load(json_file)
    else:
        gt_hkas = None
    
    if gt_ahka_path is not None:
        with open(gt_ahka_path) as json_file:
            gt_ahkas = json.load(json_file)
    else:
        gt_ahkas = None
        
    # go and display one by one 
    for name in tqdm(img_names):
        
        # display images
        ids = frame[frame['img_name'] == name].index.tolist()
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        plt.figure(figsize=(h/dpi, w/dpi), dpi=dpi)
        plt.imshow(img, cmap='gray')
        
        # display masks
        left_mask = iio.imread(os.path.join(mask_dir, f"{os.path.splitext(name)[0]}_r.png")) # png
        right_mask = iio.imread(os.path.join(mask_dir, f"{os.path.splitext(name)[0]}_l.png")) # png
        img_type = type_labels[f"{os.path.splitext(name)[0]}_l.png"] if type_labels is not None else 'a'
        with open(os.path.join(coord_change_dir, f"{os.path.splitext(name)[0]}_r.json")) as json_file:
            cg_lft = json.load(json_file)
         
        with open(os.path.join(coord_change_dir, f"{os.path.splitext(name)[0]}_l.json")) as json_file:
            cg_rght = json.load(json_file)
        
        # process and merge masks    
        left_mask = filter_annotation(left_mask)
        right_mask = filter_annotation(right_mask)
        
        mask = generete_mask(img, left_mask / 255., right_mask / 255., cg_lft, cg_rght, rszd_w=rszd_w)
        mask = cv2.resize(mask, img.shape[::-1], cv2.INTER_LANCZOS4)
        tibia_mask = shrinking_mask((mask[..., 0] * 255).astype('uint8'), ksize=7, sigma=16, shrinking_level=1)
        femur_mask = shrinking_mask((mask[..., 1] * 255).astype('uint8'), ksize=7, sigma=16, shrinking_level=1)
        label = np.zeros([*mask.shape[:2], 1], dtype='float64')
        label[tibia_mask  > 0] = 2.
        label[femur_mask > 0] = 1.
        plt.imshow(mask) 
        
        annotation = nib.Nifti1Image(np.transpose(label, axes=(1, 0, 2)), np.eye(4) * np.array([-1, -1, 1, 1]))               
        nib.save(annotation, f"{save_dir}/niis/{os.path.splitext(name)[0]}.nii")

        if not seg_only:
            # display landmarks and mechanism axis
            for idx in ids:
                landmarks = {}
                series = frame.iloc[idx]
                
                is_good_femur = series["good_femur"]
                is_good_tibia = series["good_tibia"]
                if series['side'] == 2:
                    location_tibia = [w * 3 / 4, h * 3 / 4]
                    location_femur = [w * 3 / 4, h / 4]
                else:
                    location_tibia = [w / 4, h * 3 / 4]
                    location_femur = [w / 4, h / 4]
                
                plt.text(*location_tibia, f'{is_good_tibia}', c='y', fontsize=10)
                plt.text(*location_femur, f'{is_good_femur}', c='y', fontsize=10)   
                
                if not series.isnull().any():
                    xs = [series[xk] for xk in x_keys]
                    ys = [series[yk] for yk in y_keys]
                    for i, landmark in enumerate(landmark_keys):
                        landmarks[landmark] = [xs[i], ys[i]]
                    
                    # femur mechan axes
                    femur_mechanism_axes = list(zip(landmarks['femur_head'], landmarks['condylar_midway']))
                    tibia_mechanism_axes = list(zip(landmarks['plateau_center'], landmarks['plafond_center']))
                    femur_condylar = list(zip(landmarks['condylar_leftpeak'], landmarks['condylar_rightpeak']))
                    tibia_plateau = list(zip(landmarks['plateau_left'], landmarks['plateau_right']))
                    
                    plt.plot(*femur_mechanism_axes, linewidth=1)
                    plt.scatter(*femur_mechanism_axes, s=1)
                    plt.plot(*tibia_mechanism_axes, linewidth=1)
                    plt.scatter(*tibia_mechanism_axes, s=1)
                    plt.plot(*femur_condylar)
                    plt.scatter(*femur_condylar, s=5)
                    plt.plot(*tibia_plateau)
                    plt.scatter(*tibia_plateau, s=5)
                    
                    hkaa = series["hka"]
                    ahka = series["MPTA-LDFA"]
                    
                    if series['side'] == 2:
                        location_hkaa = [tibia_plateau[0][1], tibia_plateau[1][1] - 90]
                    else:
                        location_hkaa = [tibia_plateau[0][0] - 60, tibia_plateau[1][0] - 90]
                    if img_type == 'a':
                        fontsize=8
                    else:
                        fontsize=12
                    fontsize = 10
                    plt.text(*location_hkaa, f'HKAA: {round(hkaa, 2)}', c='y', fontsize=fontsize)
                    location_cpa = [location_hkaa[0], location_hkaa[1] + 180]
                    plt.text(*location_cpa, f'aHKA: {round(ahka, 2)}', c='y', fontsize=fontsize)
                    # for gt
                    if gt_hkas is not None:
                        img_name = os.path.splitext(series['img_name'])[0]
                        if 'pre' in img_name or 'post' in img_name:
                            suffix = 'jpg'
                        else:
                            suffix = 'png'
                        if series['side'] == 2:
                            key = f"{img_name}_l.{suffix}"
                            location_gt_hkaa = [tibia_plateau[0][1], tibia_plateau[1][1] - 180]
                        else:
                            key = f"{img_name}_r.{suffix}"
                            location_gt_hkaa = [tibia_plateau[0][0] - 60, tibia_plateau[1][0] - 180]
                        gt_hka = gt_hkas.get(key, None)
                        if gt_hka is not None:
                            plt.text(*location_gt_hkaa, f'HKAA_gt: {round(gt_hka, 2)}', c='b', fontsize=fontsize)
                    
                    if gt_ahkas is not None:
                        img_name = os.path.splitext(series['img_name'])[0]
                        if series['side'] == 2:
                            key = f"{img_name}_l.{suffix}"
                        else:
                            key = f"{img_name}_r.{suffix}"
                        gt_ahka = gt_ahkas.get(key, None)
                        location_gt_ahka = [location_cpa[0], location_cpa[1] + 90]
                        if gt_ahka is not None:
                            plt.text(*location_gt_ahka, f'aHKA_gt: {round(gt_ahka, 2)}', c='b', fontsize=fontsize)
                            
                # plt.scatter(xs, ys, s=5)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches = -0.1, dpi=dpi*7)
        plt.clf() 
        
        
def generete_mask(img, mask_lft, mask_rght, cg_lft, cg_rght, rszd_w=256):
    h, w = img.shape 
    background = np.zeros((h, w, 4)) 
    background = cv2.resize(background, (rszd_w*2, int(h * rszd_w * 2 / w)))
    
    tp_pd, bttm_pd, lft_pd, rght_pd = cg_lft['top_padding'], cg_lft['bottom_padding'], cg_lft['left_padding'], cg_lft['right_padding']
    mask_lft = mask_convert(tp_pd, bttm_pd, mask_lft)

    # mask_lft = mask_lft[tp_pd:-bttm_pd, :, :2]
    background[:, :rszd_w, :2] = mask_lft
    
    tp_pd, bttm_pd, lft_pd, rght_pd = cg_rght['top_padding'], cg_rght['bottom_padding'], cg_rght['left_padding'], cg_rght['right_padding']

    # mask_rght = mask_rght[tp_pd:-bttm_pd, :, :2]
    mask_rght = mask_convert(tp_pd, bttm_pd, mask_rght)
    mask_rght = mask_rght[:, ::-1, :]
    background[:, rszd_w:, :2] = mask_rght
    
    alpha = np.logical_or(background[..., 0] > 0, background[..., 1] > 0)
    alpha = alpha.astype('float64') * 0.1
    background[..., 3] = alpha 
    return background


def mask_convert(tp_pd, bttm_pd, mask):
    if tp_pd > 0:
        mask = mask[tp_pd:]
    else:
        mask = cv2.copyMakeBorder(mask, -tp_pd, 0, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 255))
    if bttm_pd > 0:
        mask = mask[:-bttm_pd]
    else:
        mask = cv2.copyMakeBorder(mask, 0, -bttm_pd, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 255))
    return mask[..., :2]



def filter_annotation(mask):
    
    femur = filter_mask(mask[..., 1])
    tibia = filter_mask(mask[..., 0])
    mask_of_mask = np.stack([tibia, femur, np.ones(femur.shape)], axis=2).astype('uint8')
    mask = mask * mask_of_mask
    blank = np.logical_and(mask[..., 0] == 0, mask[..., 1] == 0)
    blank = np.logical_and(blank, mask[..., 2] == 0)
    mask[blank] = np.array([0, 0, 255], dtype='uint8')
    
    return mask


def shrinking_mask(mask, ksize=7, sigma=6, shrinking_level=2):
    mask = cv2.GaussianBlur(mask, 2*(ksize,), sigma)
    mask = np.where(mask > 127, 255, 0).astype('uint8')
    usdf = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    return mask * (usdf >= shrinking_level).astype('uint8')  

def filter_mask(mask):
    """Filter binary masks through connected component analysis

    Args:
        mask (np.array([h, w])): binrary masks

    Returns:
        np.array([h, w]): filtered binrary masks which keeps the largest foreground area, these are usually the bone mask we want
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_16S)
    area = stats[:, cv2.CC_STAT_AREA].squeeze()
    if area.size < 2:
        return np.ones(mask.shape)
    label = np.argsort(area)[-2]
    return (labels == label).astype('float32')