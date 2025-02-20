import os
import cv2
import json
import imageio
import pydicom

import numpy as np
import pandas as pd 
import nibabel as nib
import imageio.v2 as imageio
import PIL.Image as Image

from tqdm import tqdm

from shutil import copy, move


def split_imgs(img_dir, mask_dir, save_path, coordi_save_dir=r"12m_dicoms/coord_change", with_mask=True):
    """pre-processing the training examples by splitting them into two single-leg images

    Args:
        img_dir (str): the directory where the ordinary .dcm images saved
        mask_dir (str): the directory where the ordinary .nii masks saved
        save_path (str): the root directory where splitted images and the masks saved
        coordi_save_dir (regexp, optional): the directory where the transformation info saved. Defaults to r"12m_dicoms/coord_change".
        with_mask (bool, optional): _description_. Defaults to True.
    """
    img_type = 'dcm'
    if with_mask:
        mask_names = os.listdir(mask_dir)
    else:
        mask_names = os.listdir(img_dir)
    img_names = list(map(lambda x: f'{os.path.splitext(x)[0]}.{img_type}', mask_names))
    img_save_path = os.path.join(save_path, 'images')
    small_mask_save_path = os.path.join(save_path, 'ld_masks')
    large_mask_save_path = os.path.join(save_path, 'hd_masks')
    
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if with_mask and not os.path.exists(large_mask_save_path):
        os.makedirs(large_mask_save_path)
    if with_mask and not os.path.exists(small_mask_save_path):
        os.makedirs(small_mask_save_path)
    
    vis_path = f"{img_dir}/../jpg"
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
        
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    for img_name, mask_name in tqdm(zip(img_names, mask_names)):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        if img_type == 'dcm':
            img = pydicom.read_file(img_path)
            img = img.pixel_array
            # this is help you visualise the original images
            jpg_img = Uint162Unit8(img)
            imageio.imwrite(os.path.join(vis_path, f"{os.path.splitext(img_name)[0]}.jpg"), jpg_img)
        else:
            img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        #img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        if with_mask:
            mask = nib.load(mask_path)
            mask = mask.get_fdata()
            mask = np.squeeze(mask).T
            new_mask = np.zeros((*mask.shape, 3))
            new_mask[mask == 0] = blue
            new_mask[mask == 1] = green
            new_mask[mask == 2] = red 
        
        # right part
        right_part, ratio_h, ratio_w, tp_pd, lft_pad, bttm_pad, rgt_pad = preprocessing(img, side="right", rgb=False, enhance=True, size=(256, 2048))
        if with_mask:
            right_mask_small, _, _, _, _, _, _ = preprocessing(new_mask, side="right", rgb=True, enhance=False, size=(128, 1024))
            right_mask_large, _, _, _, _, _, _ = preprocessing(new_mask, side="right", rgb=True, enhance=False, size=(256, 2048))
        right_name = f"{os.path.splitext(img_name)[0]}_r.png"
        imageio.imwrite(os.path.join(img_save_path, right_name), right_part.astype('uint16'))
        if with_mask:
            imageio.imwrite(os.path.join(small_mask_save_path, right_name), right_mask_small.astype('uint8'))
            imageio.imwrite(os.path.join(large_mask_save_path, right_name), right_mask_large.astype('uint8'))
        save_dict = {'ratio_height': ratio_h, 'ratio_width': ratio_w, 
                     "top_padding": tp_pd, "left_padding": lft_pad, 
                     "bottom_padding": bttm_pad, 'right_padding': rgt_pad}
        json_str = json.dumps(save_dict)
        with open(os.path.join(coordi_save_dir, f"{os.path.splitext(right_name)[0]}.json"), 'w') as jf:
            jf.write(json_str)
        # left part
        left_part, ratio_h, ratio_w, tp_pd, lft_pad, bttm_pad, rgt_pad = preprocessing(img, side="left", rgb=False, enhance=True, size=(256, 2048))
        if with_mask:
            left_mask_small, _, _, _, _, _, _ = preprocessing(new_mask, side="right", rgb=True, enhance=False, size=(128, 1024))
            left_mask_large,  _, _, _, _, _, _ = preprocessing(new_mask, side="left", rgb=True, enhance=False, size=(256, 2048))
        left_name = f"{os.path.splitext(img_name)[0]}_l.png"
        imageio.imwrite(os.path.join(img_save_path, left_name), left_part.astype('uint16'))
        if with_mask:
            imageio.imwrite(os.path.join(small_mask_save_path, left_name), left_mask_small.astype('uint8'))
            imageio.imwrite(os.path.join(large_mask_save_path, left_name), left_mask_large.astype('uint8'))
        save_dict = {'ratio_height': ratio_h, 'ratio_width': ratio_w, 
                     "top_padding": tp_pd, "left_padding": lft_pad, 
                     "bottom_padding": bttm_pad, 'right_padding': rgt_pad}
        json_str = json.dumps(save_dict)
        with open(os.path.join(coordi_save_dir, f"{os.path.splitext(left_name)[0]}.json"), 'w') as jf:
            jf.write(json_str)
          
            
def preprocessing(img, side="right", rgb=True, enhance=True, size=(256, 2048)):
    '''
    preprocessing raw image for network training/inference
    :param input_path: The path of the raw images
    :param side: crop the image with right or left side, ATTENTION: the side is defined based on clinic, which is opposite of it in computer vision and our common sense
    :return: preprocessed image
    '''
    rszd_w, rszd_h = size
    if rgb:
        h, w, _ = img.shape
        img = Image.fromarray(img.astype('uint8'))
        img = img.resize((rszd_w*2, int(h * rszd_w * 2 / w)), Image.ANTIALIAS) # 256
        img = np.asarray(img, dtype='uint8')
        img = reassign_label(img)
        

    else:
        h, w = img.shape
        img = cv2.resize(img, (rszd_w * 2, int(h * rszd_w * 2 / w)), interpolation=cv2.INTER_AREA)  # 256

    # 
    h_, w_ = img.shape[:2]
    # cropped the image to half, with unilateral limb
    if side == "right":
        cropped_img = img[:, :rszd_w]  # 128
    else:
        cropped_img = img[:, rszd_w:]  # 128
        cropped_img = cv2.flip(cropped_img, 1)

    img, top, left, bottom, right = resize_with_pad(cropped_img, size, rgb)  # 256
    # image enhancement
    if enhance and (not rgb):
        gridsize = 8 if rszd_h == 1024 else 16
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(gridsize,) * 2) # (16, 16) in large imgs
        img = clahe.apply(img)
    return img, h / h_, w / w_, top, left, bottom, right


def reassign_label(img):
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    
    res = np.zeros(img.shape)
    label = np.argmax(img, axis=-1)
    res[label == 0] = red
    res[label == 1] = green
    res[label == 2] = blue
    return res.astype('uint8')


def resize_with_pad(img, target_size, rgb=False):
    """resize the image by padding at borders.
    Params:
        img: image to be resized, read by cv2.imread()
        target_size: a tuple shows the image size after padding.
        For example, a tuple could be like (width, height)
    Returns:
        image: resized image with padding
    refer to
    https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
    """
    img_size = (img.shape[1], img.shape[0])
    d_w = target_size[0] - img_size[0]
    d_h = target_size[1] - img_size[1]
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    # rgb image default read by cv2, with BGR channels
    value = (0, 0, 255) if rgb else 0
    pad_img = cv2.copyMakeBorder(img, 
                                 max(top, 0), 
                                 max(bottom, 0), 
                                 max(left, 0), 
                                 max(right, 0), cv2.BORDER_CONSTANT, value=value)
    if d_h < 0:
        pad_img = pad_img[abs(top):bottom]
    if d_w < 0:
        pad_img = pad_img[:, abs(left): right]
    return pad_img, top, left, bottom, right

def Uint162Unit8(img):
    img = img / 65535.0 * 255
    return np.uint8(255 * (img - img.min()) / (img.max() - img.min()) + 0.5)