import os
import math
import json

import pandas as pd
import numpy as np 
import imageio.v2 as imageio

from collections import defaultdict
from matplotlib import pyplot as plt

from .post_utils import bone_boxing, crop_bone, filter_mask, shape_check, check_results

from .landmarks import locate_femur_head_v0, locate_plateau_features_v1, locate_condylar_features, locate_tibia_plateau_features, tibia_plafond_distally, locate_condylar_features_v1
    
def get_points_from_seg(seg_dir, save_dir, coord_transform_dir, rsz_witdth=256, plateau_peak_neigbor=4, ankle_neigbor=12):
    filenames = os.listdir(seg_dir)
    record = defaultdict(list)
    for n in filenames:
        print("process: ", n)
        name = os.path.splitext(n)[0]
        mask = imageio.imread(os.path.join(seg_dir, n))
        with open(os.path.join(coord_transform_dir, f"{name}.json"), 'r') as jf:
            chang_info = json.load(jf)
        mask = mask_pad(mask, chang_info)

        key_points = keypoints(mask, plateau_peak_neigbor, ankle_neigbor)
        femur_head = key_points['femur_head']
        FemurHead_radius = key_points['radius']
        condylar_midway = key_points['condylar_midway']
        condylar_leftpeak = key_points['condylar_leftpeak']
        condylar_rightpeak = key_points['condylar_rightpeak']
        plateau_center = key_points['plateau_center']
        plateau_left = key_points['plateau_left']
        plateau_right = key_points['plateau_right']
        plafond_center = key_points['plafond_center']
        is_good_femur = key_points['good_femur']
        is_good_tibia = key_points['good_tibia']

        side = name.split('_')[1]
        side = 1 if side == 'r' else 2
        # if side == 2:
        #     condylar_leftpeak, condylar_rightpeak = condylar_rightpeak, condylar_leftpeak
        #     plateau_left, plateau_right = plateau_right, plateau_left
        key_point_list = [femur_head, condylar_midway, condylar_leftpeak, condylar_rightpeak,
                          plateau_center, plateau_left, plateau_right, plafond_center]
        """check for debug
        # img = imageio.imread(os.path.join("D:\\Datasets\\comp8603\\Lower_limb_Xrays\\12m_dicoms\\png\\test\\images", f"{name}.png"))
        # plt.imshow(img, cmap='gray')
        # xs, ys = zip(femur_head, condylar_midway, plateau_center, plafond_center)
        # plt.scatter(xs, ys)
        # plt.show()
        """
        femur_head, condylar_midway, condylar_leftpeak, condylar_rightpeak, plateau_center, plateau_left, plateau_right, plafond_center \
            = coordi_switch(key_point_list, chang_info, side, rsz_witdth)
        """check for debug
        # img = imageio.imread(os.path.join("D:\\Datasets\\comp8603\\Lower_limb_Xrays\\12m_dicoms\\jpg", f"{name.split('_')[0]}.jpg"))
        # xs, ys = zip(femur_head, condylar_midway, plateau_center, plafond_center)
        # plt.imshow(img, cmap='gray')
        # plt.scatter(xs, ys)
        # plt.show()
        """
        hka = calculate_hka(femur_head, condylar_midway, plateau_center, plafond_center)
        cpa = calculate_cpa(condylar_leftpeak, condylar_rightpeak, plateau_left, plateau_right)
        ldfa, mpta = calculate_ldf_mpt(condylar_leftpeak, condylar_rightpeak, plateau_left, plateau_right, femur_head, condylar_midway, plateau_center, plafond_center)
        if side == 2 and hka is not None:
            hka = -hka
        if side == 2 and cpa is not None:    
            cpa == -cpa
        if ldfa is None or mpta is None:
            plus = None 
            minus = None
        else:
            plus = ldfa +  mpta
            minus = mpta - ldfa
        tmp = {'img_name': f"{name.split('_')[0]}.jpg",
               'side': side,
               'FemurHead_radius': FemurHead_radius,
               'good_femur': is_good_femur,
               'good_tibia': is_good_tibia,
               'femur_head_x': femur_head[0], 'femur_head_y': femur_head[1],
               'condylar_midway_x': condylar_midway[0], 'condylar_midway_y': condylar_midway[1],
               'condylar_leftpeak_x': condylar_leftpeak[0], 'condylar_leftpeak_y': condylar_leftpeak[1],
               'condylar_rightpeak_x': condylar_rightpeak[0], 'condylar_rightpeak_y': condylar_rightpeak[1],          
               'plateau_center_x': plateau_center[0], 'plateau_center_y': plateau_center[1], 
               'plateau_left_x': plateau_left[0], 'plateau_left_y': plateau_left[1], 
               'plateau_right_x': plateau_right[0], 'plateau_right_y': plateau_right[1], 
               'plafond_center_x': plafond_center[0], 'plafond_center_y': plafond_center[1],
               'hka': hka,
               'cpa': cpa,
               'LDFA': ldfa,
               'MPTA': mpta,
               'LDFA+MPTA': plus,
               'MPTA-LDFA': minus}

        is_good_result = check_results(**tmp)
        if not is_good_result:
            for k in ['hka', 'cpa', 'LDFA', 'MPTA', 'LDFA+MPTA', 'MPTA-LDFA']:
                tmp[k] = None 

        for k, v in tmp.items():
            record[k].append(v)
    pdframe = pd.DataFrame(record)
    pdframe.to_csv(os.path.join(save_dir, f'measurements.csv'), index=False)
    
        
        
        
def coordi_switch(swith_tuples, change_info, side, rsz_width=256):
    res = []
    for coord in swith_tuples:
        x, y = coord
        if x is not None and y is not None:
            x -= change_info['left_padding']
            y -= change_info['top_padding']
            if side == 1:
                x *= change_info['ratio_width']
            else:
                x = (rsz_width - x) * change_info['ratio_width']
            y *= change_info['ratio_height']
            res.append((x, y))
        else:
            res.append((None, None))
        
    return res
        
            
def calculate_hka(femur_head, condylar_midway, plateau_center, plafond_center):
    '''
    calculate the hka angle. Inputs are the coordinates of the landmarks
    :return: the hka angle
    '''
    for x, y in [femur_head, condylar_midway, plateau_center, plafond_center]:
        if x is None or y is None:
            return None
    a, b, c, d = np.array(femur_head), np.array(condylar_midway), np.array(plateau_center), np.array(plafond_center)
    vector_femur = a - b
    vector_tibia = c - d
    dot_product = np.dot(vector_femur, vector_tibia)
    mod = np.linalg.norm(vector_femur) * np.linalg.norm(vector_tibia)
    if not -1 <= dot_product / mod <= 1:
        return None 
    angle = math.degrees(math.acos(dot_product / mod))
    # if np.cross(vector_femur, vector_tibia) > 0:
    #     angle = angle
    # else:
    #     angle = - angle
    return angle * np.sign(np.cross(vector_femur, vector_tibia))

def calculate_cpa(condylar_left, condylar_right, plateau_left, plateau_right):
    '''
    calculate the hka angle. Inputs are the coordinates of the landmarks
    :return: the hka angle
    '''
    for x, y in [condylar_left, condylar_right, plateau_left, plateau_right]:
        if x is None or y is None:
            return None
    a, b, c, d = np.array(condylar_left), np.array(condylar_right), np.array(plateau_left), np.array(plateau_right)
    vector_condylar = a - b
    vector_plateau = c - d
    dot_product = np.dot(vector_condylar, vector_plateau)
    mod = np.linalg.norm(vector_condylar) * np.linalg.norm(vector_plateau)
    if not -1 <= dot_product / mod <= 1:
        return None 
    angle = math.degrees(math.acos(dot_product / mod))
    # if np.cross(vector_condylar, vector_plateau) > 0:
    #     angle = angle
    # else:
    #     angle = - angle
    return angle * np.sign(np.cross(vector_condylar, vector_plateau))


def calculate_ldf_mpt(condylar_left, condylar_right, plateau_left, plateau_right, femur_head, condylar_midway, plateau_center, plafond_center):
    
    for x, y in [condylar_left, condylar_right, plateau_left, plateau_right, femur_head, condylar_midway, plateau_center, plafond_center]:
        if x is None or y is None:
            return None, None
    vector_condylar = np.array(condylar_left) - np.array(condylar_right)
    vector_femur_axis = np.array(femur_head) - np.array(condylar_midway)
    dot_product = np.dot(vector_condylar, vector_femur_axis)
    mod = np.linalg.norm(vector_condylar) * np.linalg.norm(vector_femur_axis)
    ldfa = math.degrees(math.acos(dot_product / mod))
    
    vector_plateau = np.array(plateau_right) - np.array(plateau_left)
    vector_tibia_axis = np.array(plafond_center) - np.array(plateau_center)
    dot_product = np.dot(vector_plateau, vector_tibia_axis)
    mod = np.linalg.norm(vector_plateau) * np.linalg.norm(vector_tibia_axis)
    mpta = math.degrees(math.acos(dot_product / mod))
    return ldfa, mpta
    
        
    

def keypoints(mask, plateau_peak_neigbor=4, ankle_neigbor=12, img_height=2048):
    femur_mask = filter_mask(mask[..., 1])
    tibia_mask = filter_mask(mask[..., 0])
    
    is_good_femur = shape_check(femur_mask, img_height, bone='femur')
    is_good_tibia = shape_check(tibia_mask, img_height, bone='tibia')
    
    if not (is_good_femur and is_good_tibia):
        return {'femur_head': (None, None),
                'radius': None,
                'condylar_midway': (None, None),
                'condylar_leftpeak': (None, None),
                'condylar_rightpeak': (None, None),
                'plateau_center': (None, None),
                'plateau_left': (None, None),
                'plateau_right': (None, None),
                'plafond_center': (None, None),
                'good_femur': is_good_femur,
                'good_tibia': is_good_tibia
                }
    femur_head, r = locate_femur_head_v0((femur_mask * 255).astype('uint8'))
    # femur_head, r = locate_femur_head_v1((femur_mask * 255).astype('uint8'))
    condylar_features = locate_condylar_features(femur_mask)
    # condylar_features = locate_condylar_features_v1(femur_mask)
    condylar_midway = condylar_features['condylar_midway']
    condylar_leftpeak = condylar_features['condylar_left']
    condylar_rightpeak = condylar_features['condylar_right']
    
    # tibia
    bbx = bone_boxing(tibia_mask)
    roi = crop_bone(tibia_mask, bbx)
    interg_x = np.sum(roi, axis=1)
    plateau_features = locate_tibia_plateau_features(roi, interg_x, bbx, plateau_peak_neigbor)
    # plateau_features = locate_plateau_features_v1(roi, interg_x, bbx, plateau_peak_neigbor)
    plateau_center = plateau_features['center']
    plateau_left = plateau_features['left_terminal']
    plateau_right = plateau_features['right_terminal']
    plafond_center = tibia_plafond_distally(roi, interg_x, bbx, ankle_neigbor)
    
    """debug
    plt.imshow(tibia_mask)
    plt.scatter(*plateau_center)
    plt.scatter(*plateau_left)
    plt.scatter(*plateau_right)
    plt.show()
    """
    return {'femur_head': femur_head,
             'radius': r,
             'condylar_midway': condylar_midway,
             'condylar_leftpeak': condylar_leftpeak,
             'condylar_rightpeak': condylar_rightpeak,
             'plateau_center': plateau_center,
             'plateau_left': plateau_left,
             'plateau_right': plateau_right,
             'plafond_center': plafond_center,
             'good_femur': is_good_femur,
             'good_tibia': is_good_tibia
             }

 
def mask_pad(mask, padding_dict):
    """Masking the padding region with background color

    Args:
        mask ([type]): [description]
        padding_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    if padding_dict['top_padding'] > 0:
        mask[:padding_dict['top_padding']] = np.array([0, 0, 255])
    if padding_dict['bottom_padding'] > 0:
        mask[-padding_dict['bottom_padding']:] = np.array([0, 0, 255])
    if padding_dict['left_padding'] > 0:
        mask[:, :padding_dict['left_padding']] = np.array([0, 0, 255])
    if padding_dict['right_padding'] > 0:
        mask[:, -padding_dict['right_padding']:] = np.array([0, 0, 255])
    return mask