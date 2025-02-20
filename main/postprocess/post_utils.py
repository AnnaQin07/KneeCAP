import cv2
import numpy as np 

def bone_boxing(mask):
    """boxing the bone mask with a bounding box

    Args:
        mask (np.array([h, w])): the segmentation mask

    Returns:
        tuple(x1, y1, x2, y2): the top left and bottom right coordinates of surrounding bounding box
    """
    ys, xs = np.where(mask > 0)
    return xs.min(), ys.min(), xs.max(), ys.max()

def crop_bone(mask, bbx):
    """crop the mask with the bounding box region

    Args:
        mask (np.array([h, w])): the segmentation mask
        Sequence(x1, y1, x2, y2): the top left and bottom right coordinates of surrounding bounding box

    Returns:
        np.array([h, w]): the cropped segmentation mask
    """
    return mask[bbx[1]: bbx[3], bbx[0]: bbx[2]]

def bilinear_interpolation(curve, idx):
    rate = idx - int(idx)
    return curve[int(idx)] * (1 - rate) +  curve[int(idx) + 1] * rate


def find_inclose_circle(contours, roi):
    """find internally connected circle for a curve

    Args:
        contours (np.array([m, 2])): coordinates of contours
        roi (np.array([h, w])): masks of object

    Returns:
        (xc, yc), radius: circle params
    """
    coordinates = np.argwhere(roi > 0)
    diff = contours - coordinates[None, ...]
    L2_dist = np.linalg.norm(diff, axis=-1)
    distance = np.min(L2_dist, axis=0)
    max_idx = np.argmax(distance)

    return tuple(coordinates[max_idx]), distance[max_idx]
    
    

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
        return None
    label = np.argsort(area)[-2]
    return (labels == label).astype('float32')



def shape_check(msk, img_height=2048, bone='femur'):
    """check whether the bone mask is reasonable via the pre-defined rules
    * rule 1: the 

    Args:
        msk (np.array([h, w])): the segmentation mask
        img_height (int, optional): the height of image. Defaults to 2048.
        bone (str, optional): the bone type in {femur, tibia}. Defaults to 'femur'.

    Returns:
        bool: whether the bone is good 
    """
    if np.sum(msk) == 0 or msk is None:
        return False
    bbx = bone_boxing(msk)
    msk = crop_bone(msk, bbx)
    curve = np.sum(msk, axis=1)
    # check length
    default_rate = 0.65 if bone == 'femur' else 0.5
    len_bone = curve.shape[0]
    mid_idx = len_bone // 2
    if (len_bone / img_height) < 0.2:
        return False
    # check boundary distribution
    upp_bndry = np.argmax(curve[:mid_idx])
    low_bndry = np.argmax(curve[mid_idx:]) + mid_idx
    
    if upp_bndry / len_bone > 0.15:
        return False
    if low_bndry / len_bone < 0.90:
        return False
    # check boundary distribution
    rate = min(curve[upp_bndry]/curve[low_bndry], curve[low_bndry]/curve[upp_bndry])
    if rate < default_rate:
        return False
    
    return True
    


def check_results(**kwargs):
    """Check whether the measured landmarks coordinates are reasonable

    Returns:
        bool: the results 
    """
    if None in kwargs.values():
        return False
    if not kwargs['femur_head_y'] < kwargs['condylar_midway_y'] < kwargs['plateau_center_y'] < kwargs['plafond_center_y']:
        return False
    if kwargs['condylar_leftpeak_y'] <= kwargs['condylar_midway_y'] or kwargs['condylar_rightpeak_y'] <= kwargs['condylar_midway_y']:
        return False
    if kwargs['plateau_left_y'] <= kwargs['plateau_center_y'] or kwargs['plateau_right_y'] <= kwargs['plateau_center_y']:
        return False
    if not (kwargs['condylar_leftpeak_x'] < kwargs['condylar_midway_x'] < kwargs['condylar_rightpeak_x'] or 
            kwargs['condylar_rightpeak_x'] < kwargs['condylar_midway_x'] < kwargs['condylar_leftpeak_x']):
        return False
    if not (kwargs['plateau_right_x'] < kwargs['plateau_center_x'] < kwargs['plateau_left_x'] or 
            kwargs['plateau_left_x'] < kwargs['plateau_center_x'] < kwargs['plateau_right_x']):
        return False
    
    # check knee regions (femur condylar and tibia plateau)
    if not check_knee_region(kwargs['condylar_midway_x'], kwargs['condylar_leftpeak_x'], kwargs['condylar_rightpeak_x']):
        return False
    if not check_knee_region(kwargs['plateau_center_x'], kwargs['plateau_left_x'], kwargs['plateau_right_x']):
        return False
    # if not -11 < kwargs['hka'] < 11:
    #     return False
    return True

def check_knee_region(mid, left, right):
    left_distance = abs(mid - left)
    right_distance = abs(mid - right)
    minval, maxval = min(left_distance, right_distance), max(left_distance, right_distance)
    if minval / maxval < 0.5:
        return False
    return True