import cv2
import numpy as np

from matplotlib import pyplot as plt
from .post_utils import bone_boxing, crop_bone, bilinear_interpolation, find_inclose_circle, shape_check


def locate_femur_head_v0(mask):
    """Locate the centre of femur head via usdf method 

    Args:
        mask (np.array([h, w])): the segmentation mask
    Returns:
        tuple(x, y), float: pixel coordinate of femur head, radius of estimated circle 
    """
    # split out the bone region
    # TODO: How to split out correctly
    bbx = bone_boxing(mask) # x1, y1, x2, y2
    roi = crop_bone(mask, bbx)
    bone_mid_idx = roi.shape[0] // 2
    upper_femur = roi[:bone_mid_idx]
    
    interg_x = np.sum(upper_femur, axis=1)
    if interg_x.size == 0:
        return (None, None), None
    max_y = np.argmax(interg_x) + int(0.03 * roi.shape[0])
    
    interg_y = np.sum(upper_femur, axis=0)
    upper_width = len(np.where(interg_y > 0)[0])
    if interg_y.size == 0:
        return (None, None), None
    min_x = np.argmax(interg_y)
    min_x = max(0, min_x - int(0.15 * upper_width))

    region_mask = roi[:max_y, min_x:].copy()
    region_mask[-1:] = 0
    region_mask[:, :2] = 0
    
    # locate the center point via usdf
    dist_map = cv2.distanceTransform(region_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    x_c, y_c = center
    """for debug
    plt.imshow(region_mask)
    plt.scatter(x_c, y_c)
    plt.show()
    """ 

    x_c = x_c + bbx[0] + min_x
    y_c = y_c + bbx[1]
    return (x_c, y_c), radius



def locate_femur_head_v1(mask):
    """Locate the centre of femur head via template matching

    Args:
        mask (np.array([h, w])): the segmentation mask
    Returns:
        tuple(x, y), float: pixel coordinate of femur head, radius of estimated circle 
    """
    # split out the bone region
    # TODO: How to split out correctly
    bbx = bone_boxing(mask) # x1, y1, x2, y2
    roi = crop_bone(mask, bbx)
    bone_mid_idx = roi.shape[0] // 2
    upper_femur = roi[:bone_mid_idx]
    interg_x = np.sum(upper_femur, axis=1)
    max_y = np.argmax(interg_x) + int(0.03 * roi.shape[0])
    interg_y = np.sum(upper_femur, axis=0)
    upper_width = len(np.where(interg_y > 0)[0])
    if interg_y.size == 0:
        return (None, None), None
    
    min_x = np.argmax(interg_y)
    region_mask = roi[:max_y, min_x-int(0.15*upper_width):].copy()
    region_mask[-1:] = 0
    region_mask[:, :2] = 0
    # plt.imshow(region_mask, cmap='gray')
    
    # template matching for center positioning 
    contours, hierarchy = cv2.findContours(region_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS) # cv2.CHAIN_APPROX_TC89_L1; cv2.CHAIN_APPROX_TC89_KCOS
    # (x_c, y_c), radius = cv2.minEnclosingCircle(contours[0])
    (y_c, x_c), radius = find_inclose_circle(contours[0], region_mask)
    """degug
    # center = (x_c, y_c)
    # plt.imshow(region_mask, cmap='gray')
    # circle = plt.Circle(center, radius, color='lightblue', fill=False)
    # plt.gca().add_patch(circle)
    # plt.show()
    """
    x_c = x_c + bbx[0] + min_x - int(0.15 * upper_width)
    y_c = y_c + bbx[1]
    radius=int(radius)

    return (x_c, y_c), radius


def locate_condylar_features(mask):
    """Location the condylar midway of medial and lateral condyles and condylar tangent

    Args:
        mask (H, W): seg mask of femur

    Returns:
        dict: {'condylar_midway': (float, float), center of condylar
               'condylar_left': (float, float), left point of condylar tangent
               'condylar_right': (float, float)}, right point of condylar tangent
    """
    ## TODO: improve it 
    # Crop RoI
    bbx = bone_boxing(mask) # x1, y1, x2, y2
    roi = crop_bone(mask, bbx)
    
    # Crop to distal femur region
    bone_mid_idx = roi.shape[0] // 2
    vertical_sum = np.sum(roi, axis=1)
    if vertical_sum[bone_mid_idx:].size == 0:
        return  {'condylar_midway': (None, None),
                'condylar_left': (None, None),
                'condylar_right': (None, None)}
    distal_upper_bound = np.argmax(vertical_sum[bone_mid_idx:]) + bone_mid_idx + bbx[1]
    distal_regions = mask[distal_upper_bound: bbx[3]+1, bbx[0]: bbx[2]+1]
    
    # template_mask = np.load("femur_condylar.npy")
    # IoU_score = matching_similarites(template_mask, distal_regions, method='iou')
    # np.save("femur_condylar.npy", distal_regions)
    # locating the midways
    horizontal_sum = np.sum(distal_regions, axis=0)
    
    lft_peak, rgt_peak = find_two_peaks(horizontal_sum)
    if lft_peak is None or rgt_peak is None:
        return {'condylar_midway': (None, None),
                'condylar_left': (None, None),
                'condylar_right': (None, None)}
    # minumum_id = (lft_peak + rgt_peak) / 2
    # plt.imshow(mask)
    # plt.scatter(lft_peak + bbx[0], horizontal_sum[lft_peak] + distal_upper_bound)
    # plt.scatter(rgt_peak + bbx[0], horizontal_sum[rgt_peak] + distal_upper_bound)
    # plt.show()
    # condylar_midway
    # minumum_id = np.argmin(horizontal_sum[lft_peak: rgt_peak]) + lft_peak
    minumum_id = np.mean(np.where(horizontal_sum[lft_peak: rgt_peak] == np.min(horizontal_sum[lft_peak: rgt_peak]))[0]) + lft_peak
    condylar_terminal = condylar_line(horizontal_sum, lft_peak, rgt_peak, int(minumum_id))
    condylar_left_x, condylar_left_y  = condylar_terminal['left_peak']
    condylar_right_x, condylar_right_y  = condylar_terminal['right_peak']
    """debug
    plt.plot(np.arange(horizontal_sum.shape[0]), horizontal_sum)
    plt.imshow(mask)
    plt.scatter(minumum_id + bbx[0], horizontal_sum[int(minumum_id)] + distal_upper_bound)
    plt.show()
    """
    if condylar_left_x is None and condylar_left_y is None:
        condylar_left = (None, None)
    else:
        condylar_left = (condylar_left_x + bbx[0], condylar_left_y + distal_upper_bound)
    
    if condylar_right_x is None and condylar_right_y is None:
        condylar_right = (None, None)
    else:
        condylar_right = (condylar_right_x + bbx[0], condylar_right_y + distal_upper_bound)
        
    # distal_upper_bound + bilinear_interpolation(horizontal_sum, minumum_id)
    return {'condylar_midway': (bbx[0] + minumum_id, distal_upper_bound + horizontal_sum[int(minumum_id)]),
            'condylar_left': condylar_left,
            'condylar_right': condylar_right} 
    

def find_two_peaks(horizontal_sum):
    """Location the peaks of femur condylar

    Args:
        horizontal_sum np.array([m]): the projected mask in horizontal

    Returns:
        tuple(float): ((x, y), (x, y)) coordinates of two splines peaks 
    """
    # mid_ways = horizontal_sum.shape[0] // 2
    pos_regions = np.where(horizontal_sum > 0)[0]
    # the femur condylar is invalid
    if pos_regions.shape[0] < 4:
        return None, None

    mid_ways = pos_regions[pos_regions.shape[0] // 2]
    # mid_ways = (pos_regions[0] + pos_regions[-1]) // 2
    # lft_pks, right_pks = np.max(horizontal_sum[:mid_ways]), np.max(horizontal_sum[mid_ways:])
    # arg_lfts = np.where(horizontal_sum == lft_pks)[0]
    # arg_rights = np.where(horizontal_sum == right_pks)[0]
    lft_pks = np.argmax(horizontal_sum[:mid_ways])
    right_pks = np.argmax(horizontal_sum[mid_ways:]) + mid_ways
    """for debug
    plt.plot(np.arange(horizontal_sum.shape[0]), horizontal_sum)
    plt.scatter(lft_pks, horizontal_sum[lft_pks])
    plt.scatter(right_pks, horizontal_sum[right_pks])
    plt.show()
    """
    return lft_pks, right_pks


def condylar_line(horizontal_sum, left_peak, right_peak, minumum_id, length_thresh=2):
    """Location the peaks of two splines

    Args:
        horizontal_sum np.array([m]): the projected mask in horizontal

    Returns:
        tuple(float): ((x, y), (x, y)) coordinates of two splines peaks 
    """
    condylar_region = np.where(horizontal_sum)[0]
    if condylar_region.shape[0] < 2:
        return {'left_peak': (None, None),
                'right_peak': (None, None)}
    left_bound, right_bound = condylar_region[0] + 4, condylar_region[-1] - 4
    
    # get the left peak
    max_v = horizontal_sum[left_peak] # init guess
    left_part = horizontal_sum[left_bound: minumum_id]
    ids = np.where(left_part == max_v)[0] + left_bound
    length_left = ids.shape[0]
    num_attempt = 10
    while length_left < length_thresh and num_attempt > 0:
        max_v -= 1
        ids = np.where(left_part == max_v)[0] + left_bound
        length_left = ids.shape[0]
        num_attempt -= 1
    if length_left < length_thresh and num_attempt == 0:
        max_v += 10
    ids = np.where(left_part == max_v)[0] + left_bound
    if ids.shape[0] == 0:
        left_peak = (None, None)
    else:
        left_pk_x = ids[ids.shape[0] // 2]
        left_terminal_y = bilinear_interpolation(horizontal_sum, left_pk_x)
        left_peak = (left_pk_x, left_terminal_y)
    
    # get the right peak as above
    max_v = horizontal_sum[right_peak] # init guess
    right_part = horizontal_sum[minumum_id: right_bound]
    ids = np.where(right_part == max_v)[0] + minumum_id
    length_right = ids.shape[0]
    num_attempt = 10
    while length_right < length_thresh and num_attempt > 0:
        max_v -= 1
        ids = np.where(right_part == max_v)[0] + minumum_id
        length_right = ids.shape[0]
        num_attempt -= 1
    if length_right < length_thresh and num_attempt == 0:
        max_v += 10
        
    ids = np.where(right_part == max_v)[0] + minumum_id
    if ids.shape[0] == 0:
        right_peak = (None, None)
    else:
        right_pk_x = ids[ids.shape[0] // 2]
        right_terminal_y = bilinear_interpolation(horizontal_sum, right_pk_x)
        right_peak = (right_pk_x, right_terminal_y)
    return {'left_peak': left_peak,
            'right_peak': right_peak}
          

# tibia landmarks
def locate_tibia_plateau_features(roi, interg_x, bbx, length=4):
    """Location the tibia spines centre and tibia tangents

    Args:
        roi (H, W): seg mask of tibia plateau
        interg_x float: the tibia curve in vertical direction
        bbx (x1, y1, x2, y2): The bounding box of tibia
        length: predefined mini length of tibia spines
    Returns:
        dict: {'condylar_midway': (float, float), centre of plateau spines
               'condylar_left': (float, float), left point of plateau tangent
               'condylar_right': (float, float)} right point of plateau tangent
    """
    ## TODO: improve it 
    # locate the tibia_plateau regions
    mid_tibia = interg_x.shape[0] // 2 
    if mid_tibia == 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    tibial_plateau_lower_bound = np.argmax(interg_x[:mid_tibia]) 
    
    # np.save('tibia_plateau.npy', roi[:tibial_plateau_lower_bound, :])
    # template_mask = np.load("tibia_plateau.npy")
    # IoU_score = matching_similarites(template_mask, roi[:tibial_plateau_lower_bound, :], method='other')
    plateau_curve = np.sum(roi[:tibial_plateau_lower_bound, :], axis=0)
    
    # find the plateau center
    peak_ids = np.where(plateau_curve == np.max(plateau_curve))[0]
    peak_idx = int(np.mean(peak_ids))
    if peak_idx-length <= 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    # peak_idx = np.argmax(plateau_curve)
    lft_mx_idx = np.argmax(plateau_curve[:peak_idx-length])
    
    # cannot find right peak, which idicates the segmentation mask is pretty bad, we don't output results in this case
    if plateau_curve[peak_idx+length:].shape[0] == 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    
    rght_mx_idx = np.where(plateau_curve[peak_idx+length:] == np.max(plateau_curve[peak_idx+length:]))[0][-1] + peak_idx + length
    # rght_mx_idx = np.argmax(plateau_curve[peak_idx+length:]) + peak_idx + length
    other_peak_idx = np.where(plateau_curve[lft_mx_idx] > plateau_curve[rght_mx_idx], lft_mx_idx, rght_mx_idx)
    lft_idx = np.where(peak_idx < other_peak_idx, peak_idx, other_peak_idx)
    right_idx = np.where(peak_idx >= other_peak_idx, peak_idx, other_peak_idx)
    valley_val = np.min(plateau_curve[lft_idx: right_idx+1])
    valley_ids = np.where(plateau_curve[lft_idx: right_idx+1] == valley_val)[0] + lft_idx
    x = np.mean(valley_ids)    
    if valley_val >= plateau_curve[lft_idx] or valley_val >= plateau_curve[right_idx]:
        x = (lft_mx_idx + right_idx) / 2
        valley_val = bilinear_interpolation(plateau_curve, x)
        if valley_val > plateau_curve[peak_idx]:
            x = peak_idx
            valley_val = plateau_curve[peak_idx]
    rate = (x - lft_idx)/ (right_idx - lft_idx)
    if x != peak_idx and (0< rate < 1/4 or rate > 3/4):
        x = (right_idx + lft_idx) / 2
        valley_val = bilinear_interpolation(plateau_curve, x)
        
    # find the plateau line for condylar-plateau angle:
    terminals = tibia_plateau_line(plateau_curve, int(x))   
    left_x, left_y =  terminals['left_terminal']
    right_x, right_y =  terminals['right_terminal']
    """for debug
    plt.plot(np.arange(plateau_curve.shape[0]), plateau_curve)
    plt.scatter(lft_idx, plateau_curve[lft_idx])
    plt.scatter(right_idx, plateau_curve[right_idx])
    plt.scatter(x, valley_val)
    plt.scatter(left_x, left_y)
    plt.scatter(right_x, right_y)
    plt.show()
    """
    y = tibial_plateau_lower_bound - valley_val
     
    if left_y is not None:
        left_y = tibial_plateau_lower_bound - left_y
    if right_y is not None:
        right_y = tibial_plateau_lower_bound - right_y
        
    if left_x is None and left_y is None:
        left_terminal = (None, None)
    else:
        left_terminal = (left_x + bbx[0], left_y + bbx[1])
        
    if right_x is None and right_y is None:
        right_terminal = (None, None)
    else:
        right_terminal = (right_x + bbx[0], right_y + bbx[1])
    
    return {'center': (x + bbx[0], y + bbx[1]),
            'left_terminal': left_terminal,
            'right_terminal': right_terminal}



def tibia_plateau_line(plateau_curve, midway_x, length_thresh=2):
    """Location of tibia tangents

    Args:
        plateau_curve np.array([n]): plateau curve in horizontal direction
        midway_x: init center of tibia spines
        length: predefined mini length of tibia spines
    Returns:
        dict: {'condylar_left': (float, float), left point of plateau tangent
               'condylar_right': (float, float)} right point of plateau tangent}
    """
    # left
    plateau_regions = np.where(plateau_curve > 0)[0]
    left_bound, right_bound = plateau_regions[0] + 4, plateau_regions[-1] - 4
    
    left_part = plateau_curve[left_bound:midway_x]
    if left_part.shape[0] == 0:
        return {'left_terminal': (None, None), 
            'right_terminal': (None, None)}
    min_left = np.min(left_part)
    ids = np.where(left_part == min_left)[0] + left_bound
    length = ids.shape[0]
    num_attempt = 10
    while length < length_thresh and num_attempt > 0:
        min_left += 1
        ids = np.where(left_part == min_left)[0]  + left_bound
        length = ids.shape[0]
        num_attempt -= 1
    if length < length_thresh and num_attempt == 0:
        min_left -= 10
        ids = np.where(left_part == min_left)[0]  + left_bound
    if ids.shape[0] == 0:
        left_terminal = (None, None)
    else:
        left_terminal_x = ids[ids.shape[0] // 2]
        left_terminal_y = bilinear_interpolation(plateau_curve, left_terminal_x)
        left_terminal = (left_terminal_x, left_terminal_y)
    
    # right
    right_part = plateau_curve[midway_x:right_bound]
    if right_part.shape[0]  == 0:
        return {'left_terminal': (None, None), 
                'right_terminal': (None, None)}
    min_right = np.min(right_part)
    ids = np.where(right_part == min_right)[0] + midway_x
    length = ids.shape[0]
    num_attempt = 10
    while length < length_thresh and num_attempt > 0:
        min_right += 1
        ids = np.where(right_part == min_right)[0] + midway_x
        length = ids.shape[0]
        num_attempt -= 1
    if length < length_thresh and num_attempt == 0:
        min_right -= 10
        ids = np.where(right_part == min_right)[0] + midway_x
    
    if ids.shape[0] == 0:
        right_terminal = (None, None)
    else:
        right_terminal_x = ids[ids.shape[0] // 2]
        right_terminal_y = bilinear_interpolation(plateau_curve, right_terminal_x)
        right_terminal = (right_terminal_x, right_terminal_y)
    return {'left_terminal': left_terminal, 
            'right_terminal': right_terminal}
        
    

def tibia_plafond_distally(roi, x_interg, bbx, neighbor=12):
    """Location of tibia plafond centre
    Args:
        roi [H, W]: plafond distally region
        x_interg np.array([n]): tibia curve
        bbx (x1, y1, x2, y2): bounding box of tibia
        neighbor: predefined length of tibia malleolus
    Returns:
        (x, y) plafond centre coordinates
    """
    mid_tibia = x_interg.shape[0] // 2
    if mid_tibia == 0:
        return None, None
    # get distal plafond region
    plafond_distally_upper_bound = np.argmax(x_interg[mid_tibia:]) + mid_tibia
    plafond_distally_region = roi[plafond_distally_upper_bound:]
    
    # np.save('tibia_plafond.npy', plafond_distally_region)
    # template_mask = np.load("tibia_plafond.npy")
    # IoU_score = matching_similarites(template_mask, plafond_distally_region, method='other')

    # devriate the platfond
    y_interg = np.sum(plafond_distally_region, axis=0)
    diff = np.diff(y_interg)

    plafond_ids = np.where(y_interg > 0)[0]
    mid_id = plafond_ids[plafond_ids.shape[0] // 2 - 1]
    if mid_id == 0:
        return None, None
    left_max = np.argmax(y_interg[:mid_id])
    right_max = mid_id + np.argmax(y_interg[mid_id:])    
    # find the left 
    right_foothills = right_max - neighbor
    valid_boundarys = np.where(diff[right_foothills: right_max - (neighbor // 6)] < 0.3)[0]
    if valid_boundarys.shape[0] == 0:
        return None, None
    plafond_right_boundary = valid_boundarys[-1] + right_foothills + 1
    if y_interg[left_max] / y_interg[mid_id] > 1 and y_interg[right_max] / y_interg[mid_id] > 1: # two peaks on both side
        left_foothills = left_max + neighbor
        if not (diff[left_max + (neighbor // 4): left_foothills] > -0.3).any():
            return None, None 
        plafond_left_boundary = np.where(diff[left_max + (neighbor // 4): left_foothills] > -0.3)[0][0] + left_max + (neighbor // 4) + 1
        mid_x = (plafond_left_boundary + plafond_right_boundary) / 2
        """for debug
        # plt.plot(np.arange(diff.shape[0]), diff)
        # plt.plot(np.arange(y_interg.shape[0]), y_interg)
        # plt.scatter(plafond_right_boundary, y_interg[plafond_right_boundary])
        # plt.scatter(plafond_left_boundary, y_interg[plafond_left_boundary])
        # plt.scatter(left_max, y_interg[left_max])
        # plt.show()
        """

    else: # one peak only
        # region_split = y_interg[:plafond_right_boundary].shape[0] // 2
        # lft_maxs = np.where(y_interg[:region_split+1] == np.max(y_interg[:region_split+1]))[0]
        # mid_x  = np.mean(
        #         np.where(y_interg[lft_maxs[-1]: plafond_right_boundary] == np.min(y_interg[lft_maxs[-1]: plafond_right_boundary]))[0]
        #         ) + lft_maxs[-1]
        # if y_interg[lft_maxs[-1]] - y_interg[int(mid_x)]  < 1:
        #     plafond_left_boundary = np.where(y_interg > 0)[0][0]
        #     mid_x = (plafond_left_boundary + plafond_right_boundary) / 2    
        # plafond_left_boundary = np.where(y_interg > 0)[0][0]
        """for debug
        # plt.plot(np.arange(y_interg.shape[0]), y_interg)
        # plt.scatter(plafond_right_boundary, y_interg[plafond_right_boundary])
        # plt.scatter(plafond_left_boundary, y_interg[plafond_left_boundary])
        # plt.scatter(mid_id, y_interg[mid_id])
        # plt.show()
        """
        mid_x = mid_id - neighbor // 6
    mid_y = bilinear_interpolation(y_interg, mid_x)

    return mid_x + bbx[0], mid_y + plafond_distally_upper_bound + bbx[1]

def locate_condylar_features_v1(mask):
    """Location the condylar midway of medial and lateral condyles and condylar tangent
    difference between it and previous [locate_condylar_features()] is it has rotation
       ##NOTICE :not finished yet

    Args:
        mask (H, W): seg mask of femur

    Returns:
        dict: {'condylar_midway': (float, float), center of condylar
               'condylar_left': (float, float), left point of condylar tangent
               'condylar_right': (float, float)}, right point of condylar tangent
    """
    ## TODO: GET it done
    # Crop RoI
    bbx = bone_boxing(mask) # x1, y1, x2, y2
    roi = crop_bone(mask, bbx)
    
    # Crop to distal femur region
    bone_mid_idx = roi.shape[0] // 2
    vertical_sum = np.sum(roi, axis=1)
    if vertical_sum[bone_mid_idx:].size == 0:
        return  {'condylar_midway': (None, None),
                'condylar_left': (None, None),
                'condylar_right': (None, None)}
    distal_upper_bound = np.argmax(vertical_sum[bone_mid_idx:]) + bone_mid_idx + bbx[1]
    distal_regions = mask[distal_upper_bound: bbx[3]+1, bbx[0]: bbx[2]+1]
    distal_edges = cv2.Canny((distal_regions*255).astype('uint8'), 0, 200)
    ys, xs = np.where(distal_edges > 0)
    p = np.polyfit(xs, ys, deg=1)
    theta = -np.arctan(p[0])
    Affine = cv2.getRotationMatrix2D([0, roi.shape[0]], theta, 1)
    Affine = np.append(Affine, [[0, 0, 1]], axis=0)
    inv_affine = np.linalg.inv(Affine)
    inv_affine = inv_affine / inv_affine[-1, -1]
    rotate_roi = cv2.warpAffine(roi ,inv_affine[:2], roi.shape[::-1], flags=cv2.WARP_INVERSE_MAP)
    
    distal_regions = np.copy(rotate_roi[distal_upper_bound:, :])
    
    horizontal_sum = np.sum(distal_regions, axis=0)
    
    
    lft_peak, rgt_peak = find_two_peaks(horizontal_sum)
    if lft_peak is None or rgt_peak is None:
        return {'condylar_midway': (None, None),
                'condylar_left': (None, None),
                'condylar_right': (None, None)}
    # minumum_id = (lft_peak + rgt_peak) / 2
    # plt.imshow(mask)
    # plt.scatter(lft_peak + bbx[0], horizontal_sum[lft_peak] + distal_upper_bound)
    # plt.scatter(rgt_peak + bbx[0], horizontal_sum[rgt_peak] + distal_upper_bound)
    # plt.show()
    # condylar_midway
    # minumum_id = np.argmin(horizontal_sum[lft_peak: rgt_peak]) + lft_peak
    minumum_id = np.mean(np.where(horizontal_sum[lft_peak: rgt_peak] == np.min(horizontal_sum[lft_peak: rgt_peak]))[0]) + lft_peak
    condylar_terminal = condylar_line(horizontal_sum, lft_peak, rgt_peak, int(minumum_id))
    condylar_left_x, condylar_left_y  = condylar_terminal['left_peak']
    condylar_right_x, condylar_right_y  = condylar_terminal['right_peak']
    """debug
    plt.plot(np.arange(horizontal_sum.shape[0]), horizontal_sum)
    plt.imshow(mask)
    plt.scatter(minumum_id + bbx[0], horizontal_sum[int(minumum_id)] + distal_upper_bound)
    plt.show()
    """
    if condylar_left_x is None and condylar_left_y is None:
        condylar_left = (None, None)
    else:
        condylar_left = (condylar_left_x + bbx[0], condylar_left_y + distal_upper_bound)
    
    if condylar_right_x is None and condylar_right_y is None:
        condylar_right = (None, None)
    else:
        condylar_right = (condylar_right_x + bbx[0], condylar_right_y + distal_upper_bound)
        
    # distal_upper_bound + bilinear_interpolation(horizontal_sum, minumum_id)
    return {'condylar_midway': (bbx[0] + minumum_id, distal_upper_bound + horizontal_sum[int(minumum_id)]),
            'condylar_left': condylar_left,
            'condylar_right': condylar_right} 

def locate_plateau_features_v1(roi, interg_x, bbx, length=4):
    """Location the tibia spines centre and tibia tangents
       difference between it and previous [locate_plateau_features()] is it has rotation
       ##NOTICE :abandoned since the rotation is not useful
    Args:
        roi (H, W): seg mask of tibia plateau
        interg_x float: the tibia curve in vertical direction
        bbx (x1, y1, x2, y2): The bounding box of tibia
        length: predefined mini length of tibia spines
    Returns:
        dict: {'condylar_midway': (float, float), centre of plateau spines
               'condylar_left': (float, float), left point of plateau tangent
               'condylar_right': (float, float)} right point of plateau tangent
    """
    mid_tibia = interg_x.shape[0] // 2 
    if mid_tibia == 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    tibial_plateau_lower_bound = np.argmax(interg_x[:mid_tibia]) 
    # rotate 
    plateau_edges = cv2.Canny((roi[:tibial_plateau_lower_bound]*255).astype('uint8'), 0, 200)
    ys, xs = np.where(plateau_edges > 0)
    p = np.polyfit(xs, ys, deg=1)
    theta = -np.arctan(p[0])
    Affine = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0]]).astype('float32')
    rotate_roi = cv2.warpAffine(roi ,Affine, roi.shape[::-1], flags=cv2.WARP_INVERSE_MAP)
    tibial_plateau = np.copy(rotate_roi[:tibial_plateau_lower_bound, :])
    plateau_curve = np.sum(tibial_plateau, axis=0)
    
    # find the plateau center
    peak_ids = np.where(plateau_curve == np.max(plateau_curve))[0]
    breaking_id = np.where(np.diff(peak_ids) > 1)[0]
    if breaking_id.size > 0:
        peak_ids = peak_ids[:breaking_id[0]+1]
    peak_idx = int(np.mean(peak_ids))
    if peak_idx-length <= 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    # peak_idx = np.argmax(plateau_curve)
    lft_mx_idx = np.argmax(plateau_curve[:peak_idx-length])
    
    # cannot find right peak, which idicates the segmentation mask is pretty bad, we don't output results in this case
    # TODO: in some cases (eg, bone with implant), the plateau only has one even no spines, solution needed
    if plateau_curve[peak_idx+length:].shape[0] == 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    
    rght_mx_idx = np.where(plateau_curve[peak_idx+length:] == np.max(plateau_curve[peak_idx+length:]))[0][-1] + peak_idx + length
    # rght_mx_idx = np.argmax(plateau_curve[peak_idx+length:]) + peak_idx + length
    other_peak_idx = np.where(plateau_curve[lft_mx_idx] > plateau_curve[rght_mx_idx], lft_mx_idx, rght_mx_idx)
    lft_idx = np.where(peak_idx < other_peak_idx, peak_idx, other_peak_idx)
    right_idx = np.where(peak_idx >= other_peak_idx, peak_idx, other_peak_idx)
    valley_val = np.min(plateau_curve[lft_idx: right_idx+1])
    valley_ids = np.where(plateau_curve[lft_idx: right_idx+1] == valley_val)[0] + lft_idx
    x = np.mean(valley_ids) 

    if valley_val >= plateau_curve[lft_idx] or valley_val >= plateau_curve[right_idx]:
        x = (lft_mx_idx + right_idx) / 2
        valley_val = bilinear_interpolation(plateau_curve, x)
        if valley_val > plateau_curve[peak_idx]:
            x = peak_idx
            valley_val = plateau_curve[peak_idx]
    rate = (x - lft_idx)/ (right_idx - lft_idx)
    if x != peak_idx and (0< rate < 1/4 or rate > 3/4):
        x = (right_idx + lft_idx) / 2
        valley_val = bilinear_interpolation(plateau_curve, x)
        
    # find the plateau line for condylar-plateau angle:
    terminals = tibia_plateau_line(plateau_curve, int(x))   
    left_x, left_y =  terminals['left_terminal']
    right_x, right_y =  terminals['right_terminal']
    """for debug
    plt.plot(np.arange(plateau_curve.shape[0]), plateau_curve)
    plt.scatter(lft_idx, plateau_curve[lft_idx])
    plt.scatter(right_idx, plateau_curve[right_idx])
    plt.scatter(x, valley_val)
    plt.scatter(left_x, left_y)
    plt.scatter(right_x, right_y)
    plt.show()
    """
    y = tibial_plateau_lower_bound - valley_val
    x, y = Affine[:, :2] @ np.array([[x], [y]])
    
     
    if left_y is not None:
        left_y = tibial_plateau_lower_bound - left_y
    if right_y is not None:
        right_y = tibial_plateau_lower_bound - right_y
        
    if left_x is None and left_y is None:
        left_terminal = (None, None)
    else:
        left_x, left_y = Affine[:, :2] @ np.array([[left_x], [left_y]])
        left_terminal = (left_x[0] + bbx[0], left_y[0] + bbx[1])
        
    if right_x is None and right_y is None:
        right_terminal = (None, None)
    else:
        right_x, right_y = Affine[:, :2] @ np.array([[right_x], [right_y]])
        right_terminal = (right_x[0] + bbx[0], right_y[0] + bbx[1])
    
    return {'center': (x[0] + bbx[0], y[0] + bbx[1]),
        'left_terminal': left_terminal,
        'right_terminal': right_terminal}
    
      







def locate_tibia_implant_plateau_features(roi, interg_x, bbx):
    """Location the tibia(with implant) spines centre and tibia tangents

    Args:
        roi (H, W): seg mask of tibia plateau
        interg_x float: the tibia curve in vertical direction
        bbx (x1, y1, x2, y2): The bounding box of tibia
    Returns:
        dict: {'condylar_midway': (float, float), centre of plateau spines
               'condylar_left': (float, float), left point of plateau tangent
               'condylar_right': (float, float)} right point of plateau tangent
    """
    
    # locate the tibia_plateau regions
    mid_tibia = interg_x.shape[0] // 2 
    if mid_tibia == 0:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    implant_bottom_y=len([num for num in interg_x if num > 0])//30
    implant_region=roi[:implant_bottom_y]
    contours, _ = cv2.findContours(implant_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []

    for contour in contours:
        if len(contour) >= 5:  # At least 5 points are needed to fit the ellipse
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    if ellipses:
        # Suppose we only consider the first ellipse
        ellipse = ellipses[0]
        condylar_midway = (ellipse[0][0], ellipse[0][1])  # Center point of ellipse
        axes = (ellipse[1][0] / 2, ellipse[1][1] / 2)  # The length of the half-axis of the ellipse
        angle = np.deg2rad(ellipse[2]-90)  # Rotation Angle (converted to radians)
    
        # calculate the focus
        a = max(axes)  # semi-major axis
        b = min(axes)  # semi-minor axis
        c = np.sqrt(a**2 - b**2)  # focal length

        # focus coordinates
        focal_left = (condylar_midway[0] - c * np.cos(angle), condylar_midway[1] - c * np.sin(angle))  # left focus
        focal_right = (condylar_midway[0] + c * np.cos(angle), condylar_midway[1] + c * np.sin(angle))  # right focus
        print(int(condylar_midway[0]), int(condylar_midway[1]))
        #cv2.ellipse(implant_region, ellipse, (255, 0, 0), 2)  # draw an ellipse
        cv2.circle(implant_region, (int(condylar_midway[0]), int(condylar_midway[1])), 5, (0, 255, 0), -1) 
        cv2.circle(implant_region, (int(focal_left[0]), int(focal_left[1])), 5, (0, 255, 0), -1)  # draw left focus
        cv2.circle(implant_region, (int(focal_right[0]), int(focal_right[1])), 5, (0, 255, 0), -1)  # draw right focus

        #  show results 
        #plt.imshow(implant_region)
        #plt.axis('off')
        #plt.title('Fitted Ellipse and Foci')
        #plt.show()
        
        center_terminal=(condylar_midway[0]+ bbx[0],condylar_midway[1]+ bbx[1])
        left_terminal = (focal_left[0] + bbx[0], focal_left[1] + bbx[1])
        right_terminal = (focal_right[0] + bbx[0], focal_right[1] + bbx[1])
        
        return {'center': center_terminal,
                'left_terminal': left_terminal,
                'right_terminal': right_terminal}
    
    else:
        return {'center': (None, None),
                'left_terminal': (None, None),
                'right_terminal': (None, None)}
    
