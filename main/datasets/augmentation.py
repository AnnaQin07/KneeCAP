import os
import cv2 
import torch
import random
import numpy as np 
import PIL.Image as Image
import torchvision.transforms.functional as TF

from skimage import io
from torchvision import transforms 
from matplotlib import pyplot as plt 

class To_Tensor(object):
    
    def __init__(self, std):
        self.std = std 
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, img, ld_mask, hd_mask, sdm_label):
        return self.to_tensor(img), self.to_tensor(np.array(ld_mask)), self.to_tensor(np.array(hd_mask)), torch.from_numpy(sdm_label / self.std).permute(2, 0, 1).to(torch.float32)
    
    
class Colorjitter(object):
    
    def __init__(self, brightness=0.3, contrast=0.3):
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
    
    def __call__(self, img, ld_mask, hd_mask, sdm_label):
        img = Image.fromarray(img, mode="L")
        return self.color_jitter(img), ld_mask, hd_mask, sdm_label
    

class Random_shift(object):
    
    def __init__(self, p, direction, max_sft):
        
        self.p = p
        self.direction = direction
        self.max_sft = max_sft
    
    def __call__(self, img, ld_mask, hd_mask, sdm_label):
        
        if random.uniform(0, 1) < self.p:
            h, w = img.shape
            shft_val = self.generate_param()
            # shft_val = -16

            img = self.shift_img(img, shft_val)
            hd_mask = self.shift_label(hd_mask, shft_val)
            ld_mask = Image.fromarray(hd_mask).resize((w // 2, h // 2), Image.Resampling.LANCZOS)
            ld_mask = np.asarray(ld_mask, dtype='uint8')
            ld_mask = reassign_label(ld_mask)
            sdm_label = self.generate_sdf(ld_mask)
            
    
        return img, ld_mask, hd_mask, sdm_label
    
    def generate_param(self):
        return -np.random.randint(1, self.max_sft+1, size=1)[0]
    
    def shift_img(self, img, shift_val):
        tmp = np.zeros(img.shape, dtype=img.dtype)
        if self.direction == 'horizontal':
            tmp[:, :shift_val] = img[:, -shift_val:].copy()
        else:
            tmp[:shift_val] = img[-shift_val:].copy()
        return tmp
    
    def shift_label(self, label, shift_val):
        tmp = np.zeros(label.shape, dtype=label.dtype)
        tmp[..., 2] = 255
        if self.direction == 'horizontal':


            tmp[:, :shift_val] = label[:, -shift_val:].copy()
        else:
            tmp[:shift_val] = label[-shift_val:].copy()
        return tmp
    
    def generate_sdf(self, mask):
        tibia, femur, _ = cv2.split(mask)
        usdf_femur = usdf_v1(femur)
        usdf_tibia = usdf_v1(tibia)
        sdm_femur = np.where(femur > 0, -1, 1) * usdf_femur
        sdm_tibia = np.where(tibia > 0, -1, 1) * usdf_tibia
        return np.concatenate([sdm_tibia[..., None], sdm_femur[..., None]], axis=-1)
    
class Compose(object):
    
    def __init__(self, aug_list):
        self.aug_list = aug_list 
    
    def __call__(self, img, ld_mask, hd_mask, sdm_label):
        for aug in self.aug_list:
            img, ld_mask, hd_mask, sdm_label = aug(img, ld_mask, hd_mask, sdm_label)
        return img, ld_mask, hd_mask, sdm_label
        
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


class Random_brightness(object):
    
    def __init__(self, max_k, p):
        self.k = max_k
        self.p = p
        
    def __call__(self, img, ld_mask, hd_mask, sdm_label):
        
        if random.uniform(0, 1) < self.p:
            factors = self.generate_factor(img)
            img = (img.copy() * factors.reshape(-1, 1)).astype('uint8')
        return img, ld_mask, hd_mask, sdm_label
    
    def generate_factor(self, img):
        self.img_height = img.shape[0]
        mode = random.choice([0, 1, 2, 3])
        if mode == 0:
            return self.scale_up()
        elif mode == 1:
            return self.scale_down()
        elif mode == 2:
            return self.up_down()
        else:
            return self.down_up()
    
    def scale_up(self):
        xs = np.linspace(0, 1, self.img_height)
        k = random.uniform(1, self.k + 1)
        b = random.uniform(0, 1)
        return k * xs + b
    
    def scale_down(self):
        xs = np.linspace(1, 0, self.img_height)
        k = random.uniform(1, self.k + 1)
        b = random.uniform(0, 1)

        return k * xs + b
    
    def up_down(self):
        xs = np.linspace(0, 1, self.img_height)
        basic_func = -4 * (xs - 0.5) ** 2 + 1 
        k = random.uniform(1, self.k + 1)
        b = random.uniform(0, 1)
        return k * basic_func + b
    
    def down_up(self):
        xs = np.linspace(0, 1, self.img_height)
        basic_func = 4 * (xs - 0.5) ** 2 
        k = random.uniform(1, self.k + 1)
        b = random.uniform(0, 1)
        return k * basic_func + b

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
    

if __name__ == "__main__":
    data_root = r"data/split/train"
    img = os.path.join(data_root, 'images', '00764704_l.png')
    ld_mask = os.path.join(data_root, 'ld_masks', '00764704_l.png')
    hd_mask = os.path.join(data_root, 'hd_masks', '00764704_l.png')
    sdm = os.path.join(data_root, 'sdm', '00764704_l.npy')
    
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    ld_mask = io.imread(ld_mask)
    hd_mask = io.imread(hd_mask)
    ld_sdm = np.load(sdm)[..., ::-1]
    
    random_shift = Random_shift(1, 'horizontal', 128)
    img, ld_mask, hd_mask, ld_sdm = random_shift(img, ld_mask, hd_mask, ld_sdm)
    
    plt.subplot(151)
    plt.imshow(img)
    plt.subplot(152)
    plt.imshow(hd_mask)
    plt.subplot(153)
    plt.imshow(ld_mask)
    plt.subplot(154)
    plt.imshow(ld_sdm[..., 0])
    plt.subplot(155)
    plt.imshow(ld_sdm[..., 1])
    plt.show()
    
    
            
            
            
        
        
        
        
