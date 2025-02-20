'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2023-07-28 19:10:12
LastEditors: YunxiangLiu u7191378@anu.edu.au
LastEditTime: 2023-11-24 15:23:55
FilePath: \MA_detections\datasets\png_dataset.py
Description: dataset
'''
import os
import cv2
import torch
import numpy as np

import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from skimage import io, img_as_ubyte

from .augmentation import To_Tensor, Colorjitter, Random_shift, Compose, Random_brightness


class PNG_Dataset(Dataset):
    
    def __init__(self, img_dir, coordi_save_dir, ld_mask_dir, hd_mask_dir, ld_sdm_dir, mode='train'):
        """The dataset for .jpg/.png format images, they are usually the one after pre-processing

        Args:
            img_dir (str): the directory in which images saved 
            coordi_save_dir (str): the directory in which transformation info saved
            ld_mask_dir (str): the directory in which low resolution masks saved 
            hd_mask_dir (str): the directory in which high resolution masks saved
            ld_sdm_dir (str):  the directory in which high resolution SDF map saved
            mode (str, optional): mode, in [train, val, test, inf]. Defaults to 'train'.
        """
        super(PNG_Dataset, self).__init__()


        # image directory as long as the directory of label
        # ld -> low resolution; hd -> high resolution
        self.img_dir = img_dir
        self.coordi_save_dir = coordi_save_dir
        self.ld_mask_dir = ld_mask_dir
        self.hd_mask_dir = hd_mask_dir
        self.ld_sdm_dir = ld_sdm_dir
        self.mode = mode
        if self.ld_mask_dir is None or self.hd_mask_dir is None or self.ld_sdm_dir is None:
            self.mode = 'inf'
            
        # normalise the value of sdf by default
        self.std = np.array([659.5150, 507.6584])

        self.train_transform = Compose([Random_brightness(0.6, 0.1), Random_shift(0.1, 'horizontal', 16), Colorjitter(0.3, 0.3), To_Tensor(self.std)])
            # self.transform = Compose([Colorjitter(0.3, 0.3), To_Tensor(self.std)])

        self.val_transform = Compose([Colorjitter(0.3, 0.3), To_Tensor(self.std)])

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_names = os.listdir(self.img_dir)
        self.img_names = [fname for fname in self.file_names if fname.endswith('.jpg') or fname.endswith('.png')]
        self.mask_names = list(map(lambda x: f'{os.path.splitext(x)[0]}.png', self.img_names))
        self.sdfmp_names = list(map(lambda x: f'{os.path.splitext(x)[0]}.npy', self.img_names))


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # load examples
        img_dir = os.path.join(self.img_dir, self.img_names[idx])
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        # img = io.imread(img_dir)
        # img = img_as_ubyte(img)
        if self.mode == 'inf':
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
            return {'img': img, 'img_name': self.img_names[idx]}
        
        ld_mask_dir = os.path.join(self.ld_mask_dir, self.mask_names[idx])
        hd_mask_dir = os.path.join(self.hd_mask_dir, self.mask_names[idx])
        ld_sdm_dir = os.path.join(self.ld_sdm_dir, self.sdfmp_names[idx])

        ld_mask = io.imread(ld_mask_dir)
        hd_mask = io.imread(hd_mask_dir)
        # ld_mask = Image.open(ld_mask_dir)
        # hd_mask = Image.open(hd_mask_dir)
        ld_sdm = np.load(ld_sdm_dir)[..., ::-1]

        if self.mode == "train":
            img, ld_mask, hd_mask, ld_sdfmp = self.train_transform(img, ld_mask, hd_mask, ld_sdm)
            # img = transforms.ColorJitter(brightness=0.3, contrast=0.3)(img)
        elif self.mode == 'val':
            img, ld_mask, hd_mask, ld_sdfmp = self.val_transform(img, ld_mask, hd_mask, ld_sdm)
        else:
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
            ld_mask = self.transform(ld_mask)
            hd_mask = self.transform(hd_mask)
            ld_sdfmp = torch.from_numpy(ld_sdm / self.std).permute(2, 0, 1).to(torch.float32)

        return {'img': img, 'ld_mask': ld_mask, 'hd_mask': hd_mask, 'ld_sdm': ld_sdfmp, 'img_name': self.img_names[idx]}
    
    
if __name__ == "__main__":
    
    img_dir = r"D:\Datasets\comp8603\Lower_limb_Xrays\12m_dicoms\png_large\train\images"
    coordi_save_dir = r"D:\Datasets\comp8603\Lower_limb_Xrays\12m_dicoms\coord_change_large"
    ld_mask_dir = r"D:\Datasets\comp8603\Lower_limb_Xrays\12m_dicoms\png\train\masks"
    hd_mask_dir = r"D:\Datasets\comp8603\Lower_limb_Xrays\12m_dicoms\png_large\train\masks"
    ld_sdm_dir = r"D:\Datasets\comp8603\Lower_limb_Xrays\12m_dicoms\png\train\sdm"
    dataset = PNG_Dataset(img_dir, coordi_save_dir, ld_mask_dir, hd_mask_dir, ld_sdm_dir)
    x = dataset[0]