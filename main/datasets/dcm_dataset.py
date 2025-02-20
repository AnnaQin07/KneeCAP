import os
import pydicom

import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from skimage import  img_as_ubyte


from data.split_images import preprocessing

class DCM_Dataset(Dataset):
    
    def __init__(self, img_dir):
        """The dataset for .jpg/.png format images, they are usually the one after pre-processing

        Args:
            img_dir (str): the directory in which images saved 
            coordi_save_dir (str): the directory in which transformation info saved
            ld_mask_dir (str): the directory in which low resolution masks saved 
            hd_mask_dir (str): the directory in which high resolution masks saved
            ld_sdm_dir (str):  the directory in which high resolution SDF map saved
            mode (str, optional): mode, in [train, val, inf]. Defaults to 'train'.
        """
        super(DCM_Dataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_names = os.listdir(self.img_dir)
        self.img_names = [fname for fname in self.file_names if fname.endswith('.dcm')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # load examples
        img_dir = os.path.join(self.img_dir, self.img_names[idx])
        img = pydicom.read_file(img_dir)
        img = img.pixel_array
        right_part, ratio_h, ratio_w, tp_pd, lft_pad, bttm_pad, rgt_pad = preprocessing(img, side="right", rgb=False, enhance=True, size=(256, 2048))
        meta_right = {'ratio_height': ratio_h, 'ratio_width': ratio_w, 
                     "top_padding": tp_pd, "left_padding": lft_pad, 
                     "bottom_padding": bttm_pad, 'right_padding': rgt_pad}
        left_part, ratio_h, ratio_w, tp_pd, lft_pad, bttm_pad, rgt_pad = preprocessing(img, side="left", rgb=False, enhance=True, size=(256, 2048))
        meta_left = {'ratio_height': ratio_h, 'ratio_width': ratio_w, 
                     "top_padding": tp_pd, "left_padding": lft_pad, 
                     "bottom_padding": bttm_pad, 'right_padding': rgt_pad}

        # right_part = img_as_ubyte(right_part) # uint16 -> uint8
        # right_part = self.transform(right_part)
        # left_part = img_as_ubyte(left_part) # uint16 -> uint8
        # left_part = self.transform(left_part)
        right_part = Image.fromarray(self.Uint16toUint8(right_part), mode="L")
        right_part = self.transform(right_part)
        left_part = Image.fromarray(self.Uint16toUint8(left_part), mode="L") # uint16 -> uint8
        left_part = self.transform(left_part)
        
        return {'imgs': {'right': right_part, 'left': left_part},
                'meta_infos': {'right': meta_right, 'left': meta_left}}
        
    def Uint16toUint8(self, img):
        """Convert the image dtype from uint16 to uint8
        Reproduce the result of cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        Args:
            img (np.array([h, w])): the uint16 image

        Returns:
            np.array([h, w]: the uint8 image
        """
        img = img / 65535.0 * 255
        return np.uint8(255 * (img - img.min()) / (img.max() - img.min()) + 0.5)

        

    
        
if __name__ == "__main__":
    
    dcm_dir = r"D:/Datasets/comp8603/Lower_limb_Xrays/12m_dicoms/labeled/images"
    dataset = DCM_Dataset(dcm_dir)
    x = dataset[0]
    right_img = x['imgs']['right'].cpu().numpy()
    plt.imshow(right_img[0])
    plt.show()
