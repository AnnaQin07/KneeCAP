from .dcm_dataset import DCM_Dataset
from .png_dataset import PNG_Dataset


def Build_dataset(args):
    
    if args.type == 'dcm':
        return DCM_Dataset(args.img_dir)
    elif args.type == 'png':
        return PNG_Dataset(args.img_dir, 
                           args.coordi_save_dir, 
                           args.ld_mask_dir, 
                           args.hd_mask_dir, 
                           args.ld_sdm_dir, 
                           args.mode)
    else:
        raise TypeError("Unrecognized dataset type which out of {dcm, png}")