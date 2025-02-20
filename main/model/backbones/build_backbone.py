from .asymmetric_unet import Asymmetric_UNet




def build_backbone(args):
    
    if args.type == 'asymmetric_unet':
        return Asymmetric_UNet(in_channels=args.in_channels, in_features=args.in_features, out_features=args.out_features)
    else:
        raise TypeError("Unrecognised backbone")