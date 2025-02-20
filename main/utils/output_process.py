import torch
import numpy as np


def pred2mask(pred):
    '''
    :param pred: prediction from model with size [batch=1, channel=3, h, w]
    :return: an rgb masks
    '''
    pred = torch.argmax(pred, 1).squeeze(0)

    h, w = pred.shape

    pred_red = torch.zeros((h, w))
    pred_green = torch.zeros((h, w))
    pred_blue = torch.zeros((h, w))

    pred_red[pred == 0] = 1.0
    pred_green[pred == 1] = 1.0
    pred_blue[pred == 2] = 1.0

    pred_red = pred_red.unsqueeze(2)
    pred_green = pred_green.unsqueeze(2)
    pred_blue = pred_blue.unsqueeze(2)

    return torch.cat((pred_red, pred_green, pred_blue), 2)


def batchpred2mask(preds):
    
    b, c, h, w = preds.shape
    preds = torch.argmax(preds, 1) # (B, H, W)
    pred_red = torch.zeros((b, h, w))
    pred_green = torch.zeros((b, h, w))
    pred_blue = torch.zeros((b, h, w))
    
    pred_red[preds == 0] = 255
    pred_green[preds == 1] = 255
    pred_blue[preds == 2] = 255
    
    return torch.stack((pred_red, pred_green, pred_blue), -1)
    
    

def pred2nobackground_mask(pred):
    '''
    :param pred: prediction from model with size [batch=1, channel=3, h, w]
    :return: an rgb masks
    '''
    pred = pred.squeeze(0)

    c, h, w = pred.shape

    pred_red = torch.zeros((h, w))
    pred_green = torch.zeros((h, w))
    pred_blue = torch.zeros((h, w))

    pred_red[pred[0] > 0.5] = 1.0
    pred_green[pred[1] > 0.5] = 1.0


    pred_red = pred_red.unsqueeze(2)
    pred_green = pred_green.unsqueeze(2)
    pred_blue = pred_blue.unsqueeze(2)

    return torch.cat((pred_red, pred_green, pred_blue), 2)

def sdf2mask(sdf):
    """from sdf to mask

    Args:
        sdf (torch.tensor(shape=[b, 2, h, w])): channels: [tibia, femur]
    returns:
        torch.tensor(shape=[h, w, 3]), masks
    """
    c, h, w = sdf.shape[1:]
    sdf = sdf.squeeze(0)
    res = torch.zeros((h, w, c + 1))
    for i in range(c):
        res[..., i] = torch.where(sdf[i] <= 0, 1, 0)
    return res


def contour2mask(contour):
    """from contour to mask

    Args:
        contour (torch.tensor(shape=[b, 2, h, w])): channels: [tibia, femur]
    returns:
        torch.tensor(shape=[h, w, 3]), masks
    """
    pred = torch.argmax(contour, 1).squeeze(0)
    h, w = pred.shape
    pred_red = torch.zeros((h, w))
    pred_green = torch.zeros((h, w))
    pred_blue = torch.zeros((h, w))
    
    pred_red[pred == 0] = 1.0
    pred_green[pred == 1] = 1.0
    
    pred_red = pred_red.unsqueeze(2)
    pred_green = pred_green.unsqueeze(2)
    pred_blue = pred_blue.unsqueeze(2)
    
    return torch.cat((pred_red, pred_green, pred_blue), 2)
    
    

def mask_synthesis(seg_mask, sdf_mask):
    h, w, = sdf_mask.shape[:2]
    seg_mask, sdf_mask = seg_mask.astype('bool'), sdf_mask.astype('bool')
    # res = ((seg xor sdf) or seg) and seg
    sysnthesis = np.logical_and(seg_mask, sdf_mask)
    res = np.zeros((h, w, 3))
    res[..., :2] = sysnthesis.astype('float32')
    return res
    
    
        
    
