import torch

from tqdm import tqdm
from torch.nn import functional as F
from collections import defaultdict




def evaluation(model, validation_loader, loss_funcs, device):
    
    num_examples = len(validation_loader)
    res = defaultdict(float)
    model.eval()
    model.render_head.mode = 'inference'
    loop = tqdm(validation_loader, leave=True, total=len(validation_loader), colour = 'blue')
    
    for btch in loop:
        btch = batch_to_device(btch, device)
        
        with torch.no_grad():
            # pred = # pred = {'coarse_pred': down_pred, 'fine_pred': out, 'coordi': points, 'sdm': sdm}
            pred = model(btch['img'])
            pred_mask = torch.softmax(pred['fine_pred'], dim=1)
            seg_loss = loss_funcs(pred['fine_pred'], btch['hd_mask'], 'dice').item()
            sdf_loss = loss_funcs(pred['sdm'], btch['ld_sdm'], 'mse').item() if pred.get('sdm') is not None else 0
            total_loss = seg_loss + sdf_loss
            iou_classes, miou = mIoU(pred_mask, btch['hd_mask'])
            if pred.get('sdm') is not None:
                psnr_classes, mpsnr = PSNR(pred['sdm'], btch['ld_sdm'])#  if pred.get('sdm') is not None else pred['sdm'].new_zeros(2), 0
            else:
                psnr_classes = (0, 0)
                mpsnr = 0
            
            # collect results
            res['validation_seg_loss'] += seg_loss
            res['validation_sdf_loss'] += sdf_loss
            res['validation_loss'] += total_loss
            
            res['iou_tibia'] += iou_classes[0].item()
            res['iou_femur'] += iou_classes[1].item()
            res['miou'] += miou
            
            res['psnr_tibia'] += psnr_classes[0].item()
            res['psnr_femur'] += psnr_classes[1].item()
            res['mpsnr'] += mpsnr
            
            loop.set_description("Validation")
            loop.set_postfix(loss=total_loss, mIOU=miou, mpsnr=mpsnr)
              
    for k in res:
        res[k] = res[k] / num_examples
    print(res)
    return res


def is_better(current, best, what_is_good):
    return current > best if what_is_good == 'higher' else current < best
            
def batch_to_device(example, device):

    example.update({k: v.to(device) for k, v in example.items() if hasattr(v, 'to')})
    return example

def point_sample(inputs, point_coords, **kwargs):
    """Pick up the features of inputted pixel coordinates

    Args:
        inputs (torch.tensor([B, C, H, W])): the feature map or score map
        point_coords (torch.tensor([B, num_samplers, 2])): pixel coordinates

    Returns:
        torch.tensor([B, num_samplers, C]): picked features /scores
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def mIoU(pred, mask):
    """Compute mIoU of predicted masks with ground truth

    Args:
        pred (torch.tensor): prediction
        mask (torch.tensor): ground truth

    Returns:
        tuple(torch.tensor(IoU_tibia, IoU_femur), int): IoU of tibia and femur, mIoU
    """
    IoU_classes = torch.zeros(2, dtype=torch.float32)
    one_hots = to_one_hot(pred)
    for i in range(2):
        intersec = torch.sum(one_hots[:, i, :, :] * mask[:, i, :, :])
        addition = torch.sum(one_hots[:, i, :, :] + mask[:, i, :, :])
        IoU_classes[i] = intersec / (addition - intersec)
    mIoU = torch.mean(IoU_classes).item()
    return IoU_classes, mIoU


def PSNR(pred, mask):
    """Compute PSNR of predicted sdf map with the ground truth

    Args:
        pred (torch.tensor): prediction
        mask (torch.tensor): ground truth

    Returns:
        uple(torch.tensor(psnr_tibia, psnr_femur), int): psnr of tibia and femur, mean_psnr
    """
    maxvals =  torch.tensor([1.2366, 0.7593], dtype=torch.float32).to(pred.device)
    psnr_classes = pred.new_zeros(2)
    for i in range(2):
        psnr_classes[i] = torch.mean((pred[:, i] - mask[:, i]) ** 2)
    psnr_classes = 10 * ((2 * torch.log(maxvals) - torch.log(psnr_classes)) / torch.log(torch.tensor(10., device=pred.device)))
    mpsnr = torch.mean(psnr_classes).item()
    return psnr_classes, mpsnr

def to_one_hot(pred):
    num_classes = pred.size(1)
    pred = pred.argmax(1)
    one_hots = F.one_hot(pred, num_classes=num_classes)
    return one_hots.permute(0, 3, 1, 2)
