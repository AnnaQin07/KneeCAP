import os
import torch 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as DataLoader
from utils import Losses, build_optimizer, build_scheduler, load_checkpoint, \
                  resume_training, batch_to_device, point_sample, evaluation, is_better



def train(args, train_dataset, val_dataset, model):
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=args.datasets.train.shuffle,
                                  num_workers=args.num_workers)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size,
                                shuffle=args.datasets.train.shuffle,
                                num_workers=args.num_workers)
    device = torch.device(args.device)
    
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        checkpoints = torch.load(args.checkpoint_path)
        model = load_checkpoint(model, checkpoints)
        print("successfully load model weights")
    else:
        checkpoints = {}
        print("no checkpoints detect, training from start")
        
    # set loss, optimizer and tensorboard plot drawer
    loss_funcs = Losses(args.loss)
    optimizer = build_optimizer(model, args.optimizer)
    scheduler = build_scheduler(optimizer, args.scheduler)
    optimizer, resume_epoch, best_metric = resume_training(optimizer, checkpoints, good_metric=args.good_metric)
    best_epoch = resume_epoch
    print(f'we start training from epoch {resume_epoch + 1}, {args.number_epoch - resume_epoch} epoch remains')
    writer = SummaryWriter(log_dir=f'experiments/{args.exp_name}/logs')
    model.freeze()
    model.to(device)
    best_model = model
    
    # Let's start
    for e in range(resume_epoch, args.number_epoch):
        # res = evaluation(model, val_dataloader, loss_funcs, device)
        model.render_head.mode = 'train'
        model.train()
        loop = tqdm(train_dataloader, leave=True, total=len(train_dataloader), colour='green')
        train_loss = 0
        
        # example = {'img': img, 'ld_mask': ld_mask, 'hd_mask': hd_mask, 'ld_sdm': ld_sdfmp}
        for btch in loop:
            btch = batch_to_device(btch, device)
            # pred = {'coarse_pred': down_pred, 'fine_pred': out, 'coordi': points, 'sdm': sdm}
            pred = model(btch['img'])
            
            # compute loss
            coarse_seg_loss = loss_funcs(pred['coarse_pred'], btch['ld_mask'], 'dice')
            sdf_loss = loss_funcs(pred['sdm'], btch['ld_sdm'], 'mse') if pred.get('sdm') is not None else 0
            hd_masks = btch['hd_mask'].argmax(dim=1)
            gt_points = point_sample(hd_masks.to(torch.float32).unsqueeze(1), pred['coordi'], mode="nearest", align_corners=False).squeeze(1).long()
            fine_seg_loss = loss_funcs(pred['fine_pred'], gt_points, 'ce')
            total_loss = coarse_seg_loss + sdf_loss + fine_seg_loss
            
            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            
            # training log
            loop.set_description(f"Train Epoch[{e+1}/{args.number_epoch}]")
            loop.set_postfix({'loss': total_loss.item(), 'coarse': coarse_seg_loss.item(), 'fine': fine_seg_loss.item(), 'geometry': sdf_loss.item()})
        
        train_loss = train_loss / len(train_dataloader)    
        
        # evaluate
        if e % 1 == 0:
            model.eval()
            res = evaluation(model, val_dataloader, loss_funcs, device)
            scheduler.step(res['validation_loss'])
            res['train_loss'] = train_loss
            writer.add_scalars(f"metrics_{args.exp_name}", res, e)
            checkpoint = {'model_state_dict': model.state_dict(), 
                          'optimizer_state_dict': optimizer.state_dict(), 
                          'epoch': e, 
                          "best_metric": best_metric, 
                          'good_metric': args.good_metric}
            torch.save(checkpoint, f'experiments/{args.exp_name}/latest.pth')
            current_metric = res['miou'] if args.evaluation_metric == 'iou' else res['mpsnr']
            if is_better(current_metric, best_metric, args.good_metric):
                best_metric = current_metric
                best_epoch = e
                best_model = model
                best_checkpoint = {'model_state_dict': best_model.state_dict(), 'best_epoch': best_epoch, 'eval metric': args.evaluation_metric}
                torch.save(best_checkpoint, f'experiments/{args.exp_name}/best.pth')
                print(f'\n new best model saved at epoch: {e}, whose {args.evaluation_metric} is {best_metric}')
    
    print('\n -------------------------------------------------')
    print('\n best {} achieved: {:.3f} at epoch {}'.format(args.evaluation_metric, best_metric, best_epoch))      
    writer.flush()
    writer.close()
    return best_model      
            
        
            
            
    
    
    
    
    
    
    
    
    
    
        