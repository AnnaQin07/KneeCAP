'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2023-09-12 14:25:56
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2024-03-11 22:26:22
FilePath: \MA_detections\main.py
Description: main file for program implementation
'''
###############################################
# To package the program, please:
# 1. Comment line 164-166
# 2. Uncomment line 169-171
# 3. Run in the terminal:
# >>> cd main
# >>> pyinstaller --onefile --windowed --add-data "best.pth:." --add-data "configs:configs" --add-data "guis:guis" main.py
# ## pyinstaller --windowed --name "MyApp" --add-data "best.pth:." --add-data "configs:configs" --add-data "guis:guis" main.py
# >>> cd dist
# >>> ./main
###############################################

import sys
import os
import yaml
import json
import torch
import argparse

from PyQt5.QtWidgets import QApplication, QMainWindow
from easydict import EasyDict as edict
from torch.utils.data import DataLoader as DataLoader

from train import train
from inference import inference
from guis.interface_v0 import Pip_MainWindow
from postprocess.evaluation import compute_metrics
from model import LowerLimb_Network
from utils import Losses, evaluation, load_checkpoint
from datasets import Build_dataset, Check_dataset, Split_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug',
                        help="the experiment name, which will create a fold named as setting in the 'experiment' folder")
    parser.add_argument('--device', type=str, default='cuda',
                        help="the device to run the experiment choose {cuda, cpu}")
    parser.add_argument('--config', type=str, default=r'configs/fine_tune.yaml',
                        help="the path of the config file in which contains the configurations of dataset, pipeline and hype-parameter of the model")
    parser.add_argument('--task', type=str, default='gui', choices=['train', 'test', 'inf', 'gui'],
                        help="the task implement ")
    parser.add_argument('--inf_mode', type=str, default='seg_only', choices=['seg_only', 'postprocess_only', 'all'],
                        help="the inference mode")
    parser.add_argument('--gt_hkaa_dir', type=str, default=r"data/12m_hkaa.json",
                        help="the coordinate change directory for postprocess algorithm")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--number_epoch', type=int, default=60, help='the number of epoch during training') # 100
    parser.add_argument('--batch_size', type=int, default=4, help='batch size') # 8
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers of dataloaders')
    # r"experiments/v1_shift/best.pth"
    # "experiments/finetune_head_cbr/best.pth"
    parser.add_argument('--checkpoint_path', type=str, default='experiments/finetune_v2/best.pth', help='the checkpoints')

    parser.add_argument('--good_metric', type=str, default='higher', choices=['higher', 'lower'], 
                        help='indicates the strategy for saving the checkpoint during training')
    return parser

# Function to get the path to a file, depending on whether the program is frozen (PyInstaller packed) or not
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for both dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    
    parser = parse_args()
    default_args = parser.parse_args()
    torch.manual_seed(default_args.seed)
    torch.cuda.manual_seed(default_args.seed)

    os.makedirs("log", exist_ok=True)
    config_path = resource_path('configs/inference.yaml')
    
    if not default_args.task == 'gui':
        # create experiment folder
        experiment_path = f"experiments/{default_args.exp_name}"
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        
        with open(default_args.config, 'r') as yf:
            args = edict(yaml.load(yf, Loader=yaml.SafeLoader))
        
        # create datasets
        test_dataset = Build_dataset(args.datasets.test)
        default_args = vars(default_args)
        args.update(default_args)
        
        if args.task == 'train':
            train_dataset = Build_dataset(args.datasets.train)
            train_dataset, val_dataset = Split_dataset(args.datasets.train, train_dataset)
            if val_dataset is None:
                val_dataset = test_dataset
            val_dataset.mode = 'val'
        # check dataset is complete
            Check_dataset(train_dataset)
            Check_dataset(test_dataset)
        
        model = LowerLimb_Network(args.model)
        
        if args.task == 'train':
            if hasattr(args, 'resume_path'):
                weight = torch.load(args.resume_path)
                model = load_checkpoint(model, weight)
            best_model = train(args, train_dataset, val_dataset, model)
            test_dataloader = DataLoader(test_dataset, 
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.num_workers)
            loss_funcs = Losses(args.loss)
            res = evaluation(best_model, test_dataloader, loss_funcs, device=torch.device(args.device))
            
            with open(f"experiments/{args.exp_name}/test_report.json", "w") as jf:
                json.dump(res, jf) 
            print("----test after training----")
            print(res)
        
        elif args.task == 'test':
            weight = torch.load(args.resume_path)
            model = load_checkpoint(model, weight)
            test_dataloader = DataLoader(test_dataset, 
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.num_workers)
            loss_funcs = Losses(args.loss)
            res = evaluation(model.to(args.device), test_dataloader, loss_funcs, device=torch.device(args.device))
            
            with open(f"experiments/{default_args.exp_name}/test_report.json", "w") as jf:
                json.dump(res, jf) 
            print("----test after training----")
            print(res)
            
        elif args.task == 'inf':
            if not args.inf_mode == 'postprocess_only':
                if args.checkpoint_path is not None:
                    weight = torch.load(args.checkpoint_path)
                elif hasattr(args, 'resume_path'):
                    weight = torch.load(args.resume_path)
                else:
                    # model_path = f'experiments/{args.exp_name}/best.pth'
                    model_path = f'best.pth'
                    weight = torch.load(model_path)
                model = load_checkpoint(model, weight)
            test_dataset.mode = 'inf'
            inference(args, test_dataset, model.to(args.device))
            
            # predition_path = f"experiments/{args.exp_name}/measurements.csv"
            # hkas, mae = compute_metrics(args.gt_hkaa_dir, predition_path)
            # print(f"the mean average error is {round(mae, 3)}")
    else:
        # Uncomment to run in the terminal
        # with open(default_args.config, 'r') as yf:
        #     args = edict(yaml.load(yf, Loader=yaml.SafeLoader))
        # model_path = args.checkpoint_path

        # Uncomment to package the whole program
        with open(config_path, 'r') as yf:
            args = edict(yaml.load(yf, Loader=yaml.SafeLoader))
        model_path = resource_path('best.pth')
        
        default_args = vars(default_args)
        args.update(default_args)
        
        model = LowerLimb_Network(args.model)
        device = 'cpu'

        weight = torch.load(model_path, map_location=torch.device('cpu'))
        model = load_checkpoint(model, weight)
        
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        ui = Pip_MainWindow(MainWindow, model, device)
        # ui.setupUi(MainWindow)
        MainWindow.show()
        
        sys.exit(app.exec_())


    # model.to(device)
    # checkpoint = {'model_state_dict': model.state_dict()}
    # torch.save(checkpoint, 'default.pth')
