'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2024-02-07 20:45:49
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2024-02-28 12:55:03
FilePath: \MA_detections\gui.py
Description: Gui
'''
import sys
import yaml
import torch 
from PyQt5.QtWidgets import QApplication, QMainWindow

from guis.Interface import App
from easydict import EasyDict as edict
from guis.interface_v0 import Pip_MainWindow

from model import LowerLimb_Network
from utils import load_checkpoint

if __name__ == "__main__":
    
    yaml_file = r'configs/fine_tune.yaml'
    with open(yaml_file, 'r') as yf:
        args = edict(yaml.load(yf, Loader=yaml.SafeLoader))
    
    model = LowerLimb_Network(args.model)
    device = 'cpu'
    model_path = 'experiments/finetune_v2/best.pth'
    weight = torch.load(model_path)
    model = load_checkpoint(model, weight)
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Pip_MainWindow(MainWindow, model, device)
    # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



    