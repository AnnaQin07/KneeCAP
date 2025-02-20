'''
Author: YunxiangLiu u7191378@anu.edu.au
Date: 2024-02-07 21:46:56
LastEditors: YunxiangLiu u7191378@anu.edu.au
LastEditTime: 2024-02-14 01:07:55
FilePath: \MA_detections\guis\test.py
Description: GUI pipe
'''
import sys
import os 
import PyQt5
import pydicom
import numpy as np 

o_path = os.getcwd()
sys.path.append(o_path)

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton

from model import LowerLimb_Network

class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'Mechanical Alignment Analysis Pipeline'
        self.left = 0
        self.top = 0
        self.width = 1280
        self.height = 1024
        self.initUI()
         
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        # Add a label object to display the processed image
        self.label = QLabel(self)
        self.label.resize(self.width, self.height)
        self.label.move(50, 0)
         
        # Add a button object to trigger the processing
        button = QPushButton('Input', self)
        button.move(self.width - 80, self.height - 60)
        button.clicked.connect(self.input_img)
         
        self.show()
        
    def input_img(self):
        
        img = r"F:/Datasets/comp8603/Lower_limb_Xrays/Canberra_hospital/10_Aug_2023/dcm/Ac-IM-0024-0005.dcm"
        img = pydicom.read_file(img)
        img = img.pixel_array
        img = self.Uint16toUint8(img)
        
        bytesPerLine = img.shape[1]
        qImg = QImage(img.data, *img.shape[::-1], bytesPerLine, QImage.Format_Grayscale8)
        # qImg = qImg.scaledToHeight(img.shape[0] // 4, mode=QtCore.Qt.FastTransformation)
        pixmap = QPixmap.fromImage(qImg).scaledToHeight(img.shape[0]//4, mode=QtCore.Qt.FastTransformation) 
        self.label.setPixmap(pixmap)


        
    
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
        
        




