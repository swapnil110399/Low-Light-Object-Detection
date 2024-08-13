import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset,DataLoader, Subset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import kornia

from image_utils import *



class IlluminationAdjustment(nn.Module):
    def __init__(self):
        super(IlluminationAdjustment, self).__init__()
        self.layers=32
        self.kernel_size=3
        self.padding=1
        self.stride=1
        self.alpha=None

        # Concat input and the ratio
        self.conv1=nn.Conv2d(2,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        self.relu=nn.LeakyReLU(0.2)
        self.conv2=nn.Conv2d(self.layers,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv3=nn.Conv2d(self.layers,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv4=nn.Conv2d(self.layers,1,kernel_size=3,padding=1,stride=self.stride)
        self.sigmoid=nn.Sigmoid()

    def forward(self,inp,ratio):
        # Concat input and the ratio
        
#         print(f'inp is {inp.shape}')
#         print(f'self.alpha is {self.alpha.shape}')
        concat1=torch.cat([inp,ratio],dim=1)
#         print(f'After concat shape is {concat1.shape}')
        conv1=self.conv1(concat1)
        relu1=self.relu(conv1)
#         print(f'After conv1 shape is {relu1.shape}')
        conv2=self.conv2(relu1)
        relu2=self.relu(conv2)
#         print(f'After conv2 shape is {relu2.shape}')
        conv3=self.conv3(relu2)
        relu3=self.relu(conv3)
#         print(f'After conv3 shape is {relu3.shape}')
        conv4=self.conv4(relu3)
#         print(f'After conv4 shape is {conv4.shape}')
        return self.sigmoid(conv4)
 

def AdjustmentLoss(adjustment_map,illum_high,sobel_horizontal,sobel_vertical):
    # Compute the gradient of the high light image
    gradient_high = compute_gradient(illum_high,sobel_horizontal,sobel_vertical)
    gradient_adjustment = compute_gradient(adjustment_map,sobel_horizontal,sobel_vertical)

    gradient_loss=torch.norm(torch.abs(gradient_high) - torch.abs(gradient_adjustment))

    squared_loss=torch.norm(illum_high - adjustment_map)

    return 0.5*gradient_loss + 0.2*squared_loss