import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset,DataLoader, Subset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
#from kornia.metrics import SSIM
import kornia
from image_utils import generateSobelFilters,compute_gradient



class Decomposition(nn.Module):
    def __init__(self):
        super(Decomposition, self).__init__()
        self.layers=32
        self.kernel_size=3
        self.padding=1
        self.stride=1

        # Reflectance
        self.conv1=nn.Conv2d(3,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        self.relu=nn.LeakyReLU(0.2)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(self.layers,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(self.layers*2,self.layers*4,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.deconv1=nn.ConvTranspose2d(self.layers*4,self.layers*2,kernel_size=2,padding=0,stride=2)
        # Concat deconv1 and conv2
        self.conv4=nn.Conv2d(self.layers*4,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.deconv2=nn.ConvTranspose2d(self.layers*2,self.layers,kernel_size=2,padding=0,stride=2)
        # Concat deconv2 and conv1
        self.conv5=nn.Conv2d(self.layers*2,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv6=nn.Conv2d(self.layers,3,kernel_size=1,padding=0,stride=self.stride)
        self.sigmoid=nn.Sigmoid()

        # Illumination
        self.conv7=nn.Conv2d(self.layers,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        # Concat conv7 and conv5
        self.conv8=nn.Conv2d(self.layers*2,1,kernel_size=1,padding=0,stride=self.stride)


    def forward(self,inp):
        # Reflectance
        conv1=self.conv1(inp)
        relu1=self.relu(conv1)
        pool1=self.pool1(relu1)
        conv2=self.conv2(pool1)
        relu2=self.relu(conv2)
        pool2=self.pool2(relu2)
        conv3=self.conv3(pool2)
        relu3=self.relu(conv3)
        deconv1=self.deconv1(relu3)
        concat1=torch.cat([deconv1,conv2],dim=1)
        conv4=self.conv4(concat1)
        relu4=self.relu(conv4)
        deconv2=self.deconv2(relu4)
        concat2=torch.cat([deconv2,conv1],dim=1)
        conv5=self.conv5(concat2)
        relu5=self.relu(conv5)
        conv6=self.conv6(relu5)
        reflectance=self.sigmoid(conv6)

        # Illumination
        conv7=self.conv7(relu1)
        relu6=self.relu(conv7)
        concat3=torch.cat([relu6,relu5],dim=1)
        conv8=self.conv8(concat3)
        illumination=self.sigmoid(conv8)


        return reflectance,illumination



def DecomLoss(ref_low,ref_high,illum_low,illum_high,sobel_horizontal,sobel_vertical,actual_low,actual_high):  # Modifying Loss functions
    # Reflectance Similarity
    reflectance_similarity=torch.norm(ref_low-ref_high)

    # Illumination Smoothness
    gradient_illum_low=compute_gradient(illum_low,sobel_horizontal,sobel_vertical)
    gradient_actual_low=compute_gradient(transforms.functional.rgb_to_grayscale(actual_low),sobel_horizontal,sobel_vertical)
    low_illum_loss=torch.norm(gradient_illum_low/torch.max(gradient_actual_low,torch.tensor(0.01)))

    gradient_illum_high=compute_gradient(illum_high,sobel_horizontal,sobel_vertical)
    gradient_actual_high=compute_gradient(transforms.functional.rgb_to_grayscale(actual_high),sobel_horizontal,sobel_vertical)
    high_illum_loss=torch.norm(gradient_illum_high/torch.max(gradient_actual_high,torch.tensor(0.01)))

    illumination_smoothness=low_illum_loss+high_illum_loss

    # Mutual Consistency
    M=torch.abs(gradient_illum_high)+torch.abs(gradient_illum_low)
    mutual_consistency=torch.norm(M*torch.exp(-2*M))

    # Reconstruction Loss
    reconstructed_loss=torch.norm(ref_low*illum_low.repeat(1,3,1,1)-actual_low) + torch.norm(ref_high*illum_high.repeat(1,3,1,1)-actual_high)

    return 0.009*reflectance_similarity+ 0.2*mutual_consistency + reconstructed_loss + 0.002 *illumination_smoothness
