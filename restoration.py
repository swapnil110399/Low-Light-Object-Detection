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

class Restoration(nn.Module):
    def __init__(self):
        super(Restoration, self).__init__()
        self.layers=32
        self.kernel_size=3
        self.padding=1
        self.stride=1

        #Concat reflectance and illumination
        self.conv1=nn.Conv2d(4,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        self.relu=nn.LeakyReLU(0.2)
        self.conv2=nn.Conv2d(self.layers,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)   # Img size is //2

        self.conv3=nn.Conv2d(self.layers,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv4=nn.Conv2d(self.layers*2,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)   # Img size is //4

        self.conv5=nn.Conv2d(self.layers*2,self.layers*4,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv6=nn.Conv2d(self.layers*4,self.layers*4,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)   # Img size is //8

        self.conv7=nn.Conv2d(self.layers*4,self.layers*8,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv8=nn.Conv2d(self.layers*8,self.layers*8,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)  # Img size is //16

        self.conv9=nn.Conv2d(self.layers*8,self.layers*16,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv10=nn.Conv2d(self.layers*16,self.layers*16,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu

        self.deconv1=nn.ConvTranspose2d(self.layers*16,self.layers*8,kernel_size=2,padding=0,stride=2) # Img size is //8
        # Concat deconv1 and pool3

        self.conv11=nn.Conv2d(self.layers*16,self.layers*8,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv12=nn.Conv2d(self.layers*8,self.layers*8,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu

        self.deconv2=nn.ConvTranspose2d(self.layers*8,self.layers*4,kernel_size=2,padding=0,stride=2) # Img size is //4
        # Concat deconv2 and pool2

        self.conv13=nn.Conv2d(self.layers*8,self.layers*4,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv14=nn.Conv2d(self.layers*4,self.layers*4,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu

        self.deconv3=nn.ConvTranspose2d(self.layers*4,self.layers*2,kernel_size=2,padding=0,stride=2)  # Img size is //2
        # Concat deconv3 and pool1

        self.conv15=nn.Conv2d(self.layers*4,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv16=nn.Conv2d(self.layers*2,self.layers*2,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu

        self.deconv4=nn.ConvTranspose2d(self.layers*2,self.layers,kernel_size=2,padding=0,stride=2)  # Img size is same as inp
        # Concat deconv4 and conv2

        self.conv17=nn.Conv2d(self.layers*2,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu
        self.conv18=nn.Conv2d(self.layers,self.layers,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)
        # Relu

        self.conv19=nn.Conv2d(self.layers,3,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride)

        self.sigmoid=nn.Sigmoid()


    def forward(self,ref,illum):
        inp=torch.cat([ref,illum],dim=1)
        # print(f'After concatenation inp shape is {inp.shape}')

        conv1=self.conv1(inp)
        relu1=self.relu(conv1)
        # print(f'After conv1 shape is {relu1.shape}')
        conv2=self.conv2(relu1)
        relu2=self.relu(conv2)
        # print(f'After conv2 shape is {relu2.shape}')
        pool1=self.pool1(relu2)
        # print(f'After pool1 shape is {pool1.shape}')

        conv3=self.conv3(pool1)
        relu3=self.relu(conv3)
        # print(f'After conv3 shape is {relu3.shape}')
        conv4=self.conv4(relu3)
        relu4=self.relu(conv4)
        # print(f'After conv4 shape is {relu4.shape}')
        pool2=self.pool2(relu4)
        # print(f'After pool2 shape is {pool2.shape}')

        conv5=self.conv5(pool2)
        relu5=self.relu(conv5)
        # print(f'After conv5 shape is {relu5.shape}')
        conv6=self.conv6(relu5)
        relu6=self.relu(conv6)
        # print(f'After conv6 shape is {relu6.shape}')
        pool3=self.pool3(relu6)
        # print(f'After pool3 shape is {pool3.shape}')

        conv7=self.conv7(pool3)
        relu7=self.relu(conv7)
        # print(f'After conv7 shape is {relu7.shape}')
        conv8=self.conv8(relu7)
        relu8=self.relu(conv8)
        # print(f'After conv8 shape is {relu8.shape}')
        pool4=self.pool4(relu8)
        # print(f'After pool4 shape is {pool4.shape}')

        conv9=self.conv9(pool4)
        relu9=self.relu(conv9)
        # print(f'After conv9 shape is {relu9.shape}')
        conv10=self.conv10(relu9)
        relu10=self.relu(conv10)
        # print(f'After conv10 shape is {relu10.shape}')

        deconv1=self.deconv1(relu10)
        # print(f'After deconv1 shape is {deconv1.shape}')

        concat1=torch.cat([deconv1,relu8],dim=1)
        # print(f'After concat1 shape is {concat1.shape}')
        conv11=self.conv11(concat1)
        relu11=self.relu(conv11)
        # print(f'After conv11 shape is {relu11.shape}')
        conv12=self.conv12(relu11)
        relu12=self.relu(conv12)
        # print(f'After conv12 shape is {relu12.shape}')

        deconv2=self.deconv2(relu12)
        # print(f'After deconv2 shape is {deconv2.shape}')
        concat2=torch.cat([deconv2,relu6],dim=1)
        # print(f'After concat2 shape is {concat2.shape}')
        conv13=self.conv13(concat2)
        relu13=self.relu(conv13)
        # print(f'After conv13 shape is {relu13.shape}')
        conv14=self.conv14(relu13)
        relu14=self.relu(conv14)
        # print(f'After conv14 shape is {relu14.shape}')

        deconv3=self.deconv3(relu14)
        # print(f'After deconv3 shape is {deconv3.shape}')

        concat3=torch.cat([deconv3,relu4],dim=1)
        # print(f'After concat3 shape is {concat3.shape}')
        conv15=self.conv15(concat3)
        relu15=self.relu(conv15)
        # print(f'After conv15 shape is {relu15.shape}')
        conv16=self.conv16(relu15)
        relu16=self.relu(conv16)
        # print(f'After conv16 shape is {relu16.shape}')

        deconv4=self.deconv4(relu16)
        # print(f'After deconv4 shape is {deconv4.shape}')
        concat4=torch.cat([deconv4,relu2],dim=1)
        # print(f'After concat4 shape is {concat4.shape}')
        conv17=self.conv17(concat4)
        relu17=self.relu(conv17)
        # print(f'After conv17 shape is {relu17.shape}')
        conv18=self.conv18(relu17)
        relu18=self.relu(conv18)
        # print(f'After conv18 shape is {relu18.shape}')

        conv19=self.conv19(relu18)
        # print(f'After conv19 shape is {conv19.shape}')
        restored=self.sigmoid(conv19)

        return restored



# Instead of above manual computation kornia provides inbuilt SSIM but kornia is not being able to download in the gpu environmnet
# pip install kornia installs the module but the module is not found.

def ssim_loss(img1, img2):
    sim_1=  kornia.losses.ssim_loss(img1[:,0:1,:,:], img2[:,0:1,:,:],window_size=11)
    sim_2=  kornia.losses.ssim_loss(img1[:,1:2,:,:], img2[:,1:2,:,:],window_size=11)
    sim_3=  kornia.losses.ssim_loss(img1[:,2:3,:,:], img2[:,2:3,:,:],window_size=11)
    
    final_ssim= (sim_1+sim_2+sim_3)/3.0
    
    return 1 - final_ssim


def restorationLoss(restoration_op,ref_high,sobel_horizontal,sobel_vertical):
    loss_ssim=ssim_loss(restoration_op,ref_high)
    #print(f'SSIM loss is {loss_ssim}')

    squared_loss=torch.norm(restoration_op- ref_high)
    #print(f'squared loss is {squared_loss}')
    gradient_high = compute_gradient(ref_high,sobel_horizontal,sobel_vertical)
    gradient_restoration = compute_gradient(restoration_op,sobel_horizontal,sobel_vertical)

    gradient_loss=torch.norm(gradient_high - gradient_restoration)
    #print(f'gradient loss is {gradient_loss}')
    #squared_loss=torch.nn.functional.mse_loss(restoration_op,ref_high)
    
    return 0.67*squared_loss+0.5*gradient_loss-loss_ssim
