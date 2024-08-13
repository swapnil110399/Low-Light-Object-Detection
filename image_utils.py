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

def generateSobelFilters():
    sobel_horizontal = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_vertical = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    return sobel_horizontal,sobel_vertical

def compute_gradient(image,sobel_horizontal,sobel_vertical):
    if image.shape[1]==3:
        gradient_horizontal = torch.nn.functional.conv2d(image,weight=sobel_horizontal.repeat(3, 1, 1, 1), padding=1, groups=3)
        gradient_vertical = torch.nn.functional.conv2d(image,weight=sobel_vertical.repeat(3, 1, 1, 1), padding=1, groups=3)
    else:
        gradient_horizontal = torch.nn.functional.conv2d(image,weight=sobel_horizontal,padding=1)
        gradient_vertical = torch.nn.functional.conv2d(image,weight=sobel_vertical,padding=1)

    return torch.sqrt(gradient_horizontal**2 + gradient_vertical**2 +0.001)


def getDataLoaders(dataset,batch_size):
    class_names = dataset.classes
    # Dictionary to store DataLoader for each class.
    # Since we need to process each class of images sepaartely,separating data into distcinct classes befor further processing
    class_data_loaders = {}

    for class_idx, class_name in enumerate(class_names):
        # Create a subset of the dataset for the current class
        class_indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label == class_idx]
        class_subset = Subset(dataset, class_indices)

        # shuffle=False is used to enforce corresponding images to be in pair.
        # If not used a low light image might not have its corresponding normal light image in a batch and are randomly paired

        class_data_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)

        # Store the DataLoader in the dictionary
        class_data_loaders[class_name] = class_data_loader

    return class_data_loaders

def preprocessDataset(train_images_directory,test_images_directory):

    transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=1.0),
    #transforms.RandomRotation(90),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    ])

    train_dataset=ImageFolder(train_images_directory,transform=transform)
    test_dataset=ImageFolder(test_images_directory,transform=transform)

    return train_dataset,test_dataset


def verifyDataset(high_img,low_img):
    print(f"Batch size: {high_img[0].shape[0]}")
    print(f"Shape of each image: {high_img[0].shape[2]}*{high_img[0].shape[3]}*{high_img[0].shape[1]}")

    fig = plt.figure(constrained_layout=False,figsize=(15,5))
    subplots = fig.subfigures(1,2)

    ax0 = subplots[0].subplots(1,2)
    ax1 = subplots[1].subplots(1,2)

    counter=0
    row=0
    col=0
    for i in range(high_img[0].shape[0]):
        ax0[col].imshow(high_img[0][counter].squeeze().permute(1,2,0))
        ax1[col].imshow(low_img[0][counter].squeeze().permute(1,2,0))
        col+=1
        counter+=1
        if col>=2:
            row+=1
            col=0
    fig.suptitle('Normal and Low Light Images')
    plt.tight_layout()
    plt.show()

def testData(directory):
    transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    ])

    dataset=ImageFolder(directory,transform=transform)
    
    return dataset