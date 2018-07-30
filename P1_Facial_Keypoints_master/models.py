## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import math


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # output size = 112x112

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2))
        # output size = 56x56

        self.conv3 = nn.Sequential(
             nn.Conv2d(64,64, 3, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(2, stride=2)
        )
        # output size = 28x28

        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        # output size = 14x14  
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        #output size = 7x7
        
        self.lr = nn.Sequential(            
            nn.Dropout(p=0.6),
            nn.Linear(7*7*128,136),

        )

        #The code for weight initialization refer to PyTorch documentation.
        #https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
        for m in self.modules():
            if isinstance(m,nn.Conv2d):                           
                
                I.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.0001)
                m.bias.data.zero_()
                
                  

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        
        x = self.conv4(x)
        x = self.conv5(x)
        #x = F.avg_pool2d(x,kernel_size=x.size()[2:])
        x = x.view(x.size(0),-1)
        x = self.lr(x)

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
