
#from Python
import time
import csv
import os
import math
import numpy as np
import sys
from shutil import copyfile

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

#import 3rd Party
import convlstm as ConvLSTM

#from this project
import param as p
import VisionOP



#local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1,1)



version = '22 - RefPad & padless'



################ Hyper Parameters ################
maxDataNum = p.maxDataNum #in fact, 4206
batchSize = p.batchSize

trainMode = p.trainMode
MaxCropWidth = p.MaxCropWidth
MinCropWidth = p.MinCropWidth
MaxCropHeight = p.MaxCropHeight
MinCropHeight = p.MinCropHeight

imageWidth = p.imageWidth
imageHeight = p.imageHeight
NGF = p.NGF
NDF = p.NDF
scaleLevel = p.scaleLevel

# train
MaxEpoch = p.MaxEpoch
learningRateNET = p.learningRateNET

# save
numberSaveImage = p.numberSaveImage


############################################
class resBlock(nn.Module):
    
    def __init__(self, channelDepth, windowSize=3):
        
        super(resBlock, self).__init__()

        
        self.pad = nn.ReflectionPad2d(1)

        self.IN_conv1 = nn.InstanceNorm2d(channelDepth)

        self.conv1 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, 0)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, 0)


                      
    def forward(self, x):
        
        res = x
        x = F.relu(self.IN_conv1(self.conv1(self.pad(x))))
        x = self.IN_conv1(self.conv2(self.pad(x)))
    
        x = x + res
        
        return x

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()


        self.pad = nn.ReflectionPad2d(1)

        self.IN_conv1 = nn.InstanceNorm2d(NGF * 1)
        self.IN_conv2 = nn.InstanceNorm2d(NGF * 2)

        self.encoder1 = nn.Conv2d(6, NGF * 1, 4, 2, 0)
        
        self.Enres1 = resBlock(NGF * 1)
        self.Enres2 = resBlock(NGF * 1)
        self.Enres3 = resBlock(NGF * 1)
        self.Enres4 = resBlock(NGF * 1)
        self.Enres5 = resBlock(NGF * 1)

        self.encoder2 = nn.Conv2d(NGF*1, NGF * 2, 4, 2, 0)

        self.res1 = resBlock(NGF * 2)
        self.res2 = resBlock(NGF * 2)
        self.res3 = resBlock(NGF * 2)
        self.res4 = resBlock(NGF * 2)
        self.res5 = resBlock(NGF * 2)

        self.decoder1 = nn.ConvTranspose2d(NGF * 2, NGF*1, 4, 2, 1) 

        self.Deres1 = resBlock(NGF * 1)
        self.Deres2 = resBlock(NGF * 1)
        self.Deres3 = resBlock(NGF * 1)
        self.Deres4 = resBlock(NGF * 1)
        self.Deres5 = resBlock(NGF * 1)

        self.decoder2 = nn.ConvTranspose2d(NGF * 1, 3, 4, 2, 1)


    def forward(self, x):

        x = F.relu(self.IN_conv1(self.encoder1(self.pad(x))))

        x = self.Enres5(self.Enres4(self.Enres3(self.Enres2(self.Enres1(x)))))

        x = F.relu(self.IN_conv2(self.encoder2(self.pad(x))))

        x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
                    
        x = F.relu(self.IN_conv1(self.decoder1(x)))

        x = self.Deres5(self.Deres4(self.Deres3(self.Deres2(self.Deres1(x)))))
                    
        x = self.decoder2(x)
        return x       





