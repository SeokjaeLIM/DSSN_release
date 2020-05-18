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
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

#from OpenCV
import cv2

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



version = '3-15l-7-5-3'



################ Hyper Parameters ################
maxDataNum = p.maxDataNum #in fact, 4206
batchSize = p.batchSize


imgCropWidth = p.imgCropWidth
imgCropHeight = p.imgCropHeight

imageWidth = p.imageWidth
imageHeight = p.imageHeight
NGF = p.NGF
NDF = p.NDF
ZLength = p.ZLength
# train
MaxEpoch = p.MaxEpoch
learningRateE = p.learningRateE
lrDecayRateE = p.lrDecayRateE
learningRateR = p.learningRateR  # default:0.0003
lrDecayRateR = p.lrDecayRateR #0.999308

# save
numberSaveImage = p.numberSaveImage


############################################



class VisionOPLayer(nn.Module):

    def __init__(self):
        super(VisionOPLayer, self).__init__()

    def forward(self, x, isDenorm = 1):
        if isDenorm == 1:
            x = denorm(x)
            
        #x = VisionOP.RGB2HSI(x)        
        #x[:,1,:,:] = norm(torch.clamp((x[:,1,:,:] + 0.3),0,1))
        x = norm(VisionOP.Laplacian(x,ksize=3))

        return x
        

class SharpMSELoss(nn.Module):

    def __init__(self):
        super(SharpMSELoss, self).__init__()        

    #(N,C,H,W)
    def forward(self, input, target):

        # error < 0.5 -> error / error >= 0.5 -> error^2+0.25
        input = torch.abs(input - target).cuda()

        underth = VisionOP.Nonzero((VisionOP.Min(input,0.5*torch.ones(input.size()).cuda()) - 0.5))

        input = input * underth + (input*input+0.25) * (1 - underth)
        
        Loss = torch.mean(input) #torch.mean(input.view(input.size()[0],-1),1)
        
        return Loss

class Util(nn.Module):

    def __init__(self):
        super(Util, self).__init__()

        self.downSample = nn.AvgPool2d(4,stride=4)
        self.upSample = nn.Upsample(size=(imageHeight,imageWidth),mode='bilinear')

    def forward(self, x):
        x = x.view(-1,3,imageHeight,imageWidth)
        x = self.upSample(self.downSample(x))
        return x


class Encoder_class(nn.Module):

    def __init__(self):
        super(Encoder_class, self).__init__()

        self.conv1 = nn.Conv2d(3, NDF * 1, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv2d(NDF * 1, NDF * 2, 4, 2, 1)  # 128->64
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1)  # 64->32
        self.conv4 = nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1)  # 32->16

        self.BN_conv2 = nn.BatchNorm2d(NDF * 2)
        self.BN_conv3 = nn.BatchNorm2d(NDF * 4)
        self.BN_conv4 = nn.BatchNorm2d(NDF * 8)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, x):
        x = x.view(-1, 3, imageHeight, imageWidth)

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.BN_conv2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.BN_conv3(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.BN_conv4(self.conv4(x)), 0.2)

        return x


class Decoder_class(nn.Module):

    def __init__(self):
        super(Decoder_class, self).__init__()

        self.Deconv1 = nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1)  # 16->32
        self.Deconv2 = nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1)  # 32->64
        self.Deconv3 = nn.ConvTranspose2d(NGF * 2, NGF * 1, 4, 2, 1)  # 64->128
        self.Deconv4 = nn.ConvTranspose2d(NGF * 1, 3, 4, 2, 1)  # 128->256

        self.BN_conv1 = nn.BatchNorm2d(NGF * 4)
        self.BN_conv2 = nn.BatchNorm2d(NGF * 2)
        self.BN_conv3 = nn.BatchNorm2d(NGF * 1)

        nn.init.xavier_normal_(self.Deconv1.weight)
        nn.init.xavier_normal_(self.Deconv2.weight)
        nn.init.xavier_normal_(self.Deconv3.weight)
        nn.init.xavier_normal_(self.Deconv4.weight)


    def forward(self, x):
        x = F.relu(self.BN_conv1(self.Deconv1(x)))

        x = F.relu(self.BN_conv2(self.Deconv2(x)))

        x = F.relu(self.BN_conv3(self.Deconv3(x)))

        x = F.tanh(self.Deconv4(x))

        return x


class Res_class(nn.Module):

    def __init__(self):
        super(Res_class, self).__init__()

        self.conv1 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 2)  # I/O same size
        self.conv2 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 1)
        self.conv3 = nn.Conv2d(NDF * 8, NDF * 8, 3, 1, 1)
        self.conv4 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 2)
        self.conv5 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 1)
        self.conv6 = nn.Conv2d(NDF * 8, NDF * 8, 3, 1, 1)
        self.conv7 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 2)
        self.conv8 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 1)
        self.conv9 = nn.Conv2d(NDF * 8, NDF * 8, 3, 1, 1)
        
        self.conv10 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 2)
        self.conv11 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 1)
        self.conv12 = nn.Conv2d(NDF * 8, NDF * 8, 3, 1, 1)

        
        self.conv13 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 2)
        self.conv14 = nn.Conv2d(NDF * 8, NDF * 8, 4, 1, 1)
        self.conv15 = nn.Conv2d(NDF * 8, NDF * 8, 3, 1, 1)
        
                

        self.BN_conv = nn.BatchNorm2d(NGF * 8)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)
        
        nn.init.xavier_normal_(self.conv10.weight)
        nn.init.xavier_normal_(self.conv11.weight)
        nn.init.xavier_normal_(self.conv12.weight)
        
        nn.init.xavier_normal_(self.conv13.weight)
        nn.init.xavier_normal_(self.conv14.weight)
        nn.init.xavier_normal_(self.conv15.weight)
        


    def forward(self, x):
        res = x
        x = F.leaky_relu(self.BN_conv(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv3(x)) + res, 0.2)

        res = x
        x = F.leaky_relu(self.BN_conv(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv6(x) + res), 0.2)

        res = x
        x = F.leaky_relu(self.BN_conv(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv8(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv9(x) + res), 0.2)

        res = x
        x = F.leaky_relu(self.BN_conv(self.conv10(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv11(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv12(x) + res), 0.2)
        
        res = x
        x = F.leaky_relu(self.BN_conv(self.conv13(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv14(x)), 0.2)
        x = F.leaky_relu(self.BN_conv(self.conv15(x) + res), 0.2)
        
        
        return x


class D_class(nn.Module):

    def __init__(self):
        super(D_class, self).__init__()

        self.conv1 = nn.Conv2d(3, NDF * 1, 4, 2, 1)  # 64->32
        self.conv2 = nn.Conv2d(NDF * 1, NDF * 2, 4, 2, 1)  # 32->16
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1)  # 16->8
        self.conv4 = nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1)  # 8->4
        self.conv5 = nn.Conv2d(NDF * 8, 1, 4, 1, 0)  # 4->1

        self.BN_conv2 = nn.BatchNorm2d(NDF * 2)
        self.BN_conv3 = nn.BatchNorm2d(NDF * 4)
        self.BN_conv4 = nn.BatchNorm2d(NDF * 8)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)

    def parallelPool(self, x):
        xSize = x.size()

        x = x.contiguous()

        lrX = torch.chunk(x.view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)

        lubX = torch.chunk(lrX[0].contiguous().view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)
        rubX = torch.chunk(lrX[1].contiguous().view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)

        x1 = lubX[0].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x2 = rubX[0].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x3 = lubX[1].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x4 = rubX[1].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))

        x = torch.cat((x1, x2, x3, x4), 1)  # (N,C,D,H,W)->(N,C*4,D,H/2,W/2)
        return x

    def forward(self, x):
        x = x.view(-1, 3, imageHeight, imageWidth)

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.BN_conv2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.BN_conv3(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.BN_conv4(self.conv4(x)), 0.2)

        x = F.sigmoid(self.conv5(x))

        return x





















