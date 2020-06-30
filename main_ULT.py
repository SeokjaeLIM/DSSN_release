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

#from pytorch_msssim (github/jorge_pessoa)
import pytorch_msssim

#from pytorch_ssim(github/po-hsun-su)
import pytorch_ssim

#from AdamW-pytorch(egg-west)
from adamW import AdamW

#from this project
from data_loader import get_loader
import data_loader as dl
import VisionOP
import model
import param as p

torch.backends.cudnn.benchmark = True

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


################ Hyper Parameters ################
# data Set
dataSetName = p.dataSetName
trainMode = p.trainMode

dataPath = p.dataPath
labelPath = p.labelPath

maxDataNum = p.maxDataNum #in fact, 4206
batchSize = p.batchSize

MaxCropWidth = p.MaxCropWidth
MinCropWidth = p.MinCropWidth
MaxCropHeight = p.MaxCropHeight
MinCropHeight = p.MinCropHeight

Hue_PB = p.Hue_PB
Min_Hue_Factor = p.Min_Hue_Factor
Max_Hue_Factor = p.Max_Hue_Factor
Gamma_PB = p.Gamma_PB
Min_Gamma_Factor = p.Min_Gamma_Factor
Max_Gamma_Factor = p.Max_Gamma_Factor

# model
imageWidth = p.imageWidth
imageHeight = p.imageHeight 
NGF = p.NGF
NDF = p.NDF
scaleLevel = p.scaleLevel

# train
MaxEpoch = p.MaxEpoch
learningRateNET = p.learningRateNET
learningRateE = p.learningRateE
trainRange = p.trainRange

# save
numberSaveImage = p.numberSaveImage

################################################
MODE = sys.argv[1]

#test mode#
if(MODE == 't'):
    data_loader = get_loader(dataPath + 'test/',
                                labelPath,
                                1280,1280,720,720,imageHeight,imageWidth,
                                1,
                                Hue_PB,Max_Hue_Factor,Min_Hue_Factor,
                                Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor,
                                dataSetName,'test',trainMode)

#train mode#
elif(MODE == 'y' or MODE == 'n'):
    data_loader = get_loader(dataPath + 'train/',
                                labelPath,
                                MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,256,256,
                                batchSize,
                                Hue_PB,Max_Hue_Factor,Min_Hue_Factor,
                                Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor,
                                dataSetName,'train',trainMode)


# init model
NET = []
NET2 = []
net_optimizer = []
net2_optimizer = []

for i in range(0,2):  
    NET.append(model.Network())

    net_optimizer.append(AdamW(NET[i].parameters(), lr=learningRateNET[i], weight_decay=0.00008))

models = [(NET,'NET',net_optimizer)]

# init GPU & DP
startEpoch = 0
print("Load below models")

for mdl in models:
    for i in range(scaleLevel):
        if(mdl[0][i] != None):
        
            modelObj = mdl[0]
            modelName = mdl[1]
            optimizer = mdl[2]
    
            modelObj[i].cuda()
            modelObj[i] = nn.DataParallel(modelObj[i])  
    
            if (MODE != 'n'):
                
                checkpoint = torch.load('./data/model/' + modelName + str(i) + '.pkl')
                modelObj[i].load_state_dict(checkpoint['model'],strict=False)

                if(i==0):
                    startEpoch = checkpoint['epoch']

                preTrainedDict = torch.load('./data/model/' + modelName + str(i) + '.pkl')
                optimDict = optimizer[i].state_dict()
                preTrainedDict = {k: v for k, v in preTrainedDict.items() if k in optimDict}

                optimDict.update(optimDict)
                
                optimizer[i].load_state_dict(optimDict)
                for param_group in optimizer[i].param_groups: param_group['lr'] = learningRateNET[i]
    
                paramSize = 0
                for parameter in modelObj[i].parameters():
                    paramSize = paramSize + np.prod(np.array(parameter.size()))
                print(modelName + str(i) + ' : ' + str(paramSize))        
    
            
            if (MODE == 't'):
                modelObj[i].eval()
            else:
                modelObj[i].train()

print("All model loaded.")
       


if(MODE == 't'):

    torch.set_grad_enabled(False)
    TotalImage = 0
                                                                  
   
    for i, (images, _) in enumerate(data_loader):

        torch.cuda.synchronize()
    
        with torch.no_grad():
            GTImagesOri = images[:,:,:,imageWidth:].cuda()
         
            blurryImagesOri = images[:,:,:,0:imageWidth].cuda()
    
            a=0
            batchSize = images.size(0)

            outputList = []

            w = time.perf_counter()

            tmpRst = blurryImagesOri  


                ##########   SPAT
            for Scale in range(0,2):
                
                inputblurry = torch.cat((blurryImagesOri[:,:,],tmpRst[:,:,:,:]),1)
            
                tmpRst =  NET[Scale](inputblurry) + blurryImagesOri

            oldw = w
            w = time.perf_counter()
            a = w - oldw
        
            
            output = tmpRst
            GTImages = GTImagesOri.cuda() 
            GTImages = denorm(GTImages)
  
            TotalImage = TotalImage + batchSize
    
            DeblurImages = denorm(output)
            
            save_image(DeblurImages.data, './data/result/test-%d.png' % (i + 1))
            save_image(GTImages.data, './data/result/gt-%d.png' % (i + 1))


        print('TotalImage :  %d , i : %d, Time : %.4fs' %(TotalImage,i,a),end='\r')

    print('TotalImage :  %d , i : %d, Time : %.4fs' %(TotalImage,i,a))
        
else:
    # loss and optimizer
    ae_criterion = nn.MSELoss()

    a = time.perf_counter()
    b = time.perf_counter()

    #startEpoch = 1200


    for epoch in range(startEpoch, MaxEpoch):

        # ============= Train the AutoEncoder =============#
        finali = 0

        AvgNETloss = [torch.zeros(1)]*256
        Avgtotalloss = [torch.zeros(1)]*256
        
        outputList = []
    
        for i, (images, _) in enumerate(data_loader):

            batchSize = images.size(0)
            
            NET_loss = [torch.zeros(1)]*256

            blurryImagesOri = to_var(images[:,:,:,0:256]).contiguous()
        
            with torch.no_grad():
                GTImagesOri = to_var(images[:,:,:,256:]).contiguous()

            tmpRst = blurryImagesOri

            ##########   SPAT

            for Scale in range(0,2):

                ##########   SPAT
                
                inputblurry = torch.cat((blurryImagesOri,tmpRst),1)
                
                tmpRst = NET[Scale](inputblurry) + blurryImagesOri

                if Scale == 0:
                    NET_loss[0] = ae_criterion(tmpRst,GTImagesOri)

                    NET_loss[1] = ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0])
                  
                    temp_loss = NET_loss[0] + NET_loss[1] / 32500

                    NET[Scale].zero_grad()
                    temp_loss.backward()

                    net_optimizer[Scale].step()

                if Scale == 1:
                    NET_loss[0] = ae_criterion(tmpRst,GTImagesOri)

                    NET_loss[1] = ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0])
                 
                    temp_loss = NET_loss[0] + NET_loss[1] / 32500

                    NET[Scale].zero_grad()
                    temp_loss.backward()   

                    net_optimizer[Scale].step()   


                totalLoss = NET_loss[0] + NET_loss[1] / 32500
                tmpRst = tmpRst.detach()


            if(i == math.ceil(2103/p.batchSize) - 1):
                outputList.append(tmpRst.data)
                
                
            
            AvgNETloss[0] = AvgNETloss[0] + torch.Tensor.item(NET_loss[0].data)
            AvgNETloss[1] = AvgNETloss[1] + torch.Tensor.item(NET_loss[1].data)

            Avgtotalloss[0] = Avgtotalloss[0] + torch.Tensor.item(totalLoss.data)
            
            finali = i + 1

            if (i + 1) % 1 == 0:
                olda = a
                a = time.perf_counter()
            
                print('E[%d/%d][%.2f%%]'
                      % (epoch, MaxEpoch, (i + 1) / (maxDataNum / batchSize / 100)))
       
        AvgNETloss[:] = [x / finali for x in AvgNETloss]
        Avgtotalloss[:] = [x / finali for x in Avgtotalloss]
        

        oldb = b
        b = time.perf_counter()      

        print('E[%d/%d] NET:'
              % (epoch, MaxEpoch),  end=" ")

        print('[%.6f] '
              'lr: %.6f, time: %.2f sec    '
              % (Avgtotalloss[0],learningRateNET[0],
                 (b - oldb)))
    
        # Save the trained parameters
        print('saving model...')
        for mdl in models:
            for i in range(0,scaleLevel):
                if(mdl[0][i] != None):
    
                    modelObj = mdl[0]
                    modelName = mdl[1]
                    optimizer = mdl[2]
    
                    torch.save({'model': modelObj[i].state_dict(), 'optim': optimizer[i].state_dict(), 'epoch': epoch + 1}, './data/model/' + modelName + str(i) + '.pkl')
        


    
