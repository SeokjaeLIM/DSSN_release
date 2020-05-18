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
# VERSION
version = 'final'
subversion = '1'

# data Set
dataSetName = p.dataSetName
dataSetMode = p.dataSetMode
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

############################################

############################################
############################################
print("")
print("          _____  ______ ____  _     _    _ _____  ______ _____  ")
print("         |  __ \\|  ____|  _ \\| |   | |  | |  __ \\|  ____|  __ \\ ")
print("         | |  | | |__  | |_) | |   | |  | | |__) | |__  | |  | | ")
print("         | |  | |  __| |  _ <| |   | |  | |  _  /|  __| | |  | | ")
print("         | |__| | |____| |_) | |___| |__| | | \\ \\| |____| |__| |")
print("         |_____/|______|____/|______\\____/|_|  \\_\\______|_____/ ")                       
print("")
print("TestGAN")
print("main Version : " + version)
print("sub Version : " + subversion)
print("dataloader Version : " + dl.version)
print("VisionOP Version : " + VisionOP.version)
print("model Version : " + model.version)
print("")
############################################
############################################

MODE = sys.argv[1]

if(MODE == 't' or MODE == 'to'):
    data_loader = get_loader(dataPath + 'test/',
                                labelPath,
                                MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,imageHeight,imageWidth,
                                batchSize,
                                Hue_PB,Max_Hue_Factor,Min_Hue_Factor,
                                Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor,
                                dataSetName,'test',trainMode)
elif(MODE == 'y' or MODE == 'n'):
    data_loader = get_loader(dataPath + 'train/',
                                labelPath,
                                MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,imageHeight,imageWidth,
                                batchSize,
                                Hue_PB,Max_Hue_Factor,Min_Hue_Factor,
                                Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor,
                                dataSetName,'train',trainMode)



if not os.path.exists('data/' + version):
    os.makedirs('./data/' + version)
if not os.path.exists('./data/' + version + '/model'):
    os.makedirs('./data/' + version + '/model')
if not os.path.exists('./data/' + version + '/eval'):
    os.makedirs('./data/' + version + '/eval')
if not os.path.exists('./data/' + version + '/log'):
    os.makedirs('./data/' + version + '/log')
    f = open('./data/'+version+'/log/loss.csv', 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(["Epoch","LR","NETLoss"])
    f.close() 
    
copyfile('./' + sys.argv[0], './data/' + version + '/model/' + sys.argv[0])
copyfile('./data_loader.py', './data/' + version + '/model/data_loader.py')
copyfile('./VisionOP.py', './data/' + version + '/model/VisionOP.py')
copyfile('./model.py', './data/' + version + '/model/model.py')
copyfile('./param.py', './data/' + version + '/model/param.py')


# init model

NET = []
NET2 = []
Ec = []
Dc = []
net_optimizer = []
net2_optimizer = []
ec_optimizer = []
dc_optimizer = []


#SPAT
for i in range(0,scaleLevel):  
    NET.append(model.TOG_class())
    #Ec.append(model.Encoder_class())
    #Dc.append(model.Decoder_class())
    
    net_optimizer.append(AdamW(NET[i].parameters(), lr=learningRateNET[i], weight_decay=0.00008))
    #ec_optimizer.append(AdamW(Ec[i].parameters(), lr=learningRateE[i], weight_decay=0.000015))
    #dc_optimizer.append(AdamW(Dc[i].parameters(), lr=learningRateE[i], weight_decay=0.000015))


    



#models = (Ec,'Encoder',ec_optimizer),(Dc,'Decoder',dc_optimizer),(NET,'NET',net_optimizer))
models = [(NET,'NET',net_optimizer)]
PSNR = model.PSNR()
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
                
                checkpoint = torch.load('./data/' + version + '/'  + modelName + str(i) + '.pkl')
                modelObj[i].load_state_dict(checkpoint['model'],strict=False)

                if(i==0):
                    startEpoch = checkpoint['epoch']

                preTrainedDict = torch.load('./data/' + version + '/'  + modelName + str(i) + '.pkl')
                optimDict = optimizer[i].state_dict()
                preTrainedDict = {k: v for k, v in preTrainedDict.items() if k in optimDict}

                optimDict.update(optimDict)
                
                optimizer[i].load_state_dict(optimDict)
                #optimizer[i].load_state_dict(checkpoint['optim'])
                for param_group in optimizer[i].param_groups: param_group['lr'] = learningRateNET[i]
    
                paramSize = 0
                for parameter in modelObj[i].parameters():
                    paramSize = paramSize + np.prod(np.array(parameter.size()))
                print(modelName + str(i) + ' : ' + str(paramSize))        
    
            
            if (MODE == 't'):
                modelObj[i].eval()
            else:
                modelObj[i].train()
            
            #modelObj[i].train()

print("All model loaded.")
       


if(MODE == 't'):



    torch.set_grad_enabled(False)
    SSIM = pytorch_msssim.SSIM()
    Loss1 = 0
    LossMSE = 0
    LossSSIM = 0
    Loss2 = 0
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
            for Scale in range(0,1):
                
                
                wendy = torch.cat((blurryImagesOri[:,:,],tmpRst[:,:,:,:]),1)
            
                tmpRst =  NET[Scale](wendy) + blurryImagesOri

            oldw = w
            w = time.perf_counter()
            a = w - oldw

                
                ##########   SPAT
                
            outputList.append(tmpRst.data)


            GTImages = GTImagesOri.cuda()
            blurryImages = blurryImagesOri.cuda()
            
            output = torch.cat(outputList,1)
  
            TotalImage = TotalImage + batchSize
    
            blurryImages = denorm(blurryImages)
            GTImages = denorm(GTImages)
            DeblurImages = denorm(output)
            

            save_image(DeblurImages.data, './data/' + version + '/eval/test-%d.png' % (i + 1))
            #save_image(GTImages.data, './data/' + version + '/eval/gt-%d.png' % (i + 1))
            #save_image(blurryImages.data, './data/' + version + '/eval/blur-%d.png' % (i + 1))
      


        
        
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
        AvgInputloss = [torch.zeros(1)]*256
        Avgtotalloss = [torch.zeros(1)]*256
        
        outputList = []
    
        for i, (images, _) in enumerate(data_loader):

            batchSize = images.size(0)
            
            NET_loss = [torch.zeros(1)]*256
            Input_loss = [torch.zeros(1)]*256

            blurryImagesOri = to_var(images[:,:,:,0:imageWidth]).contiguous()
        
            with torch.no_grad():
                GTImagesOri = to_var(images[:,:,:,imageWidth:]).contiguous()

            tmpRst = blurryImagesOri

            ##########   SPAT

            for Scale in range(0,scaleLevel):

                ##########   SPAT
                
                wendy = torch.cat((blurryImagesOri,tmpRst),1)
                
                tmpRst = NET[Scale](wendy) + blurryImagesOri

                if Scale == 0:
                    NET_loss[0] = ae_criterion(tmpRst,GTImagesOri)
                    Input_loss[0] = (ae_criterion(tmpRst,GTImagesOri) - ae_criterion(blurryImagesOri,GTImagesOri))*100 / ae_criterion(blurryImagesOri,GTImagesOri)

                    NET_loss[1] = ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0])
                    Input_loss[1] = (ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0]) - ae_criterion(VisionOP.polarFFT(blurryImagesOri)[0],VisionOP.polarFFT(GTImagesOri)[0]))*100 / ae_criterion(VisionOP.polarFFT(blurryImagesOri)[0],VisionOP.polarFFT(GTImagesOri)[0])

                    temp_loss = NET_loss[0] * 32500 + NET_loss[1]

                    NET[Scale].zero_grad()
                    temp_loss.backward()

                    net_optimizer[Scale].step()

                if Scale == 1:
                    NET_loss[0] = ae_criterion(tmpRst,GTImagesOri)
                    Input_loss[0] = (ae_criterion(tmpRst,GTImagesOri) - ae_criterion(blurryImagesOri,GTImagesOri))*100 / ae_criterion(blurryImagesOri,GTImagesOri)

                    NET_loss[1] = ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0])
                    Input_loss[1] = (ae_criterion(VisionOP.polarFFT(tmpRst)[0],VisionOP.polarFFT(GTImagesOri)[0]) - ae_criterion(VisionOP.polarFFT(blurryImagesOri)[0],VisionOP.polarFFT(GTImagesOri)[0]))*100 / ae_criterion(VisionOP.polarFFT(blurryImagesOri)[0],VisionOP.polarFFT(GTImagesOri)[0])

                    temp_loss = NET_loss[0] * 32500 + NET_loss[1]

                    NET[Scale].zero_grad()
                    temp_loss.backward()   

                    

                    net_optimizer[Scale].step()   


                totalLoss = NET_loss[0] * 32500 + NET_loss[1]
                tmpRst = tmpRst.detach()


            if(i == math.ceil(2103/p.batchSize) - 1):
                outputList.append(tmpRst.data)
                
                
            
            AvgNETloss[0] = AvgNETloss[0] + torch.Tensor.item(NET_loss[0].data)
            AvgInputloss[0] = AvgInputloss[0] + torch.Tensor.item(Input_loss[0].data)

            AvgNETloss[1] = AvgNETloss[1] + torch.Tensor.item(NET_loss[1].data)
            AvgInputloss[1] = AvgInputloss[1] + torch.Tensor.item(Input_loss[1].data)

            #totalLoss = Input_loss[0] * 1.2 + Input_loss[1]
            Avgtotalloss[0] = Avgtotalloss[0] + torch.Tensor.item(totalLoss.data)
            


                


            finali = i + 1

            if (i + 1) % 1 == 0:
                olda = a
                a = time.perf_counter()

            
                print('E[%d/%d][%.2f%%] NET:'
                      % (epoch, MaxEpoch, (i + 1) / (maxDataNum / batchSize / 100)),  end=" ")

                for j in range(0,2):
                    print('%.5f(%.5f)' % (AvgInputloss[j]/finali,AvgNETloss[j]/finali), end = " ")

                print('[%.5f] '
                      'lr: %.6f, time: %.2f sec    '
                      % (Avgtotalloss[0]/finali,learningRateNET[0],
                         (a - olda)), end="\r")


        

        
        AvgNETloss[:] = [x / finali for x in AvgNETloss]
        AvgInputloss[:] = [x / finali for x in AvgInputloss]
        Avgtotalloss[:] = [x / finali for x in Avgtotalloss]
        

        oldb = b
        b = time.perf_counter()      


        print('E[%d/%d] NET:'
              % (epoch, MaxEpoch),  end=" ")

        for j in range(0,2):
            print('%.6f(%.6f)' % (AvgInputloss[j],AvgNETloss[j]), end = " ")

        print('[%.6f] '
              'lr: %.6f, time: %.2f sec    '
              % (Avgtotalloss[0],learningRateNET[0],
                 (b - oldb)))
    


        print('saving deblurred image...')
    
        # Save sampled images
        blurry_images = denorm(blurryImagesOri.view(blurryImagesOri.size(0), 3, imageHeight, imageWidth)).cuda(0)

        images = denorm(GTImagesOri.view(blurryImagesOri.size(0), 3, imageHeight, imageWidth)).cuda(0)

        fake_images = torch.zeros(outputList[0].size(0), 3, imageHeight, 1).cuda(0)
        for iimg in outputList:
            fake_images = torch.cat((fake_images,denorm(iimg.view(iimg.size(0), 3, imageHeight, imageWidth)).cuda(0)),3)
        
    
        save_image(torch.cat((blurry_images.data,
                                fake_images.data,
                                images.data
                                ),3), './data/'+version+'/deblured_images-%d.png' % (epoch + 1))
        
        if (MODE != 'to'):
    
            print('saving log file...')
        
            # Save loss log
            f = open('./data/'+version+'/log/loss.csv', 'a', encoding='utf-8')
            wr = csv.writer(f)
    
            wr.writerow([epoch,learningRateNET[0]
            ,torch.Tensor.item(AvgNETloss[0])
            ,torch.Tensor.item(AvgInputloss[0])
            ,torch.Tensor.item(AvgNETloss[1])
            ,torch.Tensor.item(AvgInputloss[1])])
            f.close()    
        



        # Save the trained parameters
            print('saving model...')
            for mdl in models:
                for i in range(0,scaleLevel):
                    if(mdl[0][i] != None):
        
                        modelObj = mdl[0]
                        modelName = mdl[1]
                        optimizer = mdl[2]
        
                        torch.save({'model': modelObj[i].state_dict(), 'optim': optimizer[i].state_dict(), 'epoch': epoch + 1}, './data/'+version+'/' + modelName + str(i) + '.pkl')
        
                        if epoch % 100 == 0:
                            torch.save({'model': modelObj[i].state_dict(), 'optim': optimizer[i].state_dict(), 'epoch': epoch + 1}, './data/'+version+'/model/'+ modelName + str(i) +'-%d.pkl' % (epoch + 1))


    
