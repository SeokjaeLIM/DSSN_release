    ################ Hyper Parameters ################
# data Set
dataSetName = 'blur_dataset'
dataSetMode = 'train'
dataPath = './data/GoPro_Large/'
labelPath = './data/list_attr_celeba.txt'

trainMode = 'blur'  # all : blur + blur_gamma , gamma: only gamma , blur : only blur
maxDataNum = 2103#4206#1352 #in fact, 4206
batchSize = 33

#Crop image size  => 178*356(blur) + 178*356(sharp) = 356*356(crop image)
MaxCropWidth = 256      
MinCropWidth = 256
MaxCropHeight = 256
MinCropHeight = 256

# ex) 20% => 0.2
Hue_PB = 0
Min_Hue_Factor = -0.2
Max_Hue_Factor = 0.2
Gamma_PB = 0
Min_Gamma_Factor = -1 #2^-k
Max_Gamma_Factor = 1

# model
#imageSize = 64
#image size => 255*255(blur) + 255*255(sharp) => 510*255(image)
imageWidth = 1280
imageHeight = 720
NGF = 72
NDF = 72
scaleLevel = 2


# train
MaxEpoch = 3600
learningRateE = [0.0001,0.0001]#,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003]
learningRateNET = [0.0001,0.0001]#,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003]
trainRange = 0

# save
numberSaveImage = 20

############################################
