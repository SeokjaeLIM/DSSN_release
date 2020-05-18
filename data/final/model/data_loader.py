import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from PIL import Image

version='GAMMA v1.1'

class deblur_dataset(Dataset):
    def __init__(self,image_path,transform,mode,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,img_height,img_width,trainMode,Hue_PB,Max_Hue_Factor,Min_Hue_Factor,Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor):
        self.image_path = image_path
        self.transform = transform
        self.resize = transforms.Resize((img_height, img_width), interpolation=Image.ANTIALIAS)
        self.MaxCropWidth = MaxCropWidth
        self.MinCropWidth = MinCropWidth
        self.MaxCropHeight = MaxCropHeight
        self.MinCropHeight = MinCropHeight
        self.crop_width = 0
        self.crop_height = 0
        self.img_height = img_height
        self.img_width = img_width
        self.Hue_PB = Hue_PB
        self.Min_Hue_Factor = Min_Hue_Factor
        self.Max_Hue_Factor = Max_Hue_Factor
        self.Gamma_PB = Gamma_PB
        self.Min_Gamma_Factor = Min_Gamma_Factor
        self.Max_Gamma_Factor = Max_Gamma_Factor
        self.hue = 0
        self.trainMode = trainMode
        self.state = ['/blur/', '/blur_gamma/', '/sharp/']
        self.mode = mode
        self.num_data = []
        self.list = []
        self.Folder_list = []
        self.image_filenames = []
        self.Img_list = []
        self.Blur_image = []
        self.Sharp_image = []
        self.temp = []

        print('Start image processing')
        self.image_processing()
        print('Fineshed image processing')

        if self.mode == 'train':
            if self.trainMode == 'all':
                self.num_data = int(2*len(self.image_filenames)/3)
            elif self.trainMode == 'gamma':
                self.num_data = int(len(self.image_filenames) / 3)
            elif self.trainMode == 'blur':
                self.num_data = int(len(self.image_filenames) / 3)
            print('TrainSet loaded : %d images'%self.num_data)

        if self.mode == 'test':
            if self.trainMode == 'all':
                self.num_data = int(2 * len(self.image_filenames) / 3)
            elif self.trainMode == 'gamma':
                self.num_data = int(len(self.image_filenames) / 3)
            elif self.trainMode == 'blur':
                self.num_data = int(len(self.image_filenames) / 3)
            print('TestSet loaded : %d images'%self.num_data)
           

    def image_processing(self):
        self.list = os.listdir(self.image_path)
        self.NOF = len(self.list)

        for i in range(0,self.NOF):
            self.Folder_list = self.image_path + self.list[i] + self.state[0]

            self.Img_list = os.listdir(self.Folder_list)
            for j in range(0,len(self.Img_list)):
                self.temp.append(self.list[i] + self.state[0] + self.Img_list[j])

        self.temp = sorted(self.temp)
        self.image_filenames.extend(self.temp)
        self.temp = []

        for i in range(0, self.NOF):
            self.Folder_list = self.image_path + self.list[i] + self.state[1]

            self.Img_list = os.listdir(self.Folder_list)
            for j in range(0,len(self.Img_list)):
                self.temp.append(self.list[i] + self.state[1] + self.Img_list[j])

        self.temp = sorted(self.temp)
        self.image_filenames.extend(self.temp)
        self.temp = []

        for i in range(0, self.NOF):
            self.Folder_list = self.image_path + self.list[i] + self.state[2]

            self.Img_list = os.listdir(self.Folder_list)
            for j in range(0,len(self.Img_list)):
                self.temp.append(self.list[i] + self.state[2] + self.Img_list[j])

        self.temp = sorted(self.temp)
        self.image_filenames.extend(self.temp)
        self.temp = []


    def get_params(self,input1):

        self.w, self.h = input1.size

        self.crop_width = random.randint(self.MinCropWidth,self.MaxCropWidth)
        self.crop_height = random.randint(self.MinCropHeight, self.MaxCropHeight)
        
        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)

        return i,j
        
    def Random_Crop(self,input1,input2):

        self.i,self.j = self.get_params((input1))


        image1 = input1.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))

        image2 = input2.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))

        return image1,image2

    def Resize(self,input):

        return self.resize(input)

    def FLIP_LR(self,input1,input2):
        if random.random() > 0.5:
            input1 = input1.transpose(Image.FLIP_LEFT_RIGHT)
            input2 = input2.transpose(Image.FLIP_LEFT_RIGHT)

        return input1, input2

    def FLIP_UD(self,input1,input2):
        if random.random() > 0.5:
            input1 = input1.transpose(Image.FLIP_TOP_BOTTOM)
            input2 = input2.transpose(Image.FLIP_TOP_BOTTOM)

        return input1, input2

    def Adjust_hue(self,input1,input2):
        if random.random() < self.Hue_PB:
            self.hue = random.uniform(self.Min_Hue_Factor, self.Max_Hue_Factor)
            input1 = transforms.functional.adjust_hue(input1,self.hue)
            input2 = transforms.functional.adjust_hue(input2, self.hue)

        return input1,input2

    def Adjust_gamma(self,input1,input2):
        if random.random() < self.Gamma_PB:
            gamma = 2**random.uniform(self.Min_Gamma_Factor, self.Max_Gamma_Factor)
            input1 = transforms.functional.adjust_gamma(input1, gamma)
            input2 = transforms.functional.adjust_gamma(input2, gamma)

        return input1,input2

    def Divide_image(self,input1):
        self.w,self.h = input1.size

        self.w_divide = int((self.w-256) / 128) + 1
        self.h_divide = int((self.h-256) / 128) + 1

        if (self.w - 256) % 128 != 0:
            self.w_divide = self.w_divide + 1
        if (self.h - 256) % 128 != 0:
            self.h_divide = self.h_divide + 1


        self.check = 0

       # total_tensor = 0
        if (self.w_divide != 1):
            self.w_interval = (self.w - 256)/(self.w_divide-1)
        else:
            self.w_interval = 0
            
        if (self.h_divide != 1):    
            self.h_interval = (self.h - 256)/(self.h_divide-1)
        else:
            self.h_interval = 0
            

        for i in range(0,self.h_divide):
            for j in range(0,self.w_divide):
                self.divide_image = input1.crop((j * self.w_interval, i * self.h_interval, j * self.w_interval + 256,  i * self.h_interval + 256))

               # print(j * self.w_interval, i * self.h_interval)
                if self.w_divide * i + j == 0:
                    total_tensor = self.transform(self.divide_image)
                  #  print(total_tensor.size())
                else:
                    total_tensor = torch.cat((total_tensor,self.transform(self.divide_image)),2)
                  #  print(total_tensor.size())

        return total_tensor

    def __getitem__(self, index):

        image = Image.new('RGB', (self.img_width * 2, self.img_height))

        if self.mode == 'train':
            Labeling = self.label = 1

            if self.trainMode == 'all':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index%int(self.num_data/2) + self.num_data]))
            elif self.trainMode == 'gamma':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data + self.num_data]))
            elif self.trainMode == 'blur':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + 2 * self.num_data]))

            self.Blur_image,self.Sharp_image = self.Random_Crop(self.Blur_image,self.Sharp_image)

            #self.Blur_image = self.Resize(self.Blur_image)
            #self.Sharp_image = self.Resize(self.Sharp_image)

            #self.Blur_image, self.Sharp_image = self.FLIP_LR(self.Blur_image, self.Sharp_image)
            #self.Blur_image, self.Sharp_image = self.FLIP_UD(self.Blur_image, self.Sharp_image)

            #self.Blur_image, self.Sharp_image = self.Adjust_hue(self.Blur_image, self.Sharp_image)
            #self.Blur_image, self.Sharp_image = self.Adjust_gamma(self.Blur_image, self.Sharp_image)

            image.paste(self.Blur_image,(0,0))
            image.paste(self.Sharp_image,(self.img_width,0))

            return self.transform(image), torch.FloatTensor(Labeling)
            

        else:
            Labeling = self.label =  1
            
            if self.trainMode == 'all':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index%int(self.num_data/2) + self.num_data]))
            elif self.trainMode == 'gamma':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data + self.num_data]))
            elif self.trainMode == 'blur':
                self.Blur_image = Image.open(os.path.join(self.image_path, self.image_filenames[index]))
                self.Sharp_image = Image.open(os.path.join(self.image_path, self.image_filenames[index + 2 * self.num_data]))

            self.Blur_image, self.Sharp_image = self.Random_Crop(self.Blur_image, self.Sharp_image)

            #self.Blur_image = self.Resize(self.Blur_image)
            #self.Sharp_image = self.Resize(self.Sharp_image)

            ###total_tensor = self.Divide_image(self.Blur_image)Hue

            #print(image)

            #image.paste(self.Blur_image,(0,0))
            #image.paste(self.Sharp_image,(self.img_width,0))
            #image2.save('full_image %d.png' %(index))
            #print(self.image_filenames[index])
            image.paste(self.Blur_image,(0,0))
            image.paste(self.Sharp_image,(self.img_width,0))
            return self.transform(image), torch.FloatTensor(Labeling)
            
            ###return total_tensor,self.transform(self.Sharp_image),torch.FloatTensor(Labeling)
        
    def __len__(self):
        return self.num_data

class shadowDataset(Dataset):
    def __init__(self,image_path,transform,mode,MaxCropSize,MinCropSize,img_height,img_width):
        self.image_path = image_path
        self.transform = transform
        self.resize = transforms.Resize((img_height, img_width), interpolation=Image.ANTIALIAS)
        self.colorjitter = transforms.ColorJitter()
        self.rotation = transforms.RandomRotation(20)
        self.crop_width = MaxCropSize
        self.crop_height = MinCropSize
        self.img_height = img_height
        self.img_width = img_width
        self.state = ['train_A/', 'train_B/', 'train_C/']
        self.mode = mode
        self.num_data = []
        self.list = []
        self.Folder_list = []
        self.image_filenames = []
        self.Img_list = []

        print('Start image processing')
        self.image_processing()
        print('Fineshed image processing')
        if self.mode == 'train':
            self.num_data = int(len(self.image_filenames)/2)
        if self.mode == 'test':
            self.num_data = int(len(self.image_filenames) / 3)

    def image_processing(self):
        self.list = os.listdir(self.image_path)
        self.NOF = len(self.list)

        self.Folder_list = self.image_path + self.state[0]
        self.Img_list = os.listdir(self.Folder_list)
        for j in range(0, len(self.Img_list)):
            self.image_filenames.append(self.state[0] + self.Img_list[j])


        self.Folder_list = self.image_path + self.state[2]
        self.Img_list = os.listdir(self.Folder_list)
        for j in range(0, len(self.Img_list)):
            self.image_filenames.append(self.state[2] + self.Img_list[j])

    def __getitem__(self, index):
        if self.mode == 'train':
            Labeling = self.label = 1
            self.w,self.h = Image.open(os.path.join(self.image_path, self.image_filenames[index])).size
            self.i = random.randint(0, self.h - self.crop_height)
            self.j = random.randint(0, self.w - self.crop_width)
            image = Image.new('RGB', (self.img_width * 2,self.img_height))
            if random.random() > 0.5 :
                image.paste(self.resize((Image.open(os.path.join(self.image_path, self.image_filenames[index])).crop(
                    (self.j, self.i, self.w, self.h)).transpose(Image.FLIP_LEFT_RIGHT))), (0, 0))
                image.paste(
                    self.resize(Image.open(os.path.join(self.image_path, self.image_filenames[index + self.num_data])).crop(
                        (self.j, self.i, self.w, self.h)).transpose(Image.FLIP_LEFT_RIGHT)), (self.img_width, 0))
            else:
                image.paste(self.resize(Image.open(os.path.join(self.image_path, self.image_filenames[index])).crop(
                    (self.j, self.i, self.w, self.h))), (0, 0))
                image.paste(self.resize(Image.open(
                    os.path.join(self.image_path, self.image_filenames[index + self.num_data])).crop(
                    (self.j, self.i, self.w, self.h))), (self.img_width, 0))

        elif self.mode in ['test']:
            self.w, self.h = Image.open(os.path.join(self.image_path, self.image_filenames[index + 2*self.num_data])).size
            self.i = random.randint(0, self.h - self.crop_height)
            self.j = random.randint(0, self.w - self.crop_width)
            image = Image.open(os.path.join(self.image_path, self.image_filenames[index + 2*self.num_data])).crop(
                            (self.j, self.i, self.w, self.h))
      #  image.save('full_image %d.png' %(index))
        return self.transform(image),torch.FloatTensor(Labeling)
    def __len__(self):
        return self.num_data





class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}
        print('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print('Finished preprocessing dataset..!')
        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)
    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []
        lines = self.lines[2:]
        random.shuffle(lines)  # random shuffling
        for i, line in enumerate(lines):
            splits = line.split()
            filename = splits[0]
            values = splits[1:]
            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]
                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)
            if (i + 1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)
    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]
        return self.transform(image), torch.FloatTensor(label)
    def __len__(self):
        return self.num_data

def get_loader(image_path, metadata_path, 
                MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight, img_height,img_width,
                batch_size,
                Hue_PB,Max_Hue_Factor,Min_Hue_Factor,
                Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor,
                dataset='CelebA', mode='train',trainMode = 'blur'):
                
    """Build and return data loader."""
    if dataset == 'blur_dataset':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      
    elif dataset =='shadow':
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else :
        if mode == 'train':
            transform = transforms.Compose([
                transforms.CenterCrop((crop_height,crop_width)),
                transforms.Resize((img_height,img_width), interpolation=Image.ANTIALIAS),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop((crop_height,crop_width)),
                transforms.Scale((img_height,img_width), interpolation=Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, transform, mode)
    elif dataset == 'blur_dataset':
        dataset = deblur_dataset(image_path, transform, mode,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,img_height,img_width,trainMode,Hue_PB,Max_Hue_Factor,Min_Hue_Factor,Gamma_PB,Max_Gamma_Factor,Min_Gamma_Factor)

    elif dataset == 'RaFD':
        dataset = ImageFolder(image_path, transform)
    elif dataset == 'Mnist':
        dataset = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=True)
    elif dataset == 'shadow':
        dataset = shadowDataset(image_path, transform, mode, crop_width, crop_height, img_height, img_width)

    shuffle = False
    if mode == 'train':
        shuffle = True
        
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=2)
    return data_loader

