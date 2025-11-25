import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import yaml
from easydict import EasyDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, augmentations, train=True, train_ratio=0.8):
        self.image_root = image_root
        self.gt_root = gt_root
        self.samples   = [name for name in os.listdir(image_root) if name[0]!="."]
        if train==True:
            self.samples = self.samples[:int(len(self.samples) * train_ratio)]
        else:
            self.samples = self.samples[int(len(self.samples) * train_ratio):]
        self.transform = augmentations
        
        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.image_root+'/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if len(self.color2) != 0 and np.random.rand()<0.7:
            name2 = self.color2[idx%len(self.color2)]
        elif len(self.color1) != 0:
            name2 = self.color1[idx%len(self.color1)]
        else:
            name2 = name  
        image2 = cv2.imread(self.image_root+'/'+name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
        mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
        image = np.uint8((image-mean)/std*std2+mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask  = cv2.imread(self.gt_root+'/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair  = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'].unsqueeze(0)

    def __len__(self):
        return len(self.samples)

# def get_image_num(image_root):
def give_augmentations(config):
    augmentations = A.Compose([
            A.Normalize(),
            A.Resize(config.image_size, config.image_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            ToTensorV2()
        ])
    return augmentations

def get_dataloader(config, dataset_choose = 'CVC_ClinicDB'):
    if dataset_choose == 'CVC_ClinicDB':
        config = config.dataset.CVC_ClinicDB
        data_root = config.data_root
        image_root = data_root + '/' + 'Original'
        gt_root = data_root + '/' + 'GroundTruth'
    else:
        config = config.dataset.Kvasir_SEG
        data_root = config.data_root
        image_root = data_root + '/' + 'images'
        gt_root = data_root + '/' + 'masks'
    augmentation = give_augmentations(config)
    train_dataset = PolypDataset(image_root, gt_root, augmentation, train=True, train_ratio=config.train_ratio)
    test_dataset = PolypDataset(image_root, gt_root, augmentation, train=False, train_ratio=config.train_ratio)
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True)
    return train_loader, test_loader

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
                                 ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



if __name__ == '__main__':
    image_root = '/workspace/Encvis/Code/CVC-ClinicDB/Original'
    gt_root = '/workspace/Encvis/Code/CVC-ClinicDB/Ground_Truth'
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    train_loader,test_loader = get_dataloader(config)   
    train_num = 0
    for i, image_batch in enumerate(train_loader):
        print(image_batch[0].size())
        print(image_batch[1].size())
        train_num += 1
    test_num = 0
    for i, image_batch in enumerate(test_loader):
        print(image_batch[0].size())
        print(image_batch[1].size())
        test_num += 1
    print(train_num)
    print(test_num)
    print(train_num+test_num)
    
    