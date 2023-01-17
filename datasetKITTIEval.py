import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from torchvision.io import read_video
import random
import cv2
import math

resolution = (1248,368)
dresolution = (312,92)

class KITTIDataset(Dataset):
    def __init__(self, split='train', root = None):
        super(KITTIDataset, self).__init__()
        self.resolution = resolution
        self.root_dir = root

        self.rgb_dir = os.path.join(self.root_dir,'rgb')
        self.instance_dir = os.path.join(self.root_dir,'instance')


        self.files = os.listdir(self.rgb_dir)
        self.files.sort()
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.files[index]
        image = cv2.imread(os.path.join(self.rgb_dir,path))
        mask = cv2.imread(os.path.join(self.instance_dir,path),-1)

        image = cv2.resize(image, resolution, interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dresolution, interpolation = cv2.INTER_NEAREST)

        mask = torch.Tensor(mask).long()
        image = torch.Tensor(image).float()

        image = image / 255.0
        image = image.permute(2,0,1)
        image = self.img_transform(image)
            
        sample = {'image': image, 'mask':mask}
        return sample
            
    
    def __len__(self):
        return len(self.files)
