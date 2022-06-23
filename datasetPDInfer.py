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

banned_scenes = ['scene_000100','scene_000002','scene_000008','scene_000012','scene_000018','scene_000029',
'scene_000038','scene_000040','scene_000043','scene_000044','scene_000049','scene_000050','scene_000053','scene_000063',
'scene_000079','scene_000090','scene_000094','scene_000103','scene_000106','scene_000111','scene_000112',
'scene_000124','scene_000125','scene_000127','scene_000148','scene_000159','scene_000166','scene_000169',
'scene_000170','scene_000171','scene_000187', 'scene_000191','scene_000200','scene_000202','scene_000217',
'scene_000218','scene_000225','scene_000229','scene_000232','scene_000236','scene_000237','scene_000245',
'scene_000249'
]

class PDDataset(Dataset):
    def __init__(self, split='train', idx = 0, root = None):
        super(PDDataset, self).__init__()
        
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        # self.files = self.files[:154]
        self.files.pop(77)
        # self.files = self.files[:10]
        if split == 'train':
            self.files = self.files[1:]
        else:
            self.files = self.files[split:split+1]

        self.real_files = []
        self.mask_files = []
        for f in self.files:
            if f in banned_scenes:
                continue
            for i in [1,5,6,7,8,9]:
                if os.path.exists(os.path.join(self.root_dir,f+'/rgb/camera_0{}'.format(i))):
                    self.real_files.append(f+'/rgb/camera_0{}'.format(i))
                    self.mask_files.append(f+'/moving_masks/camera_0{}'.format(i))
        self.img_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.path = self.real_files[idx]
        self.mask_path = self.mask_files[idx]

    def __getitem__(self, index):
        path = self.path 
        mask_path = self.mask_path
        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_images.sort()
        # all_masks = os.listdir(os.path.join(self.root_dir,mask_path))
        # all_masks.sort()
        # rand_id = random.randint(0,190)
        # rand_id = random.randint(0,190)
        idd = index 
        image = cv2.imread(os.path.join(os.path.join(self.root_dir, path),all_images[idd]))
        mask = cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path),all_images[idd]),-1)
        downsampling_ratio = 0.58
        crop = 158
        width = int(math.ceil(image.shape[1] * downsampling_ratio))
        height = int(math.ceil(image.shape[0] * downsampling_ratio))
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
        image = image[crop:, :, :]
        image = torch.Tensor(image).float()
        image = image / 255.0
        image = image.permute(2,0,1)
        image = self.img_transform(image)
        sample = {'image': image}
        return sample
            
    
    def __len__(self):
        return len(self.real_files)
