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


class KITTIDataset(Dataset):
    def __init__(self, split='train', root = None, resolution = (1248,368)):
        super(KITTIDataset, self).__init__()
        self.resolution = (resolution[1],resolution[0])
        self.dresolution = (resolution[1]//4, resolution[0]//4)
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        self.files = self.files[:151]
        if split == 'train':
            self.files = self.files[5:]
        else:
            self.files = self.files[0:5]
        self.real_files = []
        self.mask_files = []
        self.flow_files = []
        self.depth_files = []
        for f in self.files:
            for i in ['image_02','image_03']:
                if os.path.exists(os.path.join(self.root_dir,f+'/{}/'.format(i))):
                    self.real_files.append(f+'/{}/data'.format(i))
                    self.mask_files.append(f+'/{}/raft_seg'.format(i))
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.real_files[index]
        mask_path = self.mask_files[index]

        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_images.sort()
        N = len(all_images)
        rand_id = random.randint(0,N-10)
        # rand_id = 0
        real_idx = [rand_id + j for j in range(5)]
        ims = []
        masks = []
        for idd in real_idx:
            image = cv2.imread(os.path.join(os.path.join(self.root_dir, path),all_images[idd]))
            mask = cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path),all_images[idd]),-1)

            image = cv2.resize(image, self.resolution, interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.dresolution, interpolation = cv2.INTER_NEAREST)

            mask = torch.Tensor(mask).long()
            image = torch.Tensor(image).float()
            image = image / 255.0
            image = image.permute(2,0,1)
            image = self.img_transform(image)
            ims.append(image)
            masks.append(mask)
        ims = torch.stack(ims)
        masks = torch.stack(masks)
        sample = {'image': ims, 'mask':masks}
        return sample
            
    
    def __len__(self):
        return len(self.real_files)
