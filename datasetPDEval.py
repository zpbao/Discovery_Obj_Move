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

class PDDataset(Dataset):
    def __init__(self, split='train', root = None):
        super(PDDataset, self).__init__()
        
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()

        self.real_files = []
        self.mask_files = []
        for f in self.files:
            for i in [1,5,6,7,8,9]:
                if os.path.exists(os.path.join(self.root_dir,f+'/rgb/camera_0{}'.format(i))):
                    self.real_files.append(f+'/rgb/camera_0{}'.format(i))
                    self.mask_files.append(f+'/ari_masks/camera_0{}'.format(i))
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.real_files[index]
        mask_path = self.mask_files[index]

        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_images.sort()
        rand_id = 0
        real_idx = [rand_id + 1*j for j in range(200)]
        ims = []
        masks = []
        mapping = {0:0}
        mapping_count = 1
        for idd in real_idx:
            image = cv2.imread(os.path.join(os.path.join(self.root_dir, path),all_images[idd]))
            mask = cv2.imread(os.path.join(os.path.join(self.root_dir, mask_path),all_images[idd]),-1)
            
            downsampling_ratio = 0.5
            crop = 128
            width = int(math.ceil(image.shape[1] * downsampling_ratio))
            height = int(math.ceil(image.shape[0] * downsampling_ratio))
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
            image = image[crop:, :, :]
            mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            mask = mask[crop:,:]

            values, indices, counts = np.unique(mask, return_inverse=True, return_counts=True)
            for i in range(len(values)):
                if values[i] not in mapping:
                    if counts[i] > 500:
                        mapping[values[i]] = mapping_count
                        mapping_count += 1
            cur_mapping = []
            for i in range(len(values)):
                value = values[i]
                if value not in mapping:
                    cur_mapping.append(0)
                else:
                    cur_mapping.append(mapping[value])
            cur_mapping = np.array(cur_mapping)
            _h, _w = mask.shape
            mask = cur_mapping[indices].reshape((_h, _w))

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
