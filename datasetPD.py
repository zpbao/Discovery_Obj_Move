import os
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import math
import numpy as np

banned_scenes = ['scene_000100','scene_000002','scene_000008','scene_000012','scene_000018','scene_000029',
'scene_000038','scene_000040','scene_000043','scene_000044','scene_000049','scene_000050','scene_000053','scene_000063',
'scene_000079','scene_000090','scene_000094','scene_000100','scene_000103','scene_000106','scene_000111','scene_000112',
'scene_000124','scene_000125','scene_000127','scene_000148','scene_000159','scene_000166','scene_000169',
'scene_000170','scene_000171','scene_000187', 'scene_000191','scene_000200','scene_000202','scene_000217',
'scene_000218','scene_000225','scene_000229','scene_000232','scene_000236','scene_000237','scene_000245',
'scene_000249'
]

class PDDataset(Dataset):
    def __init__(self, split='train', root = None, supervision = 'moving'):
        super(PDDataset, self).__init__()       
        self.root_dir = root
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        if split == 'train':
            self.files = self.files[1:]
        elif split == 'eval':
            self.files = self.files[0:1]
        else:
            self.files = self.files
        self.annotation = None 
        if supervision == 'moving':
            self.annotation = 'moving_masks'
        elif supervision == 'all':
            self.annotation = 'ari_masks'
        elif supervision == 'est':
            self.annotation = 'raft_seg'
        else:
            raise ValueError("Need to choose from moving masks, all masks, or estimated masks. Or revise the code for customized setting.")
        self.real_files = []
        self.mask_files = []
        for f in self.files:
            if f in banned_scenes:
                continue
            for i in [1,5,6,7,8,9]:
                if os.path.exists(os.path.join(self.root_dir,f+'/rgb/camera_0{}'.format(i))):
                    self.real_files.append(f+'/rgb/camera_0{}'.format(i))
                    self.mask_files.append(f+'/{}/camera_0{}'.format(self.annotation, i))
        self.img_transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        path = self.real_files[index]
        mask_path = self.mask_files[index]
        all_images = os.listdir(os.path.join(self.root_dir,path))
        all_images.sort()
        rand_id = random.randint(0,190)
        real_idx = [rand_id + j for j in range(5)]
        ims = []
        masks = []
        for idd in real_idx:
            mapping = {0:0}
            mapping_count = 1
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
            for value in values:
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
