import os
import argparse
from datasetPDInfer import *
from modelinf import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch.nn as nn

import random

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

import wandb
from utils import adjusted_rand_index as ARI 
from utils import temporal_loss as t_loss 

from matplotlib.patches import Polygon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resolution = (548, 1123)

data_path = '/mnt/fsx/pd_v2'
test_set = PDDataset(split = 'test', root = data_path)

model = SlotAttentionAutoEncoder(resolution, 45, 3,  64).to(device)
model = nn.DataParallel(model)
# model.load_state_dict(torch.load('./model.ckpt')['model_state_dict'])
model.load_state_dict(torch.load('./tmp/f11-FT_epooch400.ckpt')['model_state_dict'])
model.eval()

for param in model.module.parameters():
    param.requires_grad = False

print('load finished!')

# sample = test_set[0]

# image = sample['image'].to(device)
# image = image.unsqueeze(0)

# print(image.shape)
h_state = None 
s_state = None 

cmap = plt.get_cmap('rainbow')
colors = [cmap(ii) for ii in np.linspace(0, 1, 45)]

for i in range(25):
    print(i)
    ims = [test_set[k]['image'] for k in range(i*5,i*5+5)]
    image = torch.stack(ims)
    image = image.to(device)
    cur_image = image.unsqueeze(0)
    recon_combined, masks, slots,h_state, s_state = model(cur_image,h_state, s_state)
    masks = masks.detach()
    index_mask = masks.argmax(dim = 2)
    index_mask = F.one_hot(index_mask,num_classes = 45)
    index_mask = index_mask.permute(0,1,4,2,3)
    # masks = masks * index_mask
    cur_image = F.interpolate(cur_image, (3,137,281))
    # h_state = h_state.detach()
    # s_state = s_state.detach()
    masks =   masks[0]
    cur_image = cur_image[0]
    for j in range(5):
        image_j = cur_image[j].permute(1,2,0).cpu().numpy()
        image_j = image_j * 0.5 + 0.5
        masks_j = masks[j]
        tk = 45
        masks_j = masks_j.cpu().numpy()
        image_j = image_j[:,:,-1::-1]       
        
        
        for seg in range(tk):
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(image_j, alpha = 1)
            msk = masks_j[seg]
            # threshold = 0
            threshold = (msk[msk > 0]).mean()*1.5

            e = (msk > threshold).astype('uint8')
            contour, hier = cv2.findContours(
                        e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cmax = None 
            for c in contour:
                if cmax is None:
                    cmax = c
                if len(c) > len(cmax):
                    cmax = c
            if cmax is not None:
                polygon = Polygon(
                        cmax.reshape((-1, 2)),
                        fill=True, facecolor=colors[seg],
                        edgecolor='r', linewidth=0.0,
                        alpha=0.5)
                ax.add_patch(polygon)
            fig.savefig('./infer/tune400/frame-{}-slot{}.png'.format(i*5+j,seg))
            plt.close(fig)
    # print(torch.cuda.memory_allocated(0))
    del recon_combined, masks, slots, image, image_j, masks_j, cur_image, index_mask
    # print(torch.cuda.memory_allocated(0))
    torch.cuda.empty_cache()
