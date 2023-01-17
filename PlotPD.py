from datasetPDEval import PDDataset
from models.model import *
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

resolution = (480, 968)

model_path = './PD_est_500.ckpt'
data_path = '/mnt/fsx/data/test_video'
test_set = PDDataset(split = 'test', root = data_path)

model = SlotAttentionAutoEncoder(resolution, 40, 64, 3).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_path)['model_state_dict'])

print('model load finished!')

for param in model.module.parameters():
    param.requires_grad = False

sample = test_set[0]
images = sample['image']

cmap = plt.get_cmap('rainbow')
colors = [cmap(ii) for ii in np.linspace(0, 1, 40)]

image = images[:10]

image = image.to(device)
image = image.unsqueeze(0)
recon_combined, masks, slots= model(image)
masks = masks.detach()
index_mask = masks.argmax(dim = 2)
index_mask = F.one_hot(index_mask,num_classes = 40)
index_mask = index_mask.permute(0,1,4,2,3)
masks = masks * index_mask
cur_image = F.interpolate(image, (3,120,242))
    
masks =   masks[0]
cur_image = cur_image[0]
for j in range(10):
    image_j = cur_image[j].permute(1,2,0).cpu().numpy()
    image_j = image_j * 0.5 + 0.5
    masks_j = masks[j]
    tk = 40
    masks_j = masks_j.cpu().numpy()
    image_j = image_j[:,:,-1::-1]       
    
    
    for seg in range(tk):
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(image_j, alpha = 1)
        msk = masks_j[seg]
        threshold = 0

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
        fig.savefig('./infer/frame-{}-slot{}.png'.format(j,seg))
        plt.close(fig)
