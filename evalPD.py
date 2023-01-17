import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from datasetPDEval import PDDataset
from utils import adjusted_rand_index as ARI 

from models.model import *

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='./model.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--test_path', default = '/mnt/fsx/pd_v2', type = str, help = 'path of PD test set')
parser.add_argument('--num_slots', default=40, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')

resolution = (480, 968)

def main():
    opt = parser.parse_args()
    model_path = opt.ckpt_path
    data_path = opt.test_path
    test_set = PDDataset(split = 'test', root = data_path)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim, 3).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    print('model load finished!')

    for param in model.module.parameters():
        param.requires_grad = False


    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                shuffle=False, num_workers=4, drop_last=False)

    ARIs = []

    for sample in tqdm(test_dataloader):
        image = sample['image'].to(device)
        mask_gt = sample['mask']
        mask_gts = F.interpolate(mask_gt.float(), (120, 242)).long()

        for i in range(40):
            _, masks, _ = model(image[:,i*5:i*5+5,:,:,:])
            mask_gt = mask_gts[:,i*5:i*5+5,:,:]
            masks = masks.detach().cpu()

            gt_msk = mask_gt[0]
            pred_msk = masks[0]
            gt_msk = gt_msk.view(5,-1)
            pred_msk = pred_msk.view(5,opt.num_slots,-1).permute(1,0,2)
            gt_msk = gt_msk.view(-1)
            pred_msk = pred_msk.reshape(opt.num_slots,-1)

            pred_msk = pred_msk.permute(1,0)
            gt_msk = F.one_hot(gt_msk)
            _,n_cat = gt_msk.shape 
            if n_cat <= 2:
                continue
            gt_msk = gt_msk[:,1:]
            ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
            ARIs.append(ari)
        del image, mask_gt, masks
    print('final ARI:',sum(ARIs) / len(ARIs))

if __name__ == '__main__':
    main()