import os
import argparse
from dataset import *
from models.model import *
from tqdm import tqdm
import time
import datetime


from utils import adjusted_rand_index as ARI 
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--ckpt_path', default='./model.ckpt', type=str, help='pre-trained model path' )
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--test_path', default = '/mnt/fsx/pd_v2', type = str, help = 'path of PD test set')
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')


def main():
    opt = parser.parse_args()
    resolution = (548, 1123)
    ckpt_path = opt.ckpt_path
    root_path = opt.test_path
    test_set = PDDataset(split = 'test', root = opt.test_path)
    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    for param in model.module.parameters():
        param.requires_grad = False

    print('load finished!')

    aris = []
    for i in range(len(test_set)):
        sample = test_set[i]
        img = sample['image']
        msk = sample['mask']
        img = img.unsqueeze(0)
        img = img.to(device)
        msk = msk.to(device).unsqueeze(0).float()
        msk = F.interpolate(msk, (137, 281))
        msk = msk.long().squeeze(0)
        recon_combined, masks, slots = model(img)
        masks = masks[0]
        for j in range(5):
            gt_msk = msk[j]
            gt_msk = gt_msk.view(-1)
            idx = gt_msk>0
            gt_msk = gt_msk[idx]
            if len(gt_msk) == 0:
                continue 
            pred_msk = masks[j]
            pred_msk = pred_msk.view(45, -1)
            pred_msk = pred_msk[:,idx]
            gt_msk = F.one_hot(gt_msk)
            # gt_msk = gt_msk[:,1:]
            pred_msk = pred_msk.permute(1,0)
            ari = ARI(gt_msk.unsqueeze(0), pred_msk.unsqueeze(0))
            aris.append(ari)
    print('ARI Score:', sum(aris) / len(aris))

if __name__ == '__main__':
    main()



