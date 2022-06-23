import os
import argparse
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import scipy.optimize
import torch.nn.functional as F
import numpy as np 

from dataset import *
from models.model import *
from utils import adjusted_rand_index as ARI 
from utils import temporal_loss as t_loss 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
# basic configurations
parser.add_argument('--model_dir', default='./tmp/', type=str, help='where to save models' )
parser.add_argument('--sample_dir', default = './samples/', type = str, help = 'where to save the plots')
parser.add_argument('--exp_name', default='', type=str, help='experiment name, used for model saving/plotting/wand ect' )
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_path', default = '/mnt/fsx/pd_v2', type = str, help = 'path of PD dataset')
parser.add_argument('--supervision',  default = 'moving', choices=['moving', 'all'], help = 'type of supervision, currently available: moving and all')
# model parameters
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_slots', default=45, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_epochs', default=500, type=int, help='number of workers for loading data')
parser.add_argument('--weight_mask', default = 1.0, type = float, help = 'weight for the mask loss')
parser.add_argument('--weight_temporal', default = 1.0, type = float, help = 'weight for the temporal loss')
# wandb
parser.add_argument('--wandb', default=False, type = bool)
parser.add_argument('--entity', default='zpbao', type = str, help = 'wandb name')



def main():
    opt = parser.parse_args()

    resolution = (548, 1123)

    if opt.wandb:
        import wandb
        wandb.init(project=opt.exp_name, entity="")
        wandb.config = {
        "learning_rate": opt.learning_rate,
        "mask": opt.weight_mask,
        "temporal": opt.weight_temporal,
        "Dataset": 'PD'
        }
    
    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    if not os.path.exists(opt.sample_dir):
        os.mkdir(opt.sample_dir)
    if not os.path.exists(os.path.join(opt.model_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.model_dir, opt.exp_name))
    if not os.path.exists(os.path.join(opt.sample_dir, opt.exp_name)):
        os.mkdir(os.path.join(opt.sample_dir, opt.exp_name))

    data_path = opt.data_path
    train_set = PDDataset(split = 'train', root = data_path, supervision = opt.supervision)
    test_set = PDDataset(split = 'eval', root = data_path, supervision = opt.supervision)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.hid_dim).to(device)
    model = nn.DataParallel(model)


    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    optimizer = optim.Adam(params, lr=opt.learning_rate)

    start = time.time()
    step = 0
    print('Model build finished!')
    for epoch in range(opt.num_epochs):
        model.train()

        total_loss = 0
        M_loss = 0
        T_loss = 0
        en_loss = 0

        for sample in tqdm(train_dataloader):
            step += 1
        
            if step < opt.warmup_steps:
                learning_rate = opt.learning_rate * (step / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                step / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample['image'].to(device)
            mask_gt = sample['mask'].to(device)

            recon_combined, masks, slots = model(image)
            image = F.interpolate(image, (3,137,281))
            mask_gt = F.interpolate(mask_gt.float(), (137,281)).long()
            # reconstruction loss
            loss = criterion(recon_combined, image)
            # mask loss
            loss_mask = 0.0
            mask_detach = masks.detach().flatten(3,4)
            mask_detach = mask_detach * 0.999 + 1e-8
            mask_detach = mask_detach.cpu().numpy()
            n_objects = mask_gt.max()
            mask_gt = F.one_hot(mask_gt, n_objects+1)
            mask_gt = mask_gt.permute(0,1,4,2,3)
            mask_gt_np = mask_gt.flatten(3,4)
            
            mask_gt_np = mask_gt_np.detach().cpu().numpy()


            scores = np.zeros((opt.batch_size, 5, opt.num_slots, n_objects+1))
            for i in range(opt.batch_size):
                for j in range(5):
                    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
                    scores[i,j] += cross_entropy_cur
            scores = scores.sum(axis = 1)
            for i in range(opt.batch_size):
                matches = scipy.optimize.linear_sum_assignment(-scores[i])
                id_slot, id_gt = matches 
                loss_mask += ( -torch.log(masks[i,:,id_slot,:,:]) * mask_gt[i,:,id_gt,:,:] - (1- mask_gt[i,:,id_gt,:,:]) * torch.log(1-masks[i,:,id_slot,:,:])).mean()
            time_loss = t_loss(slots)

            whole_loss = loss +  opt.weight_mask * loss_mask + opt.weight_temporal* time_loss

            optimizer.zero_grad()
            whole_loss.backward()
            optimizer.step()

            total_loss += loss.item()

            T_loss += time_loss.item()
            M_loss += loss_mask.item()

            entropy = -masks * torch.log(masks)
            entropy = entropy.detach()
            entropy = entropy.sum(dim = 2)
            entropy_loss = torch.mean(entropy)
            en_loss += entropy_loss.item()

            
            del recon_combined, masks, image, mask_gt, loss, whole_loss, entropy_loss, entropy, slots

        total_loss /= len(train_dataloader)
        M_loss /= len(train_dataloader)
        T_loss /= len(train_dataloader)
        en_loss /= len(train_dataloader)
        if opt.wandb:
            wandb.log({"recon_loss": total_loss, "mask_loss":M_loss, "Temporal_loss": T_loss, "entropy": en_loss})

        print ("Epoch: {}, Loss: {}, Loss_mask: {}, Loss_time: {}, Entropy: {}, Time: {}".format(epoch, total_loss, M_loss,T_loss,en_loss,
            datetime.timedelta(seconds=time.time() - start)))
        
        sample = test_set[0]
        image = sample['image'].to(device)
        image = image.unsqueeze(0)
        recon_combined, masks, slots = model(image)

        index_mask = masks.argmax(dim = 2)
        index_mask = F.one_hot(index_mask,num_classes = opt.num_slots)
        index_mask = index_mask.permute(0,1,4,2,3)
        masks = masks * index_mask

        image = image[0]
        image = F.interpolate(image, (137,281))
        recon_combined, masks =  recon_combined[0], masks[0]

        recon_combined = recon_combined.detach()
        masks = masks.detach()

        
        fig, ax = plt.subplots(math.ceil((opt.num_slots+2) / 10), 10, figsize=(45, 5 * math.ceil((opt.num_slots +2)/ 10)))
        for i in range(1):
            image_i = image[i]
            recon_combined_i = recon_combined[i]
            masks_i = masks[i].cpu().numpy()
            image_i = image_i.permute(1,2,0).cpu().numpy()
            image_i = image_i * 0.5 + 0.5
            recon_combined_i = recon_combined_i.permute(1,2,0)
            recon_combined_i = recon_combined_i.cpu().numpy()
            recon_combined_i = recon_combined_i * 0.5 + 0.5
            ax[i,0].imshow(image_i)
            ax[i,0].set_title('Image-f{}'.format(i))
            ax[i,1].imshow(recon_combined_i)
            ax[i,1].set_title('Recon.')
            for j in range(opt.num_slots):               
                ax[(j+2)//10,(j + 2)%10].imshow(image_i)
                ax[(j+2)//10,(j + 2)%10].imshow(masks_i[j], cmap = 'viridis', alpha = 0.6)
                ax[(j+2)//10,(j + 2)%10].set_title('Slot %s' % str(j + 1))
            for j in range(math.ceil((opt.num_slots+2) / 10) * 10):
                ax[(j)//10,(j)%10].grid(False)
                ax[(j)//10,(j)%10].axis('off')
        eval_name = os.path.join(opt.sample_dir,opt.exp_name,'epoch_{}.png'.format(epoch))
        fig.savefig(eval_name)
        plt.close(fig)
        del masks, recon_combined, image, slots
        
        if not epoch % 10:
            torch.save({
                'model_state_dict': model.state_dict(),
                }, os.path.join(opt.model_dir, opt.exp_name, 'epoch_{}.ckpt'.format(epoch))
                )


if __name__ == '__main__':
    main()
