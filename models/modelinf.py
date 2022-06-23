import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

import resnet
from ConvGRU import *
from math import ceil 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).to(device)
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim)).to(device)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        n_s = self.num_slots 
        
        mu = self.slots_mu.expand(1, n_s, -1)
        sigma = self.slots_sigma.expand(1, n_s, -1)
        slots = torch.normal(mu, sigma)

        slots = slots.contiguous()
        self.register_buffer("slots", slots)

    def get_attention(self, slots, inputs):
        slots_prev = slots
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        # updates = torch.einsum('bjd,bij->bid', v, attn)

        # slots = self.gru(
        #     updates.reshape(-1, d),
        #     slots_prev.reshape(-1, d)
        # )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots = None):
        b, nf,  n, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b,-1,-1)
        slots, _ = self.get_attention(slots, inputs[:,0,:,:])
        for f in range(nf):
            cur_slots, cur_attn = self.get_attention(slots,inputs[:,f,:,:])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3)
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1,0,2,3)
        return slots_out, attns

    def infer(self, inputs, num_slots = None, slot_state = None):
        b, nf,  n, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b,-1,-1)
        if slot_state is None:
            slots, _ = self.get_attention(slots, inputs[:,0,:,:])
        else:
            slots = slot_state
        for f in range(nf):
            cur_slots, cur_attn = self.get_attention(slots,inputs[:,f,:,:])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3)
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1,0,2,3)
        return slots_out, attns, slots.detach()


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1).to(device)


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.resolution = resolution
        self.hid_dim = hid_dim
        self.resnet = resnet.resnet18(pretrained=False)
        self.convGRU = ConvGRU(input_size=(ceil(resolution[0]/4), ceil(resolution[1]/4)),
                input_dim=128,
                hidden_dim=hid_dim,
                kernel_size=(3,3),
                num_layers=2,
                dtype = torch.cuda.FloatTensor,
                batch_first=True,
                bias = True,
                return_all_layers = False)
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        # self.register_buffer("grid", build_grid((resolution[0], resolution[1])))
        self.encoder_pos = SoftPositionEmbed(hid_dim, (ceil(resolution[0]/4), ceil(resolution[1]/4)))

    def forward(self, x):
        bs, n_frames, N, H, W = x.shape
        x = x.view(-1, N, H,W)
        x = self.resnet(x)
        x = x.view(bs, n_frames, x.shape[1], x.shape[2],x.shape[3])
        x,_ = self.convGRU(x)
        x = x[0]
        x = x.view(bs*n_frames, -1, x.shape[3], x.shape[4])
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

    def infer(self, x, h = None):
        bs, n_frames, N, H, W = x.shape
        x = x.view(-1, N, H,W)
        x = self.resnet(x)
        x = x.view(bs, n_frames, x.shape[1], x.shape[2],x.shape[3])
        x,_,h_out = self.convGRU.infer(x,h)
        x = x[0]
        x = x.view(bs*n_frames, -1, x.shape[3], x.shape[4])
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x, h_out

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=3)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=3)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=1)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 3, 3, stride=(1, 1), padding=1)
        self.resolution = resolution

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.pad(x, (4,4,4,4))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.dresolution = (ceil(resolution[0]/4), ceil(resolution[1]/4))
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.dresolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)
        self.LN = nn.LayerNorm([self.dresolution[0] * self.dresolution[1], 64])

    def forward(self, image, hidden_state=None, slot_state=None):
        bs, n_frames, _,_,_ = image.shape
        x,hidden_state_out = self.encoder_cnn.infer(image, hidden_state)  # CNN Backbone.
        x = self.LN(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].
        # x shape: [(bs*n_frames), width*height, input_size
        _, H, W = x.shape
        x = x.view(bs, n_frames, H, W)
        
        # Slot Attention module.
        slots_ori, attn_masks, slot_state_out = self.slot_attention.infer(x,slot_state)
        # print(slots_ori.shape, attn_masks.shape)
        attn_masks = attn_masks.reshape(bs*n_frames,self.num_slots, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1],self.dresolution[0], self.dresolution[1])
        a_mask = F.interpolate(attn_masks, (ceil(self.dresolution[0]/16), ceil(self.dresolution[1]/16)))
        a_mask = a_mask.unsqueeze(4).repeat(1,1,1,1,self.hid_dim)
        # exit()

        # # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots_ori.reshape(bs*n_frames, self.num_slots, -1)
        # slots = slots_ori.view(bs*n_frames, self.num_slots, -1)
        slots = slots.unsqueeze(2).unsqueeze(3)
        slots = slots.repeat((1, 1, ceil(self.dresolution[0]/16), ceil(self.dresolution[1]/16), 1))
        # print(attn_masks.sum(dim = 1).max())
        slots_combine = slots * a_mask
        slots_combine = slots_combine.sum(dim = 1)
        # print(slots_combine.shape)

        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        recons = self.decoder_cnn(slots_combine)
        recons = recons.permute(0,3,1,2)
        recons = recons.view(bs, n_frames, 3, self.dresolution[0],self.dresolution[1])
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].
        # print(recons.shape, attn_masks.shape, a_mask.shape,slots.shape)

        return recons, attn_masks.view(bs,n_frames, self.num_slots, attn_masks.shape[2], attn_masks.shape[3]),slots_ori,hidden_state_out,slot_state_out

