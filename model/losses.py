import torch
import torch.nn as nn
import pdb
import math
import wandb


import matplotlib.pyplot as plt

import torch.nn.functional as F

class MultiLoss (nn.Module):
    
    def __init__(self, *args, **kw):
        nn.Module.__init__(self)
        assert len(args) % 2 == 0
        self.weights = []
        self.losses = []
        self.cfg = kw
        for i in range(len(args)//2):
            weight = float(args[2*i+0])
            loss = eval(args[2*i+1])
            self.weights.append(weight)
            self.losses.append(loss)

    def forward(self, **kw):
        cum_loss = 0
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses),1):
            val = loss_func()(**{**kw, **self.cfg})
            cum_loss += weight * val
            if kw['metadata']['wandb_log']: wandb.log({'train/{}'.format(loss_func.__name__): val}, step=kw['step'])
            
        if kw['metadata']['wandb_log']: wandb.log({'train/Loss': float(cum_loss)}, step=kw['step'])
        return cum_loss
    
class PeakyLoss(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        loss = (self.forward_each(src_feats) + self.forward_each(trg_feats)) / 2
        return loss
        
    def forward_each(self, feats):
        local_mean = feats.mean(dim=1)
        local_max = feats.max(dim=1).values
        return 1 - (local_max - local_mean).mean()
    
class EntropyLoss(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        src_entropy, trg_entropy = (-src_feats * torch.log(src_feats + 1e-30)).mean(), (-trg_feats * torch.log(trg_feats + 1e-30)).mean()
        loss = src_entropy + trg_entropy 
        return loss
    
class KLLoss(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        scale = kw['scale'][0]
        bsz, dim, H, W = src_feats.shape

        shift_step = math.floor(math.log2(scale) / math.log2(kw['scale_factor']) * dim)
        src_feats = src_feats[:, -shift_step:, :, :] if shift_step < 0 else src_feats[:, :dim-shift_step, :, :]
        trg_feats = trg_feats[:, :dim+shift_step, :, :] if shift_step < 0 else trg_feats[:, shift_step:, :, :]
        overlap_dim = len(src_feats[0, :, 0, 0])
        src_feats = F.normalize(src_feats, p=1, dim=1)
        trg_feats = F.normalize(trg_feats, p=1, dim=1)
        loss = (src_feats * (torch.log2(src_feats / trg_feats + 1e-30))).mean()
        
        if kw['metadata']['wandb_log']: 
            if kw['step'] % kw['metadata']['plot_train_every'] == 0:
                center_src_feat, center_trg_feat = src_feats[0, :, H//2, W//2], trg_feats[0, :, H//2, W//2]
                wandb.log({'train/feature_src_distribution': wandb.Histogram(center_src_feat.detach().cpu().numpy())}, step=kw['step'])
                wandb.log({'train/feature_trg_distribution': wandb.Histogram(center_trg_feat.detach().cpu().numpy())}, step=kw['step'])
        
        return loss
    
    
class CELoss(nn.Module):
        
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        scale = kw['scale'][0]
        bsz, dim, H, W = src_feats.shape

        shift_step = math.floor(math.log2(scale) / math.log2(kw['scale_factor']) * dim)
        src_feats = src_feats[:, -shift_step:, :, :] if shift_step < 0 else src_feats[:, :dim-shift_step, :, :]
        trg_feats = trg_feats[:, :dim+shift_step, :, :] if shift_step < 0 else trg_feats[:, shift_step:, :, :]
        overlap_dim = len(src_feats[0, :, 0, 0])
        src_feats = F.normalize(src_feats, p=1, dim=1)
        trg_feats = F.normalize(trg_feats, p=1, dim=1)
        loss = -(src_feats * (torch.log2(trg_feats + 1e-30))).mean()
        
        if kw['metadata']['wandb_log']: 
            if kw['step'] % kw['metadata']['plot_train_every'] == 0:
                center_src_feat, center_trg_feat = src_feats[0, :, H//2, W//2], trg_feats[0, :, H//2, W//2]
                wandb.log({'train/feature_src_distribution': wandb.Histogram(center_src_feat.detach().cpu().numpy())}, step=kw['step'])
                wandb.log({'train/feature_trg_distribution': wandb.Histogram(center_trg_feat.detach().cpu().numpy())}, step=kw['step'])
        
        return loss
