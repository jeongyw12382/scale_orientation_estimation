import torch
import torch.nn as nn
import pdb
import math

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
        if kw['metadata']['tbd_log']: kw['writer'].add_scalar('train/Loss', float(cum_loss), kw['step'])
        return cum_loss
    
class PeakyLoss(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        loss = (self.forward_each(src_feats) + self.forward_each(trg_feats)) / 2
        
        if kw['metadata']['tbd_log']: kw['writer'].add_scalar('train/PeakyLoss', loss.item(), kw['step'])
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
        if kw['metadata']['tbd_log']: kw['writer'].add_scalar('train/EntropyLoss', loss.item(), kw['step'])
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
        
        if kw['metadata']['tbd_log']: 
            kw['writer'].add_scalar('train/KLLoss', loss.item(), kw['step'])
            if kw['step'] % kw['metadata']['plot_train_every'] == 0:
                center_src_feat, center_trg_feat = src_feats[0, :, H//2, W//2], trg_feats[0, :, H//2, W//2]
                fig, ax = plt.subplots()
                ax.set(xlabel='scale', ylabel='feature', title='Features\' distribution')
                ax.plot(range(overlap_dim), center_src_feat.cpu().detach().numpy())
                ax.plot(range(overlap_dim), center_trg_feat.cpu().detach().numpy())
                ax.set_ylim([0,1])
                ax.grid()
                kw['writer'].add_figure('train/feature_distribution', fig, kw['step'])
                plt.close()
        
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
        
        if kw['metadata']['tbd_log']: 
            kw['writer'].add_scalar('train/CELoss', loss.item(), kw['step'])
            if kw['step'] % kw['metadata']['plot_train_every'] == 0:
                center_src_feat, center_trg_feat = src_feats[0, :, H//2, W//2], trg_feats[0, :, H//2, W//2]
                fig, ax = plt.subplots()
                ax.set(xlabel='scale', ylabel='feature', title='Features\' distribution')
                ax.plot(range(overlap_dim), center_src_feat.cpu().detach().numpy())
                ax.plot(range(overlap_dim), center_trg_feat.cpu().detach().numpy())
                ax.set_ylim([0,1])
                ax.grid()
                kw['writer'].add_figure('train/feature_distribution', fig, kw['step'])
                plt.close()
        
        return loss
