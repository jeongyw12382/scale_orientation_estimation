import torch
import torch.nn as nn
import pdb
import math

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

    def forward(self, **variables):
        d = dict()
        cum_loss = 0
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses),1):
            val = loss_func()(**{**variables, **self.cfg})
            cum_loss += weight * val
            d['train/loss_'+loss_func.__name__] = float(val)
        d['train/loss'] = float(cum_loss)
        return cum_loss, d
    
class PeakyLoss(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        loss = self.forward_each(src_feats) + self.forward_each(trg_feats)
        return loss / 2
        
    def forward_each(self, feats):
        local_mean = feats.mean(dim=1)
        local_max = feats.max(dim=1).values
        return 1 - (local_max - local_mean).mean()
    
class CSELoss(nn.Module):
        
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, **kw):
        src_feats, trg_feats = kw['src_feat'], kw['trg_feat']
        scale = kw['scale'][0]
        bsz, dim, H, W = src_feats.shape
        shift_step = int(math.log2(scale) * dim)
        if shift_step < 0: 
            src_feats = F.normalize(src_feats[:, -shift_step:, :, :], p=2, dim=1)
            trg_feats = F.normalize(trg_feats[:, :dim+shift_step, :, :], p=2, dim=1)
        else:
            src_feats = F.normalize(src_feats[:, :dim-shift_step, :, :], p=2, dim=1)
            trg_feats = F.normalize(trg_feats[:, shift_step:, :, :], p=2, dim=1)
        loss = ((src_feats - trg_feats) ** 2).sum() / (dim*H*W)
        return loss