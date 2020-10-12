import torch
import tqdm
import math
import pdb

import torch.nn.functional as F

def shifting_validation(dataloader, model, iscuda, cfg):
    
    corr = 0
    cnt = 0
    for data in tqdm.tqdm(dataloader):
        feat = model(data, iscuda)
        bsz, dim, H, W = feat['src_feat'].shape
        sim_list = []
        best_shift = 0

        for i in range(dim-1, -1, -1):
            src_feat = F.normalize(feat['src_feat'][:, i:, :, :], p=2, dim=1)
            trg_feat = F.normalize(feat['trg_feat'][:, :dim-i, :, :], p=2, dim=1)
            sim = ((src_feat -  trg_feat) ** 2).sum(dim=[1,2,3]) / (H * W * (dim-i))
            sim_list.append(sim)
            
        for i in range(1, dim):
            src_feat = F.normalize(feat['src_feat'][:, :dim-i, :, :], p=2, dim=1)
            trg_feat = F.normalize(feat['trg_feat'][:, i:, :, :], p=2, dim=1)
            sim = ((src_feat -  trg_feat) ** 2).sum(dim=[1,2,3]) / (H * W)
            sim_list.append(sim) 
            
        sim_list = torch.stack(sim_list)
        best_shift_step = torch.argmax(sim_list, dim=0) -dim + 1
        
        scale = feat['scale'].cuda()
        shift_step = (torch.log2(scale) * dim).int()
        
        cnt += bsz
        corr += int((torch.abs(best_shift_step - shift_step) < cfg.validation.correct_threshold).sum())
        
    return corr / cnt
        
    
def argmax_validation(dataloader, model, iscuda, cfg):
    corr = 0
    cnt = 0
    for data in tqdm.tqdm(dataloader):
        feat = model(data, iscuda)
        bsz, dim, H, W = feat['src_feat'].shape
        sim_list = []
        best_shift = 0

        for i in range(dim-1, -1, -1):
            src_feat = F.normalize(feat['src_feat'][:, i:, :, :], p=2, dim=1)
            trg_feat = F.normalize(feat['trg_feat'][:, :dim-i, :, :], p=2, dim=1)
            sim = ((src_feat -  trg_feat) ** 2).sum(dim=[1,2,3]) / (H * W * (dim-i))
            sim_list.append(sim)
            
        for i in range(1, dim):
            src_feat = F.normalize(feat['src_feat'][:, :dim-i, :, :], p=2, dim=1)
            trg_feat = F.normalize(feat['trg_feat'][:, i:, :, :], p=2, dim=1)
            sim = ((src_feat -  trg_feat) ** 2).sum(dim=[1,2,3]) / (H * W)
            sim_list.append(sim) 
            
        sim_list = torch.stack(sim_list)
        best_shift_step = torch.argmax(sim_list, dim=0) -dim + 1
        
        scale = feat['scale'].cuda()
        shift_step = (torch.log2(scale) * dim).int()
        
        cnt += bsz
        corr += int((torch.abs(best_shift_step - shift_step) < cfg.validation.correct_threshold).sum())
        
    return corr / cnt