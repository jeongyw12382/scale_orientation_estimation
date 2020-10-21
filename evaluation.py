import torch
import tqdm
import math
import pdb

import torch.nn.functional as F

def argmax_validation(**kw_val):
    
    dataloader, model, iscuda, cfg = kw_val['dataloader'], kw_val['model'], kw_val['iscuda'], kw_val['cfg']
    
    corr_easy, corr_hard, corr_extreme = 0, 0, 0
    cnt = 0
    for data in tqdm.tqdm(dataloader):
        feat = model(data, iscuda)
        bsz, dim, H, W = feat['src_feat'].shape
        sim_list = []
        best_shift = 0

        src_feat = torch.argmax(feat['src_feat'][:, :, H//2, W//2], dim=1)
        trg_feat = torch.argmax(feat['trg_feat'][:, :, H//2, W//2], dim=1)
        predicted = trg_feat - src_feat

        scale = feat['scale'].cuda()
        shift_step = torch.floor(torch.log2(scale) / math.log2(cfg['dataset']['scale']['scale_factor']) * dim)
        
        cnt += bsz
        corr_easy += (torch.abs(predicted - shift_step) < cfg.validation.correct_threshold[2]).sum()
        corr_hard += (torch.abs(predicted - shift_step) < cfg.validation.correct_threshold[1]).sum()
        corr_extreme += (torch.abs(predicted - shift_step) < cfg.validation.correct_threshold[0]).sum()

    easy_val, hard_val, ext_val = corr_easy.item() / cnt, corr_hard.item() / cnt, corr_extreme.item() / cnt
    
    if cfg.metadata.tbd_log: 
        writer = kw_val['writer']
        writer.add_scalar('validation/easy_thr', easy_val, kw_val['step'])
        writer.add_scalar('validation/hard_thr', hard_val, kw_val['step'])
        writer.add_scalar('validation/extreme_thr', ext_val, kw_val['step'])
        
    return corr_easy.item() / cnt
        