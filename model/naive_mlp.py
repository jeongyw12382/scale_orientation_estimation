from .backbone import L2Net
import pdb
import torch.nn as nn
import torchvision.transforms as TF
import torch.nn.functional as F
import pdb

'''
Experiment 1: 
    Backbone: L2Net
    - Interpolate original feature

'''

class L2NetNaiveModel(nn.Module):
    
    def __init__(self, **kw):
        super(L2NetNaiveModel, self).__init__()
        self.backbone = L2Net(**kw)
        
    def forward(self, data, iscuda):
        
        src_data = data['src_img'].cuda() if iscuda else data['src_img']
        trg_data = data['trg_img'].cuda() if iscuda else data['trg_img']
        
        src_feat = self.backbone(src_data)
        trg_feat = self.backbone(trg_data)
        trg_feat = F.interpolate(trg_feat, src_feat.shape[2:4])
        
        src_feat, trg_feat = F.normalize(src_feat**2, dim=1, p=2), F.normalize(trg_feat**2, dim=1, p=2)
        return {
            'src_feat': src_feat, 
            'trg_feat': trg_feat,
            'scale': data['scale']
        }