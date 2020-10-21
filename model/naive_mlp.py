from .backbone import *
import pdb
import torch
import torch.nn as nn
import torchvision.transforms as TF
import torch.nn.functional as F
import pdb


class Model(nn.Module):
    
    def __init__(self, **model):
        nn.Module.__init__(self)
        self.backbone = None
        self.normalize = F.softmax if 'normalize' not in model else eval('F.{}'.format(model['normalize']))
        self.kw = {'dim':1} if 'normalize_kw' not in model else model['normalize_kw']
    
    def forward(self, data, iscuda):
        
        src_data = data['src_img'].cuda() if iscuda else data['src_img']
        trg_data = data['trg_img'].cuda() if iscuda else data['trg_img']
        
        bsz, C, H, W = src_data.shape
        H_resize, W_resize = H // data['scale'], W // data['scale']
        
        src_feat = self.backbone(src_data)
        trg_feat = self.backbone(trg_data)
        
        src_feat = torch.stack([nn.Upsample(trg_feat.shape[-2:], mode='bilinear', align_corners=False)(feat[None, :, ((H - h)/2).int():((H+h)/2).int(), ((W-w)/2).int():((W+w)/2).int()]).squeeze(0) for (feat, h, w) in zip(src_feat, H_resize, W_resize)])
        
        src_feat, trg_feat = self.normalize(src_feat, **self.kw), self.normalize(trg_feat, **self.kw)
        
        return {
            'src_feat': src_feat, 
            'trg_feat': trg_feat,
            'scale': data['scale']
        }
    
    

class L2NetNaiveModel(Model):
    
    def __init__(self, **model):
        Model.__init__(self, **model)
        self.backbone = L2Net(**model['model_kw'])
        
    
class ResNet18NaiveModel(Model):
    
    def __init__(self, **model):
        Model.__init__(self, **model)
        self.backbone = ResNet18(**model['model_kw'])

