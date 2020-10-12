import torch
import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset

import pdb

class RandomRescale:
    
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
    
    def __call__(self, img):
        H, W = img.size
        resized_H, resized_W = int(H * self.scale), int(W * self.scale)
        return TF.functional.resize(img, (resized_H, resized_W))
    
class CIFARSelfSupScale(Dataset):
    
    # kw = cfg.scale
    def __init__(self, train=True, **kw):
        Dataset.__init__(self)
        self.min_scale = kw['scale']['min_scale']
        self.max_scale = kw['scale']['max_scale']
        self.cifar = kw['cifar_version']
        assert self.min_scale < self.max_scale
        assert self.cifar == 10 or self.cifar == 100
        self.dataset = eval('CIFAR{}'.format(str(self.cifar)))(root='..\data', download=True, train=train)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_img, cls_num = self.dataset[idx]
        return {
            'src_img': src_img, 
            'cls_num': cls_num
        }
    
    def collate_fn(self, samples): 
        random_scaler = RandomRescale(self.min_scale, self.max_scale)
        src_img = torch.stack([TF.ToTensor()(sample['src_img']) for sample in samples])
        trg_img = torch.stack([TF.ToTensor()(random_scaler(sample['src_img'])) for sample in samples])
        scale = torch.tensor([random_scaler.scale for i in range(len(samples))])
        cls_num = torch.tensor([sample['cls_num'] for sample in samples])
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'scale': scale,
            'cls_num': cls_num
        }