import torch
import torchvision.transforms as TF
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset

import pdb
import os
import json
import tqdm
import numpy as np

from PIL import Image 

class RandomRescale:
    
    def __init__(self, min_scale, max_scale):
        self.min_scale = 1
        self.max_scale = max_scale
    
    def __call__(self, img):
        H, W = img.size
        scale = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
        resized_H, resized_W = int(H * scale), int(W * scale)
        return (TF.functional.resize(img, (resized_H, resized_W)), scale)
    
class CIFARSelfSupScale(Dataset):
    
    # kw = cfg.scale
    def __init__(self, root='data', train=True, **kw):
        self.rep = kw['rep']
        dir_name = 'cifar{}'.format(kw['cifar_version'])
        mode = 'train' if train else 'test'
        img_dir = os.path.join(root, dir_name, mode)
        self.src_img, self.trg_img, self.trg_gt = [], [], []
        img_idx = 0
        while(os.path.exists(os.path.join(img_dir, '{}_orig.jpg'.format(img_idx)))):
            for rep in range(self.rep):
                self.src_img.append(os.path.join(img_dir, '{}_orig.jpg'.format(img_idx)))
                self.trg_img.append(os.path.join(img_dir, '{}_{}.jpg'.format(img_idx, rep)))
                self.trg_gt.append(os.path.join(img_dir, '{}_{}.json'.format(img_idx, rep)))
            img_idx += 1
        self.mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
        self.std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
                
        
    def __len__(self):
        return len(self.trg_img)
    
    def __getitem__(self, idx):
        src_img = Image.open(self.src_img[idx])
        trg_img = Image.open(self.trg_img[idx])
        scale = json.load(open(self.trg_gt[idx], 'r'))['scale']
        return {
            'src_img': src_img,
            'trg_img': TF.CenterCrop(src_img.size)(trg_img),
            'cls_num': json.load(open(self.trg_gt[idx], 'r'))['cls_num'],
            'scale': scale
        }
    
    def collate_fn(self, samples): 
        src_img = torch.stack([((TF.ToTensor()(sample['src_img']) / 255.0) - self.mean[:, None, None]) / self.std[:, None, None]  for sample in samples])
        trg_img = torch.stack([((TF.ToTensor()(sample['trg_img']) / 255.0) - self.mean[:, None, None]) / self.std[:, None, None] for sample in samples])
        scale = torch.tensor([sample['scale'] for sample in samples])
        cls_num = torch.tensor([sample['cls_num'] for sample in samples])
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'scale': scale,
            'cls_num': cls_num
        }
    
def download_cifar(version, min_scale, max_scale, repeat=1):
    
    data_train = eval('CIFAR{}'.format(str(version)))(root='data', train=True, download=True)
    data_test = eval('CIFAR{}'.format(str(version)))(root='data', train=False, download=True)
    
    random_scale = RandomRescale(min_scale, max_scale)
    cifar_dir = 'cifar{}'.format(version)
    train_path, test_path = os.path.join('data', cifar_dir, 'train'), os.path.join('data', cifar_dir, 'test')
    os.mkdir(os.path.join('data', cifar_dir))
    os.mkdir(train_path)
    os.mkdir(test_path)
    for data_type in [data_train, data_test]:
        dir_name = train_path if data_type == data_train else test_path
        for (i, data) in tqdm.tqdm(enumerate(data_type)):
            src_path = os.path.join(dir_name, '{}_orig.jpg'.format(i))
            data[0].save(src_path)
            for rep in range(repeat):
                trg_path = os.path.join(dir_name, '{}_{}.jpg'.format(i, rep))
                trg_img, scale = random_scale(data[0])
                trg_img.save(trg_path)
                trg_gt = {
                    'scale' : scale.item(),
                    'cls_num' : data[1]
                }
                with open(os.path.join(dir_name, '{}_{}.json'.format(i, rep)), 'w') as f:
                    f.write(json.dumps(trg_gt))
    
    
if __name__ == '__main__': 
    download_cifar(10, 1/3, 1*3, 2)
    download_cifar(100, 1/3, 1*3, 2)
    