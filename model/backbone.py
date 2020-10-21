import torch
import torch.nn as nn
import pdb

from torchvision.models import resnet18


# Following the same implementation to R2D2. 
class Net(nn.Module):
    
    def __init__(self, **model_kw):
        nn.Module.__init__(self)
        self.dilation = 1
        self.bin_num = model_kw['bin_num']
    
    def create_conv_bn_relu_block(self, in_channel, out_channel, kernel_size, bn=True, relu=True, stride=1, dilation=1):
        module_list = []
        self.dilation *= dilation
        module_list.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride,
                                     padding = (kernel_size - 1) * self.dilation // 2, dilation=self.dilation))
        if bn: module_list.append(nn.BatchNorm2d(out_channel))
        if relu: module_list.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*module_list)

    def forward(self, img):
        return self.module_list(img)
    
class L2Net(Net): 
    
    def __init__(self, **model_kw):
        
        Net.__init__(self, **model_kw)
        module_list = []
        module_list.append(self.create_conv_bn_relu_block(3, 32, 3))
        module_list.append(self.create_conv_bn_relu_block(32, 32, 3))
        module_list.append(self.create_conv_bn_relu_block(32, 64, 3, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(64, 64, 3))
        module_list.append(self.create_conv_bn_relu_block(64, 128, 3, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 3))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, self.bin_num, 1))
        
        self.module_list = nn.Sequential(*module_list)

class ResNet18(Net):
    
    def __init__(self, **kw):
        
        Net.__init__(self)
        network = resnet18(pretrained=True)
        module_list = list(network.children())[:-3]
        module_list.append(self.create_conv_bn_relu_block(512, self.bin_num, 3))
        self.module_list = nn.Sequential(*module_list)
