import torch
import torch.nn as nn

# Following the same implementation to R2D2. 
class L2Net(nn.Module): 
    
    def __init__(self, **kw):
        self.dilation = 1
        nn.Module.__init__(self)
        module_list = []
        module_list.append(self.create_conv_bn_relu_block(3, 32, 3))
        module_list.append(self.create_conv_bn_relu_block(32, 32, 3))
        module_list.append(self.create_conv_bn_relu_block(32, 64, 3, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(64, 64, 3))
        module_list.append(self.create_conv_bn_relu_block(64, 128, 3, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 3))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, relu=False, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, relu=False, dilation=2))
        module_list.append(self.create_conv_bn_relu_block(128, 128, 2, bn=False, relu=False, dilation=2))

        self.module_list = nn.Sequential(*module_list)
    
    def create_conv_bn_relu_block(self, in_channel, out_channel, kernel_size, bn=True, relu=True, stride=1, dilation=1):
        module_list = []
        self.dilation *= dilation
        module_list.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = (kernel_size - 1) * self.dilation // 2, dilation=self.dilation))
        if bn: module_list.append(nn.BatchNorm2d(out_channel))
        if relu: module_list.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*module_list)
    
    def forward(self, img):
        return self.module_list(img)