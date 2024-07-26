import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Norm_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Norm_ReLU, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv_Norm_ReLU(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class SynapseSegmentationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_filters=64):
        super(SynapseSegmentationModel, self).__init__()
        self.inc = nn.Conv3d(in_channels, n_filters, kernel_size=3, padding=1)
        self.down1 = Conv_Norm_ReLU(n_filters, n_filters*2)
        self.down2 = Conv_Norm_ReLU(n_filters*2, n_filters*4)
        self.down3 = Conv_Norm_ReLU(n_filters*4, n_filters*8)
        self.down4 = Conv_Norm_ReLU(n_filters*8, n_filters*16)
        self.up1 = Up(n_filters*16, n_filters*8)
        self.up2 = Up(n_filters*8, n_filters*4)
        self.up3 = Up(n_filters*4, n_filters*2)
        self.up4 = Up(n_filters*2, n_filters)
        self.outc = nn.Conv3d(n_filters, out_channels, kernel_size=1)  
        self.activate = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool3d(x1, 2))
        x3 = self.down2(F.max_pool3d(x2, 2))
        x4 = self.down3(F.max_pool3d(x3, 2))
        x5 = self.down4(F.max_pool3d(x4, 2))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        mask = self.activate(x)
        return mask