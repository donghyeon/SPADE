import torch.nn as nn
from models.networks.architecture import SPADEResnetBlock


class SEBlock(nn.Module):
    def __init__(self, fin, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(fin, fin // reduction, 1)
        self.actv1 = nn.ReLU()
        self.fc2 = nn.Conv2d(fin // reduction, fin, 1)
        self.actv2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.actv1(y)
        y = self.fc2(y)
        y = self.actv2(y)
        return x * y


class SPADEResnetSEBlock(nn.Module):
    def __init__(self, fin, fout, opt, reduction=8):
        super().__init__()
        self.spade_resnet_block = SPADEResnetBlock(fin, fout, opt)
        self.se_block = SEBlock(fout, reduction)

    def forward(self, x, seg):
        spade = self.spade_resnet_block

        x_s = spade.shortcut(x, seg)

        dx = spade.conv_0(spade.actvn(spade.norm_0(x, seg)))
        dx = spade.conv_1(spade.actvn(spade.norm_1(dx, seg)))

        out = x_s + self.se_block(dx)

        return out
