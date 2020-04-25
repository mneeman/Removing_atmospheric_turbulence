import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

from unet_parts import *

def weights_init(m):
    classname = m.__class__.__name__
    # for every Conve layer in a model..
    if classname.find('Conv') != -1:
        # apply a uniform distribution to the weights
        m.weight.data.normal_(0.0, 0.02)
    
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
    
    # for every Norm layer in a model..
    if classname.find('Norm') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class UNet(nn.Module):
    def __init__(self, sample_num, n_channels, batch_size, alpha):
        self.batch_size = batch_size
        self.n_channels = n_channels
        super(UNet, self).__init__()
        self.inc = inconv(sample_num, 32, alpha)
        self.down1 = down(32, 64, alpha)
        self.down2 = down(64, 128, alpha)
        self.down3 = down(128, 256, alpha)
        self.down4 = down(256, 512, alpha)
        self.down5 = down(512, 1024, alpha)
        self.down6 = down(1024, 2048, alpha)
        self.up1 = up(2048, 1024, alpha)
        self.up2 = up(1024, 512, alpha)
        self.up3 = up(512, 256, alpha)
        self.up4 = up(256, 128, alpha)
        self.up5 = up(128, 64, alpha)
        self.up6 = up(64, 32, alpha)
        self.outc = outconv(32, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outc(x)
        x = x.view(self.batch_size,self.n_channels,256,256)
        return x


class Discriminator(nn.Module):
    def __init__(self, batch_size, alpha):
        self.batch_size = batch_size
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(64*8*8, 512)
        #self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = x6.view(self.batch_size, -1)
        x8 = self.fc1(x7)
        #x9 = self.fc2(x8)

        return x8
"""
class Discriminator_(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        """