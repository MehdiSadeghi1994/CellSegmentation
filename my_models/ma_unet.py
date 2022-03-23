""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *
from .att_block import Channel_ATT





class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear, attention=True)
        self.up2 = Up(512, 256 // factor, bilinear, attention=True)
        self.up3 = Up(256, 128 // factor, bilinear, attention=True)
        self.up4 = Up(128, 64, bilinear, attention=True)
        self.outc = OutConv(64, n_classes)
        self.CA_1 = Channel_ATT()
        self.CA_2 = Channel_ATT()
        self.CA_3 = Channel_ATT()
        self.CA_4 = Channel_ATT()





    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.CA_1(x)        
        x = self.up2(x, x3)
        x = self.CA_2(x)
        x = self.up3(x, x2)
        x = self.CA_3(x)
        x = self.up4(x, x1)
        x = self.CA_4(x)
        logits = self.outc(x)
        return logits,0


