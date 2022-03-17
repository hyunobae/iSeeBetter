import argparse

import torch
import torch.nn as nn

import math
import torch
from torch import nn
from module import *
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.config = config
        self.fsize = 64
        self.version = self.config.version
        self.model = self.phase(self.version)

    def forward(self, x):
        out = self.model(x)
        out = torch.sigmoid(out)
        return out

    def make_model(self, version):
        if version == 0:
            fsize = int(self.fsize * 0.25)

        elif version == 1:
            fsize = int(self.fsize*0.5)

        elif version == 2:
            fsize = self.fsize

        model = nn.Sequential(
            nn.Conv2d(3, fsize, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize, fsize, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fsize),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize, fsize*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fsize*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize*2, fsize*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fsize*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize*2, fsize*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(fsize*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize*4, fsize*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fsize*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize*4, fsize*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(fsize*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(fsize*8, fsize*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fsize*8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),  # 512개의 feature map이 512*1*1로 바뀜
            nn.Conv2d(fsize*8, fsize*16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(fsize*16, 1, kernel_size=1)
        )

        return model

    def phase(self, version):
        ver = version
        return self.make_model(ver)






