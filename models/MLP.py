import torch
import torch.nn as nn


class SELayer_1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_1d, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLP_score(nn.Module):
    def __init__(self, in_channel=64, out_channel=1):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)
        self.selayer = SELayer_1d(in_channel)

    def forward(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        x = self.selayer.forward(x.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer3(x)
        return output
