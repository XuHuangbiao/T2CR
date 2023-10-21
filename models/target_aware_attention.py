import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        # self.act = LFTM(hidden_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class TAA(nn.Module):
    def __init__(self, conv_input=1024):
        super(TAA, self).__init__()

        self.selayer = SELayer_1d(conv_input)
        self.inconv = double_conv(8, 10)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.activation_1 = nn.ReLU()
        self.mlp = Mlp(64)
        self.cbam = CBAMBlock(10, 10, 3)
        self.fc_score = nn.Linear(64, 1)
        self.fc_class = nn.Linear(64, 58)
        self.ca = ChannelAttention(10, 10)
        self.soft = nn.Softmax()

    def forward(self, x):
        if x.shape[1] == 8:
            x = self.inconv.forward(x)

        attn = self.selayer.forward(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.cbam.forward(attn)

        x = self.activation_1(self.fc1(x))
        x = self.activation_1(self.fc2(x))

        score = self.fc_score(x).mean(1)
        pred_class = self.soft(self.fc_class(x).mean(1))

        x1, x2 = torch.chunk(x, 2, dim=0)
        score1, score2 = torch.chunk(score, 2, dim=0)
        pred_class1, pred_class2 = torch.chunk(pred_class, 2, dim=0)
        return x1, x2, score1, score2, pred_class1, pred_class2
