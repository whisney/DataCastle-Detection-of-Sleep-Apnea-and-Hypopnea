import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 9, 4
k4, p4 = 17, 8
leak_rate = 0.1


class BranchConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv, self).__init__()
        C = out_channels // 4
        self.b1 = nn.Conv2d(in_channels, C, [1, k1], [1, stride], [0, p1], bias=False)
        self.b2 = nn.Conv2d(in_channels, C, [1, k2], [1, stride], [0, p2], bias=False)
        self.b3 = nn.Conv2d(in_channels, C, [1, k3], [1, stride], [0, p3], bias=False)
        self.b4 = nn.Conv2d(in_channels, C, [1, k4], [1, stride], [0, p4], bias=False)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.operation = nn.Sequential(
            BranchConv(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leak_rate, inplace=True),
            nn.Dropout2d(0.3),
            BranchConv(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels))
        self.k = 3
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, [1, stride], [1, stride], bias=False),
                nn.BatchNorm2d(out_channels))
        self.seq_len = 2742
        self.activate = nn.LeakyReLU(leak_rate, inplace=True)
        self.softmax = nn.Softmax(dim=0)
        self.period_weight = nn.Parameter(torch.tensor([0.2, 0.2, 0.2]))

    def forward(self, x):
        B,  N, T = x.size()
        period_list = [150, 200, 250]


        self.seq_len = x.shape[-1]

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0],  x.shape[1], (length - (self.seq_len))]).to(x.device)
                out = torch.cat([x, padding], dim=2)
            else:
                length = (self.seq_len)
                out = x
            # reshape
            out = out.reshape(B, N,  length // period, period).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out = self.activate(self.operation(out) + self.shortcut(out))
            # reshape back
            out = out.reshape(B, out.shape[1], -1)
            res.append(out[:, :, :int(x.shape[2]/2)])


        res = torch.stack(res, dim=-1)


        # adaptive aggregation
        period_weight = self.softmax(self.period_weight)
        res = res[:, :, :, 0]*period_weight[0]+res[:, :, :, 1]*period_weight[1]+res[:, :, :, 2]*period_weight[2]

        return res


class ECGNet(nn.Module):

    def __init__(self, num_classes=1, init_channels=12, growth_rate=16, base_channels=32, stride=2):
        super(ECGNet, self).__init__()
        self.num_channels = base_channels
        self.num_channels = init_channels

        block_n = 4
        block_c = [base_channels + i * growth_rate for i in range(block_n)]


        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock(self.num_channels, C, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("InsidePool", module)


        self.fc = nn.Linear(self.num_channels, num_classes)


    def forward(self, x):

        out = self.blocks(x)
        out = torch.squeeze(out)
        out = self.fc(out)

        return out



if __name__ == "__main__":
#=======================================================================================================================
    x = torch.randn([15, 2, 180])
    net = ECGNet(growth_rate=16, num_classes=3, init_channels=2)
    y = net(x)
    # print(net)
    print(y.shape)
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
# =======================================================================================================================