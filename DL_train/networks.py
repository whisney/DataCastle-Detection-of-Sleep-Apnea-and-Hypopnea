import torch
from torch import nn
from Networks.Network_utils import BasicBlock1D, ResBasicBlock1D
from Networks.TimeNet import ECGNet
from Networks.Sit import SiT, SiT_sincosPE, Transformer_sincosPE, Transformer
from einops import rearrange, repeat
from Networks.TimesNet_utils import TimesNet
from Networks.Nonstationary_Transformer_utils import NSTransformer
from monai.networks.nets import DenseNet, EfficientNetBN, SENet, ResNet

# FC
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(180 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

# MSDNN
class Net2(nn.Module):
    def __init__(self, num_classes=3, init_channels=2, growth_rate=16, base_channels=32,
                 stride=2, drop_out_rate=0.2):
        super(Net2, self).__init__()
        self.num_channels = init_channels
        self.num_classes = num_classes

        block_n = 4
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)
        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)

        feature = torch.squeeze(out)

        out = self.fc(feature)

        return out

# ECGNet
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.net = ECGNet(growth_rate=16, num_classes=3, init_channels=2)

    def forward(self, x):
        x = self.net(x)
        return x

# SiT
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.net = SiT(num_classes=3, signal_length=180, channels=2)

    def forward(self, x):
        x = self.net(x)
        return x

# MSDNN for Manually extracted features
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.net = Net2(num_classes=3, init_channels=8, growth_rate=16, base_channels=32,
                 stride=2, drop_out_rate=0.0)

    def forward(self, x):
        x = x.unfold(dimension=2, size=30, step=3)
        x = torch.cat([torch.mean(x, dim=3, keepdim=False),
                       torch.std(x, dim=3, keepdim=False),
                       torch.max(x, dim=3, keepdim=False).values,
                       torch.min(x, dim=3, keepdim=False).values], dim=1)
        x = self.net(x)
        return x

# MSDNN for Manually extracted features
class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.net = Net2(num_classes=3, init_channels=8, growth_rate=16, base_channels=32,
                 stride=2, drop_out_rate=0.2)

    def forward(self, x):
        x = x.unfold(dimension=2, size=30, step=3)
        x = torch.cat([torch.mean(x, dim=3, keepdim=False),
                       torch.std(x, dim=3, keepdim=False),
                       torch.max(x, dim=3, keepdim=False).values,
                       torch.min(x, dim=3, keepdim=False).values], dim=1)
        x = self.net(x)
        return x

# SiT
class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.net = SiT(num_classes=3, signal_length=180, patch_length=3, dim=16, depth=4, heads=2,
                 mlp_dim=128, pool='cls', channels=2, dim_head=32, dropout=0.2, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# SiT
class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.net = SiT(num_classes=3, signal_length=180, patch_length=6, dim=30, depth=4, heads=2,
                 mlp_dim=256, pool='cls', channels=2, dim_head=64, dropout=0.0, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# SiT
class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        self.net = SiT(num_classes=3, signal_length=180, patch_length=15, dim=128, depth=8, heads=4,
                 mlp_dim=256, pool='cls', channels=2, dim_head=64, dropout=0, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# SiT_sincosPE
class Net10(nn.Module):
    def __init__(self):
        super(Net10, self).__init__()
        self.net = SiT_sincosPE(num_classes=3, signal_length=180, patch_length=15, dim=128, depth=8, heads=4,
                 mlp_dim=256, pool='cls', channels=2, dim_head=64, dropout=0, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# MSDNN
class Net11(nn.Module):
    def __init__(self, num_classes=3, init_channels=2, growth_rate=32, base_channels=32,
                 stride=2, drop_out_rate=0.2):
        super(Net11, self).__init__()
        self.num_channels = init_channels
        self.num_classes = num_classes

        block_n = 4
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = ResBasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)
        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)

        feature = torch.squeeze(out)

        out = self.fc(feature)

        return out

# MSDNN
class Net12(nn.Module):
    def __init__(self, num_classes=3, init_channels=2, growth_rate=16, base_channels=16,
                 stride=2, drop_out_rate=0.2):
        super(Net12, self).__init__()
        self.num_channels = init_channels
        self.num_classes = num_classes

        block_n = 4
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = ResBasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)
        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)

        feature = torch.squeeze(out)

        out = self.fc(feature)

        return out

# Transformer_sincosPE
class Net13(nn.Module):
    def __init__(self):
        super(Net13, self).__init__()
        self.net = Transformer_sincosPE(num_classes=3, signal_length=180, patch_length=15, dim=64, depth=6, heads=4,
                 mlp_dim=128, pool='cls', channels=2, dropout=0, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# Transformer_sincosPE
class Net14(nn.Module):
    def __init__(self):
        super(Net14, self).__init__()
        self.net = Transformer_sincosPE(num_classes=3, signal_length=180, patch_length=15, dim=128, depth=4, heads=4,
                 mlp_dim=256, pool='cls', channels=2, dropout=0, emb_dropout=0)

    def forward(self, x):
        x = self.net(x)
        return x

# CNN + Transformer
class Net15(nn.Module):
    def __init__(self):
        super(Net15, self).__init__()
        self.block1 = BasicBlock1D(in_channels=2, out_channels=32, drop_out_rate=0, stride=2)

        self.pos_embedding = nn.Parameter(torch.randn(1, 90 + 1, 32))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))

        self.transformer = Transformer(dim=32, depth=4, heads=4, dim_head=256, mlp_dim=64, dropout=0)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.block1(x).permute(0, 2, 1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

# CNN + BiLSTM
class Net16(nn.Module):
    def __init__(self):
        super(Net16, self).__init__()
        self.block1 = BasicBlock1D(in_channels=2, out_channels=32, drop_out_rate=0, stride=2)
        self.LSTM = nn.LSTM(input_size=32, hidden_size=128, num_layers=4, bias=True,
                            batch_first=True, dropout=0, bidirectional=True)
        self.mlp_head = nn.Linear(128 * 2, 3)

    def forward(self, x):
        x = self.block1(x).permute(0, 2, 1)
        b, n, _ = x.shape
        x, _ = self.LSTM(x)
        x = x[:, -1]
        x = self.mlp_head(x)
        return x

# TimesNet
class Net17(nn.Module):
    def __init__(self):
        super(Net17, self).__init__()
        self.net = TimesNet(seq_len=180, e_layers=3, enc_in=2, d_model=32, dropout=0, num_class=3, top_k=3, d_ff=32)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x

# TimesNet
class Net18(nn.Module):
    def __init__(self):
        super(Net18, self).__init__()
        self.net = NSTransformer(seq_len=180, e_layers=3, enc_in=2, d_model=32, d_ff=32, dropout=0, n_heads=8,
                 activation='gelu', num_class=3, p_hidden_dims=[128, 128], p_hidden_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x

# MSDNN
class Net19(nn.Module):
    def __init__(self, num_classes=3, init_channels=2, growth_rate=32, base_channels=64,
                 stride=2, drop_out_rate=0.2):
        super(Net19, self).__init__()
        self.num_channels = init_channels
        self.num_classes = num_classes

        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)
        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)
        feature = torch.squeeze(out)
        out = self.fc(feature)
        return out

# ResNet34
class Net20(nn.Module):
    def __init__(self):
        super(Net20, self).__init__()
        self.net = ResNet(block='basic', layers=[3, 4, 6, 3], block_inplanes=[64, 128, 256, 512], spatial_dims=1,
                          n_input_channels=2, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B',
                          widen_factor=1.0, num_classes=3, feed_forward=True, bias_downsample=True,
                          act=('relu', {'inplace': True}))

    def forward(self, x):
        x = self.net(x)
        return x

# DenseNet
class Net21(nn.Module):
    def __init__(self):
        super(Net21, self).__init__()
        self.net = DenseNet(spatial_dims=1, in_channels=2, out_channels=3, init_features=32, growth_rate=16,
        block_config=(6, 12, 24, 16), bn_size=4, act=("relu", {"inplace": True}), norm="batch", dropout_prob=0.0)

    def forward(self, x):
        x = self.net(x)
        return x

# DenseNet
class Net22(nn.Module):
    def __init__(self):
        super(Net22, self).__init__()
        self.net = DenseNet(spatial_dims=1, in_channels=2, out_channels=3, init_features=32, growth_rate=32,
        block_config=(6, 12, 24, 16), bn_size=4, act=("relu", {"inplace": True}), norm="batch", dropout_prob=0.0)

    def forward(self, x):
        x = self.net(x)
        return x

# SENet
class Net23(nn.Module):
    def __init__(self):
        super(Net23, self).__init__()
        self.net = SENet(spatial_dims=1, in_channels=2, block='se_resnet_bottleneck', layers=[3, 4, 6, 3], groups=1,
                         reduction=16, dropout_prob=None, dropout_dim=1, inplanes=64, downsample_kernel_size=1,
                         input_3x3=False, num_classes=3)

    def forward(self, x):
        x = self.net(x)
        return x

# Efficientnet-b0
class Net24(nn.Module):
    def __init__(self):
        super(Net24, self).__init__()
        self.net = EfficientNetBN('efficientnet-b0', pretrained=True, progress=True, spatial_dims=1, in_channels=2,
                         num_classes=3, norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False).cuda()

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    with torch.no_grad():
        x = torch.rand(256, 2, 180)
        net = Net20()
        y = net(x)
        print(y.shape)