import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

# helpers_

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SiT(nn.Module):
    def __init__(self, num_classes, signal_length=180, patch_length=30, dim=128, depth=4, heads=2,
                 mlp_dim=256, pool='cls', channels=12, dim_head=64, dropout=0, emb_dropout=0):
        super().__init__()

        assert signal_length % patch_length == 0, 'ECG dimensions must be divisible by the patch size.'

        num_patches = signal_length // patch_length
        patch_dim = channels * patch_length

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_length),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, ecg):

        x = self.to_patch_embedding(ecg)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        z = torch.sum(x, dim=2)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x= self.mlp_head(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model:embedding的维度
        :param dropout: Dropout的置零比例
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始一个位置编码矩阵，大小是max_len*d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，词汇的位置就是用它的索引表示max_len*1
        position = torch.arange(0, max_len).unsqueeze(
            1)  # 由[0,1,2...max_len][max_len] -> [[0],[1]...[max_len]][max_len,1]
        # 目的是要把position的信息放到pe里面去

        # 定义一个变换矩阵使得position的[max_len,1]*变换矩阵得到pe[max_len,d_model]->变换矩阵格式[1,d_model]
        # 除以这个是为了加快收敛速度
        # div_term格式是[0,1,2...d_model/2],分成了两个部分，步长为2
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # print(div_term.shape)
        # print(position * div_term)
        # a = position*div_term
        # print(a.shape)
        # 将前面定义好的矩阵进行奇数偶数赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 此时pe[max_len,d_model]
        # embedding三维(可以是[batch_size,vocab,d_model])#vocab就是max_len
        # 将pe升起一个维度扩充成三维张量
        pe = pe.unsqueeze(0)

        # 位置编码矩阵注册成模型的buffer，它不是模型中的参数，不会跟随优化器进行优化
        # 注册成buffer后我们就可以在模型的保存和加载时，将这个位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x:x代表文本序列的词嵌入
        pe编码过长将第二个维度也就是max_len的维度缩小成句子的长度
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to(x.device)
        return self.dropout(x)

class SiT_sincosPE(nn.Module):
    def __init__(self, num_classes, signal_length=180, patch_length=30, dim=128, depth=4, heads=2,
                 mlp_dim=256, pool='cls', channels=12, dim_head=64, dropout=0, emb_dropout=0):
        super().__init__()

        assert signal_length % patch_length == 0, 'ECG dimensions must be divisible by the patch size.'

        num_patches = signal_length // patch_length
        patch_dim = channels * patch_length

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_length),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = PositionalEncoding(d_model=dim, dropout=0, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, ecg):

        x = self.to_patch_embedding(ecg)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        x = self.transformer(x)

        z = torch.sum(x, dim=2)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x= self.mlp_head(x)

        return x

class Transformer_sincosPE(nn.Module):
    def __init__(self, num_classes, signal_length=180, patch_length=30, dim=128, depth=4, heads=2,
                 mlp_dim=256, pool='cls', channels=12, dropout=0, emb_dropout=0):
        super().__init__()

        assert signal_length % patch_length == 0, 'ECG dimensions must be divisible by the patch size.'

        num_patches = signal_length // patch_length
        patch_dim = channels * patch_length

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_length),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.pos_embedding = PositionalEncoding(d_model=dim, dropout=0, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
                                                   dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth, norm=None)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, ecg):

        x = self.to_patch_embedding(ecg)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x= self.mlp_head(x)
        return x

if __name__ == "__main__":
    ecgs = torch.randn([64, 2, 180])
    net = Transformer_sincosPE(num_classes=3, signal_length=180, patch_length=15, dim=64, depth=6, heads=4,
                 mlp_dim=128, pool='cls', channels=2, dropout=0, emb_dropout=0)
    y = net(ecgs)
    # print(net)
    print(y.shape)
