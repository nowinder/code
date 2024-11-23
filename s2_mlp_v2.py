import torch
from torch import nn
# from einops.layers.torch import Reduce
# from .utils import pair
from torchinfo import summary
from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple
from typing import List, Tuple, Optional, Union

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x

def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel = 512, k = 3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias = False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias = False)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)          #bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)       #bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  #bs,kc
        hat_a = hat_a.reshape(b, self.k, c)         #bs,k,c
        bar_a = self.softmax(hat_a)                 #bs,k,c
        attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
        out = attention * x_all                     # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c) # bs,h,w,c
        return out

class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, h, w, c = x.size()
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        return x

def ffn(dim, expansion_factor=4, dropout=0., layer = nn.Linear):
    return nn.Sequential(
                    layer(dim, dim * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    layer(dim * expansion_factor, dim),
                    nn.Dropout(dropout)
                )


class ConvFFN(nn.Module):
    """Convolutional FFN Module. from https://github.com/apple/ml-fastvit/blob/main/models/fastvit.py#L348"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        ks: int = 7,
        change_chan = True
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                padding=(ks-1)//2,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.chac = change_chan
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chac: x = x.permute(0,3,1,2)
        short_cut = x
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = (x+short_cut).permute(0,2,3,1)
        # PreNormResidual 有残差 不过不确定需不需要里面用了BN还在外面包一层LN再残差
        if not self.chac:return x+short_cut
        x = x.permute(0,2,3,1)
        return x
    
class S2Block(nn.Module):
    def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, S2Attention(d_model)),
                PreNormResidual(d_model, ConvFFN(in_channels=d_model,hidden_channels=d_model*expansion_factor,ks=3,drop=dropout))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes)
    )

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]),requires_grad=True)

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x    

class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, init_values=1e-2):
        super().__init__()
        # ori_img_size = img_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 这里源代码三种选择的卷积都是3的输入
        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:  
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 2:  
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim, 2),
                nn.GELU(),
            )
        else:
            raise("For convolutional projection, patch size has to be in [2, 4, 16]")
        self.pre_affine = Affine(in_chans)
        self.post_affine = Affine(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape 
        # 这里源代码直接设为affine(3)
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)

        # Hp, Wp = x.shape[2], x.shape[3]
        # x = x.flatten(2).transpose(1, 2)

        return x
      
class RandomConcatLayer(nn.Module):
    '''对输入张量根据指定维度dim生成随机张量并拼接，这个随机在每个训练epoch应该是不重复的'''
    def __init__(self, dim=1):
        super(RandomConcatLayer, self).__init__()
        self.dim = dim
        # self.device = device
        # self.list = list()

        # self.random_matrix = torch.rand()
        # self.cat = torch.cat()

    def forward(self, x):
        # 获取输入的形状
        shape = list(x.shape)
        
        # 生成随机矩阵
        shape[self.dim] = 1
        random_matrix = torch.rand(*shape, device=x.device)
        
        # 将随机矩阵与输入图像在通道维度上拼接
        x = torch.cat((x, random_matrix), dim=self.dim)
        return x
    
class ConcatenateLayer(nn.Module):
    '''根据指定维度dim叠加两个张量，等效tensorflow concatenate'''
    def __init__(self, dim=1):
        super(ConcatenateLayer, self).__init__()
        self.dim = dim
        # self.cat = torch.cat()
    def forward(self,x1,x2):
        return torch.cat((x1,x2),self.dim)
    

class ConvConca(nn.Module):
    def __init__(self, ini_channels, ks=1, filters_num: list = None,activation=nn.GELU):
        super(ConvConca, self).__init__()
        self.filters_num = filters_num
        self.ini_channels = ini_channels
        self.act = activation
        self.ndim = len(self.filters_num)
        # self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bicubic') for _ in range(self.ndim)])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=ini_channels+sum(filters_num[:i]), out_channels=filters_num[i],
                                              kernel_size=ks, padding='same' if ks>1 else 0, padding_mode='replicate' if ks>1 else 'zeros' ) for i in range(self.ndim)])
        self.conca = ConcatenateLayer()
    def forward(self,x):
        for con in self.convs:
            x = self.conca(x, self.act()(con(x)))
        return x
    

class S2MLPv2(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=[4, 2],
        in_channels=3,
        # num_classes=1000,
        d_model=[192, 384],
        depth=[4, 14],#感觉14有点多
        filters_num = [16,8],
        expansion_factor = [3, 3],
        ups = 3,
        drop=0.
    ):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()
        self.dm = [d_model[-1]+sum(filters_num)*i for i in range(ups)]
        # self.random_concat_layer = RandomConcatLayer(dim=1)
        self.stage = len(patch_size)
        # self.stages = nn.Sequential(
        #     *[nn.Sequential(
        #         nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=patch_size[i], stride=patch_size[i]),
        #         S2Block(d_model[i], depth[i], expansion_factor[i], dropout = 0.)
        #     ) for i in range(self.stage)]
        # )
        self.stages = nn.Sequential(
            *[nn.Sequential(
                # 这里是否有必要让patchembed和S2放一起呢？还是只在最开始放？
                # depth更重要
                ConvPatchEmbed(img_size=image_size,patch_size=patch_size[i], in_chans= in_channels if i==0 else d_model[i-1],embed_dim=d_model[i]),
                S2Block(d_model[i], depth[i], expansion_factor[i], dropout = drop)
            ) for i in range(self.stage)]
        )

        # self.mlp_head = nn.Sequential(
        #     Reduce('b c h w -> b c', 'mean'),
        #     nn.Linear(d_model[-1], num_classes)
        # )
        # self.conv_conca = ConvConca(ini_channels=d_model[-1], filters_num=filters_num, ks=1)
        # self.upsample1 = nn.Upsample(scale_factor= 2, mode='bicubic', align_corners=True)
        
        # self.upconv = nn.Sequential(
        #     *[nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
        #         ConvConca(ini_channels=self.dm[i], filters_num=filters_num, ks=1)
        #     ) for i in range(ups)]
        # )
        self.upcfn = nn.Sequential(
            *[nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
                ConvFFN(in_channels=d_model[-1],hidden_channels=d_model[-1]*4,ks=3,drop=drop,change_chan=False)
            ) for _ in range(ups)]
        )
        self.emb2dep = nn.Conv2d(in_channels=d_model[-1],out_channels=24, kernel_size=1, bias=True)
        self.out = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=5, bias=False)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        # self.conv_out = nn.Conv2d(in_channels=d_model[-1]+sum(filters_num), out_channels=2, kernel_size=1, bias=False, padding='valid')



    def forward(self, x):
        # x = self.random_concat_layer(x)
        embedding = self.stages(x)
        # convc = self.conv_conca(self.upsample1(embedding))
        # convc = self.upconv(embedding)
        convc = self.upcfn(embedding)
        dep = self.emb2dep(convc)
        dep = dep.unsqueeze(1)
        out = self.out(dep)
        # out = self.mlp_head(embedding)
        # out = self.conv_out(convc)
        # out = self.conv_out(self.upsample1(embedding))
        return out


# model2 = S2MLPv2(in_channels=20,patch_size=[4],expansion_factor=[3],d_model=[192],depth=[4]) 
# summary(model2, input_size=(16, 20, 128,128))