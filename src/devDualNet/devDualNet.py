import os
import math
import torch
import warnings
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.registry import register_model
from mmengine.model import constant_init, kaiming_init
from timm.models.layers import DropPath, to_2tuple, make_divisible, trunc_normal_
warnings.filterwarnings('ignore')

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))

# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Mlp(nn.Module):
    def __init__(self, dim, shallow = False):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)
        self.dwconv = nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):   
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att,_ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)
        output = output + x
        return output

class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class DLKBlock(nn.Module):
    def __init__(self, dim, shallow=False, drop_path=0.):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DLKModule(dim)
        self.mlp = Mlp(dim, shallow)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6         
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)

        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i >=2:
                shallow = False
            else:
                shallow = True
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[i], drop_path=dp_rates[cur + j], shallow = shallow) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = nn.LayerNorm(dims[i], eps=1e-6)
            self.norm_layers.append(norm_layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.stages[i](x)
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Convblock(nn.Module):
    def __init__(self, input_dim, dim, shallow=False):
        super().__init__()
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            self.act
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            self.act
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output

def channel_to_last(x):
    return x.permute(0, 2, 3, 1)

def channel_to_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 3, 1, 2)

class Attention(nn.Module):
    def __init__(self, 
                 in_dim=1, 
                 out_dim=32, 
                 d_state = 16, 
                 d_conv = 4, 
                 expand = 2, 
                 head=4, 
                 num_slices=4, 
                 step = 1,
                 goble = True):
        super(Attention, self).__init__()

        # 大核全局扫描
        if goble == True:
            # 大核全局扫描
            self.att_conv = nn.Conv2d(in_dim, in_dim, kernel_size=7, stride=1, padding=9, groups=in_dim, dilation=3)
        else:
            # 小核局部扫描
            self.att_conv = nn.Conv2d(in_dim, in_dim, kernel_size=5, stride=1, padding=2, groups=in_dim)


        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Mamba
        self.norm = nn.LayerNorm(in_dim)
        self.mamba = Mamba(
                d_model=in_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",   # TODO: set 154 assert bimamba_type=="v3" as none
                nslices = num_slices
        )
        
        # 调整通道Conv
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)  
    
    def forward(self, x):

        # 保持原始x
        att1 = x

        # 全局特征提取
        x = self.att_conv(x)

        # 维度记录
        B, C, H, W = x.shape

        # 特征Norm
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # Mamba处理
        out, q, k, v = self.mamba(x_norm)

        # 维度转换
        att2 = out.transpose(-1, -2).reshape(B, C, *img_dims)
        
        # 调整通道
        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att,_ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:].unsqueeze(1) + att2 * att[:,1,:,:].unsqueeze(1)

        x = self.conv(output)

        return x



class AttentionBlock(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, kernel_size=3, num_slices=4, shallow=True):
        super(AttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()

        self.gobel_attention = Attention(in_dim=in_dim//2, out_dim=out_dim, num_slices=num_slices, goble=True)

        self.local_attention = Attention(in_dim=in_dim//2, out_dim=out_dim, num_slices=num_slices, goble=False)

        self.downsample = Convblock(out_dim*2, out_dim, shallow=shallow)

    def forward(self, x):
        x_0, x_1 = x.chunk(2,dim = 1)
        x_0 = self.gobel_attention(x_0)
        x_1 = self.local_attention(x_1)
        x = torch.cat([x_0, x_1], dim=1)
        x = self.downsample(x)
        return x


class dkDualNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], kernel_size=3, out_dim=64, num_slices_list = [64, 32, 16, 8], drop_path_rate=0.3):
        super().__init__()

        self.dnet_down = Encoder(
            in_chans=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate
        )


        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.block2 = AttentionBlock(in_dim=c2_in_channels, out_dim=out_dim, kernel_size=kernel_size,  shallow=True, num_slices=num_slices_list[1])
        self.block3 = AttentionBlock(in_dim=c3_in_channels, out_dim=out_dim, kernel_size=kernel_size, shallow=False, num_slices=num_slices_list[2])
        self.block4 = AttentionBlock(in_dim=c4_in_channels, out_dim=out_dim, kernel_size=kernel_size,  shallow=False, num_slices=num_slices_list[3])

        self.fuse = Convblock(out_dim, out_dim, shallow=True)
        self.fuse2 = nn.Sequential(Convblock(out_dim*2, out_dim, shallow=False),nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False))
        
        self.L_feature = Convblock(c1_in_channels, out_dim, shallow=True)
        
        self.o1_u = nn.ConvTranspose2d(1, out_channels, kernel_size=4, stride=4)
        self.o2_u = nn.ConvTranspose2d(out_dim*2, out_channels, kernel_size=2, stride=2)

        self.head = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False)

    def Upsample(self, x, size, align_corners = False):
        """
        Wrapper Around the Upsample Call
        """
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)
    def forward(self, x):
        c1, c2, c3, c4 = self.dnet_down(x)

        _c4 = self.block4(c4) 
        _c4 = self.Upsample(_c4, c3.size()[2:])
        _c3 = self.block3(c3)
        _c2 = self.block2(c2) 

        output = self.fuse2(torch.cat([self.Upsample(_c4, c2.size()[2:]), self.Upsample(_c3, c2.size()[2:])], dim=1))

        L_feature = self.L_feature(c1)  # [1, 64, 88, 88]
        H_feature = self.fuse(_c2)
        H_feature = self.Upsample(H_feature, L_feature.size()[2:])


        output2 = torch.cat((H_feature,L_feature), dim=1)

        output = self.o1_u(output)
        output2 = self.o2_u(output2)

        out = self.head(torch.cat((output, output2), dim=1))

        return out

if __name__ == '__main__':
    device = 'cuda:1'
    
    x = torch.randn(size=(1, 3, 608, 608)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)
    
    model = dkDualNet(in_channels=3,out_channels=1,).to(device)
    # module = AttentionBlock(in_dim=64, out_dim=32, kernel_size=3, mlp_ratio=4, shallow=True).to(device)
    
    print(model(x).size())
    # print(module(test_x).size())
    
    