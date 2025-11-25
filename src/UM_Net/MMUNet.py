import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
import einops
from mamba_ssm import Mamba


class MMConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device = "cuda",
        num_slices=4
    ):
        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.mamba = Mamba(
                d_model=kernel_size, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
                bimamba_type="v1",
                # bimamba_type="v3",   # TODO: set 154 assert bimamba_type=="v3" as none
                nslices = num_slices
        )

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )
        self.altho = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))

    def two_row_columnwise_flatten_grad_safe(self, x):
        """
        输入: x [H, W]，其中 H 为任意整数（>=1）
        输出: 1D tensor，按“每两行按列交错”的方式展平，支持梯度追踪
        """
        B, C, H, W = x.shape
        even_rows = H // 2 * 2
        has_tail = H % 2 == 1

        x_main = x[:, :, :even_rows, :]  # [B, C, even_rows, W]
        x_tail = x[:, :, even_rows:, :]  # [B, C, 1, W] or empty

        # Reshape → [B, C, even_rows//2, 2, W]
        x_pair = x_main.view(B, C, even_rows // 2, 2, W)

        # Permute → [B, C, even_rows//2, W, 2]
        x_pair = x_pair.permute(0, 1, 2, 4, 3)

        # Reshape → [B, C, even_rows * W]
        x_flat = x_pair.reshape(B, C, -1)  # [B, C, even_rows*W]

        # 加上最后一行（如果存在）
        if has_tail:
            x_flat = torch.cat([x_flat, x_tail.reshape(B, C, -1)], dim=2)

        return x_flat

    def inverse_two_row_columnwise_flatten(self, x_flat, H, W):
        """
        x_flat: 1D tensor 展平后的结果
        H, W: 原始目标形状
        返回: [H, W] tensor，支持 autograd
        """
        B, C, L = x_flat.shape
        even_rows = H // 2 * 2
        even_count = even_rows * W
        has_tail = H % 2 == 1

        x_main = x_flat[:, :, :even_count]  # [B, C, even_rows*W]
        x_tail = x_flat[:, :, even_count:] if has_tail else None  # [B, C, W] if exists

        # Reshape main → [B, C, even_rows//2, W, 2]
        x_pair = x_main.view(B, C, even_rows // 2, W, 2)

        # Permute → [B, C, even_rows//2, 2, W]
        x_pair = x_pair.permute(0, 1, 2, 4, 3)

        # Reshape → [B, C, even_rows, W]
        x_restored = x_pair.reshape(B, C, even_rows, W)

        if has_tail:
            x_restored = torch.cat([x_restored, x_tail.view(B, C, 1, W)], dim=2)

        return x_restored
    def get_coordinate_map_2D(self,
        offset: torch.Tensor,
        morph: int,
        extend_scope: float = 1.0,
        device = "cuda",
    ):
        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        batch_size, _, width, height = offset.shape
        kernel_size = offset.shape[1] // 2
        center = kernel_size // 2
        device = torch.device(device)

        y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)
        y_keep = y_offset_
        y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

        x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        # Mamba
        _, _, width, height = y_keep.shape
        y_keep = self.two_row_columnwise_flatten_grad_safe(y_keep)

        y_keep = y_keep.transpose(-1, -2)
        y_keep, _, _, _ = self.mamba(y_keep)
        y_keep = y_keep.transpose(-1, -2)
        y_keep = self.inverse_two_row_columnwise_flatten(y_keep, width, height)
        
        # Mamba
        weight = torch.nn.functional.softplus(self.altho)
        weight = torch.clamp(weight, min=0.01)
        y = weight * y_keep + y_new_

        y_coordinate_map = einops.rearrange(y, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

        return y_coordinate_map, x_coordinate_map


    def get_interpolated_feature(self,
        input_feature: torch.Tensor,
        y_coordinate_map: torch.Tensor,
        x_coordinate_map: torch.Tensor,
        interpolate_mode: str = "bilinear",
    ):
        if interpolate_mode not in ("bilinear", "bicubic"):
            raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

        y_max = input_feature.shape[-2] - 1
        x_max = input_feature.shape[-1] - 1

        y_coordinate_map_ = self._coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
        x_coordinate_map_ = self._coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

        y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
        x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

        # Note here grid with shape [B, H, W, 2]
        # Where [:, :, :, 2] refers to [x ,y]
        grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

        interpolated_feature = nn.functional.grid_sample(
            input=input_feature,
            grid=grid,
            mode=interpolate_mode,
            padding_mode="zeros",
            align_corners=True,
        )

        return interpolated_feature


    def _coordinate_map_scaling(self,
        coordinate_map: torch.Tensor,
        origin: list,
        target: list = [-1, 1],
    ):
        min, max = origin
        a, b = target

        coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

        scale_factor = (b - a) / (max - min)
        coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

        return coordinate_map_scaled


    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = self.get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = self.get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        # output = self.relu(output)

        return output



class HPPF(nn.Module):
    def __init__(self, in_channels):
        super(HPPF, self).__init__()
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 64, 1, 1), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(MMConv(in_channels, in_channels // 16, num_slices=64, kernel_size=1), nn.ReLU(inplace=True))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max1 = nn.AdaptiveMaxPool2d(4)
        self.max2 = nn.AdaptiveMaxPool2d(8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid())
        self.feat_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1),
                                       nn.BatchNorm2d(in_channels // 3),
                                       nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        feat = torch.cat((x1, x2, x3), 1)

        b, c, h, w = feat.size()
        y1 = self.avg(feat)
        y2 = self.conv1(self.max1(feat))
        y3 = self.conv2(self.max2(feat))
        y2 = y2.reshape(b, c, 1, 1)
        y3 = y3.reshape(b, c, 1, 1)
        z = (y1 + y2 + y3) / 3
        attention = self.mlp(z)
        output1 = attention * feat
        output2 = self.feat_conv(output1)
        return output2


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c_avg = self.mlp(self.avg_pool(x))
        c_max = self.mlp(self.max_pool(x))
        c_out = self.sigmoid(c_avg + c_max)
        y1 = c_out * x

        s_avg = torch.mean(y1, dim=1, keepdim=True)
        s_max, _ = torch.max(y1, dim=1, keepdim=True)
        s_out = torch.cat((s_max, s_avg), 1)
        s_out = self.sigmoid(self.conv(s_out))
        output = s_out * y1

        return output


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_slices=4):
        super(SideoutBlock, self).__init__()
        self.conv1 = nn.Sequential(MMConv(in_channels, in_channels//4, num_slices = num_slices, kernel_size=3), nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class RCG(nn.Module):
    def __init__(self, d_state = 16, d_conv = 4, expand = 2, head=4, num_slices=4, step = 1):
        super(RCG, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(MMConv(128, 64, num_slices=num_slices, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.upsample = nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,           # 常见设置
                    output_padding=0     # 保持标准尺寸
                )
        
        self.downsample = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.mamba = Mamba(
                d_model=64, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v1",
                bimamba_type="v3",   # TODO: set 154 assert bimamba_type=="v3" as none
                nslices = num_slices
        )
        
        self.mlp = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, pre, edge, f):
        f_att = torch.sigmoid(pre)
        r_att = -1 * f_att + 1
        r = r_att * f

        edge1 = F.interpolate(edge, size=f.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat((edge1, r), 1)
        x2 = self.conv1(x1)
        
        ######### Mamba ###########
        x0 = self.upsample(x2)
        B, C, H, W = x0.shape
        
        n_tokens = x0.shape[2:].numel()
        img_dims = x0.shape[2:]
        
        x_flat = x0.reshape(B, C, n_tokens).transpose(-1, -2)
        
        out, q, k, v = self.mamba(x_flat)
        
        out_m = out.transpose(-1, -2).reshape(B, C, *img_dims)
        
        x0 = self.downsample(out_m)
        ######### Mamba ###########
        
        x3 = self.mlp(x2)
        x4 = x0 * x3 * x2 
        output = x4 + f

        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_slices=4):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(MMConv(in_channels, in_channels // 4, kernel_size=3, num_slices=num_slices),nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(MMConv(in_channels//4, out_channels, kernel_size=3,num_slices=num_slices), nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        return x3

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_slices,downsample=False):
        super(ResidualBlock, self).__init__()

        self.downsample = downsample
        if downsample:
            self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            MMConv(out_channels, out_channels, num_slices=num_slices, kernel_size=3),
            nn.BatchNorm2d(out_channels)
        )
            self.block2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block1 = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            MMConv(in_channels, out_channels, num_slices=num_slices,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            MMConv(out_channels, out_channels, num_slices=num_slices,kernel_size=3),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.block1(x)
        if  self.downsample:
            return self.relu(self.block2(x) + x1)
        return self.relu(x1 + x)
    





class MM_Net(nn.Module):
    def __init__(self, num_classes, num_slices_list = [64, 32, 16, 8],
                 out_indices=[0, 1, 2, 3], heads=[1, 2, 4, 4]):
        super(MM_Net, self).__init__()
        # resnet = models.resnet34(pretrained=True)
        print('loading pretrained model--MMUNET')
        # checkpoint_path = "resnet/pytorch_model.bin"
        # resnet.load_state_dict(torch.load(checkpoint_path, weights_only=False))
        # Encoder
        # self.encoder1_conv = resnet.conv1
        # self.encoder1_bn = resnet.bn1
        # self.encoder1_relu = resnet.relu  # 64
        # self.maxpool = resnet.maxpool
        # self.encoder2 = resnet.layer1  # 64
        # self.encoder3 = resnet.layer2  # 128
        # self.encoder4 = resnet.layer3  # 256
        # self.encoder5 = resnet.layer4  # 512
        # print(self.encoder3, self.encoder4, self.encoder5)
        self.encoder1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.encoder2 = nn.Sequential(ResidualBlock(64, 64, num_slices=num_slices_list[0]),ResidualBlock(64, 64, num_slices=num_slices_list[0]),ResidualBlock(64, 64, num_slices=num_slices_list[0]))
        self.encoder3 = nn.Sequential(ResidualBlock(64, 128, downsample=True, num_slices=num_slices_list[1]),ResidualBlock(128, 128, num_slices=num_slices_list[1]),ResidualBlock(128, 128, num_slices=num_slices_list[1]),ResidualBlock(128, 128, num_slices=num_slices_list[1]))
        self.encoder4 = nn.Sequential(ResidualBlock(128, 256, downsample=True,num_slices=num_slices_list[2]),ResidualBlock(256, 256,num_slices=num_slices_list[2]),ResidualBlock(256, 256,num_slices=num_slices_list[2]),ResidualBlock(256, 256,num_slices=num_slices_list[2]),ResidualBlock(256, 256,num_slices=num_slices_list[2]),ResidualBlock(256, 256,num_slices=num_slices_list[2]))
        self.encoder5 = nn.Sequential(ResidualBlock(256, 512, downsample=True,num_slices=num_slices_list[3]),ResidualBlock(512, 512,num_slices=num_slices_list[3]),ResidualBlock(512, 512,num_slices=num_slices_list[3]))
        # self.down3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.down4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.down5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.down3 = nn.Sequential(MMConv(128,64,num_slices=num_slices_list[-1],kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(MMConv(256,64,num_slices=num_slices_list[-1],kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down5 = nn.Sequential(MMConv(512,64,num_slices=num_slices_list[-1],kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.hpp = HPPF(192)

        self.cbam = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),CBAM(64),nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


        self.line_predict = nn.Conv2d(64, 1, 3, 1, 1)

        self.side2 = SideoutBlock(64, 1, num_slices=num_slices_list[0])
        self.side3 = SideoutBlock(64, 1, num_slices=num_slices_list[1])
        self.side4 = SideoutBlock(64, 1, num_slices=num_slices_list[2])
        self.side5 = SideoutBlock(64, 1, num_slices=num_slices_list[3])

        self.rcg2 = RCG( num_slices=num_slices_list[0], head = heads[0])
        self.rcg3 = RCG( num_slices=num_slices_list[1], head = heads[1])
        self.rcg4 = RCG( num_slices=num_slices_list[2], head = heads[2])

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=64, out_channels=64, num_slices=num_slices_list[3])
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=64, num_slices=num_slices_list[2])
        self.decoder3 = DecoderBlock(in_channels=128, out_channels=64, num_slices=num_slices_list[1])
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64, num_slices=num_slices_list[0])

        # self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        #                            nn.Dropout2d(0.1),
        #                            nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        # e1 = self.encoder1_conv(x)
        # e1 = self.encoder1_bn(e1)
        # e1 = self.encoder1_relu(e1)
        e1 = self.encoder1(x)
        e1_pool = self.maxpool(e1)

        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        e3 = self.down3(e3)  # 64
        e4 = self.down4(e4)  # 64
        e5 = self.down5(e5)  # 64

        # decoder5
        d5 = self.decoder5(e5)
        out5 = self.side5(d5)

        # e1_Contour
        c1 = self.cbam(e1)
        # c1 = e1

        p_c = self.line_predict(c1)

        # decoder4
        r4 = self.rcg4(out5, c1, e4)
        d41 = torch.cat((d5, r4), dim=1)
        d4 = self.decoder4(d41)
        out4 = self.side4(d4)

        # decoder3
        r3 = self.rcg3(out4, c1, e3)
        d31 = torch.cat((d4, r3), dim=1)
        d3 = self.decoder3(d31)
        out3 = self.side3(d3)

        # decoder2
        r2 = self.rcg2(out3, c1, e2)
        d21 = torch.cat((d3, r2), dim=1)
        d2 = self.decoder2(d21)
        out2 = self.side2(d2)

        # final_output
        other_out = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)+ F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)+ F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)+ F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)+ F.interpolate(p_c, size=x.size()[2:], mode='bilinear', align_corners=True)

        # p = self.hpp(d2, d3, d4)
        # out1 = self.final(p)
        # out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)

        
        # return out1 + other_out
        return other_out


if __name__ == '__main__':
    # from thop import profile, clever_format
    device = 'cuda:0'
    
    x = torch.randn(size=(1, 3, 608, 608)).to(device)
    model = MM_Net(num_classes=1).to(device)
    print(model(x).size())
    print(model(x))