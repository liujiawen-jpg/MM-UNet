import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
from src.UM_Net.DSC_conv import DSConv_pro
# from DSC_conv import DSConv_pro
from mamba_ssm import Mamba

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)
        self.g = DSConv_pro(in_channels=self.in_channels, out_channels=self.inter_channels,)

        if bn_layer:
            self.W = nn.Sequential(
                # nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                #           kernel_size=1, stride=1, padding=0),
                DSConv_pro(in_channels=self.inter_channels, out_channels=self.in_channels,),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            # self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
            #                    kernel_size=1, stride=1, padding=0)
            self.W = DSConv_pro(in_channels=self.inter_channels, out_channels=self.in_channels,)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        self.theta = DSConv_pro(in_channels=self.in_channels, out_channels=self.inter_channels,)
        # self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                        kernel_size=1, stride=1, padding=0)
        self.phi = DSConv_pro(in_channels=self.in_channels, out_channels=self.inter_channels,)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class HPPF(nn.Module):
    def __init__(self, in_channels):
        super(HPPF, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 64, 1, 1), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(DSConv_pro(in_channels, in_channels // 16, ), nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(DSConv_pro(in_channels, in_channels // 64, ), nn.ReLU(inplace=True))
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


class ALGM(nn.Module):
    def __init__(self, mid_ch, pool_size=(), out_list=(), cascade=False, y_flag=True):
        super(ALGM, self).__init__()
        in_channels = mid_ch // 4
        self.cascade = cascade
        self.out_list = out_list
        size = [1, 2, 3]
        LGlist = []
        LGoutlist = []

        LGlist.append(NonLocalBlock(in_channels))
        for i in size:
            LGlist.append(nn.Sequential(
                nn.Conv2d(in_channels*i, in_channels, 3, stride=1, padding=pool_size[i-1], dilation=pool_size[i-1]),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)))
        self.LGmodule = nn.ModuleList(LGlist)

        for j in range(len(self.out_list)):
            LGoutlist.append(nn.Sequential(SELayer(in_channels*4),
                                           nn.Conv2d(in_channels * 4, self.out_list[j], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_list[j]),
                                           nn.ReLU(inplace=True)))
        self.LGoutmodel = nn.ModuleList(LGoutlist)
        self.conv1 = nn.Sequential(nn.Conv2d(mid_ch, in_channels, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        if y_flag == True:
            self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x, y=None):
        xsize = x.size()[2:]
        x = self.conv1(x)
        lg_context = []
        for i in range(2):
            lg_context.append(self.LGmodule[i](x))
        x1 = torch.cat((x, lg_context[0]), 1)
        lg_context.append(self.LGmodule[2](x1))
        x2 = torch.cat((x, lg_context[1], lg_context[2]), 1)
        lg_context.append(self.LGmodule[3](x2))
        lg_context = torch.cat(lg_context, dim=1)

        output = []
        for i in range(len(self.LGoutmodel)):
            out = self.LGoutmodel[i](lg_context)
            if self.cascade is True and y is not None:
                m = self.conv2(abs(F.interpolate(y[i], xsize, mode='bilinear', align_corners=True) - out))
                
                out = out + m
                # out = out
            output.append(out)

        return output


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


class RCG(nn.Module):
    def __init__(self, d_state = 16, d_conv = 4, expand = 2, head=4, num_slices=4, step = 1):
        super(RCG, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(DSConv_pro(128, 64, ), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
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
                # bimamba_type="v3",   # TODO: set 154 assert bimamba_type=="v3" as none
                # nslices = num_slices
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
        
        out = self.mamba(x_flat)
        
        out_m = out.transpose(-1, -2).reshape(B, C, *img_dims)
        
        x0 = self.downsample(out_m)
        ######### Mamba ###########
        
        x3 = self.mlp(x2)
        x4 = x0 * x3 * x2 
        output = x4 + f

        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(DSConv_pro(in_channels, in_channels // 4, ),#nn.Conv2d(in_channels, in_channels // 4,  3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(DSConv_pro(in_channels//4, out_channels),#nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        return x3


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SideoutBlock, self).__init__()
        self.conv1 = nn.Sequential(DSConv_pro(in_channels, in_channels//4), #nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


# class UM_Net(nn.Module):
#     def __init__(self, num_classes, num_slices_list = [64, 32, 16, 8],
#                  out_indices=[0, 1, 2, 3], heads=[1, 2, 4, 4]):
#         super(UM_Net, self).__init__()
#         resnet = models.resnet34(pretrained=True)
#         print('loading pretrained model')
#         # checkpoint_path = "resnet/pytorch_model.bin"
#         # resnet.load_state_dict(torch.load(checkpoint_path, weights_only=False))

#         # Encoder
#         self.encoder1_conv = resnet.conv1
#         self.encoder1_bn = resnet.bn1
#         self.encoder1_relu = resnet.relu  # 64
#         self.maxpool = resnet.maxpool
#         self.encoder2 = resnet.layer1  # 64
#         self.encoder3 = resnet.layer2  # 128
#         self.encoder4 = resnet.layer3  # 256
#         self.encoder5 = resnet.layer4  # 512

#         self.down3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.down4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.down5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

#         self.hpp = HPPF(192)
#         self.cbam = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#                                   CBAM(64),
#                                   nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.line_predict = nn.Conv2d(64, 1, 3, 1, 1)

#         self.lg5 = ALGM(64, pool_size=[1, 3, 5], out_list=[64, 64, 64, 64], y_flag=False)
#         self.lg4 = ALGM(64, pool_size=[2, 6, 10], out_list=[64, 64, 64], cascade=True)
#         self.lg3 = ALGM(64, pool_size=[3, 9, 15], out_list=[64, 64], cascade=True)
#         self.lg2 = ALGM(64, pool_size=[4, 12, 20], out_list=[64], cascade=True)

#         self.side2 = SideoutBlock(64, 1)
#         self.side3 = SideoutBlock(64, 1)
#         self.side4 = SideoutBlock(64, 1)
#         self.side5 = SideoutBlock(64, 1)

#         self.rcg2 = RCG( num_slices=num_slices_list[0], head = heads[0])
#         self.rcg3 = RCG( num_slices=num_slices_list[1], head = heads[1])
#         self.rcg4 = RCG( num_slices=num_slices_list[2], head = heads[2])

#         # Decoder
#         self.decoder5 = DecoderBlock(in_channels=64, out_channels=64)
#         self.decoder4 = DecoderBlock(in_channels=192, out_channels=64)
#         self.decoder3 = DecoderBlock(in_channels=192, out_channels=64)
#         self.decoder2 = DecoderBlock(in_channels=192, out_channels=64)

#         self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#                                    nn.Dropout2d(0.1),
#                                    nn.Conv2d(32, num_classes, kernel_size=1))

#     def forward(self, x):
#         e1 = self.encoder1_conv(x)
#         e1 = self.encoder1_bn(e1)
#         e1 = self.encoder1_relu(e1)
#         e1_pool = self.maxpool(e1)
#         e2 = self.encoder2(e1_pool)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#         e5 = self.encoder5(e4)

#         e3 = self.down3(e3)  # 64
#         e4 = self.down4(e4)  # 64
#         e5 = self.down5(e5)  # 64

#         lg5 = self.lg5(e5)
#         lg4 = self.lg4(e4, lg5[1:])
#         lg3 = self.lg3(e3, lg4[1:])
#         lg2 = self.lg2(e2, lg3[1:])

#         # decoder5
#         d5 = self.decoder5(lg5[0])
#         out5 = self.side5(d5)

#         # e1_Contour
#         c1 = self.cbam(e1)
#         p_c = self.line_predict(c1)

#         # decoder4
#         r4 = self.rcg4(out5, c1, e4)
#         d41 = torch.cat((d5, lg4[0], r4), dim=1)
#         d4 = self.decoder4(d41)
#         out4 = self.side4(d4)

#         # decoder3
#         r3 = self.rcg3(out4, c1, e3)
#         d31 = torch.cat((d4, lg3[0], r3), dim=1)
#         d3 = self.decoder3(d31)
#         out3 = self.side3(d3)

#         # decoder2
#         r2 = self.rcg2(out3, c1, e2)
#         d21 = torch.cat((d3, lg2[0], r2), dim=1)
#         d2 = self.decoder2(d21)
#         out2 = self.side2(d2)

#         # final_output
#         p = self.hpp(d2, d3, d4)
#         out1 = self.final(p)
#         out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)

#         other_out = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(p_c, size=x.size()[2:], mode='bilinear', align_corners=True)
#         return out1+other_out
        # return torch.sigmoid(out1)

class UM_Net(nn.Module):
    def __init__(self, num_classes, num_slices_list = [64, 32, 16, 8],
                 out_indices=[0, 1, 2, 3], heads=[1, 2, 4, 4]):
        super(UM_Net, self).__init__()
        resnet = models.resnet34(pretrained=True)
        print('loading pretrained model')
        # checkpoint_path = "resnet/pytorch_model.bin"
        # resnet.load_state_dict(torch.load(checkpoint_path, weights_only=False))

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu  # 64
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512

        self.down3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.hpp = HPPF(192)
        self.cbam = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  CBAM(64),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.line_predict = nn.Conv2d(64, 1, 3, 1, 1)

        # self.lg5 = ALGM(64, pool_size=[1, 3, 5], out_list=[64, 64, 64, 64], y_flag=False)
        # self.lg4 = ALGM(64, pool_size=[2, 6, 10], out_list=[64, 64, 64], cascade=True)
        # self.lg3 = ALGM(64, pool_size=[3, 9, 15], out_list=[64, 64], cascade=True)
        # self.lg2 = ALGM(64, pool_size=[4, 12, 20], out_list=[64], cascade=True)

        self.side2 = SideoutBlock(64, 1)
        self.side3 = SideoutBlock(64, 1)
        self.side4 = SideoutBlock(64, 1)
        self.side5 = SideoutBlock(64, 1)

        self.rcg2 = RCG( num_slices=num_slices_list[0], head = heads[0])
        self.rcg3 = RCG( num_slices=num_slices_list[1], head = heads[1])
        self.rcg4 = RCG( num_slices=num_slices_list[2], head = heads[2])

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=64, out_channels=64)
        # self.decoder4 = DecoderBlock(in_channels=192, out_channels=64)
        # self.decoder3 = DecoderBlock(in_channels=192, out_channels=64)
        # self.decoder2 = DecoderBlock(in_channels=192, out_channels=64)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder3 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)

        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        e3 = self.down3(e3)  # 64
        e4 = self.down4(e4)  # 64
        e5 = self.down5(e5)  # 64

        # lg5 = self.lg5(e5)
        # lg4 = self.lg4(e4, lg5[1:])
        # lg3 = self.lg3(e3, lg4[1:])
        # lg2 = self.lg2(e2, lg3[1:])

        # decoder5
        d5 = self.decoder5(e5)
        out5 = self.side5(d5)

        # e1_Contour
        c1 = self.cbam(e1)
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
        p = self.hpp(d2, d3, d4)
        out1 = self.final(p)
        out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)

        other_out = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)+F.interpolate(p_c, size=x.size()[2:], mode='bilinear', align_corners=True)
        return out1+other_out

if __name__ == '__main__':
    # from thop import profile, clever_format
    device = 'cuda:0'
    
    x = torch.randn(size=(1, 3, 608, 608)).to(device)
    model = UM_Net(num_classes=1).to(device)
    print(model(x).size())
    print(model(x))