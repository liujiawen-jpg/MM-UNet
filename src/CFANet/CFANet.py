import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch
import math
import torch.utils.model_zoo as model_zoo

class Res2Net_Ours(nn.Module):

    def __init__(self, in_class, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net_Ours, self).__init__()
        
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_class, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)


        return x0,x1,x2,x3,x4

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)

        return x7

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'],map_location='cpu'))
        
        model_state = torch.load('./lib/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
        
    return model


def res2net50_v1b_Ours(in_class=1, pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net_Ours(in_class, Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def Res2Net_model(in_class=1, ind=50):

    model = res2net50_v1b_Ours(in_class = in_class)
    return model

class global_module(nn.Module):
    def __init__(self, channels=64, r=4):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att
      
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xg  = self.global_att(x)
        out = self.sig(xg)

        return out
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
  

class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()
        
        self.gate_1 = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        
        ###
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion


class BAM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(BAM, self).__init__()
        
        
        self.relu       = nn.ReLU(True)
        
        self.global_att = global_module(channel)

        self.conv_layer = BasicConv2d(channel*2, channel, 3, padding=1)


    def forward(self, x, x_boun_atten):
               
        out1 = self.conv_layer(torch.cat((x, x_boun_atten), dim=1))
        out2 = self.global_att(out1)
        out3 = out1.mul(out2)
        
        out = x + out3
        
        return out
    

class CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(CFF, self).__init__()
        

        act_fn         = nn.ReLU(inplace=True)
                
        ## ---------------------------------------- ##
        self.layer0    = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1    = BasicConv2d(in_channel2, out_channel // 2, 1)
        
        self.layer3_1  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),  nn.BatchNorm2d(out_channel // 2),act_fn)
        self.layer3_2  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),  nn.BatchNorm2d(out_channel // 2),act_fn)
        
        self.layer5_1  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),  nn.BatchNorm2d(out_channel // 2),act_fn)
        self.layer5_2  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),  nn.BatchNorm2d(out_channel // 2),act_fn)
        
        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel),act_fn)


    def forward(self, x0, x1):
        
        ## ------------------------------------------------------------------ ##
        x0_1  = self.layer0(x0)
        x1_1  = self.layer1(x1)
        
        x_3_1 = self.layer3_1(torch.cat((x0_1,  x1_1),  dim=1))    
        x_5_1 = self.layer5_1(torch.cat((x1_1,  x0_1),  dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))
        
        out   = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))
        
        return out

    
    
###############################################################################
## 2022/01/03
###############################################################################
class CFANet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, in_class=1, out_class=1, channel=64, opt=None):
        super(CFANet, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)

        self.resnet     = Res2Net_model(in_class, 50)
        self.downSample = nn.MaxPool2d(2, stride=2)
        
        ## ---------------------------------------- ##
        
        self.layer0 = nn.Sequential(nn.Conv2d(64,  channel,  kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer1 = nn.Sequential(nn.Conv2d(256, channel, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(channel),act_fn)

        self.low_fusion  = GateFusion(channel)

        self.high_fusion1 = CFF(256,  512,  channel)
        self.high_fusion2 = CFF(1024, 2048, channel)
         
        ## ---------------------------------------- ##
        self.layer_edge0 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_edge1 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_edge2 = nn.Sequential(nn.Conv2d(channel, 64,   kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),act_fn)
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, out_class,   kernel_size=1))
        
        ## ---------------------------------------- ##
        # self.layer_cat_ori1 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig01 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)

        # self.layer_cat11 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig11 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        
        # self.layer_cat21 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig21 = nn.Sequential(nn.Conv2d(channel, 64,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),act_fn)

        # self.layer_cat31 = nn.Sequential(nn.Conv2d(64*2, 64,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),act_fn)
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, out_class,  kernel_size=1))

        # self.layer_cat_ori2 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig02 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)

        # self.layer_cat12 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig12 = nn.Sequential(nn.Conv2d(channel, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        
        # self.layer_cat22 = nn.Sequential(nn.Conv2d(channel*2, channel,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_hig22 = nn.Sequential(nn.Conv2d(channel, 64,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),act_fn)

        # self.layer_cat32 = nn.Sequential(nn.Conv2d(64*2, 64,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),act_fn)
        self.layer_hig32 = nn.Sequential(nn.Conv2d(64, out_class,  kernel_size=1))
        
        self.layer_fil = nn.Sequential(nn.Conv2d(64, out_class,  kernel_size=1))
        
        ## ---------------------------------------- ##
        
        self.atten_edge_0 = ChannelAttention(channel)
        self.atten_edge_1 = ChannelAttention(channel)
        self.atten_edge_2 = ChannelAttention(channel)
        self.atten_edge_ori = ChannelAttention(channel)
        
        
        self.cat_01 = BAM(channel)
        self.cat_11 = BAM(channel)
        self.cat_21 = BAM(channel)
        self.cat_31 = BAM(channel)

        self.cat_02 = BAM(channel)
        self.cat_12 = BAM(channel)
        self.cat_22 = BAM(channel)
        self.cat_32 = BAM(channel)
        
        
        ## ---------------------------------------- ##
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        


    def forward(self, xx):
        
        # ---- feature abstraction -----
        
        x0, x1, x2, x3, x4 = self.resnet(xx)
         
      
        
        ## -------------------------------------- ## 
        
        x0_1  = self.layer0(x0)
        x1_1  = self.layer1(x1)
        
        low_x = self.low_fusion(x0_1, x1_1)                 # 64*44

        
        edge_out0 = self.layer_edge0(self.up_2(low_x))      # 64*88
        edge_out1 = self.layer_edge1(self.up_2(edge_out0))  # 64*176
        edge_out2 = self.layer_edge2(self.up_2(edge_out1))  # 64*352
        edge_out3 = self.layer_edge3(edge_out2)
        
        
        etten_edge_ori = self.atten_edge_ori(low_x)
        etten_edge_0   = self.atten_edge_0(edge_out0)
        etten_edge_1   = self.atten_edge_1(edge_out1)
        etten_edge_2   = self.atten_edge_2(edge_out2)
        
 
        ## -------------------------------------- ##
        high_x01   = self.high_fusion1(self.downSample(x1), x2)
        high_x02   = self.high_fusion2(self.up_2(x3), self.up_4(x4))
        
        ## --------------- high 1 ----------------------- # 
        cat_out_01 = self.cat_01(high_x01,low_x.mul(etten_edge_ori))
        hig_out01  = self.layer_hig01(self.up_2(cat_out_01))
        
        cat_out11  = self.cat_11(hig_out01,edge_out0.mul(etten_edge_0))
        hig_out11  = self.layer_hig11(self.up_2(cat_out11))
        
        cat_out21  = self.cat_21(hig_out11,edge_out1.mul(etten_edge_1))
        hig_out21  = self.layer_hig21(self.up_2(cat_out21))

        cat_out31  = self.cat_31(hig_out21,edge_out2.mul(etten_edge_2))
        sal_out1   = self.layer_hig31(cat_out31)   
        
        ## ---------------- high 2 ---------------------- ##
        cat_out_02 = self.cat_02(high_x02,low_x.mul(etten_edge_ori))
        hig_out02  = self.layer_hig02(self.up_2(cat_out_02))
        
        cat_out12  = self.cat_12(hig_out02,edge_out0.mul(etten_edge_0))  
        hig_out12  = self.layer_hig12(self.up_2(cat_out12))
        
        cat_out22  = self.cat_22(hig_out12,edge_out1.mul(etten_edge_1))    
        hig_out22  = self.layer_hig22(self.up_2(cat_out22))

        cat_out32  = self.cat_32(hig_out22,edge_out2.mul(etten_edge_2))    
        sal_out2   = self.layer_hig32(cat_out32) 
        
        ## --------------------------------------------- ##
        sal_out3   = self.layer_fil(cat_out31+cat_out32)
        
        # ---- output ----
        return edge_out3+sal_out1+sal_out2+sal_out3   

if __name__ == '__main__':
    device = 'cpu'
    x = torch.randn(size=(2, 3, 352, 352)).to(device)
    model = CFANet(in_class=3, channel=64).to(device)
#    summary(model, (1, 3, 352, 352))
    print(model(x).size())