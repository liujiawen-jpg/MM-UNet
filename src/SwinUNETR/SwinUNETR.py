import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class swin_unetr(SwinUNETR):
    pass

if __name__ == '__main__':
    device = 'cuda:0'
    
    x = torch.randn(size=(2, 3, 352, 352)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)
    
    model = swin_unetr(img_size=(352,352), in_channels=3, out_channels=1, use_checkpoint=True, spatial_dims=2).to(device)
    # module = AttentionBlock(in_dim=64, out_dim=32, kernel_size=3, mlp_ratio=4, shallow=True).to(device)
    
    print(model(x).size())
    # print(module(test_x).size())