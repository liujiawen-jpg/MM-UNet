import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class u_netr(UNETR):
    pass

if __name__ == '__main__':
    device = 'cuda:0'
    
    x = torch.randn(size=(2, 3, 352, 352)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)
    
    model = u_netr(spatial_dims=2,
                   img_size=352,
                in_channels=3,
                out_channels=1).to(device)
    # module = AttentionBlock(in_dim=64, out_dim=32, kernel_size=3, mlp_ratio=4, shallow=True).to(device)
    
    print(model(x).size())
    # print(module(test_x).size())