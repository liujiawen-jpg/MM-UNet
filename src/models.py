
from src.DuAT.DuAT import DuAT
from src.Unetr.Unetr import u_netr
from src.SwinUNETR.SwinUNETR import swin_unetr
from src.CFPnet.CFPnet import CFPNet
from src.TransUnet.TransUnet import TransUNet
from src.CVCUNETR.CVCUNETR import CVCUnetr
from src.Unet.Unet import UNet
from src.FCBFormer.models import FCBFormer
from src.CVCUNETR.NewCVC import CVC_Unetr
from src.CFANet.CFANet import CFANet
from src.PVT_CA.PVT_CA import PVT_CASCADE
from src.UM_Net.UM_Net import UM_Net
from src.BMANet.BMANet import BMANet
from src.VANet.VANet import VANet
# from src.FRUNet.FRUNet import FR_UNet
from src.ConvUneXt.ConvNeXt import ConvUNeXt
# from src.UNet3Plus.UNet3Plus import UNet3Plus
# from src.ATTUNet.ATTUNet import ATTUNet
from src.UM_Net.MMUNet import MM_Net
from src.devDualNet.devDualNet import dkDualNet
def give_model(config):
    if config.finetune.model_choose == 'CVC_UNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = CVC_Unetr(**config.models.cvc_unetr.branch1)
        else:
            model = CVC_Unetr(**config.models.cvc_unetr.branch5)
    elif config.finetune.model_choose == 'TransUNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = TransUNet(**config.models.trans_unet.branch1)
        else:
            model = TransUNet(**config.models.trans_unet.branch5)
    elif config.finetune.model_choose == 'CFPNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = CFPNet(**config.models.cfp_net.branch1)
        else:
            model = CFPNet(**config.models.cfp_net.branch5)
    elif config.finetune.model_choose == 'UNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = u_netr(**config.models.u_netr.branch1)
        else:
            model = u_netr(**config.models.u_netr.branch5)
    elif config.finetune.model_choose == 'SWINUNETR':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = swin_unetr(**config.models.swin_unetr.branch1)
        else:
            model = swin_unetr(**config.models.swin_unetr.branch5)
    elif config.finetune.model_choose == 'DuAT':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = DuAT(**config.models.duat.branch1)
        else:
            model = DuAT(**config.models.duat.branch5)
    elif config.finetune.model_choose == 'UNet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = UNet(**config.models.unet.branch1)
        else:
            model = UNet(**config.models.unet.branch5)
    elif config.finetune.model_choose == 'FCBFormer':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = FCBFormer(**config.models.FCBFormer.branch1)
        else:
            model = FCBFormer(**config.models.FCBFormer.branch5)
    elif config.finetune.model_choose == 'CFANet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = CFANet(**config.models.cfa_net.branch1)
        else:
            model = CFANet(**config.models.cfa_net.branch5)
    elif config.finetune.model_choose == 'PVT_CASCADE':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = PVT_CASCADE(**config.models.pvt_ca.branch1)
        else:
            model = PVT_CASCADE(**config.models.pvt_ca.branch5)
    elif config.finetune.model_choose == 'UM_Net':
        model = UM_Net(**config.models.um_net.branch1)
    elif config.finetune.model_choose == 'MM_Net':
        model = MM_Net(**config.models.MM_Net.branch1)
    elif config.finetune.model_choose == 'dkDualNet':
        model = dkDualNet(**config.models.dkDualNet.branch1)
    elif config.finetune.model_choose == 'FRUNet':
        model = FR_UNet(**config.models.FRUNet.branch1)
    elif config.finetune.model_choose == 'ConvUNetXt':
        model = ConvUNeXt(**config.models.ConvUNetXt.branch1)
    elif config.finetune.model_choose == 'UNet3Plus':
        model = UNet3Plus(**config.models.UNet3Plus.branch1)
    elif config.finetune.model_choose == 'ATTUNet':
        model = ATTUNet(**config.models.ATTUNet.branch1)
    elif config.finetune.model_choose == 'ATTUNet':
        model = ATTUNet(**config.models.ATTUNet.branch1)
    elif config.finetune.model_choose == 'BMANet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = BMANet(**config.models.bmanet.branch1)
        else:
            model = BMANet(**config.models.bmanet.branch5)
    
    elif config.finetune.model_choose == 'VANet':
        if config.trainer.dataset_choose != 'EDD_seg':
            model = VANet(**config.models.vanet.branch1)
        else:
            model = VANet(**config.models.vanet.branch5)
    return model