import json
import os
from typing import Tuple, List, Mapping, Hashable, Dict

import monai
from monai.networks.utils import one_hot
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from PIL import Image
import cv2
from monai.transforms import MapTransform
from monai.config import KeysCollection


class ConvertToMultiChannelBasedOnBratsClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: monai.config.KeysCollection, is2019: bool = False, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [(img == 2) | (img == 3), (img == 1) | (img == 2) | (img == 3), (img == 2)]
        else:
            # TC WT ET
            result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]) -> Dict[
        Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(monai.transforms.MapTransform):
    """
       TC WT ET
       Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
       Convert labels to multi channels based on brats18 classes:
       label 1 is the necrotic and non-enhancing tumor core
       label 2 is the peritumoral edema
       label 4 is the GD-enhancing tumor
       The possible classes are TC (Tumor core), WT (Whole tumor)
       and ET (Enhancing tumor).
       """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: monai.config.KeysCollection,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [(img == 1), (img == 2)]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]) -> Dict[
        Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + '/' + path + '/' + path
        flair_img = image_path + '_flair.nii.gz'
        t1_img = image_path + '_t1.nii.gz'
        t1ce_img = image_path + '_t1ce.nii.gz'
        t2_img = image_path + '_t2.nii.gz'
        seg_img = image_path + '_seg.nii.gz'
        images_list.append({
            'image': [flair_img, t1_img, t1ce_img, t2_img],
            'label': seg_img
        })
    return images_list


def load_brats2019_dataset_images(root):
    root_dir = root + '/dataset.json'
    # 读打开文件
    with open(root_dir, encoding='utf-8') as a:
        # 读取文件
        images_list = json.load(a)['training']
        for image in images_list:
            image['image'] = image['image'].replace('./', root + '/')
            image['label'] = image['label'].replace('./', root + '/')
    return images_list


def get_Brats_transforms(config: EasyDict) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"], is2019=config.trainer.is_brats2019),
        monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        monai.transforms.SpatialPadD(keys=["image", "label"], spatial_size=(255, 255, config.trainer.image_size.BraTS),
                                     method='symmetric', mode='constant'),

        monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        monai.transforms.CenterSpatialCropD(keys=["image", "label"],
                                            roi_size=ensure_tuple_rep(config.trainer.image_size.BraTS, 3)),

        # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
        monai.transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", num_samples=2,
                                                spatial_size=ensure_tuple_rep(config.trainer.image_size.BraTS, 3), pos=1,
                                                neg=1,
                                                image_key="image", image_threshold=0),
        # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=config.model.image_size, random_size=False),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        monai.transforms.ToTensord(keys=["image", "label"]),
    ])
    val_transform = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", is2019=config.trainer.is_brats2019),
        monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
        monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    return train_transform, val_transform


def load_MSD_dataset_images(root):
    img_dir = root + '/imagesTr'
    lab_dir = root + '/labelsTr'
    file_list = os.listdir(img_dir)
    label_list = os.listdir(lab_dir)
    images_list = []
    for file in file_list:
        file_dict = {}
        if '._' not in file:
            if file in label_list:
                file_dict['image'] = img_dir + '/' + file
                file_dict['label'] = lab_dir + '/' + file
                images_list.append(file_dict)
    return images_list


def load_dataset_images(root):
    root_dir = root + '/dataset.json'
    # 读打开文件
    with open(root_dir, encoding='utf-8') as a:
        # 读取文件
        images_list = json.load(a)['training']
        # images_val_list = json.load(a)['test']
        for image in images_list:
            image['image'] = image['image'].replace('./', root + '/')
            image['label'] = image['label'].replace('./', root + '/')
        # for image in images_val_list:
        #     image['image'] = image['image'].replace('./', root + '/')
        #     image['label'] = image['label'].replace('./', root + '/')
    return images_list


def get_MSD_transforms(config: EasyDict) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            # 前景裁剪
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # 强度限制
            monai.transforms.ScaleIntensityRanged(keys=["image", "label"], a_min=0.0, a_max=230.0, b_min=0.0,
                                                  b_max=230.0, clip=True),
            monai.transforms.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", num_samples=2,
                                                    spatial_size=monai.utils.ensure_tuple_rep(config.trainer.image_size.MSD,3), pos=2,neg=0,
                                                    image_key="image", image_threshold=0),
            # monai.transforms.RandScaleCropD(keys=["image", "label"],roi_scale=monai.utils.ensure_tuple_rep(config.trainer.image_size,3)),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            monai.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.5),
            monai.transforms.RandRotated(keys=["image", "label"], prob=0.25),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            monai.transforms.ToTensord(keys=['image', 'label'])
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            # 强度限制
            monai.transforms.ScaleIntensityRanged(keys=["image", "label"], a_min=0.0, a_max=230.0, b_min=0.0,
                                                  b_max=230.0, clip=True),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform, val_transform


def get_dataloader(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.trainer.is_HepaticVessel:
        train_images = load_MSD_dataset_images(config.trainer.MSD_HepaticVessel)
        train_transform, val_transform = get_MSD_transforms(config)
    else:
        if config.trainer.is_brats2019:
            train_images = load_brats2019_dataset_images(config.trainer.brats2019)
        else:
            train_images = load_brats2021_dataset_images(config.trainer.brats2021)
        train_transform, val_transform = get_Brats_transforms(config)
    train_dataset = monai.data.Dataset(data=train_images[:int(len(train_images) * config.trainer.train_ratio)],
                                       transform=val_transform, )
    val_dataset = monai.data.Dataset(data=train_images[int(len(train_images) * config.trainer.train_ratio):],
                                     transform=val_transform, )

    train_loader = monai.data.DataLoader(train_dataset, num_workers=config.trainer.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)

    if config.trainer.is_HepaticVessel:
        batch_size = 1
    else:
        batch_size = config.trainer.batch_size
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.trainer.num_workers, batch_size=batch_size,
                                       shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    import yaml
    import os

    config = EasyDict(yaml.load(open('/workspace/Brats/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

    train_loader, val_loader = get_dataloader(config)
    nan_num = 0

    T = 0
    V = 0
    for i, batch in enumerate(train_loader):
        print('================== {} ==================='.format(i))
        # print(batch['image'])
        # print(batch['label'])
        # if batch['label'][0][0].max() == 0:
        #     nan_num += 1

        if batch['label'][0][0].max() == 0:
            continue
        if batch['label'][0][1].max() == 0:
            continue

        all_num = torch.nonzero(batch['label'][0][0]).size(0) + torch.nonzero(batch['label'][0][0])

        T += torch.nonzero(batch['label'][0][0]).size(0)
        V += torch.nonzero(batch['label'][0][1]).size(0)

    for i, batch in enumerate(val_loader):
        print('================== {} ==================='.format(i))
        # print(batch['image'])
        # print(batch['label'])
        # if batch['label'][0][0].max() == 0:
        #     nan_num += 1

        if batch['label'][0][0].max() == 0:
            continue
        if batch['label'][0][1].max() == 0:
            continue
        T += torch.nonzero(batch['label'][0][0]).size(0)
        V += torch.nonzero(batch['label'][0][1]).size(0)
    print(T)
    print(V)
