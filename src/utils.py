import os
import random
import sys
from collections import OrderedDict

import math
import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn


class MetricSaver(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = nn.Parameter(torch.zeros(1), requires_grad=False)

def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict


# 用以恢复中断训练
def resume_train_state(model, path: str, optimizer, scheduler, train_loader: torch.utils.data.DataLoader,
                       accelerator: Accelerator, mode="checkpoint"):
    try:
        # Get the most recent checkpoint
        model_store = os.path.join("model_store")
        model_path = os.path.join(model_store, path)

        checkpoint_path = os.path.join(model_path, "checkpoint")
        best_path = os.path.join(model_path, "best")

        epoch_file = os.path.join(checkpoint_path, "epoch.pth.tar")
        epoch_checkpoint = torch.load(epoch_file, map_location='cpu')

        starting_epoch = epoch_checkpoint['epoch'] + 1
        best_acc = epoch_checkpoint['best_acc']
        best_class = epoch_checkpoint['best_class']
        step = starting_epoch * len(train_loader)

        accelerator.load_state(best_path if mode == 'best' else checkpoint_path)
        accelerator.print(f'Loading training state successfully! Start training from {starting_epoch}, Best Acc: {best_acc}')
        return model, optimizer, scheduler, starting_epoch, step, best_acc, best_class
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load training state!')
        return model, optimizer, scheduler, 0, 0, torch.tensor(0), []

def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f'Successfully loaded the training model！')
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load the training model！')
        return model


def patchify(imgs: torch.Tensor, path_size: int):
    """
    把输入图片变成patch, 图片形状为立方体
    imgs: (N, modality, H, W, D)
    x: (N, L, patch_size_hw**2 * patch_size_depth) [batch_size, num_patches, 每个patch大小]
    """

    imgs = Rearrange('b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', p1=path_size, p2=path_size, p3=path_size)(imgs)
    # x = imgs.reshape(shape=(imgs.shape[0], modality, h, path_size, w, path_size, d, path_size))  # [N, modality, patch_h, patch_size, patch_w, patch_size, patch_d, patch_size]
    # x = torch.einsum('nmhowpdq->nhwdopqm', x)  # [N, patch_h, patch_w, patch_d, patch_size, patch_size, patch_size, modality]
    # imgs = x.reshape(shape=(x.shape[0], num_patches, path_size ** 3 * modality))  # [N, num_patches, pixel_patches]
    return imgs


def unpatchify(x: torch.Tensor, image_size: int, path_size: int, modality: int):
    """
    从图片patch映射回原图
    x: (N, num_patches, 每个patch大小)
    imgs: (N, C, H, W, D)
    """
    h = w = d = image_size // path_size
    x = x.reshape(shape=(x.shape[0], h, w, d, path_size, path_size, path_size, modality))  # [N, patch_h, patch_w, patch_d, patch_size, patch_size, patch_size, modality]
    x = torch.einsum('nhwdopqm->nmhowpdq', x)  # [N, modality, patch_h, patch_size, patch_w, patch_size, patch_d, patch_size]
    imgs = x.reshape(shape=(x.shape[0], modality, image_size, image_size, image_size))  # [N, modality, img_h, img_w, img_d]
    return imgs


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


# 提供高斯模糊
def give_gaussian(img, p=0.1):
    do_it = random.random() <= p
    if do_it:
        return img
    else:
        blur_layer = get_gaussian_kernel().cuda()
        return blur_layer(img)


# 提供高斯模糊
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=4):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size * kernel_size)
    x_grid = x_grid.view(kernel_size, kernel_size, kernel_size)
    # x_grid = x_coord.repeat(kernel_size )
    # x_grid = x_grid.view(kernel_size, kernel_size)
    y_grid = x_grid.mT
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

    gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


# 随机增加对比度
def Random_contrast(image, p=0.3, lower=0.7, upper=1.3):
    if random.random() < p:
        alpha = random.uniform(lower, upper)
        image *= alpha
        image = image.clip(min=0, max=255)
    return image


# 提供数据增强
def give_data_change(image):
    image1 = give_gaussian(image.clone().cuda(), p=0.5)
    image2 = give_gaussian(image.clone().cuda(), p=0.1)
    image2 = Random_contrast(image2, p=0.2)
    return [image1, image2]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, seg_length, dim
    len_keep = int(L * (1 - mask_ratio))  # 需要保留的patch数目

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] noise.shape: [N, L]

    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove; dim=1按照seq_length这个维度排序
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # 通过ids_restore来将序列还原

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # 要保留下来的
    # Gathers values along an axis specified by dim. https://zhuanlan.zhihu.com/p/352877584
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # mask之后保留下来的token序列

    # generate the binary mask for decoder: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0  # nomask的值为0，mask的值为1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # 原始img得到的token序列的mask顺序

    return x_masked, mask, ids_restore


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def common_params(student_model: nn.Module, teacher_model: nn.Module, accelerator: Accelerator):
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in accelerator.unwrap_model(student_model).named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in accelerator.unwrap_model(teacher_model).named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
    return params_q, params_k


def get_pred_ratio(pred_ratio, pred_ratio_var):
    if isinstance(pred_ratio, list):
        pred_ratio = []
        for prm, prv in zip(pred_ratio, pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)
    else:
        assert pred_ratio >= pred_ratio_var
        pred_ratio = random.uniform(pred_ratio - pred_ratio_var, pred_ratio + pred_ratio_var) if pred_ratio_var > 0 else pred_ratio

    return pred_ratio


def give_masks(images, patch_size, pred_ratio, pred_ratio_var):
    keep_flag = False
    masks = []
    for images_data in images:
        img = images_data
        try:
            H, W, Z = img.shape[1] // patch_size, img.shape[2] // patch_size, img.shape[3] // patch_size
        except:
            # skip non-image
            continue

        high = get_pred_ratio(pred_ratio, pred_ratio_var) * H * W * Z
        high = int(high)
        log_aspect_ratio = tuple(map(lambda x: math.log(x), (0.3, 1 / 0.3)))
        mask = np.zeros((H, W, Z), dtype=bool)
        mask_count = 0
        # 总体值开3次方
        mask_num = int(pow(high, float(1) / float(3)))
        # while mask_count < high:
        #     max_mask_patches = high - mask_count
        # delta = 0
        #     for attempt in range(10):
        while mask_count < high:
            # low = (min(H, W, Z) // 3) ** 2
            # target_area = random.uniform(low, max_mask_patches)
            # aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = mask_num
            # w = int(round(math.sqrt(target_area / aspect_ratio)))
            w = mask_num
            z = mask_num
            if w < W and h < H and z < Z:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                hig = random.randint(0, Z - z)
                num_masked = mask[top: top + h, left: left + w, hig: hig + z].sum()
                if 0 < h * w * z - num_masked <= high:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            for k in range(hig, hig + z):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    mask_count += 1
                                    if mask_count >= high:
                                        break
            # if delta == 0:
            #     break
            # else:
            #     mask_count += delta
        mask = torch.from_numpy(mask)
        if keep_flag == False:
            masks = mask.unsqueeze(0)
            keep_flag = True
        else:
            masks = torch.cat([masks, mask.unsqueeze(0)], dim=0)

        # return_mask.append(masks)
    # stack_img = torch.stack((flair_img, t1_img, t2_img), 0)
    return masks.to(images.device)


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_world_size(accelerator):
    return accelerator.num_processes
