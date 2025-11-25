import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model
import warnings
import torch.nn.functional as F
import torch.nn as nn
warnings.filterwarnings('ignore')



def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    config, metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int, loss_weights):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch[0])
        total_loss = 0
        log = ''
        for name, loss_fn in loss_functions.items():
            current_loss = loss_fn(logits, image_batch[1])
            weighted_loss = loss_weights[name] * current_loss
            accelerator.log({'Train/' + name: float(current_loss)}, step=step)
            accelerator.log({'Train/Weighted_' + name: float(weighted_loss)}, step=step)
            total_loss += weighted_loss
            log += f'{name}: {current_loss:.4f} '
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])

        
        
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
        f'Train/mean {metric_name}': float(batch_acc.mean())})
        
    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}')
    accelerator.log(metric, step=epoch)
    return step

import numpy as np
@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int):
    # 验证
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch[0], model)
        total_loss = 0
        log = ''
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch[1])
            accelerator.log({'Val/' + name: float(loss)}, step=step)
            log += f' {name} {float(loss):1.5f} '
            total_loss += loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])
        accelerator.log({
            'Val/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        step += 1
        np.save("./visualization/DRIVE/output/numpy" + '/' + str(i) + '.npy', val_outputs.cpu().detach().numpy())
    metric = {}
    if config.trainer.dataset_choose != 'EDD_seg':
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metrics[metric_name].reset()
            metric.update({
            f'Val/mean {metric_name}': float(batch_acc.mean())})
            
        accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}')
        accelerator.log(metric, step=epoch)
    else:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metrics[metric_name].reset()
            if batch_acc.size()==torch.Size([]) or batch_acc.size()==torch.Size([1]):
                metric.update({
                    f'Val/mean {metric_name}': float(batch_acc.mean()),
                    })
            else:
                metric.update({
                    f'Val/mean {metric_name}': float(batch_acc.mean()),
                    f'Val/BE {metric_name}': float(batch_acc[0]),
                    f'Val/cancer {metric_name}': float(batch_acc[1]),
                    f'Val/HGD {metric_name}': float(batch_acc[2]),
                    f'Val/polyp {metric_name}': float(batch_acc[3]),
                    f'Val/suspicious {metric_name}': float(batch_acc[4])})
    return torch.Tensor([metric['Val/mean f1']]).to(accelerator.device), metric, step

class Dropoutput_Layer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = (
            -((torch.sum(w * y_true * torch.log(y_pred + smooth)) /
               torch.sum(w * y_true + smooth)) +
              (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) /
               torch.sum(w * (1 - y_true) + smooth))) / 2)
        return loss_ce


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.path.join(os.getcwd(), 'logs',
                               config.finetune.checkpoint + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)

    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('Load Model...')
    model = give_model(config)

    accelerator.print('Load Dataloader...')

    if config.trainer.dataset_choose == 'DRIVE':
        from src.VesselLoader import get_dataloader
        train_loader, val_loader = get_dataloader(config)
        image_size = config.dataset.DRIVE.image_size
        include_background = True

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=include_background,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'miou_metric': monai.metrics.MeanIoU(include_background=include_background, reduction="mean_channel"),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name='f1 score'),
        'precision': monai.metrics.ConfusionMatrixMetric(include_background=include_background,
                                                         metric_name="precision"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="recall"),
        'MCC':monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="matthews correlation coefficient"),
        'ACC':monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="accuracy"),
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)

    class_weight = torch.tensor([1.0, 13.0]).to(accelerator.device)

    # loss_functions = {
    #     'focal_tversky': monai.losses.TverskyLoss(
    #         alpha=0.7, beta=0.3,
    #         sigmoid=True,
    #         to_onehot_y=False
    #     ),

    #     'boundary_focal': monai.losses.FocalLoss(
    #         include_background=True,
    #         weight=class_weight,
    #         gamma=3.0,
    #         to_onehot_y=False
    #     ),


    #     'generalized_dice': monai.losses.GeneralizedDiceLoss(
    #         w_type='square',
    #         sigmoid=True,
    #         to_onehot_y=False
    #     ),
    #     # 'TCLloss': Dropoutput_Layer()


    # }

    loss_functions ={
        'dice_focal_loss': monai.losses.DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }

    loss_weights = {
        'dice_focal_loss': 1.0
    }

    # loss_weights = {
    #     'focal_tversky': 0.5,
    #     'boundary_focal': 0.3,
    #     'generalized_dice': 0.2,
    #     # 'TCLloss':1.0
    # }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = torch.tensor(0)
    best_class = []

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step, best_acc, best_class = utils.resume_train_state(model,
                                                                                                                 '{}'.format(
                                                                                                                     config.finetune.checkpoint),
                                                                                                                 optimizer,
                                                                                                                 scheduler,
                                                                                                                 train_loader,
                                                                                                                 accelerator)
        val_step = train_step

    best_acc = best_acc.to(accelerator.device)

    # 开始训练
    accelerator.print("Start Training! ")

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # 训练
        step = train_one_epoch(model, loss_functions, train_loader,
                               optimizer, scheduler, config, metrics,
                               post_trans, accelerator, epoch, step, loss_weights)

        mean_acc, batch_acc, val_step = val_one_epoch(model, loss_functions, inference, val_loader,
                                                      config, metrics, val_step,
                                                      post_trans, accelerator, epoch)

        # 保存模型
        if mean_acc > best_acc:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best")
            best_acc = mean_acc
            best_class = batch_acc
            best_eopch = epoch
        accelerator.print('Cheakpoint...')
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_acc': best_acc, 'best_class': batch_acc},
                   f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
        
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] best acc:{best_acc}, Now : mean acc: {mean_acc}, mean class: {batch_acc}')


    accelerator.print(f"best acc: {best_acc}")
    accelerator.print(f"best class : {best_class}")
    accelerator.print(f"best epochs: {best_eopch}")
    sys.exit(1)
