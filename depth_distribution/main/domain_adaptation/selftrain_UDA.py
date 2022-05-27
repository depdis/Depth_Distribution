import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import ctypes
from ctypes import *
from PIL import Image

from depth_distribution.main.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from depth_distribution.main.utils.func import loss_calc, loss_calc_self, bce_loss, prob_2_entropy
from depth_distribution.main.utils.func import loss_berHu
from depth_distribution.main.utils.viz_segmask import colorize_mask

from depth_distribution.main.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from depth_distribution.main.model.discriminator import  get_fc_discriminator


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'  {full_string}')


def selftrain_depdis(model, optimizer, target_loader, test_loader, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET

    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    best_miou = -1
    best_model = ''

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)

    for i_iter in tqdm(range(cfg.MAX_ITERS_SELFTRAIN)):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)
        _, batch1 = targetloader_iter.__next__()
        images, _, _, _, label_pseudo = batch1

        _, pred_seg_trg, _, _ = model(images.cuda(device))
        pred_seg_trg = interp_target(pred_seg_trg)
        loss_seg_trg_main =  loss_calc_self(pred_seg_trg, label_pseudo, device)  # seg loss
        loss_seg_trg_main.backward()
        optimizer.step()

        current_losses = {
            "loss_seg_trg_main": loss_seg_trg_main,
        }
        print_losses(current_losses, i_iter)

        if (i_iter+1) % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            state1 = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state1, snapshot_dir / f"model_{i_iter}.pth")
            # eval, can be annotated
            model.eval()
            best_miou, best_model = evaluate_domain_adaptation(model, test_loader, cfg, i_iter, best_miou, best_model)
            model.train()
        sys.stdout.flush()