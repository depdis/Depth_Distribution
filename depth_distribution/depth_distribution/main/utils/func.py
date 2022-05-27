import os.path
import os
import numpy as np
import torch
import torch.nn as nn

from depth_distribution.main.utils.loss import cross_entropy_2d, cross_entropy_2d_self


def loss_berHu(pred, label, device):
    n, c, h, w = pred.size()
    assert c == 1

    pred = pred.squeeze()
    label = label.squeeze().cuda(device)

    adiff = torch.abs(pred - label)
    batch_max = 0.2 * torch.max(adiff).item()
    t1_mask = adiff.le(batch_max).float()
    t2_mask = adiff.gt(batch_max).float()
    t1 = adiff * t1_mask
    t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
    t2 = t2 * t2_mask
    return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def loss_calc(pred, label, expid,  device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label, expid, device)

def loss_calc_self(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d_self(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def split_premodelname(modelname):
    iternum = int(os.path.basename(modelname).replace('.pth','').split('_')[-1])
    model_adv_name = os.path.dirname(modelname) + os.sep + os.path.basename(modelname).replace('.pth','').split('_')[0] + '_D_' + str(iternum) + '.pth'
    return iternum, model_adv_name
