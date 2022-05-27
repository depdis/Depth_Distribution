import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from depth_distribution.main.utils.func import per_class_iu, fast_hist
from depth_distribution.main.utils.serialization import pickle_dump, pickle_load

def evaluate_domain_adaptation( models, test_loader, cfg, restore_from,fixed_test_size=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'best':
        eval_best(cfg, models, device, test_loader, interp, fixed_test_size, restore_from)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict['model'])
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))

def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, restore_from):
    assert len(models) == 1, 'Not yet supported multi models in this mode'

    cur_best_miou = -1
    cur_best_model = ''

    if not osp.exists(restore_from):
        print('---Model does not exist!---')
        return
    print("Evaluating model", restore_from)

    load_checkpoint_for_evaluation(models[0], restore_from, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
    for index in tqdm(range(len(test_loader))):
        image, label, _, name = next(test_iter)
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred_main = models[0](image.cuda(device))[1]
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if index > 0 and index % 100 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(
                index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
    inters_over_union_classes = per_class_iu(hist)

    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    if cur_best_miou < computed_miou:
        cur_best_miou = computed_miou
        cur_best_model = restore_from
    print('\tCurrent mIoU:', computed_miou)
    print('\tCurrent best model:', cur_best_model)
    print('\tCurrent best mIoU:', cur_best_miou)
    display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)



