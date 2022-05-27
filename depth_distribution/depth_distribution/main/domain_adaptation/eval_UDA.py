import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from depth_distribution.main.utils.func import per_class_iu, fast_hist

def evaluate_domain_adaptation( models, test_loader, cfg, i_iter, best_miou, best_model):
    device = cfg.GPU_ID
    # eval
    if cfg.TEST.MODE == 'best':
        best_miou, best_model = eval_best(cfg, models, device, test_loader, i_iter, best_miou, best_model)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")
    return best_miou, best_model

def record_result(text):
    print(text)
    with open("record.txt", "a") as f:
        f.write(text)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict['model'])
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        record_result(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)) + '\n')


def eval_best(cfg, models,device, test_loader, i_iter, best_miou, best_model):
    cur_best_miou = best_miou
    cur_best_model = best_model
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
    for index in tqdm(range(len(test_loader))):
        image, label, _, name = next(test_iter)
        interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred_main = models(image.cuda(device))[1]
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if index > 0 and index % 100 == 0:
            record_result('{:d} / {:d}: {:0.2f}\n'.format(index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            # break
    inters_over_union_classes = per_class_iu(hist)
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    if cur_best_miou < computed_miou:
        cur_best_miou = computed_miou
        cur_best_model =  f'model_{i_iter}.pth'

    record_result(f'Current mIoU:{computed_miou}\n')
    record_result(f'Current model: model_{i_iter}.pth\n')
    record_result(f'Current best mIoU:{cur_best_miou}\n')
    record_result(f'Current best model:{cur_best_model}\n')

    display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)
    record_result('\n')
    return cur_best_miou, cur_best_model


