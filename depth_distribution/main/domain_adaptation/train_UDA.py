import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from depth_distribution.main.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from depth_distribution.main.utils.func import loss_calc, bce_loss, prob_2_entropy
from depth_distribution.main.utils.func import loss_berHu
from depth_distribution.main.utils.denpro import getTargetDensity_16, getTargetDensity_7, getTargetDensity_7_small, getSourceDensity
from depth_distribution.main.domain_adaptation.eval_UDA import evaluate_domain_adaptation


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'  {full_string}')


def train_depdis(model, d_main, optimizer, optimizer_d_main, source_loader, target_loader,test_loader, cfg, expid, iternum):
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET

    device = cfg.GPU_ID
    cudnn.benchmark = True
    cudnn.enabled = True

    # interpolate output segmaps
    interp_source = nn.Upsample(
        size=(input_size_source[1], input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    source_label = 0
    target_label = 1

    best_miou = -1
    best_model = ''

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        n_iter, batch = trainloader_iter.__next__()
        images_source, labels_source, density_pre_source, depthvalue_source, _, _ = batch

        _, batch1 = targetloader_iter.__next__()
        images_target, _, _, _, _= batch1

        if n_iter <= iternum and iternum != 0:
            continue

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False

        # train on source(1)############################################################################################
        _, pred_seg_src, pred_density_src, pred_depthvalue_src = model(images_source.cuda(device))
        #L_dep
        pred_depthvalue_src = interp_source(pred_depthvalue_src)
        loss_depth_src = loss_berHu(pred_depthvalue_src, depthvalue_source, device)
        #L_bal_src
        density_source_result = getSourceDensity(density_pre_source, pred_seg_src, expid)
        loss_bal_src = loss_berHu(pred_density_src, density_source_result, device)
        #L_seg
        pred_seg_src = interp_source(pred_seg_src)
        loss_seg_src = loss_calc(pred_seg_src, labels_source, expid, device)
        #source losses
        losses = (cfg.TRAIN.LAMBDA_SEG_SRC * loss_seg_src + cfg.TRAIN.LAMBDA_BAL_SRC * loss_bal_src + cfg.TRAIN.LAMBDA_DEP_SRC * loss_depth_src)
        losses.backward()

        #train on target, adversarial training to fool the discriminator(2)#############################################
        _, pred_seg_trg, pred_density_trg, pred_depthvalue_trg = model(images_target.cuda(device))
        #L_bal_tar
        pred_seg_trg_1 = pred_seg_trg.detach()
        pred_depthvalue_trg_1 = pred_depthvalue_trg.detach()
        if expid == 1:
            density_Target_result = getTargetDensity_16(pred_seg_trg_1, pred_depthvalue_trg_1)
        elif expid == 2 or expid == 4:
            density_Target_result = getTargetDensity_7(pred_seg_trg_1, pred_depthvalue_trg_1)
        else:
            density_Target_result = getTargetDensity_7_small(pred_seg_trg_1, pred_depthvalue_trg_1)
        loss_bal_trg = loss_berHu(pred_density_trg, density_Target_result, device)
        #L_adv_tar
        pred_seg_trg_2 = interp_target(pred_seg_trg)
        pred_density_trg = interp_target(pred_density_trg)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_seg_trg_2)) * pred_density_trg)
        loss_adv_trg = bce_loss(d_out_main, source_label)
        #target losses
        losses = cfg.TRAIN.LAMBDA_ADV_TAR * loss_adv_trg + cfg.TRAIN.LAMBDA_BAL_TAR * loss_bal_trg
        losses.backward()


        # enable training mode on discriminator networks,Train discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source(3)##########################################################################################
        pred_density_src = pred_density_src.detach()
        pred_density_src = interp_source(pred_density_src)
        pred_seg_src = pred_seg_src.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_seg_src)) * pred_density_src)
        loss_d_src = bce_loss(d_out_main, source_label)
        loss_d_src = loss_d_src
        loss_d_src.backward()

        # train with target(4)##########################################################################################
        pred_density_trg = pred_density_trg.detach()
        pred_seg_trg_3 = pred_seg_trg_2.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_seg_trg_3)) * pred_density_trg)
        loss_d_tar = bce_loss(d_out_main, target_label)
        loss_d_tar.backward()

        #optimizer jointly
        optimizer.step()
        optimizer_d_main.step()

        current_losses = {
            'loss_seg_src': loss_seg_src,
            'loss_bal_src': loss_bal_src,
            'loss_depth_src':loss_depth_src,
            'loss_bal_trg': loss_bal_trg,
            'loss_adv_trg':loss_adv_trg
        }

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            state1 = {'model':model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state1, snapshot_dir / f"model_{i_iter}.pth")
            state2 = {'model':d_main.state_dict(), 'optimizer': optimizer_d_main.state_dict()}
            torch.save(state2, snapshot_dir / f"model_D_{i_iter}.pth")

            # eval model, can be annotated
            model.eval()
            best_miou, best_model = evaluate_domain_adaptation(model, test_loader, cfg, i_iter, best_miou, best_model)
            model.train()

            if i_iter >= cfg.TRAIN.EARLY_STOP:
                break

        sys.stdout.flush()