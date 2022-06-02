import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import ctypes
from ctypes import *
from PIL import Image

from depth_distribution.main.utils import project_root

class StructPointer(ctypes.Structure):  # Structure在ctypes中是基于类的结构体
    _fields_ = [("revalue", ctypes.c_int * 972800)]  # 定义二维数组

class StructPointer_small(ctypes.Structure):  # Structure在ctypes中是基于类的结构体
    _fields_ = [("revalue", ctypes.c_int * 204800)]  # 定义二维数组

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


def gene_pseudo_labels_1(model, target_loader, output_dir, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)

    for i_iter in tqdm(range(cfg.MAX_ITERS_PSEUDO)):
        with torch.no_grad():
            _, batch1 = targetloader_iter.__next__()
            images, labeltrue, _, target_file_name = batch1
            target_file_name = os.path.basename(target_file_name[0])

            _, pred_seg_trg, _, _ = model(images.cuda(device))
            pred_seg_trg = interp_target(pred_seg_trg)
            pred_trg_main_1 = F.softmax(pred_seg_trg, dim=1)
            conf, target_2_1 = torch.max(pred_trg_main_1, 1)
            confidence_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

            for i in range(16):
                mask = (target_2_1 == i) * (conf < confidence_list[i])
                target_2_1[mask] = 255

            sudo_labels = target_2_1.detach().cpu().numpy()
            sudo_labels = sudo_labels.flatten()
            aaa = sudo_labels.tolist()
            my_array_u = (c_int * 972800)(*aaa)
            dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'spatial_prior_algorithm.so')
            dll.Add1.restype = ctypes.POINTER(StructPointer)
            array_count = [3165, 3311, 5000, 89, 89, 219, 50, 61, 1788, 1202, 762, 123, 730, 302, 78, 81]
            my_array_count = (c_int * 16)(*array_count)
            p = dll.Add1(my_array_u, my_array_count)
            newlabel = np.array(p.contents.revalue[:])
            newlabel = newlabel.reshape((760, 1280))
            ccImage = Image.fromarray(newlabel.astype('uint8'))
            ccImage.save(output_dir + os.sep + '{}.tiff'.format(target_file_name))


def gene_pseudo_labels_2(model, target_loader, output_dir, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.MAX_ITERS_PSEUDO)):
        with torch.no_grad():
            _, batch1 = targetloader_iter.__next__()
            images, labeltrue, _, target_file_name = batch1
            target_file_name = os.path.basename(target_file_name[0])

            _, pred_seg_trg, _, _ = model(images.cuda(device))
            pred_seg_trg = interp_target(pred_seg_trg)
            pred_trg_main_1 = F.softmax(pred_seg_trg, dim=1)
            conf, target_2_1 = torch.max(pred_trg_main_1, 1)
            confidence_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

            for i in range(7):
                mask = (target_2_1 == i) * (conf < confidence_list[i])
                target_2_1[mask] = 255

            sudo_labels = target_2_1.detach().cpu().numpy()
            sudo_labels = sudo_labels.flatten()
            aaa = sudo_labels.tolist()
            my_array_u = (c_int * 972800)(*aaa)
            dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'spatial_prior_algorithm.so')
            dll.Add1.restype = ctypes.POINTER(StructPointer)
            array_count = [1000, 772, 100, 316, 234, 173, 173]
            my_array_count = (c_int * 7)(*array_count)
            p = dll.Add1(my_array_u, my_array_count)
            newlabel = np.array(p.contents.revalue[:])
            newlabel = newlabel.reshape((760, 1280))
            ccImage = Image.fromarray(newlabel.astype('uint8'))
            ccImage.save(output_dir + os.sep + '{}.tiff'.format(target_file_name))


def gene_pseudo_labels_3(model, target_loader, output_dir, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.MAX_ITERS_PSEUDO)):
        with torch.no_grad():
            _, batch1 = targetloader_iter.__next__()
            images, labeltrue, _, target_file_name = batch1
            target_file_name = os.path.basename(target_file_name[0])

            _, pred_seg_trg, _, _ = model(images.cuda(device))
            pred_seg_trg = interp_target(pred_seg_trg)
            pred_trg_main_1 = F.softmax(pred_seg_trg, dim=1)
            conf, target_2_1 = torch.max(pred_trg_main_1, 1)
            confidence_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

            for i in range(7):
                mask = (target_2_1 == i) * (conf < confidence_list[i])
                target_2_1[mask] = 255

            sudo_labels = target_2_1.detach().cpu().numpy()
            sudo_labels = sudo_labels.flatten()
            aaa = sudo_labels.tolist()
            my_array_u = (c_int * 204800)(*aaa)
            dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'spatial_prior_algorithm_small.so')
            dll.Add1.restype = ctypes.POINTER(StructPointer_small)
            array_count =[1000, 772, 100, 316, 234, 173, 173]
            my_array_count = (c_int * 7)(*array_count)
            p = dll.Add1(my_array_u, my_array_count)
            newlabel = np.array(p.contents.revalue[:])
            newlabel = newlabel.reshape((320, 640))
            ccImage = Image.fromarray(newlabel.astype('uint8'))
            ccImage.save(output_dir + os.sep + '{}.tiff'.format(target_file_name))


def gene_pseudo_labels_4(model, target_loader, output_dir, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.MAX_ITERS_PSEUDO)):
        with torch.no_grad():
            _, batch1 = targetloader_iter.__next__()
            images, labeltrue, _, target_file_name = batch1
            target_file_name = os.path.basename(target_file_name[0])

            _, pred_seg_trg, _, _ = model(images.cuda(device))
            pred_seg_trg = interp_target(pred_seg_trg)
            pred_trg_main_1 = F.softmax(pred_seg_trg, dim=1)
            conf, target_2_1 = torch.max(pred_trg_main_1, 1)
            confidence_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

            for i in range(7):
                mask = (target_2_1 == i) * (conf < confidence_list[i])
                target_2_1[mask] = 255

            sudo_labels = target_2_1.detach().cpu().numpy()
            sudo_labels = sudo_labels.flatten()
            aaa = sudo_labels.tolist()
            my_array_u = (c_int * 972800)(*aaa)
            dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'spatial_prior_algorithm.so')
            dll.Add1.restype = ctypes.POINTER(StructPointer)
            array_count = [100, 77, 10, 32, 23, 17, 17]
            my_array_count = (c_int * 7)(*array_count)
            p = dll.Add1(my_array_u, my_array_count)
            newlabel = np.array(p.contents.revalue[:])
            newlabel = newlabel.reshape((760, 1280))
            ccImage = Image.fromarray(newlabel.astype('uint8'))
            ccImage.save(output_dir + os.sep + '{}.tiff'.format(target_file_name))


def gene_pseudo_labels_5(model, target_loader, output_dir, cfg):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    targetloader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.MAX_ITERS_PSEUDO)):
        with torch.no_grad():
            _, batch1 = targetloader_iter.__next__()
            images, labeltrue, _, target_file_name = batch1
            target_file_name = os.path.basename(target_file_name[0])

            _, pred_seg_trg, _, _ = model(images.cuda(device))
            pred_seg_trg = interp_target(pred_seg_trg)
            pred_trg_main_1 = F.softmax(pred_seg_trg, dim=1)
            conf, target_2_1 = torch.max(pred_trg_main_1, 1)
            confidence_list = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

            for i in range(7):
                mask = (target_2_1 == i) * (conf < confidence_list[i])
                target_2_1[mask] = 255

            sudo_labels = target_2_1.detach().cpu().numpy()
            sudo_labels = sudo_labels.flatten()
            aaa = sudo_labels.tolist()
            my_array_u = (c_int * 204800)(*aaa)
            dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'spatial_prior_algorithm_small.so')
            dll.Add1.restype = ctypes.POINTER(StructPointer_small)
            array_count = [100, 77, 10, 32, 23, 17, 17]
            my_array_count = (c_int * 7)(*array_count)
            p = dll.Add1(my_array_u, my_array_count)
            newlabel = np.array(p.contents.revalue[:])
            newlabel = newlabel.reshape((320, 640))
            ccImage = Image.fromarray(newlabel.astype('uint8'))
            ccImage.save(output_dir + os.sep + '{}.tiff'.format(target_file_name))