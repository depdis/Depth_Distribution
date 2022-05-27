# -*-coding:utf-8-*-
import os.path as osp
import numpy as np
from easydict import EasyDict
from depth_distribution.main.utils import project_root

cfg = EasyDict()

# source domain
cfg.SOURCE = "SYNTHIA"

# target domain
cfg.TARGET = "Mapillary"

# Number of workers for dataloading
cfg.NUM_WORKERS = 4

#pseudo labels number
cfg.MAX_ITERS_PSEUDO = 9000
#self_training number
cfg.MAX_ITERS_SELFTRAIN = 30000

# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / "main/dataset/synthia_list/{}.txt")
cfg.DATA_LIST_TARGET = str(project_root / "main/dataset/mapillary_list/{}.txt")

#Data Directories
cfg.DATA_DIRECTORY_SOURCE = str(project_root / "data/SYNTHIA")
cfg.DATA_DIRECTORY_TARGET = str(project_root / "data/Mapillary")

# Number of object classes
cfg.NUM_CLASSES = 7
cfg.USE_DEPTH = True

# Exp dirs
cfg.EXP_NAME = "SYNTHIA2Mapillary_DeepLabv2_Depdis"
cfg.EXP_ROOT = project_root / "experiments"
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, "snapshots")

# CUDA
cfg.GPU_ID = 0

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = "all"
cfg.TRAIN.SET_TARGET = "train"
cfg.TRAIN.SET_TARGET_SEL = "selftrain"
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 760)
cfg.TRAIN.INPUT_SIZE_TARGET = (1280, 760)

# Class info
cfg.TRAIN.INFO_SOURCE = ""
cfg.TRAIN.INFO_TARGET = str(project_root / "main/dataset/mapillary_list/info.json")

# Segmentation network params
cfg.TRAIN.MODEL = "DeepLabv2_depth"

# cfg.TRAIN.MULTI_LEVEL = False  # in DADA paper we turn off this feature
cfg.TRAIN.RESTORE_FROM = "../../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth"
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9

#Loss weight
cfg.TRAIN.LAMBDA_SEG_SRC = 1.0  # weight of source seg loss
cfg.TRAIN.LAMBDA_BAL_SRC = 0.001  # weight of source balance loss
cfg.TRAIN.LAMBDA_DEP_SRC = 0.0001 # weight of source depth loss
cfg.TRAIN.LAMBDA_ADV_TAR = 0.001  # weight of target adv loss
cfg.TRAIN.LAMBDA_BAL_TAR = 0.001 # weight of target balance loss

# Domain adaptation
cfg.TRAIN.DA_METHOD = "Depdis"

# Adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-4

# Other params
cfg.TRAIN.MAX_ITERS = 90000
cfg.TRAIN.EARLY_STOP = 90000
cfg.TRAIN.SAVE_PRED_EVERY = 1000
cfg.TRAIN.SNAPSHOT_DIR = ""
cfg.TRAIN.RANDOM_SEED = 1234

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = "best"

# model
cfg.TEST.MODEL = ("DeepLabv2_depth",)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Test sets
cfg.TEST.SET_TARGET = "validation"
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1280, 760)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / "main/dataset/mapillary_list/info.json")