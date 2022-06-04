import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim

from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1, CityscapesDataSet_2
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1, MapillaryDataSet_2
from depth_distribution.main.model.deeplabv2_depth import get_deeplab_v2_depth
from depth_distribution.main.domain_adaptation.selftrain_UDA import selftrain_depdis

warnings.filterwarnings("ignore")

def main():
    # LOAD ARGS
    args = get_arguments()
    print("Called with args:")
    print(args)

    expid = args.expid
    if expid == 1:
        from depth_distribution.configs.synthia_to_cityscapes_16cls  import cfg
    elif expid == 2:
        from depth_distribution.configs.synthia_to_cityscapes_7cls  import cfg
    elif expid == 3:
        from depth_distribution.configs.synthia_to_cityscapes_7cls_small  import cfg
    elif expid == 4:
        from depth_distribution.configs.synthia_to_mapillary_7cls  import cfg
    elif expid == 5:
        from depth_distribution.configs.synthia_to_mapillary_7cls_small  import cfg

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == "":
        cfg.EXP_NAME = f"{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}"

    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == "":
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME, 'self_train_model')
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    print("Using config:")
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if cfg.TRAIN.MODEL == "DeepLabv2_depth":
        # SEGMNETATION NETWORK
        model = get_deeplab_v2_depth(num_classes=cfg.NUM_CLASSES)
        if args.pret_model != '':
            saved_state_dict = torch.load(args.pret_model)
            model.load_state_dict(saved_state_dict['model'])
        else:
            saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
            if "DeepLab_resnet_pretrained_imagenet" in cfg.TRAIN.RESTORE_FROM:
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split(".")
                    if not i_parts[1] == "layer5":
                        new_params[".".join(i_parts[1:])] = saved_state_dict[i]
                model.load_state_dict(new_params)
            else:
                model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    model.train()
    model.to(cfg.GPU_ID)

    # segnet's optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    if cfg.TARGET == 'Cityscapes':
        target_dataset = CityscapesDataSet_2(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_SELFTRAIN,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN
        )
    elif cfg.TARGET == 'Mapillary':
        target_dataset = MapillaryDataSet_2(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_SELFTRAIN,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )
    else:
        raise NotImplementedError(f"Not yet supported dataset {cfg.TARGET}")

    target_loader = data.DataLoader(
        target_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    if cfg.TARGET == 'Cityscapes':
        test_dataset = CityscapesDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        )
    elif cfg.TARGET == 'Mapillary':
        test_dataset = MapillaryDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    selftrain_depdis(model, optimizer, target_loader,test_loader, cfg)

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation training")
    '''expid
    Syn_City_16cls ---> 1
    Syn_City_7cls ---> 2
    Syn_City_7cls_small ---> 3
    Syn_Map_7cls ---> 4
    Syn_Map_7cls_small ---> 5
    '''
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument("--random-train", action="store_true", help="not fixing random seed.")
    parser.add_argument("--pret-model", type=str, default='', help="pretrained weights to be used for initialization")
    return parser.parse_args()

if __name__ == "__main__":
    main()