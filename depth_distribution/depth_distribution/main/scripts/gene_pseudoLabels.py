import argparse
import pprint
import warnings
import numpy as np
import torch
from torch.utils import data

from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1
from depth_distribution.main.model.deeplabv2_depth import get_deeplab_v2_depth
from depth_distribution.main.domain_adaptation.gene_UDA import gene_pseudo_labels_1, gene_pseudo_labels_2,\
gene_pseudo_labels_3, gene_pseudo_labels_4, gene_pseudo_labels_5


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

    print("Using config:")
    pprint.pprint(cfg)

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)


    if cfg.TRAIN.MODEL == "DeepLabv2_depth":
        # SEGMNETATION NETWORK
        model = get_deeplab_v2_depth(num_classes=cfg.NUM_CLASSES)

        if args.pret_model != '':
            saved_state_dict = torch.load(args.pret_model)
            model.load_state_dict(saved_state_dict['model'])
        else:
            raise NotImplementedError(f"Not pret_model!")
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    model.eval()
    model.to(cfg.GPU_ID)

    if cfg.TARGET == 'Cityscapes':
        target_dataset = CityscapesDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_PSEUDO,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
        )
    elif cfg.TARGET == 'Mapillary':
        target_dataset = MapillaryDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_PSEUDO,
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

    if expid == 1:
        gene_pseudo_labels_1(model, target_loader, args.output_dir, cfg)
    elif expid == 2:
        gene_pseudo_labels_2(model, target_loader, args.output_dir, cfg)
    elif expid == 3:
        gene_pseudo_labels_3(model, target_loader, args.output_dir, cfg)
    elif expid == 4:
        gene_pseudo_labels_4(model, target_loader, args.output_dir, cfg)
    elif expid == 5:
        gene_pseudo_labels_5(model, target_loader, args.output_dir, cfg)

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
    parser.add_argument("--pret-model", type=str, default='', help="pretrained weights to be used for initialization")
    parser.add_argument('--output-dir', type=str,  default='../../data/Cityscapes/pseudo_labels', help='folder where pseudo labels are stored')
    return parser.parse_args()

if __name__ == "__main__":
    main()