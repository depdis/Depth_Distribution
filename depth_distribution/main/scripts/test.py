import warnings

import torch.cuda
from torch.utils import data
import argparse

from depth_distribution.main.domain_adaptation.eval_UDA1 import evaluate_domain_adaptation
from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1
from depth_distribution.main.model.deeplabv2_depth import get_deeplab_v2_depth


warnings.filterwarnings("ignore")

def main(args):

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


    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == "best":
        assert n_models == 1, "Not yet supported"
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == "DeepLabv2_depth":
            model = get_deeplab_v2_depth(num_classes=cfg.NUM_CLASSES)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    # dataloaders
    fixed_test_size = True
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
        fixed_test_size = False

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    # eval
    evaluate_domain_adaptation(models, test_loader, cfg, args.pret_model, fixed_test_size=fixed_test_size)


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    '''expid
    Syn_City_16cls ---> 1
    Syn_City_7cls ---> 2
    Syn_City_7cls_small ---> 3
    Syn_Map_7cls ---> 4
    Syn_Map_7cls_small ---> 5
    '''
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument("--pret-model", type=str, default='', help="pretrained weights to be used for test")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(torch.cuda.is_available())
    print("Called with args:")
    print(args)
    main(args)
