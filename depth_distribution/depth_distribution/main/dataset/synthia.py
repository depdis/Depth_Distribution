import numpy as np
import os

import torch.nn

from depth_distribution.main.dataset.base_dataset import BaseDataset
import cv2
from depth_distribution.main.dataset.depth import get_depth
import math

class SYNTHIADataSetDepth(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        set="all",
        num_classes=16,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        iternum=0,
        use_depth=True,
        expid = 1
    ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        self.realbeginNum = 0
        self.iternum = iternum
        self.expid = expid
        self.num_classes = num_classes
        # map to cityscape's ids
        if num_classes == 16:
            self.id_to_trainid = {
                3: 0,
                4: 1,
                2: 2,
                21: 3,
                5: 4,
                7: 5,
                15: 6,
                9: 7,
                6: 8,
                1: 9,
                10: 10,
                17: 11,
                8: 12,
                19: 13,
                12: 14,
                11: 15,
            }
        elif num_classes == 7:
            self.id_to_trainid = {
                1:4, 
                2:1, 
                3:0, 
                4:0, 
                5:1, 
                6:3, 
                7:2, 
                8:6, 
                9:2, 
                10:5, 
                11:6, 
                15:2, 
                22:0}
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} classes")
        self.use_depth = use_depth
        if self.use_depth:
            for (i, file) in enumerate(self.files):
                img_file, label_file, name = file
                density_file = self.root / "source_density_maps" / name
                depth_file_value = self.root / "Depth" / name
                self.files[i] = (img_file, label_file, density_file, depth_file_value, name)
            # disable multi-threading in opencv. Could be ignored
            import os
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

    def get_metadata(self, name):
        img_file = self.root / "RGB" / name
        label_file = self.root / "parsed_LABELS" / name
        return img_file, label_file

    def get_gaosipro(self, name):
        name1 = os.path.basename(name).replace('.png', '')
        name2 = os.path.dirname(name)
        xadd = None
        for i in range(self.num_classes):
            name = name2 + os.sep + name1 + '-' + str(i) + '.tiff'
            depthPro = cv2.imread(name, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            x = np.expand_dims(depthPro, 0)
            if i == 0:
                xadd = x
            else:
                xadd = np.append(xadd, x, axis=0)
        return xadd

    def __getitem__(self, index):
        if self.iternum > 0 and (self.realbeginNum + 5) < self.iternum:
            self.realbeginNum += 1
            return 1, 2, 3, 4, 5, 6

        if self.use_depth:
            img_file, label_file, density_file,depth_file_value, name = self.files[index]
        else:
            img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        if self.use_depth:
                #mapping
                density_pre_source = self.get_gaosipro(density_file)* 1e6
                density_pre_source = (1 - np.exp(-density_pre_source))* 255
                depthvalue = self.get_depth(depth_file_value)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        image = image.copy()
        label_copy = label_copy.copy()
        shape = np.array(image.shape)
        if self.use_depth:
            return image, label_copy, density_pre_source.copy(), depthvalue.copy(), shape, name

    def get_depth(self, file):
        return get_depth(self, file)
