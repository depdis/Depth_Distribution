from PIL import Image
import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DADA.dada.utils.viz_segmask import colorize_mask
import copy

def get_depth(file):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # depth = cv2.resize(depth, (1280,), interpolation=cv2.INTER_NEAREST)
    depth = 65536.0 / (depth + 1)  # inverse depth
    return depth

def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)

fileLabel = '/home/liang/liang/Semantic-Segmentation/DADA/dada/SYNTHIA/parsed_LABELS/0000200.png'
size = (1280, 760)
rec = _load_img(fileLabel, size, Image.BICUBIC, rgb=False)
rec = (rec == 2)

fileDepth = '/home/liang/liang/Semantic-Segmentation/DADA/dada/SYNTHIA/Depth/0000200.png'
dep = get_depth(fileDepth)
# recA = copy.deepcopy(recA)
# recA = np.ones((760,1280))

depA = dep * rec
# depA = (dep * depA)
x = np.arange(0, 1280, 1)
y = np.arange(0, 760, 1)
x, y = np.meshgrid(x, y)
# z = depA[x][y]

label = rec
label_img = colorize_mask(16, np.asarray(label, dtype=np.uint8)).convert("RGB")
label_img.save(f"/home/liang/liang/Semantic-Segmentation/DADA/groundtruth_101.png")


# x1 = np.arange(0, 12, 1)
# y1 = np.arange(0, 10, 1)
# x1, y1 = np.meshgrid(x1, y1)
# z1 = np.ones((10,12))
# z1[6][2] = 1.1
# z1[7][2] = 1.1
# z1[6][3] = 1.1
# z1[7][3] = 1.1


# x1 = np.random.randn(2,2)
# # x2 = (x1 == 3)
# x2 = np.ones((2,2))
# x2 = (x2 == 1)
# x3 = x1 * x2
fig = plt.figure()
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(x,y,depA, cmap = cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# print(depA.size())