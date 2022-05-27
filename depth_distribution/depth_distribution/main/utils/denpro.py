# -*-coding:utf-8-*-
import math
import torch
import numpy as np
from ctypes import *
import ctypes
from PIL import Image
import os
from depth_distribution.main.utils import project_root

def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)


def getTargetDensity_16(pred_src_main, pred_depth_src_main):
    pred_src_main1 = pred_src_main.data
    pred_src_main1 = np.argmax(pred_src_main1.cpu(), axis=1)
    pred_src_main1 = pred_src_main1.squeeze(0)
    pred_depth_src_main1 = pred_depth_src_main.data.cpu()
    lastResult = None
    for index in range(16):
        pred_src_main2 = (pred_src_main1 == index)
        depA_resize = pred_depth_src_main1 * pred_src_main2

        x = np.arange(0, 161, 1)
        y = np.arange(0, 96, 1)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        xsub = x
        ysub = y
        depsub = depA_resize.flatten()
        plist= []

        # <editor-fold desc="折叠后要显示的内容">
        muu = [
            torch.tensor([[[45.0559, 60.1081, 57.0025],
                           [114.0175, 75.2374, 124.1213],
                           [80.5587, 88.9656, 208.2797],
                           [40.0585, 76.2665, 133.4725],
                           [108.5040, 56.9469, 50.9103]]]),
            torch.tensor([[[113.9093, 63.7743, 66.6802],
                           [75.8250, 42.7728, 33.0366],
                           [121.1666, 75.5658, 130.9216],
                           [41.5063, 71.4912, 104.0191],
                           [79.1152, 88.0335, 212.4189]]]),
            torch.tensor([[[114.5265, 21.8564, 26.1106],
                           [96.2255, 64.8322, 544.8223],
                           [139.4009, 34.9028, 89.8108],
                           [40.5167, 20.3220, 24.6087],
                           [21.8174, 37.8606, 109.1096]]]),
            torch.tensor([[[29.6328, 49.0054, 60.8076],
                           [119.9048, 44.1251, 50.5367],
                           [46.7740, 64.7614, 391.2308],
                           [134.4677, 57.3391, 120.4063],
                           [59.8972, 29.9913, 13.9626]]]),
            torch.tensor([[[28.9301, 71.4160, 206.5515],
                           [79.4456, 61.8408, 137.9468],
                           [46.2622, 48.4535, 44.8676],
                           [110.4898, 75.6002, 205.5559],
                           [111.2881, 51.2958, 52.9173]]]),
            torch.tensor([[[9.0436e+01, 5.6475e+01, 2.7783e+02],
                           [1.1532e+02, 4.3277e+01, 1.0265e+02],
                           [9.0934e+01, 5.0583e+01, 6.5536e+04],
                           [2.9525e+01, 3.9098e+01, 8.9003e+01],
                           [8.0403e+01, 3.1777e+01, 3.4423e+01]]]),
            torch.tensor([[[37.9588, 28.7442, 71.8573],
                           [128.0957, 32.1735, 96.1469],
                           [19.4525, 78.8301, 100.6571],
                           [139.3851, 12.6489, 213.3244],
                           [97.0319, 29.8436, 43.8376]]]),
            torch.tensor([[[8.2085e+01, 2.6254e+01, 2.0966e+01],
                           [3.5553e+01, 3.9207e+01, 8.1205e+01],
                           [8.2421e+01, 4.1523e+01, 2.3374e+02],
                           [1.2303e+02, 3.5640e+01, 8.2283e+01],
                           [8.7057e+01, 1.3229e+01, 6.5536e+04]]]),
            torch.tensor([[[7.2550e+01, 3.4876e+01, 6.5536e+04],
                           [3.4817e+01, 3.8018e+01, 9.5027e+01],
                           [9.2544e+01, 5.3230e+01, 2.7350e+02],
                           [1.2477e+02, 3.2073e+01, 7.5220e+01],
                           [6.9709e+01, 2.8874e+01, 3.2891e+01]]]),
            torch.tensor([[[106.0791, 10.1975, 1.0000],
                           [37.5669, 9.5305, 1.0000],
                           [7.3670, 9.9203, 1.0003],
                           [77.0665, 11.0956, 1.0000],
                           [139.1296, 8.8760, 1.0000]]]),
            torch.tensor([[[6.6884e+01, 7.4482e+01, 2.4131e+02],
                           [7.0455e+01, 5.2893e+01, 9.2652e+01],
                           [1.3538e+02, 6.5898e+01, 1.4679e+02],
                           [8.3391e+01, 7.6162e+01, 6.5536e+04],
                           [8.3005e+01, 3.9445e+01, 3.5659e+01]]]),
            torch.tensor([[[7.8033e+01, 7.7850e+01, 2.5261e+02],
                           [7.3109e+01, 5.7845e+01, 1.2220e+02],
                           [1.3026e+02, 4.8912e+01, 5.9756e+01],
                           [6.6062e+01, 4.0893e+01, 3.9038e+01],
                           [4.1768e+01, 7.6874e+01, 6.4270e+04]]]),
            torch.tensor([[[7.6052e+01, 4.0662e+01, 3.5233e+01],
                           [7.4207e+01, 5.8333e+01, 9.2355e+01],
                           [8.0688e+01, 8.3670e+01, 6.5536e+04],
                           [7.4924e+01, 8.5251e+01, 3.2237e+02],
                           [7.6566e+01, 7.1255e+01, 2.1230e+02]]]),
            torch.tensor([[[5.7026e+01, 7.4345e+01, 3.1372e+02],
                           [2.1018e+01, 4.2093e+01, 6.2388e+01],
                           [6.5602e+01, 5.6513e+01, 1.5054e+02],
                           [6.6345e+01, 7.3251e+01, 6.5536e+04],
                           [8.5299e+01, 3.7680e+01, 3.5366e+01]]]),
            torch.tensor([[[132.2186, 62.3686, 90.3784],
                           [45.8700, 60.0349, 94.6009],
                           [84.0503, 50.0939, 50.6526],
                           [93.3389, 82.6434, 219.1840],
                           [74.3177, 37.7007, 28.0798]]]),
            torch.tensor([[[64.8546, 85.6705, 245.9444],
                           [115.8203, 69.0320, 119.7422],
                           [31.8875, 68.7924, 137.7239],
                           [116.0786, 47.6162, 47.4288],
                           [52.3060, 45.8709, 45.4255]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        varr = [
            torch.tensor([[[661.5723, 271.0027, 544.7488],
                           [728.8984, 86.6836, 1254.5605],
                           [2022.8379, 18.2153, 1152.0352],
                           [571.5184, 71.2012, 1108.4570],
                           [768.1670, 265.7231, 429.4902]]]),
            torch.tensor([[[766.1152, 161.0847, 371.5420],
                           [1739.6348, 140.4017, 147.3030],
                           [586.2246, 76.2119, 1182.9414],
                           [626.1613, 124.5786, 1587.1631],
                           [2022.4243, 23.8096, 1527.0898]]]),
            torch.tensor([[[6.9765e+02, 1.6257e+02, 2.2057e+02],
                           [2.4774e+03, 7.2309e+02, 8.6693e+06],
                           [2.4550e+02, 4.9909e+02, 2.0947e+03],
                           [5.9088e+02, 1.5445e+02, 2.3166e+02],
                           [2.6187e+02, 5.2965e+02, 4.9394e+03]]]),
            torch.tensor([[[4.8563e+02, 1.5305e+02, 5.8974e+02],
                           [7.7107e+02, 6.7375e+01, 2.3907e+02],
                           [1.5894e+03, 6.7042e+02, 2.4177e+06],
                           [3.2484e+02, 4.7829e+01, 1.4948e+03],
                           [1.7261e+03, 1.1018e+02, 1.8138e+01]]]),
            torch.tensor([[[380.7434, 94.0190, 1252.1914],
                           [887.6455, 65.8413, 650.2227],
                           [788.6682, 244.7957, 438.2778],
                           [780.8008, 79.7871, 1508.8242],
                           [726.8301, 226.6731, 534.6460]]]),
            torch.tensor([[[2.4201e+03, 7.8699e+02, 1.7043e+04],
                           [7.9671e+02, 5.5926e+02, 1.8446e+03],
                           [2.0809e+03, 5.6149e+02, 1.0000e-06],
                           [3.4449e+02, 5.2484e+02, 1.4397e+03],
                           [1.7579e+03, 1.4399e+02, 2.6099e+02]]]),
            torch.tensor([[[489.0304, 156.4670, 1208.8286],
                           [405.1758, 519.3010, 1011.8906],
                           [116.3325, 58.4004, 94.6514],
                           [146.0742, 54.3930, 1365.0508],
                           [767.6318, 85.5063, 469.5856]]]),
            torch.tensor([[[1.4202e+03, 1.6962e+02, 2.2693e+02],
                           [5.0887e+02, 2.5644e+02, 1.4352e+03],
                           [2.1379e+03, 4.1569e+02, 1.5467e+04],
                           [5.4637e+02, 3.2353e+02, 1.3516e+03],
                           [6.3256e+01, 5.3719e+01, 1.0000e-06]]]),
            torch.tensor([[[4.4996e+03, 6.8744e+02, 1.0000e-06],
                           [5.1321e+02, 5.6183e+02, 2.2231e+03],
                           [2.4300e+03, 7.6645e+02, 1.8899e+04],
                           [5.0941e+02, 3.8729e+02, 1.2241e+03],
                           [1.5235e+03, 1.2239e+02, 2.1508e+02]]]),
            torch.tensor([[[1.5497e+02, 6.2394e+01, 1.1192e-06],
                           [4.0395e+02, 5.7941e+01, 1.0000e-06],
                           [2.2937e+01, 7.0089e+01, 2.1319e-04],
                           [1.5495e+02, 7.0302e+01, 1.0000e-06],
                           [1.5325e+02, 5.1135e+01, 1.0000e-06]]]),
            torch.tensor([[[2.0008e+03, 1.5812e+02, 1.3336e+04],
                           [1.7613e+03, 6.2840e+01, 1.1364e+03],
                           [2.5461e+02, 1.9121e+02, 4.1018e+03],
                           [2.2399e+03, 1.7110e+02, 1.0000e-06],
                           [1.6354e+03, 8.8167e+01, 2.4415e+02]]]),
            torch.tensor([[[2.3238e+03, 1.1140e+02, 1.5103e+04],
                           [1.9478e+03, 6.4391e+01, 2.3571e+03],
                           [3.6243e+02, 5.2539e+01, 4.0862e+02],
                           [1.3112e+03, 1.0269e+02, 3.3175e+02],
                           [4.8093e+02, 1.2223e+02, 3.9894e+07]]]),
            torch.tensor([[[1.8022e+03, 1.2144e+02, 2.0885e+02],
                           [2.1219e+03, 9.4064e+01, 1.2461e+03],
                           [1.7424e+03, 9.9879e+01, 1.0000e-06],
                           [2.2328e+03, 4.9172e+01, 2.4464e+04],
                           [2.4377e+03, 1.3114e+02, 8.4092e+03]]]),
            torch.tensor([[[2.3299e+03, 2.9604e+02, 1.6808e+04],
                           [1.8924e+02, 7.5994e+01, 1.1610e+03],
                           [2.6715e+03, 3.0404e+02, 1.9106e+03],
                           [2.7585e+03, 3.9783e+02, 1.0000e-06],
                           [1.7342e+03, 1.4299e+02, 2.5175e+02]]]),
            torch.tensor([[[285.6934, 88.5417, 666.7842],
                           [727.8357, 60.7764, 997.6631],
                           [1543.4077, 27.9370, 216.9060],
                           [1239.5557, 68.7896, 7950.0508],
                           [1488.6714, 97.0282, 101.1750]]]),
            torch.tensor([[[2310.1704, 39.0835, 4191.9297],
                           [742.3525, 103.4634, 1921.3828],
                           [403.1223, 92.3013, 2763.9551],
                           [666.9688, 97.6538, 376.8877],
                           [789.5596, 130.0447, 481.6218]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        pii = [
            torch.tensor([[[0.2489],
                           [0.2154],
                           [0.1603],
                           [0.1513],
                           [0.2241]]]),
            torch.tensor([[[0.1900],
                           [0.1895],
                           [0.1689],
                           [0.2654],
                           [0.1862]]]),
            torch.tensor([[[0.3477],
                           [0.0314],
                           [0.1573],
                           [0.3230],
                           [0.1406]]]),
            torch.tensor([[[0.2909],
                           [0.2113],
                           [0.1851],
                           [0.1093],
                           [0.2034]]]),
            torch.tensor([[[0.1476],
                           [0.2478],
                           [0.2269],
                           [0.0915],
                           [0.2863]]]),
            torch.tensor([[[2.0666e-01],
                           [3.0312e-01],
                           [2.3175e-04],
                           [1.9208e-01],
                           [2.9792e-01]]]),
            torch.tensor([[[0.3407],
                           [0.1859],
                           [0.0493],
                           [0.1397],
                           [0.2845]]]),
            torch.tensor([[[2.4801e-01],
                           [2.7592e-01],
                           [1.6129e-01],
                           [3.1454e-01],
                           [2.3739e-04]]]),
            torch.tensor([[[1.7174e-04],
                           [2.4903e-01],
                           [1.2951e-01],
                           [2.7711e-01],
                           [3.4418e-01]]]),
            torch.tensor([[[0.2382],
                           [0.3316],
                           [0.0036],
                           [0.2646],
                           [0.1621]]]),
            torch.tensor([[[3.0564e-01],
                           [2.6041e-01],
                           [1.4270e-01],
                           [4.3112e-05],
                           [2.9121e-01]]]),
            torch.tensor([[[3.1354e-01],
                           [3.2594e-01],
                           [9.6051e-02],
                           [2.6416e-01],
                           [3.0358e-04]]]),
            torch.tensor([[[0.2453],
                           [0.3105],
                           [0.0005],
                           [0.2123],
                           [0.2314]]]),
            torch.tensor([[[1.6673e-01],
                           [1.3625e-01],
                           [3.3683e-01],
                           [2.4782e-04],
                           [3.5995e-01]]]),
            torch.tensor([[[0.1143],
                           [0.2577],
                           [0.1771],
                           [0.3091],
                           [0.1418]]]),
            torch.tensor([[[0.1364],
                           [0.2331],
                           [0.1859],
                           [0.1895],
                           [0.2550]]])]
        # </editor-fold>

        mu = muu[index]
        var = varr[index]
        pi = pii[index]

        #linux
        dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'getDensity_16.so')
        dll.Add1.restype = c_float

        abc = mu.size()
        if len(abc) == 2:
            j, k = mu.size()
            munew = mu
        else:
            _, j, k = mu.size()
            munew = mu[0]

        numc = c_int(j)
        ucollect = c_float * (j * 3)
        my_array_u = ucollect()
        sigmacollect = c_float * (j * 3)
        my_array_sigma = sigmacollect()
        picollect = c_float * j
        my_array_pi = picollect()

        for i in range(j):
            my_array_pi[i] = pi[0][i][0].item()
            for h in range(3):
                my_array_u[i * 3 + h] = munew[i][h].item()
                my_array_sigma[i * 3 + h] = math.sqrt(var[0][i][h].item())

        for x0, y0, d0 in zip(xsub, ysub, depsub):
            if d0 > 0:
                p13 = dll.Add1(numc, my_array_u, my_array_sigma, my_array_pi, c_float(x0), c_float(y0), c_float(d0))
                plist.append(p13)
            else:
                plist.append(0)

        nplist = np.array(plist).reshape(96, 161)

        if index == 0:
            lastResult = np.expand_dims(nplist, 0)
        else:
            sub = np.expand_dims(nplist, 0)
            lastResult = np.append(lastResult, sub, axis=0)

    lastResult = np.max(lastResult, axis = 0)
    lastResult = np.expand_dims(lastResult, 0)
    lastResult = np.expand_dims(lastResult, 0) * 1e6
    lastResult = (1 - np.exp(-lastResult)) * 255
    lastResult = torch.tensor(lastResult, dtype=torch.float32)
    return lastResult


def getTargetDensity_7(pred_src_main, pred_depth_src_main):
    pred_src_main1 = pred_src_main.data
    pred_src_main1 = np.argmax(pred_src_main1.cpu(), axis=1)
    pred_src_main1 = pred_src_main1.squeeze(0)
    pred_depth_src_main1 = pred_depth_src_main.data.cpu()
    lastResult = None
    for inde in range(7):
        pred_src_main2 = (pred_src_main1 == inde)
        pred_depth_src_main2 = pred_depth_src_main1 * pred_src_main2

        # <editor-fold desc="折叠后要显示的内容">
        muu = [
            torch.tensor([[[ 80.9201,  88.6915, 212.2823],
         [ 39.8787,  75.3854, 125.6814],
         [112.1707,  54.9504,  49.9108],
         [115.9290,  74.8578, 120.9333],
         [ 45.8411,  55.8685,  51.2369]]]),
            torch.tensor([[[ 22.3049,  39.0390, 111.2544],
         [105.0419,  21.1523,  21.3978],
         [136.3934,  32.8147,  77.6158],
         [ 35.3923,  20.4677,  27.2980],
         [ 98.9900,  63.9776, 526.2227]]]),
            torch.tensor([[[1.2097e+02, 3.5675e+01, 6.2861e+01],
         [4.4094e+01, 3.3643e+01, 5.5709e+01],
         [9.5833e+01, 5.2100e+01, 3.3166e+02],
         [8.4427e+01, 5.3389e+01, 1.5948e+02],
         [9.0582e+01, 4.7187e+01, 6.5536e+04]]]),
            torch.tensor([[[8.8821e+01, 5.2598e+01, 2.7234e+02],
         [7.2550e+01, 3.4876e+01, 6.5536e+04],
         [3.2900e+01, 3.7672e+01, 9.1706e+01],
         [1.2441e+02, 3.3777e+01, 8.4977e+01],
         [7.7932e+01, 2.8784e+01, 3.4923e+01]]]),
            torch.tensor([[[ 83.0977,  10.9095,   1.0000],
         [ 19.0352,   9.5200,   1.0000],
         [110.3146,  10.0167,   1.0000],
         [140.7229,   8.9811,   1.0000],
         [ 53.2517,   9.9357,   1.0000]]]),
            torch.tensor([[[8.1149e+01, 3.6202e+01, 2.8441e+01],
         [3.4854e+01, 7.0933e+01, 2.0987e+02],
         [1.2154e+02, 7.1593e+01, 2.0477e+02],
         [8.3444e+01, 4.9496e+01, 7.3980e+01],
         [8.3391e+01, 7.6162e+01, 6.5536e+04]]]),
            torch.tensor([[[3.9423e+01, 6.3927e+01, 1.1360e+02],
         [8.0688e+01, 8.3670e+01, 6.5536e+04],
         [7.4208e+01, 8.0954e+01, 3.1531e+02],
         [7.6146e+01, 4.2733e+01, 3.9763e+01],
         [1.2496e+02, 6.3865e+01, 1.2064e+02]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        varr = [
            torch.tensor([[[2018.9785, 19.7197, 1298.3672],
                     [576.8694, 86.1704, 1360.2979],
                     [706.2090, 246.8770, 423.8750],
                     [701.7227, 91.4927, 1366.1523],
                     [676.1948, 264.2642, 488.8052]]]),
            torch.tensor([[[2.8000e+02, 5.4070e+02, 5.0085e+03],
         [9.1836e+02, 1.4877e+02, 1.1831e+02],
         [3.1846e+02, 4.5064e+02, 1.7491e+03],
         [4.8607e+02, 1.6047e+02, 2.8431e+02],
         [2.4774e+03, 7.3378e+02, 8.0812e+06]]]),
            torch.tensor([[[5.3505e+02, 3.3572e+02, 1.2748e+03],
         [7.1946e+02, 2.6699e+02, 1.0249e+03],
         [2.4148e+03, 8.4543e+02, 1.8514e+04],
         [2.2926e+03, 7.1894e+02, 3.0054e+03],
         [1.8987e+03, 6.3065e+02, 1.0000e-06]]]),
            torch.tensor([[[2.4625e+03, 7.6392e+02, 1.8712e+04],
         [4.4996e+03, 6.8744e+02, 1.0000e-06],
         [4.5577e+02, 5.5073e+02, 2.1684e+03],
         [5.3936e+02, 4.6581e+02, 1.2097e+03],
         [1.6940e+03, 1.2964e+02, 2.5526e+02]]]),
            torch.tensor([[[9.7861e+01, 6.8730e+01, 1.0000e-06],
                     [1.2264e+02, 6.0046e+01, 6.1260e-06],
                     [9.6138e+01, 6.0870e+01, 1.0000e-06],
                     [1.2361e+02, 5.2529e+01, 1.0000e-06],
                     [1.3391e+02, 6.0372e+01, 1.0000e-06]]]),
            torch.tensor([[[1.5719e+03, 8.0275e+01, 1.2307e+02],
         [5.1515e+02, 1.7773e+02, 1.1932e+04],
         [6.1765e+02, 1.7254e+02, 1.1671e+04],
         [1.8959e+03, 5.5904e+01, 8.4064e+02],
         [2.2399e+03, 1.7110e+02, 1.0000e-06]]]),
            torch.tensor([[[6.2604e+02, 1.5160e+02, 2.5305e+03],
         [1.7424e+03, 9.9881e+01, 1.0000e-06],
         [2.2163e+03, 1.0931e+02, 1.7437e+04],
         [1.7417e+03, 1.2796e+02, 3.0374e+02],
         [5.2574e+02, 1.3467e+02, 2.6676e+03]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        pii = [
            torch.tensor([[[0.1674],
         [0.1827],
         [0.2077],
         [0.2290],
         [0.2133]]]),
            torch.tensor([[[0.1377],
         [0.3453],
         [0.2091],
         [0.2744],
         [0.0334]]]),
            torch.tensor([[[3.1434e-01],
         [3.4673e-01],
         [1.1499e-01],
         [2.2372e-01],
         [2.2461e-04]]]),
            torch.tensor([[[1.3178e-01],
         [1.7174e-04],
         [2.3923e-01],
         [2.2898e-01],
         [3.9984e-01]]]),
            torch.tensor([[[0.2609],
         [0.1482],
         [0.1994],
         [0.1526],
         [0.2389]]]),
            torch.tensor([[[1.9000e-01],
         [2.1929e-01],
         [2.6065e-01],
         [3.3002e-01],
         [4.3112e-05]]]),
            torch.tensor([[[0.2325],
         [0.0004],
         [0.2962],
         [0.2985],
         [0.1724]]])]
        # </editor-fold>

        mu = muu[inde]
        var = varr[inde]
        pi = pii[inde]

        x = np.arange(0, 161, 1)
        y = np.arange(0, 96, 1)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = pred_depth_src_main2.flatten()
        xsub = x
        ysub = y
        zsub = z

        plist= []
        dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'getDensity_7.so')
        dll.Add1.restype = c_float

        abc = mu.size()

        if len(abc) == 2:
            j, k = mu.size()
            munew = mu
        else:
            _, j, k = mu.size()
            munew = mu[0]

        numc = c_int(j)

        ucollect = c_float * (j * 3)
        my_array_u = ucollect()

        sigmacollect = c_float * (j * 3)
        my_array_sigma = sigmacollect()

        picollect = c_float * j
        my_array_pi = picollect()

        for i in range(j):
            my_array_pi[i] = pi[0][i][0].item()
            for h in range(3):
                my_array_u[i * 3 + h] = munew[i][h].item()
                my_array_sigma[i * 3 + h] = math.sqrt(var[0][i][h].item())

        for x0, y0, z0 in zip(xsub, ysub, zsub):
            if z0 > 0:
                p13 = dll.Add1(numc, my_array_u, my_array_sigma, my_array_pi, c_float(x0), c_float(y0), c_float(z0))
                plist.append(p13)
            else:
                plist.append(0)

        nplist = np.array(plist).reshape(96, 161)
        if inde == 0:
            lastResult = np.expand_dims(nplist, 0)
        else:
            sub = np.expand_dims(nplist, 0)
            lastResult = np.append(lastResult, sub, axis=0)

    lastResult = np.max(lastResult, axis = 0)
    lastResult = np.expand_dims(lastResult, 0)
    lastResult = np.expand_dims(lastResult, 0) * 1e6
    lastResult = (1 - np.exp(-lastResult)) * 255
    lastResult = torch.tensor(lastResult, dtype=torch.float32)
    return lastResult

def getTargetDensity_7_small(pred_src_main, pred_depth_src_main):
    pred_src_main1 = pred_src_main.data
    pred_src_main1 = np.argmax(pred_src_main1.cpu(), axis=1)
    pred_src_main1 = pred_src_main1.squeeze(0)
    pred_depth_src_main1 = pred_depth_src_main.data.cpu()
    lastResult = None
    for inde in range(7):
        pred_src_main2 = (pred_src_main1 == inde)
        pred_depth_src_main2 = pred_depth_src_main1 * pred_src_main2

        # <editor-fold desc="折叠后要显示的内容">
        muu = [
            torch.tensor([[[59.1910, 29.1475, 80.7967],
                           [39.2703, 19.2301, 36.1118],
                           [21.6726, 29.2775, 81.0389],
                           [40.4963, 32.8856, 153.6384],
                           [40.1653, 37.8269, 213.7003]]]),
            torch.tensor([[[5.7690e+01, 9.0026e+00, 2.4266e+01],
                           [4.2754e+01, 1.3737e+01, 7.8367e+01],
                           [1.8829e+01, 8.3679e+00, 2.2378e+01],
                           [5.5065e+01, 2.6483e+01, 6.5536e+04],
                           [3.8255e+01, 2.3030e+01, 2.6072e+02]]]),
            torch.tensor([[[4.5856e+01, 2.0223e+01, 1.4626e+02],
                           [4.6585e+01, 2.2706e+01, 3.3155e+02],
                           [4.3791e+01, 1.9674e+01, 6.5536e+04],
                           [3.9209e+01, 1.3040e+01, 3.1395e+01],
                           [3.8391e+01, 1.6291e+01, 7.8229e+01]]]),
            torch.tensor([[[5.7694e+01, 1.1909e+01, 4.7474e+01],
                           [4.4623e+01, 2.1726e+01, 3.3302e+02],
                           [3.8442e+01, 1.8333e+01, 1.2281e+02],
                           [1.9656e+01, 1.2636e+01, 4.3790e+01],
                           [3.5102e+01, 1.4604e+01, 6.5536e+04]]]),
            torch.tensor([[[56.0852, 3.9617, 1.0000],
                           [42.4536, 4.3896, 1.0000],
                           [27.1019, 3.9922, 1.0000],
                           [71.3339, 3.5425, 1.0000],
                           [8.9980, 3.7794, 1.0000]]]),
            torch.tensor([[[4.0836e+01, 1.5334e+01, 2.9207e+01],
                           [2.6177e+01, 2.1178e+01, 7.7986e+01],
                           [4.1375e+01, 3.1889e+01, 6.5536e+04],
                           [6.4418e+01, 2.1845e+01, 8.4458e+01],
                           [4.0155e+01, 3.0828e+01, 2.1826e+02]]]),
            torch.tensor([[[1.6035e+01, 2.6969e+01, 1.1116e+02],
                           [5.8080e+01, 2.7132e+01, 1.2232e+02],
                           [3.8589e+01, 1.8115e+01, 4.0432e+01],
                           [4.0649e+01, 3.5529e+01, 6.5536e+04],
                           [3.6777e+01, 3.4210e+01, 3.1308e+02]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        varr = [
            torch.tensor([[[155.5898, 27.5684, 667.1572],
                           [418.9441, 29.1173, 181.9380],
                           [164.8805, 27.8533, 677.6689],
                           [501.6674, 8.4346, 678.4258],
                           [529.2122, 3.0503, 1343.6445]]]),
            torch.tensor([[[1.8410e+02, 2.9126e+01, 1.6432e+02],
                           [7.9766e+02, 7.5763e+01, 8.6585e+02],
                           [1.3849e+02, 2.8042e+01, 1.5488e+02],
                           [5.2172e+02, 1.5578e+02, 1.0000e-06],
                           [8.8286e+02, 1.4165e+02, 2.0825e+04]]]),
            torch.tensor([[[5.6327e+02, 1.2807e+02, 2.8461e+03],
                           [6.3815e+02, 1.5206e+02, 1.7461e+04],
                           [4.4207e+02, 1.0741e+02, 1.0000e-06],
                           [4.8192e+02, 2.6626e+01, 2.2902e+02],
                           [5.5971e+02, 8.0185e+01, 5.8087e+02]]]),
            torch.tensor([[[1.6337e+02, 3.3909e+01, 5.5100e+02],
                           [6.8422e+02, 1.4516e+02, 1.7520e+04],
                           [6.3007e+02, 1.2523e+02, 2.3742e+03],
                           [1.4402e+02, 3.8992e+01, 5.3705e+02],
                           [1.0976e+03, 1.2818e+02, 1.0000e-06]]]),
            torch.tensor([[[2.5936e+01, 1.0882e+01, 1.0000e-06],
                           [2.2917e+01, 1.2533e+01, 1.3576e-06],
                           [4.0304e+01, 1.1065e+01, 1.0000e-06],
                           [2.7043e+01, 9.4719e+00, 8.8079e-07],
                           [2.9422e+01, 1.0874e+01, 9.2254e-06]]]),
            torch.tensor([[[3.7908e+02, 1.4685e+01, 1.3361e+02],
                           [2.3438e+02, 1.1152e+01, 9.7655e+02],
                           [5.7587e+02, 3.0127e+01, 1.0000e-06],
                           [9.4835e+01, 1.3516e+01, 1.3003e+03],
                           [5.9544e+02, 3.0129e+01, 1.1916e+04]]]),
            torch.tensor([[[1.0643e+02, 2.8050e+01, 2.3480e+03],
                           [2.0958e+02, 2.4077e+01, 2.5510e+03],
                           [4.4334e+02, 2.3851e+01, 3.1951e+02],
                           [4.1418e+02, 1.5705e+01, 1.0000e-06],
                           [5.6144e+02, 2.0499e+01, 1.6939e+04]]])]
        # </editor-fold>

        # <editor-fold desc="折叠后要显示的内容">
        pii = [
            torch.tensor([[[0.2128],
                           [0.2120],
                           [0.2093],
                           [0.2088],
                           [0.1570]]]),
            torch.tensor([[[3.4481e-01],
                           [2.6930e-01],
                           [2.9454e-01],
                           [6.1655e-05],
                           [9.1284e-02]]]),
            torch.tensor([[[2.7142e-01],
                           [1.1941e-01],
                           [2.3347e-04],
                           [2.9912e-01],
                           [3.0982e-01]]]),
            torch.tensor([[[3.3203e-01],
                           [8.3727e-02],
                           [2.7867e-01],
                           [3.0540e-01],
                           [1.6715e-04]]]),
            torch.tensor([[[0.2047],
                           [0.2456],
                           [0.2699],
                           [0.1381],
                           [0.1417]]]),
            torch.tensor([[[1.9996e-01],
                           [2.1343e-01],
                           [5.4337e-05],
                           [1.5858e-01],
                           [4.2797e-01]]]),
            torch.tensor([[[0.1890],
                           [0.2019],
                           [0.3075],
                           [0.0004],
                           [0.3011]]])]
        # </editor-fold>

        mu = muu[inde]
        var = varr[inde]
        pi = pii[inde]

        x = np.arange(0, 81, 1)
        y = np.arange(0, 41, 1)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = pred_depth_src_main2.flatten()
        xsub = x
        ysub = y
        zsub = z

        plist= []
        dll = ctypes.cdll.LoadLibrary(str(project_root) + os.sep + 'getDensity_7.so')
        dll.Add1.restype = c_float

        abc = mu.size()

        if len(abc) == 2:
            j, k = mu.size()
            munew = mu
        else:
            _, j, k = mu.size()
            munew = mu[0]

        numc = c_int(j)

        ucollect = c_float * (j * 3)
        my_array_u = ucollect()

        sigmacollect = c_float * (j * 3)
        my_array_sigma = sigmacollect()

        picollect = c_float * j
        my_array_pi = picollect()

        for i in range(j):
            my_array_pi[i] = pi[0][i][0].item()
            for h in range(3):
                my_array_u[i * 3 + h] = munew[i][h].item()
                my_array_sigma[i * 3 + h] = math.sqrt(var[0][i][h].item())

        for x0, y0, z0 in zip(xsub, ysub, zsub):
            if z0 > 0:
                p13 = dll.Add1(numc, my_array_u, my_array_sigma, my_array_pi, c_float(x0), c_float(y0), c_float(z0))
                plist.append(p13)
            else:
                plist.append(0)

        nplist = np.array(plist).reshape(41, 81)
        if inde == 0:
            lastResult = np.expand_dims(nplist, 0)
        else:
            sub = np.expand_dims(nplist, 0)
            lastResult = np.append(lastResult, sub, axis=0)

    lastResult = np.max(lastResult, axis = 0)
    lastResult = np.expand_dims(lastResult, 0)
    lastResult = np.expand_dims(lastResult, 0) * 1e6
    lastResult = (1 - np.exp(-lastResult)) * 255
    lastResult = torch.tensor(lastResult, dtype=torch.float32)
    return lastResult

def getSourceDensity(density_pre_source, pred_seg_src, expid):
    density_source = density_pre_source.numpy()
    if expid == 1 or expid == 2 or expid == 4:
        xf = np.arange(0, 161, 1)
        yf = np.arange(0, 96, 1)
        xf, yf = np.meshgrid(xf, yf)
        a0 = [0] * 15456
        pred_seg_src_sub = pred_seg_src.data
        pred_seg_src_sub_1 = np.argmax(pred_seg_src_sub.cpu(), axis=1)
        a1 = pred_seg_src_sub_1.flatten()
        a2 = yf.flatten()
        a3 = xf.flatten()
        choose = [a0, a1, a2, a3]
        density_source_result = density_source[choose].reshape(96, 161)
    else:
        xf = np.arange(0, 81, 1)
        yf = np.arange(0, 41, 1)
        xf, yf = np.meshgrid(xf, yf)
        a0 = [0] * 3321
        pred_seg_src_sub = pred_seg_src.data
        pred_seg_src_sub_1 = np.argmax(pred_seg_src_sub.cpu(), axis=1)
        a1 = pred_seg_src_sub_1.flatten()
        a2 = yf.flatten()
        a3 = xf.flatten()
        choose = [a0, a1, a2, a3]
        density_source_result = density_source[choose].reshape(41, 81)
    density_source_result = torch.from_numpy(density_source_result).unsqueeze(0).unsqueeze(0)
    return density_source_result









