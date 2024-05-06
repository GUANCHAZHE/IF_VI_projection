#!/usr/bin/python
# -*- coding: UTF-8 -*-
from PIL import Image
import cv2
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image


def re_img(path):
    img = cv2.imread(path)
    a = int(20)  # x start
    b = int(1280)  # x end
    c = int(0)  # y start
    d = int(960)  # y end
    cropimg = img[c:d, a:b]
    imgresize = cv2.resize(cropimg, (1280, 960))
    return imgresize


if __name__ == '__main__':
    num = 9
    h = np.zeros((3, 3))

    # 读取红外图.
    inf_Fig = "./rgb1/" + str(num) + "_dual.png"
    img_dst = re_img(f"IR_1.png")
    pl.figure(), pl.imshow(img_dst[:, :, ::-1]), pl.title('dual')
    # 红外图中的4个点
    dst_point = plt.ginput(4)
    dst_point = np.float32(dst_point)

    # 读取RGB图像.
    RGB_Fig_1 = "./rgb1/" + str(num) + ".png"
    im_src_1 = cv2.imread(f"RGB_1.png")
    pl.figure(), pl.imshow(im_src_1[:, :, ::-1]), pl.title('rgb1')
    # RGB图中的4个点
    src_point_1 = plt.ginput(4)
    src_point_1 = np.float32(src_point_1)

    # Calculate Homography
    h, status = cv2.findHomography(src_point_1, dst_point)

    # Warp source image to destination based on homography
    im_out_1 = cv2.warpPerspective(im_src_1, h, (1280, 960), borderValue=(255, 255, 255))

    # pl.imshow(im_out_1[:, :, ::-1]), pl.title('out')
    # pl.show()  # show dst


    # colorize
    t_warp = cv2.applyColorMap(im_out_1, cv2.COLORMAP_JET)

    # mix rgb and thermal
    alpha = 0.5
    merge = cv2.addWeighted(im_src_1, alpha, t_warp, 1 - alpha, gamma=0)

    cv2.imshow("warp", merge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





    # 保存H矩阵
    H1_name = 'h' + str(num) + '.mat'
    print('H1:', h)
    sio.savemat(H1_name, {'Homography_Mat_1': h})