#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
    #1 这个程序的目的就是对两个图像进行畸变矫正，
    # 可见光的分辨率是1920*1080，红外的分辨率是1280*960，
    # 内参
    intrinsic_rgb = np.array([[2.84693e+03, 0, 1.05265e+03],
                              [0, 2.14208e+03, 5.63435e+02],
                              [0, 0, 1]])

    distortion_rgb = np.array([-0.460342, 0.306823, 0, 0, 0])

    intrinsic_ir = np.array([[2.90291e+03, 0, 6.56707e+02],
                             [0, 2.89747e+03, 4.83656e+02],
                             [0, 0, 1]])

    distortion_ir = np.array([-0.460342, 0.306823, 0, 0, 0])

    # 加载图像

    vi_img_ori = cv2.imread('RGB_1.png')
    ir_img_ori = cv2.imread('IR_1.png')
    # 畸变矫正
    vi_img_ud = cv2.undistort(vi_img_ori, intrinsic_rgb, distortion_rgb)
    ir_img_ud = cv2.undistort(ir_img_ori, intrinsic_ir, distortion_ir)

    # 可见光 选取四个点
    img_dst = re_img(f"IR_1.png")
    pl.figure(), pl.imshow(img_dst[:, :, ::-1]), pl.title('dual')
    vi_points = plt.ginput(4)
    vi_points = np.float32(vi_points)

    # 红外 选取四个点
    im_src_1 = cv2.imread(f"RGB_1.png")
    pl.figure(), pl.imshow(im_src_1[:, :, ::-1]), pl.title('rgb1')
    ir_points = plt.ginput(4)
    ir_points = np.float32(ir_points)


    # 3 然后计算单应矩阵，
    # 4 将红外图像投影到可见光图像上。
    # 5 将可见光裁切成红外图像大小，