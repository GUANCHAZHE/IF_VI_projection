#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import scipy

if __name__ == '__main__':

    # height = 640
    # width = 480
    # rgb_img = cv2.imread(f"vi.jpg")
    # t_img = cv2.imread(f"ir.jpg")

    # height = 2048
    # width = 1053
    # rgb_img = cv2.imread(f"test2.png")
    # t_img = cv2.imread(f"test1.png")

    height = 1920
    width = 1080
    rgb_img = cv2.imread(f"RGB_1.png")
    t_img = cv2.imread(f"IR_1.png")


    # 这就是单应矩阵的参数
    h = np.loadtxt("H.txt")
    h = np.transpose(h)

    # 测试算法流程代码
    # h = np.eye(3)

    # data = scipy.io.loadmat('h9.mat')
    # H1_data = data['Homography_Mat_1']
    # h = np.array(H1_data)

    t_warp = cv2.warpPerspective(t_img, h, (height, width), borderValue=(255, 255, 255))

    # colorize
    t_warp = cv2.applyColorMap(t_warp, cv2.COLORMAP_JET)

    # mix rgb and thermal
    alpha = 0.5
    merge = cv2.addWeighted(rgb_img, alpha, t_warp, 1 - alpha, gamma=0)

    cv2.imshow("warp", merge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

