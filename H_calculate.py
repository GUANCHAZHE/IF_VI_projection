#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def find_chessboard(filename, pattern=(9, 8), wind_name="rgb"):
    # read input image
    img = cv2.imread(filename)
    # cv2.imshow("raw", img)
    # img = cv2.undistort(img, camera_matrix, distortion_coefficients)

    # convert the input image to a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)

    # if chessboard corners are detected
    if ret == True:
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, pattern, corners, ret)

        # Draw number，打印角点编号，便于确定对应点
        corners = np.ceil(corners[:, 0, :])
        for i, pt in enumerate(corners):
            cv2.putText(img, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imshow(wind_name, img)

        return corners

    return None



if __name__ == '__main__':

    idx = 1  # 0~71
    rgb_img = cv2.imread(f"RGB_{idx}.png")
    t_img = cv2.imread(f"IR_{idx}.png")

    # chessboard grid nums in rgb ,注意观察，同一块标定板在RGB相机和红外相机中的格子说可能不一样

    # 目前可行的rgb参数 9，6 这是应为他的板子颜色反了
    rgb_width, rgb_height = 9 ,6
    rgb_corners = find_chessboard(f"RGB_{idx}.png", (rgb_width, rgb_height), "rgb")
    rgb_corners_flip = np.flip(rgb_corners, axis=0)
    rgb_corners = np.flip(rgb_corners, axis=0)
    rgb_corners[:,1] += 38

    # chessboard grid nums in thermal
    thermal_width, thermal_height = 11, 8
    t_corners = find_chessboard(f"IR_{idx}.png", (thermal_width, thermal_height), "thermal")

    if rgb_corners is not None and t_corners is not None:
        # test the id correspondence between rgb and thermal corners
        rgb_idx = 27  # 可视化一个点，确认取对应点的过程是否正确
        row, col = rgb_idx // rgb_width, rgb_idx % rgb_width
        t_idx = row * thermal_width + col + 1

        pt = rgb_corners[rgb_idx]
        cv2.putText(rgb_img, str(rgb_idx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        pt = t_corners[t_idx]
        cv2.putText(t_img, str(t_idx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imshow(f"Point {rgb_idx} on rgb", rgb_img)
        cv2.imshow(f"Point {t_idx} on thermal", t_img)

        # Calculate Homography
        src_pts = []
        for rgb_idx in range(len(rgb_corners)):
            row, col = rgb_idx // 9, rgb_idx % 9
            t_idx = row * 11 + col + 1
            src_pts.append(t_corners[t_idx])
        h, status = cv2.findHomography(np.array(src_pts)[:, None, :], rgb_corners[:, None, :])

        np.savetxt("calib.param", h)

        # Warp source image to destination based on homography
        t_warp = cv2.warpPerspective(t_img, h, (1920, 1080), borderValue=(255, 255, 255))

        # colorize
        t_warp = cv2.applyColorMap(t_warp, cv2.COLORMAP_JET)

        # mix rgb and thermal
        alpha = 0.5
        merge = cv2.addWeighted(rgb_img, alpha, t_warp, 1 - alpha, gamma=0)

        cv2.imshow("warp", merge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()