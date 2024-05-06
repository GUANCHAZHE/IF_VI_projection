import numpy as np
import cv2
from scipy.optimize import least_squares

def calculate_fundamental_matrix(points_rgb, points_ir):
    """
    计算两个相机之间的基础矩阵F
    """
    def objective_function(x, points_rgb, points_ir):
        F = x.reshape(3, 3)
        residuals = []
        for p_rgb, p_ir in zip(points_rgb, points_ir):
            p_rgb = np.append(p_rgb, 1)
            p_ir = np.append(p_ir, 1)
            residuals.append(np.dot(p_ir.T, np.dot(F, p_rgb)))
        return np.array(residuals).flatten()

    # 初始化F矩阵
    F_init = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    res = least_squares(objective_function, F_init.flatten(), args=(points_rgb, points_ir))
    F = res.x.reshape(3, 3)
    return F

def project_image(ir_img, F, intrinsic_rgb, distortion_rgb):
    """
    使用基础矩阵F和内参将整个IR图像投影到RGB图像上
    """
    h, w = ir_img.shape[:2]
    rgb_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            p_ir = np.array([x, y, 1])
            l = np.dot(F.T, p_ir)
            l = l / np.linalg.norm(l[:2])
            x_rgb, y_rgb = cv2.undistortPoints(np.array([[l[0] / l[2], l[1] / l[2]]]), intrinsic_rgb, distortion_rgb)[0][0]
            x_rgb, y_rgb = int(x_rgb), int(y_rgb)

            if 0 <= x_rgb < 1920 and 0 <= y_rgb < 1080:
                rgb_img[y_rgb, x_rgb] = ir_img[y, x]

    return rgb_img


intrinsic_rgb = np.array([[2.84693e+03, 0, 1.05265e+03],
[0, 2.14208e+03, 5.63435e+02],
[0, 0, 1]])

distortion_rgb = np.array([-0.460342, 0.306823, 0, 0, 0])

intrinsic_ir = np.array([[2.90291e+03, 0, 6.56707e+02],
[0, 2.89747e+03, 4.83656e+02],
[0, 0, 1]])

distortion_ir = np.array([-0.460342, 0.306823, 0, 0, 0])

# 加载图像
rgb_img = cv2.imread('RGB_1.png')
ir_img = cv2.imread('IR_1.png', cv2.IMREAD_GRAYSCALE)

# 提取特征点
sift = cv2.SIFT_create()
kp_rgb, des_rgb = sift.detectAndCompute(rgb_img, None)
kp_ir, des_ir = sift.detectAndCompute(ir_img, None)

# 匹配特征点
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_rgb, des_ir, k=2)

# 筛选好的匹配点
good_matches = []
points_rgb = []
points_ir = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        points_rgb.append(kp_rgb[m.queryIdx].pt)
        points_ir.append(kp_ir[m.trainIdx].pt)

# 计算基础矩阵F
F = calculate_fundamental_matrix(np.array(points_rgb), np.array(points_ir))

# 使用F将整个IR图像投影到RGB图像上
projected_img = project_image(ir_img, F, intrinsic_rgb, distortion_rgb)

# 显示结果
cv2.imshow('Projected IR Image', projected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()