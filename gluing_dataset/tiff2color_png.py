import cv2
import numpy as np




for i in range(1,257):
    print(i)
    # 读取TIFF深度图像
    depth_image = cv2.imread('gluing_dataset/depth/depth'+str(i)+'.tiff', cv2.IMREAD_ANYDEPTH)

    min_value = 0
    max_value = 1
    depth_image = np.clip(depth_image, min_value, max_value)

    # 将NaN值替换为0
    # depth_image[np.isnan(depth_image)] = 0

    # 将深度图像线性映射到0-255范围
    normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 使用伪彩色映射创建深度图像
    colormap = cv2.COLORMAP_JET
    pseudo_color_depth_image = cv2.applyColorMap(normalized_depth_image, colormap)

    # 显示和保存结果
    # cv2.imshow('Pseudo Color Depth Image', pseudo_color_depth_image)
    # cv2.waitKey(0)
    cv2.imwrite('gluing_dataset/colored_depth/depth'+str(i)+'.png', pseudo_color_depth_image)