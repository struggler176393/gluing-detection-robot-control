import os
import numpy as np
import cv2
from PIL import Image
from time import time
import open3d as o3d


def NPImgTocolorPointMap(color, depth, ifvalid=False, ifvis=False):
    # im = Image.open('Data/dp1.tiff')
    color = np.array(color)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    color = color.reshape(-1,3)/255
    
    # 将图像转化成numpy数组
    depth = np.array(depth)




    intrinsic_matrix = np.array([[2.40189575e+03, 0.00000000e+00, 7.41044678e+02],
                                  [0.00000000e+00, 2.40110034e+03, 5.95356873e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    
    distortion_cv = np.array([-3.33325416e-02,  3.83851230e-02,  5.59388369e-04,  4.35525930e-04,  6.31685734e-01])
    width = 1440
    height = 1080
    X = range(width)
    Y = range(height)
    XY = np.array(np.meshgrid(X, Y), dtype=float).reshape((2, -1)).T
    undistorted_XY = cv2.undistortPoints(
        XY, intrinsic_matrix, distortion_cv).reshape((height, width, 2))
    convert_pm = np.array([undistorted_XY[:, :, 0] * depth,
                            undistorted_XY[:, :, 1] * depth, depth]).reshape((3, -1)).T
    
    
    

    
    if ifvalid:
        valid_color_img = color[~np.isnan(convert_pm[:, 2])]
        valid_pcd_data = convert_pm[~np.isnan(convert_pm[:, 2])]


        data = np.hstack([valid_pcd_data,valid_color_img])
    else:
        data = np.hstack([convert_pm,color])
    # np.savetxt("Data/point_map1.xyz", convert_pm)

    if ifvis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)
        pcd.colors = o3d.utility.Vector3dVector(valid_color_img)

        # o3d.io.write_point_cloud("test1.ply",pcd)
        # pcd = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # cl, ind = pcd.remove_radius_outlier(nb_points=2, radius=1)
        # pcd = pcd.select_by_index(ind)
        # pcd = pcd.voxel_down_sample(voxel_size=0.1)
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                             std_ratio=0.1)
        # pcd = pcd.select_by_index(ind)

        o3d.visualization.draw_geometries([pcd])
    
    return data       # 三行xyz，三行rgb









