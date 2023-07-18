
# coding:utf-8
import numpy as np
import open3d as o3d
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading
import os
import shutil
import sys
sys.path.append("./seam_point_extract/")
sys.path.append("./")
import itertools
from seam_point_extract.path_line_gen import cal_gluepath



def vis_cut_save(folder_path, point_cloud):
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print("文件夹删除成功！")
    os.makedirs(folder_path)


    while True:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.run()  # user picks points
        vis.destroy_window()

        pcd = vis.get_cropped_geometry()
        print(len(np.array(pcd.points)))

        if len(np.array(pcd.points)) == len(np.array(point_cloud.points)):
            
            print(len(np.array(point_cloud.points)))
            break
        # else:
        #     o3d.io.write_point_cloud(folder_path + str(i) + '.ply',pcd)
        #     i = i+1

def read_pcd_extract_point(folder_path):

    file_list = os.listdir(folder_path)
    file_dict = {}
    for file_name in file_list:
        if file_name.endswith(".ply"):
            category = file_name[0].lower()

            if category in file_dict:
                file_dict[category].append(file_name)
            else:
                file_dict[category] = [file_name]

    all_trajectory_points = np.array([]).reshape((0, 3))

    n_piece = 3
    for category, files in file_dict.items():
        print(f"分类 {category} 中的文件：")

        class_trajectory_points = np.array([]).reshape((0, 3))
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            point_cloud = o3d.io.read_point_cloud(file_path)
            print(f"成功读取文件：{file_path}")

            trajectory_part = cal_gluepath(point_cloud,n_piece)
            sorted_indices = np.argsort(trajectory_part[:, 0])
            trajectory_part = trajectory_part[sorted_indices]

            print(trajectory_part)


            if list(class_trajectory_points)!=[]:
                point_11 = class_trajectory_points[-4-n_piece]
                point_12 = class_trajectory_points[-1]

                point_21 = trajectory_part[0]
                point_22 = trajectory_part[-1]


                # 创建所有可能的组合对
                point_combinations = list(itertools.product([point_11, point_12], [point_21, point_22]))

                # 初始化最小距离和对应的组合
                min_distance = float('inf')
                min_combination = None

                # 遍历所有组合对，计算距离并更新最小距离和组合
                for combination in point_combinations:
                    distance = np.linalg.norm(combination[0] - combination[1])
                    if distance < min_distance:
                        min_distance = distance
                        min_combination = combination

                if min_combination[0].any() == point_11.any():
                    class_trajectory_points[-4-n_piece:] = np.flip(class_trajectory_points[-4-n_piece:], axis=0)
                if min_combination[1].any() == point_22.any():
                    trajectory_part = np.flip(trajectory_part, axis=0)






                # distance_11_21 = np.linalg.norm(point_21 - point_11)
                # distance_11_22 = np.linalg.norm(point_22 - point_11)
                # distance_12_21 = np.linalg.norm(point_21 - point_12)
                # distance_12_22 = np.linalg.norm(point_22 - point_12)

                # if (distance_11_21 + distance_11_22) > (distance_12_21 + distance_12_22):


                
                

            class_trajectory_points = np.vstack((class_trajectory_points,trajectory_part))





        all_trajectory_points = np.vstack((all_trajectory_points,class_trajectory_points))



        all_trajectory_points = np.vstack((all_trajectory_points,np.array([0,0,0])))
    all_trajectory_points = np.array(all_trajectory_points)

    # print(all_trajectory_points)
    
    return all_trajectory_points


            







if __name__ == '__main__':

    folder_path = "seam_point_extract/extract_pcd/"
    point_cloud = o3d.io.read_point_cloud("seam_point_extract/test0.ply")
    # vis_cut_save(folder_path, point_cloud)
    read_pcd_extract_point(folder_path)


