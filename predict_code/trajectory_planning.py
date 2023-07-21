import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc



def project_point_to_line(point, line_origin, line_direction):
    # point: 三维点
    # line_origin: 直线上一点
    # line_direction: 直线的方向向量

    dx, dy, dz = point[0]-line_origin[0], point[1]-line_origin[1], point[2]-line_origin[2]
    t = (dx*line_direction[0] + dy*line_direction[1] + dz*line_direction[2]) / (line_direction[0]**2 + line_direction[1]**2 + line_direction[2]**2)
    return line_origin[0]+t*line_direction[0], line_origin[1]+t*line_direction[1], line_origin[2]+t*line_direction[2]









def cluster(points):
    
    # 随机采样一定比例的点(N * ratio)
    ratio = 1 
    n_samples = int(points.shape[0] * ratio)    
    indices = np.random.choice(points.shape[0], n_samples, replace=False)  
    points = points[indices]  

    dbscan = DBSCAN(eps=0.01, min_samples=20)     # eps代表相邻点的距离，eps越小，聚类的点越紧凑
    dbscan.fit(points)

    labels = dbscan.labels_   # N * ratio个点聚类后得到的标签，K个有效类别，则标签范围为-1, 0, 1, ..., K-1，-1为噪声点，所以labels共K+1个值。
    unique_labels, counts = np.unique(labels, return_counts=True)  # unique_labels为[-1, 0, 1, ..., K-1], counts为对应点的数量

    idx = list(range(len(unique_labels)))   # 创建索引列表
    sorted_idx = sorted(idx, key=lambda i: counts[i])   # 根据b排序索引列表
    # 根据排序后的索引列表重新排序a和b
    unique_labels = [unique_labels[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    print(unique_labels)
    print(counts)
    class_points = []
    

    N = 5000
    for i in range(len(unique_labels)): 
        if counts[i] > N:   
            class_points.append(points[labels == unique_labels[i]]  )
            print(class_points[-1].shape)
    return class_points




def trajectory_xyz_cal(class_points):

    all_pcd = []

    color_idx = 0
    color_map = np.array([
       [0.97649597, 0.55249099, 0.80433385],
       [0.63619032, 0.90842269, 0.5682129 ],
       [0.85903557, 0.77274305, 0.90438683],
       [0.72182724, 0.74963459, 0.81973374],
       [0.82467131, 0.69890571, 0.71902453],])
    

    class_points = [class_points[0]]


    for points in class_points:

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color_map[color_idx].tolist())
        color_idx = color_idx + 1


        # o3d.io.write_point_cloud("seam_point_extract/test1.ply",pcd)


        pcd_points = np.asarray(pcd.points)







        line = pyrsc.Line()
        A, B, inliers = line.fit(pcd_points, thresh=0.1, maxIteration=500)

        pcd_box = pcd.get_oriented_bounding_box()
        box_points = np.array(pcd_box.get_box_points())
        
        min_index = np.argmin(box_points[:, 0])
        min_xyz = box_points[min_index]
        max_index = np.argmax(box_points[:, 0])
        max_xyz = box_points[max_index]
        min_points = [min_xyz[0], min_xyz[1], min_xyz[2]]
        max_points = [max_xyz[0], max_xyz[1], max_xyz[2]]

        min_align = np.array(project_point_to_line(min_points, B, A))
        max_align = np.array(project_point_to_line(max_points, B, A))

        R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], A)

        # 端点

        mesh_sphere_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        mesh_sphere_1.paint_uniform_color([0.1, 0.1, 0.7])
        mesh_sphere_1 = mesh_sphere_1.translate(min_align)
        mesh_sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        mesh_sphere_2.paint_uniform_color([0.1, 0.1, 0.7])
        mesh_sphere_2 = mesh_sphere_2.translate(max_align)


        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin = [0,0,0])





        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=np.linalg.norm(min_align - max_align))
        mesh_cylinder.paint_uniform_color([1, 0, 0])
        mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
        mesh_cylinder = mesh_cylinder.translate((min_align + max_align)/2)

        all_pcd.append(pcd)
        all_pcd.append(mesh_cylinder)
        all_pcd.append(mesh_sphere_1)
        all_pcd.append(mesh_sphere_2)
        all_pcd.append(coord_mesh)

        all_pcd.append(pcd_box)

    o3d.visualization.draw_geometries(all_pcd)
    




