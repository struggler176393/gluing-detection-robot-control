import open3d as o3d
import numpy as np





直角分段思路：将点云等分成多段（有问题，直角怎么分），得到中心点，根据前后点向量余弦来判断转折
# import numpy as np

# def split_point_cloud_at_corners(point_cloud, threshold):
#     segments = []  # 存储分割后的线段
    
#     # 第一个点作为当前线段的起点
#     start_point = point_cloud[0]
#     current_segment = [start_point]
    
#     for i in range(1, len(point_cloud)-1):
#         prev_point = point_cloud[i-1]
#         current_point = point_cloud[i]
#         next_point = point_cloud[i+1]
        
#         # 计算前后两个向量
#         prev_vector = current_point - prev_point
#         next_vector = next_point - current_point
        
#         # 计算前后两个向量的夹角余弦值
#         cos_angle = np.dot(prev_vector, next_vector) / (np.linalg.norm(prev_vector) * np.linalg.norm(next_vector))
        
#         if cos_angle <= threshold:
#             # 如果夹角余弦值小于等于阈值，说明位于转折处，将当前点添加到当前线段
#             current_segment.append(current_point)
#         else:
#             # 否则，结束当前线段，将其添加到segments列表，并重新开始一个新的线段
#             segments.append(np.array(current_segment))
#             start_point = current_point
#             current_segment = [start_point]
    
#     # 将最后一个线段添加到segments列表
#     current_segment.append(point_cloud[-1])
#     segments.append(np.array(current_segment))
    
#     return segments

# # 示例点云数据
# point_cloud = np.array([[1, 2, 3],
#                        [3, 4, 5],
#                        [6, 7, 8],
#                        [8, 9, 10],
#                        [12, 15, 16]])

# # 执行分割操作
# segments = split_point_cloud_at_corners(point_cloud, threshold=0.98)

# # 打印结果
# for i, segment in enumerate(segments):
#     print("线段 {}: {}".format(i+1, segment))










def adaptive_split_point_cloud(point_cloud):
    # 计算点云在各个轴上的方差
    variances = np.var(point_cloud, axis=0)
    
    # 选择方差最大的轴作为划分轴
    split_axis = np.argmax(variances)
    
    # 根据划分轴的范围选择划分点
    axis_values = point_cloud[:, split_axis]
    min_val = np.min(axis_values)
    max_val = np.max(axis_values)
    
    # 找到直角所在位置的划分点
    split_point = find_right_angle(split_axis, axis_values)
    
    return split_axis, split_point

def find_right_angle(axis, values):
    # 找到直角所在位置的划分点
    diff = np.diff(values)
    right_angles = np.where(diff == 0)[0]
    
    if len(right_angles) > 0:
        # 如果存在直角，选择第一个直角所在位置作为划分点
        split_point = values[right_angles[0]]
    else:
        # 如果不存在直角，则选择轴范围的中点作为划分点
        split_point = (np.min(values) + np.max(values)) / 2
    
    return split_point



def split_point_cloud(point_cloud, axis, split_point):
    # 根据选择的轴进行排序
    sorted_points = point_cloud[np.argsort(point_cloud[:, axis])]
    
    # 划分点云
    left_cloud = sorted_points[sorted_points[:, axis] <= split_point]
    right_cloud = sorted_points[sorted_points[:, axis] > split_point]
    
    return left_cloud, right_cloud



def fit_plane(point_cloud):
    # 平面拟合算法
    plane_model1, inliers = point_cloud.segment_plane(distance_threshold=0.0005, ransac_n=10, num_iterations=100)


    inlier_cloud = point_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 0, 1.0])
    print(inlier_cloud)
    # 平面外点点云
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1.0, 0, 0])
    print(outlier_cloud)
    # 可视化平面分割结果
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])









# 加载点云数据
point_cloud = o3d.io.read_point_cloud("seam_point_extract/test1.ply")
print(np.array(point_cloud.points).shape)



# 示例点云数据
np_pcd = np.array(point_cloud.points)

# 执行自适应划分操作
split_axis, split_value = adaptive_split_point_cloud(np_pcd)

left_cloud, right_cloud = split_point_cloud(np_pcd, split_axis, split_value)



left_pcd = o3d.geometry.PointCloud()
left_pcd.points = o3d.utility.Vector3dVector(left_cloud)
left_pcd.paint_uniform_color([1,0,0])

right_pcd = o3d.geometry.PointCloud()
right_pcd.points = o3d.utility.Vector3dVector(right_cloud)
right_pcd.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([left_pcd,right_pcd])