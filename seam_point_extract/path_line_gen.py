import open3d as o3d
import numpy as np
import math
import copy


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) + math.pow((p2[2] - p1[2]), 2))


def pathpoint(insertp, start, end):
    c = insertp + 1
    path = []
    path.append(start)
    for i in range(1, c):
        x = start[0] + (end[0] - start[0]) * (i / c)
        y = start[1] + (end[1] - start[1]) * (i / c)
        z = start[2] + (end[2] - start[2]) * (i / c)
        point = [x, y, z]
        path.append(np.asarray(point))
    path.append(end)
    return path


def plane_param(point_1, point_2, point_3):
    """
    不共线的三个点确定一个平面
    :param point_1: 点1
    :param point_2: 点2
    :param point_3: 点3
    :return: 平面方程系数:a,b,c,d
    """
    p1p2 = point_2 - point_1
    p1p3 = point_3 - point_1
    n = np.cross(p1p2, p1p3)  # 计算法向量
    n1 = n / np.linalg.norm(n)  # 单位法向量
    A = n1[0]
    B = n1[1]
    C = n1[2]
    D = -A * point_1[0] - B * point_1[1] - C * point_1[2]
    return A, B, C, D




# 没有输入两个端点
def cal_gluepath(pointcloud, insertpoint):
    '''
    :param pointcloud: 输入读取后的点云
    :param insertpoint: 除起始点和终止点，路径中间插入点的个数
    :return: 路径点（array格式）  个数：insertpoint+2
    '''

    aabb = pointcloud.get_oriented_bounding_box()
    aabb.color = (1, 0, 0)
    point = np.asarray(pointcloud.points)
    bbpoints = np.asarray(aabb.get_box_points())
    # print(bbpoints)

    # TODO 包围盒点需要根据情况修改提取条件，包围盒点输出格式是无序的
    inx = bbpoints[:, 0] < aabb.get_center()[0]
    # TODO !!!!!
    rinx = np.ones(len(bbpoints), bool)
    rinx[inx] = False

    bbx1 = o3d.geometry.PointCloud()
    bbx1.points = o3d.utility.Vector3dVector(bbpoints[inx])
    start = bbx1.get_center()
    bbx2 = o3d.geometry.PointCloud()
    bbx2.points = o3d.utility.Vector3dVector(bbpoints[rinx])
    end = bbx2.get_center()

    cutline = pathpoint(insertpoint, start, end)
    # print(len(pathline))
    path1 = o3d.geometry.PointCloud()
    path1.points = o3d.utility.Vector3dVector(cutline)

    cut_plane = []
    cut_plane.append(bbpoints[inx])
    pl = bbx1
    for i in range(insertpoint):
        pl = copy.deepcopy(pl).translate((cutline[i + 1] - cutline[i]))
        cut_plane.append(np.asarray(pl.points))
        # print(np.asarray(pl.points))
    cut_plane.append(bbpoints[rinx])

    line_points = []
    line_points.append(np.asarray(start))
    for i in range(len(cut_plane)):
        plane = cut_plane
        P1 = plane[i][0]
        P2 = plane[i][1]
        P3 = plane[i][2]
        # 2.计算P1,P2,P3三点确定的平面，以此作为切片
        a, b, c, d = plane_param(P1, P2, P3)

        point_size = point.shape[0]
        idx = []
        # 3.设置切片厚度阈值，此值为切片厚度的一半
        Delta = cal_distance(cutline[0], cutline[1])
        # 4.循环迭代查找满足切片的点
        for i in range(point_size):
            Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
            Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
            if Wr * Wl <= 0:
                idx.append(i)
        # 5.提取切片点云
        slicing_cloud = (pointcloud.select_by_index(idx))
        line_points.append(slicing_cloud.get_center())
        # slicing_cloud.paint_uniform_color((1, 0, 0))
    line_points.append(np.asarray(end))

    return np.array(line_points)


if __name__ == '__main__':
    pt1 = o3d.io.read_point_cloud("seam_point_extract/yuefan_code/1.ply")
    pt2 = o3d.io.read_point_cloud('seam_point_extract/yuefan_code/2.ply')
    # aabb = pt.get_oriented_bounding_box()
    # aabb.color = (1, 0, 0)
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # print(aabb.get_center()[0])
    # print(np.asarray(aabb.get_box_points()))

    line_points1 = cal_gluepath(pt1, 2)
    path1 = o3d.geometry.PointCloud()
    path1.points = o3d.utility.Vector3dVector(np.asarray(line_points1))

    line_points2 = cal_gluepath(pt2, 15)
    path2 = o3d.geometry.PointCloud()
    path2.points = o3d.utility.Vector3dVector(np.asarray(line_points2))

    print(line_points1,line_points2)

    o3d.visualization.draw_geometries_with_editing([pt2])
