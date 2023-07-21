from PIL import Image
import numpy as np
import open3d as o3d
import cv2
import pyransac3d as pyrsc
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import time
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading


cut_flag = False

# # 键盘好像用不了
def on_press(key):
    global key_listener
    global cut_flag

    if key == KeyCode.from_char('m'):
        cut_flag = True
        print("cut!")

def key_thread():
    key_listener = keyboard.Listener(
            on_press=on_press
        )
    key_listener.start()



def generate_pointcloud(color_image, depth_image,width=1280,height=720):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,convert_rgb_to_intensity=False,depth_trunc=100.0)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault )


    intrinsic.set_intrinsics(width=width, height=height, fx=605.8051, fy=605.6255, cx=641.7172, cy=363.2258)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # point_cloud.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    return point_cloud



def depth_to_point_cloud_kinect(depth_img, mask, fx, fy, cx, cy, depth_scale):  

    

    # 将mask为1的像素点的坐标转换为相机坐标系下的三维坐标  
    mask_points = np.where(mask == 1)  # 获取mask为1的像素点坐标  
    depth_values = depth_img[mask_points]  # 获取对应像素点的深度值  
    depth_values = depth_values / depth_scale  # 将深度值转换为真实尺度下的值  
    u_values = mask_points[1]  # 获取对应像素点的u坐标  
    v_values = mask_points[0]  # 获取对应像素点的v坐标  
    x_values = (u_values - cx) * depth_values / fx  # 计算x坐标  
    y_values = (v_values - cy) * depth_values / fy  # 计算y坐标  
    z_values = depth_values  # 深度值就是z坐标  
    
    points_3d = np.vstack((x_values, y_values, z_values)).T  # 组合为nx3的数组 


  
    return points_3d

def project_point_to_line(point, line_origin, line_direction):
    # point: 三维点
    # line_origin: 直线上一点
    # line_direction: 直线的方向向量

    dx, dy, dz = point[0]-line_origin[0], point[1]-line_origin[1], point[2]-line_origin[2]
    t = (dx*line_direction[0] + dy*line_direction[1] + dz*line_direction[2]) / (line_direction[0]**2 + line_direction[1]**2 + line_direction[2]**2)
    return line_origin[0]+t*line_direction[0], line_origin[1]+t*line_direction[1], line_origin[2]+t*line_direction[2]



def get_pointcloud():

    depth_image_o3d = o3d.io.read_image('test_rgbd/1_depth.png')
    mask_img = cv2.imread('test_rgbd/1_mask.png', cv2.IMREAD_GRAYSCALE)

    color_image = cv2.imread('test_rgbd/1_color.png')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image[mask_img==100]=[255,0,0]

    color_image_o3d = o3d.geometry.Image(color_image)
    depth_image = cv2.imread('test_rgbd/1_depth.png', -1)

    point_cloud = generate_pointcloud(color_image=color_image_o3d, depth_image=depth_image_o3d)

    points = np.array(point_cloud.points)
    points_colors = (color_image/255).reshape(-1,3)

    data = np.hstack([points,points_colors])
    return data



def DepthMapToPointMap(file_path):
    # im = Image.open('Data/dp1.tiff')
    im = Image.open(file_path)
    # 将图像转化成numpy数组
    dp = np.array(im)

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
    convert_pm = np.array([undistorted_XY[:, :, 0] * dp,
                            undistorted_XY[:, :, 1] * dp, dp]).reshape((3, -1)).T
    
    return convert_pm



# def get_pointcloud_rvbust():

    color_img = cv2.imread('Data/test1.png')
    color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)
    color_img = color_img.reshape(-1,3)

    # pcd_data = np.loadtxt("Data/point_map1.xyz")
    pcd_data = DepthMapToPointMap('Data/dp1.tiff')

    valid_color_img = color_img[~np.isnan(pcd_data[:, 2])]/255
    valid_pcd_data = pcd_data[~np.isnan(pcd_data[:, 2])]

    data = np.hstack([valid_pcd_data,valid_color_img])
    return data

from camera_capture import color_pointcloud_capture,capture_predict

def get_seam_points_rvbust():
    _, _, color_points = capture_predict()
    
    


    points = color_points[:,:3]
    colors = color_points[:,3:]
    # colors[mask_img==1]=[1,0,0]

    index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    # print(index)

    colors = colors[index]
    points = points[index]
    # print(colors.shape)

    
    valid_color_img = colors[~np.isnan(points[:, 2])]
    valid_pcd_data = points[~np.isnan(points[:, 2])]

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)
    seam_pcd.colors = o3d.utility.Vector3dVector(valid_color_img)
    # o3d.visualization.draw_geometries([seam_pcd])


    line = pyrsc.Line()

    A, B, inliers = line.fit(valid_pcd_data, thresh=0.0001, maxIteration=500)


    box_points = np.array(seam_pcd.get_oriented_bounding_box().get_box_points())
                
    min_index = np.argmin(box_points[:, 0])
    min_xyz = box_points[min_index]
    max_index = np.argmax(box_points[:, 0])
    max_xyz = box_points[max_index]
    min_points = [min_xyz[0], min_xyz[1], min_xyz[2]]
    max_points = [max_xyz[0], max_xyz[1], max_xyz[2]]

    min_align = np.array(project_point_to_line(min_points, B, A))
    max_align = np.array(project_point_to_line(max_points, B, A))


    return min_align,max_align





    # seam_points = depth_to_point_cloud_kinect(de)


# global captured_pointcloud
captured_pointcloud = 0



def pcd_callback(data):
    global captured_pointcloud
    captured_pointcloud = np.frombuffer(data.data, dtype=np.float32)
    captured_pointcloud = captured_pointcloud.reshape(-1,6)



def get_seam_points_subscribe():


    global captured_pointcloud
    rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,pcd_callback)

    points = captured_pointcloud[:,:3]
    colors = captured_pointcloud[:,3:]
    # colors[mask_img==1]=[1,0,0]

    index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    # print(index)

    colors = colors[index]
    points = points[index]
    # print(colors.shape)

    
    valid_color_img = colors[~np.isnan(points[:, 2])]
    valid_pcd_data = points[~np.isnan(points[:, 2])]

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)
    seam_pcd.colors = o3d.utility.Vector3dVector(valid_color_img)
    # o3d.visualization.draw_geometries([seam_pcd])


    line = pyrsc.Line()

    A, B, inliers = line.fit(valid_pcd_data, thresh=0.01, maxIteration=500)


    box_points = np.array(seam_pcd.get_oriented_bounding_box().get_box_points())
                
    min_index = np.argmin(box_points[:, 0])
    min_xyz = box_points[min_index]
    max_index = np.argmax(box_points[:, 0])
    max_xyz = box_points[max_index]
    min_points = [min_xyz[0], min_xyz[1], min_xyz[2]]
    max_points = [max_xyz[0], max_xyz[1], max_xyz[2]]

    min_align = np.array(project_point_to_line(min_points, B, A))
    max_align = np.array(project_point_to_line(max_points, B, A))

    min_align = np.hstack((min_align,[0,0,1]))
    max_align = np.hstack((max_align,[0,0,1]))

    trajectory_points = np.vstack((min_align,max_align))


    return trajectory_points


# 单个点云提取路径点
from seam_point_extract.path_line_gen import cal_gluepath

def subscribe_pcd_cal_trajectory():


    global captured_pointcloud
    rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,pcd_callback)

    points = captured_pointcloud[:,:3]
    colors = captured_pointcloud[:,3:]

    index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    points = points[index]
    valid_pcd_data = points[~np.isnan(points[:, 2])]

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)



    line_points = cal_gluepath(seam_pcd, 5)
    line_points_rgb = np.tile(np.array([0, 0, 1]), (len(line_points), 1))
    trajectory_points = np.hstack((line_points,line_points_rgb))



    return trajectory_points


# 手动切割点云
from seam_point_extract.pcd_cut_save import vis_cut_save,read_pcd_extract_point,read_pcd_extract_point_test
def pcd_cut_trajectory():
    global cut_flag


    global captured_pointcloud
    rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,pcd_callback)

    points = captured_pointcloud[:,:3]
    colors = captured_pointcloud[:,3:]

    index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    points = points[index]
    valid_pcd_data = points[~np.isnan(points[:, 2])]

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)

    folder_path = "seam_point_extract/extract_pcd/"
    if cut_flag:
        vis_cut_save(folder_path, seam_pcd)
        cut_flag = False
    # trajectory_points = read_pcd_extract_point(folder_path)
    trajectory_points = read_pcd_extract_point_test(folder_path)



    # line_points = cal_gluepath(seam_pcd, 5)
    # line_points_rgb = np.tile(np.array([0, 0, 1]), (len(line_points), 1))
    # trajectory_points = np.hstack((line_points,line_points_rgb))



    return trajectory_points




def get_seam_points():
    mask_img = cv2.imread('test_rgbd/1_mask.png', cv2.IMREAD_GRAYSCALE)

    color_image = cv2.imread('test_rgbd/1_color.png')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image[mask_img==100]=[255,0,0]

    depth_image = cv2.imread('test_rgbd/1_depth.png', -1)

    seam_points = depth_to_point_cloud_kinect(depth_image, mask_img/100, fx=605.805115, fy=605.625549, cx=641.717163, cy=363.225800, depth_scale=1000) 

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(seam_points)

    line = pyrsc.Line()

    A, B, inliers = line.fit(seam_points, thresh=0.01, maxIteration=500)


    box_points = np.array(seam_pcd.get_oriented_bounding_box().get_box_points())
                
    min_index = np.argmin(box_points[:, 0])
    min_xyz = box_points[min_index]
    max_index = np.argmax(box_points[:, 0])
    max_xyz = box_points[max_index]
    min_points = [min_xyz[0], min_xyz[1], min_xyz[2]]
    max_points = [max_xyz[0], max_xyz[1], max_xyz[2]]

    min_align = np.array(project_point_to_line(min_points, B, A))
    max_align = np.array(project_point_to_line(max_points, B, A))
    return min_align,max_align

    


# def get_pointcloud_111():

    depth_image_o3d = o3d.io.read_image('test_rgbd/1_depth.png')
    mask_img = cv2.imread('test_rgbd/1_mask.png', cv2.IMREAD_GRAYSCALE)

    color_image = cv2.imread('test_rgbd/1_color.png')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image[mask_img==100]=[255,0,0]

    color_image_o3d = o3d.geometry.Image(color_image)
    depth_image = cv2.imread('test_rgbd/1_depth.png', -1)

    point_cloud = generate_pointcloud(color_image=color_image_o3d, depth_image=depth_image_o3d)

    points = np.array(point_cloud.points)
    points_colors = (color_image/255).reshape(-1,3)

    data = np.hstack([points,points_colors])


    seam_points = depth_to_point_cloud_kinect(depth_image, mask_img/100, fx=605.805115, fy=605.625549, cx=641.717163, cy=363.225800, depth_scale=1000)  
    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(seam_points)

    line = pyrsc.Line()

    A, B, inliers = line.fit(seam_points, thresh=0.01, maxIteration=500)


    box_points = np.array(seam_pcd.get_oriented_bounding_box().get_box_points())
                
    min_index = np.argmin(box_points[:, 0])
    min_xyz = box_points[min_index]
    max_index = np.argmax(box_points[:, 0])
    max_xyz = box_points[max_index]
    min_points = [min_xyz[0], min_xyz[1], min_xyz[2]]
    max_points = [max_xyz[0], max_xyz[1], max_xyz[2]]

    min_align = np.array(project_point_to_line(min_points, B, A))
    max_align = np.array(project_point_to_line(max_points, B, A))

    print(min_align)
    print(max_align)

    # R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], A)


    # mesh_sphere_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    # mesh_sphere_1.paint_uniform_color([0.1, 0.1, 0.7])
    # mesh_sphere_1 = mesh_sphere_1.translate(min_align)
    # mesh_sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    # mesh_sphere_2.paint_uniform_color([0.1, 0.1, 0.7])
    # mesh_sphere_2 = mesh_sphere_2.translate(max_align)

    # o3d.visualization.draw_geometries([mesh_sphere_1,mesh_sphere_2,seam_pcd])
    o3d.visualization.draw_geometries([point_cloud])

    



    return data, min_align, max_align
# print(data.shape)



def publish_seam():
    while not rospy.is_shutdown():
        try:
            pub_seam = rospy.Publisher('/seam_points', PointCloud2, queue_size=1)
            


            # data, min_align, max_align = get_pointcloud()
            # data = get_pointcloud()
            # print(data.shape)
            # min_align,max_align = get_seam_points()
            # min_align,max_align = get_seam_points_rvbust()



            # seam_data = get_seam_points_subscribe()
            # seam_data = subscribe_pcd_cal_trajectory()
            seam_data = pcd_cut_trajectory()


            print(seam_data)

        
            
            
            msg_seam = PointCloud2()
            msg_seam.header.stamp = rospy.Time.now()
            msg_seam.header.frame_id = "camera_base"
            msg_seam.height = 1

            
            msg_seam.width = len(seam_data)
            msg_seam.is_dense = True

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
            ]
            msg_seam.fields = fields

            msg_seam.is_bigendian = False
            msg_seam.point_step = 12
            msg_seam.row_step = msg_seam.point_step * seam_data.shape[0]

            # 将点和颜色信息填充到消息中
            

            msg_seam.data = np.asarray(seam_data, dtype=np.float32).tobytes()

            # 发布消息
            pub_seam.publish(msg_seam)


            # print("published...")
            rospy.sleep(1.0)
        except:
            continue

if __name__ =="__main__":
    # get_pointcloud()
    rospy.init_node('seam_publisher', anonymous=False)

    add_thread1 = threading.Thread(target = key_thread)
    add_thread1.start()

    add_thread2 = threading.Thread(target = publish_seam)
    add_thread2.start()