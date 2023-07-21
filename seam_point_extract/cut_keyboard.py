from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import rospy,sys


global cut_flag
cut_flag = False


from seam_point_extract.pcd_cut_save import vis_cut_save
def cut_callback(data):
    captured_pointcloud = np.frombuffer(data.data, dtype=np.float32)
    captured_pointcloud = captured_pointcloud.reshape(-1,6)
    points = captured_pointcloud[:,:3]
    colors = captured_pointcloud[:,3:]

    index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
    points = points[index]
    valid_pcd_data = points[~np.isnan(points[:, 2])]

    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)
    folder_path = "seam_point_extract/extract_pcd/"
    vis_cut_save(folder_path, seam_pcd)



# def cut_pcd():
#     global cut_flag
#     if cut_flag:
#         rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,cut_callback)
#     cut_flag = False


def on_press(key):
    global key_listener
    global cut_flag

    if key == KeyCode.from_char('m'):
        # cut_flag = True
        rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,cut_callback)
        print("cut!")
        # cut_pcd()


# def thread_job1():
#     key_listener = keyboard.Listener(
#             on_press=on_press
#         )
#     key_listener.start()





if __name__ =="__main__":
    rospy.init_node('cut_pcd', anonymous=False)
    key_listener = keyboard.Listener(
            on_press=on_press
        )
    key_listener.start()
    

