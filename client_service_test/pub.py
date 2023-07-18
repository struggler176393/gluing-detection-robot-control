import rospy,sys
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


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
            seam_data = np.array([[1,2,3,0,0,1],[4,5,6,0,0,1],[7,8,9,0,0,1]])


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
            msg_seam.point_step = 24
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
    publish_seam()
