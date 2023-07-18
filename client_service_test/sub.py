import rospy,sys
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import time


a = []
def callback(data):
    points = np.frombuffer(data.data, dtype=np.float32)
    print(points)
    time.sleep(2)
    # rospy.spin()


if __name__ =="__main__":
    rospy.init_node('test', anonymous=False)
    while True:
        rospy.Subscriber('/seam_points', PointCloud2,callback)
    