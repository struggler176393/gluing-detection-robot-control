from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
import rospy, sys
import moveit_commander
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, translation_matrix,concatenate_matrices
import math
import tf
import numpy as np
from geometry_msgs.msg import Vector3, Quaternion, Transform
import tf2_ros
import geometry_msgs.msg
from threading import Thread, current_thread
# global calibration_xyzrpy
# calibration_xyzrpy = [-0.5, 0, 0.52, 0, 0, 0]



def get_trans(base_frame,end_frame):
    listener = tf.TransformListener()
    listener.waitForTransform(base_frame,end_frame,  rospy.Time(), rospy.Duration(5))
    # 这里要记住：要使用rospy.Time()，这样会从tf缓存中一个个读取，而不能用rospy.Time.now()，这样会从当前时间读取tf，要等很久
    t = listener.getLatestCommonTime(base_frame,end_frame)
    position, quaternion = listener.lookupTransform(base_frame,end_frame, t)
    
    trans_matrix = translation_matrix(position)
    qua_matrix = quaternion_matrix(quaternion)
    pose_matrix = concatenate_matrices(trans_matrix,qua_matrix)  # 与np.dot(trans_matrix, qua_matrix)等价

    rx,ry,rz = euler_from_quaternion(quaternion)
    xyzrpy = [position[0],position[1],position[2], rx,ry,rz]


    print(pose_matrix)
    print([position[0],position[1],position[2], rx,ry,rz])

    return pose_matrix, xyzrpy

def get_endpos():

    moveit_commander.roscpp_initialize(sys.argv)
    arm = moveit_commander.MoveGroupCommander('arm')

    current_pose = arm.get_current_pose()
    # print(current_pose)
    current_joints = arm.get_current_joint_values()
    print(current_joints)


    x = current_pose.pose.position.x*1000
    y = current_pose.pose.position.y*1000
    z = current_pose.pose.position.z*1000

    qx = current_pose.pose.orientation.x
    qy = current_pose.pose.orientation.y
    qz = current_pose.pose.orientation.z
    qw = current_pose.pose.orientation.w

    rx,ry,rz = euler_from_quaternion([qx,qy,qz,qw])
    xyzrpy = [x,y,z,rx,ry,rz]
    print(xyzrpy)
    return xyzrpy


def publish_tf(parent_frame,child_frame,xyzrpy):
    # rospy.init_node(parent_frame+"2"+child_frame+"_publisher")
    while rospy.get_time() == 0.0:
        pass

    x,y,z,rx,ry,rz = xyzrpy
    [qx,qy,qz,qw] = tf.transformations.quaternion_from_euler(rx,ry,rz)

    trans = Transform(Vector3(x,y,z), Quaternion(qx,qy,qz,qw))

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform = trans

    print(static_transformStamped)

    broadcaster.sendTransform(static_transformStamped)
    rospy.spin()



def publish_calibration_tf(parent_frame,child_frame):
    # rospy.init_node(parent_frame+"2"+child_frame+"_publisher")
    global calibration_xyzrpy
    while rospy.get_time() == 0.0:
        pass

    x,y,z,rx,ry,rz = calibration_xyzrpy
    [qx,qy,qz,qw] = tf.transformations.quaternion_from_euler(rx,ry,rz)

    trans = Transform(Vector3(x,y,z), Quaternion(qx,qy,qz,qw))

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform = trans

    print(static_transformStamped)

    broadcaster.sendTransform(static_transformStamped)
    rospy.spin()




if __name__ =="__main__":
    rospy.init_node('moveit_control_server', anonymous=False)
    # get_endpos()
    # get_trans("base","base_link")


    calibration_xyzrpy = [-0.5, 0, 0.52, 0, 0, 0]
    # publish_tf("base","camera_base",xyzrpy)


    # # 创建线程
    # thread01 = Thread(target=publish_tf, args=["base","camera_base",xyzrpy], name="线程1")
    # thread02 = Thread(target=publish_tf, args=["camera_base","object",xyzrpy], name="线程2")
    # thread01.start() 
    # thread02.start() 
    # print(333)