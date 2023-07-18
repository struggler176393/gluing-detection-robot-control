import tf
import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from moveit_msgs.msg import  PlanningScene, ObjectColor,CollisionObject, AttachedCollisionObject,Constraints,OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import quaternion_from_euler,euler_from_quaternion, euler_matrix, euler_from_matrix, translation_matrix,concatenate_matrices,quaternion_from_matrix
from copy import deepcopy
import numpy as np
import math
from copy import deepcopy
from tf_utils import get_trans
import open3d as o3d
import cv2
from PIL import Image
from scipy.optimize import least_squares
from time import time




if __name__ =="__main__":


    # xyz = [0.766037,	-0.295099,	1.10878-0.512]
    # rpy = [-2.13701,	-0.142609,	0.261697]
    
    # trans_matrix = translation_matrix(xyz)
    # rot_matrix = euler_matrix(rpy[0],rpy[1],rpy[2],axes='sxyz')
    # # camera2base_pose_matrix = concatenate_matrices(rot_matrix,trans_matrix)  # 与np.dot(trans_matrix, rot_matrix)等价
    # camera2baselink_pose_matrix = concatenate_matrices(trans_matrix,rot_matrix)

    # trans_matrix = translation_matrix([0,0,0])
    # rot_matrix = euler_matrix(0,0,-np.pi,axes='sxyz')
    # # camera2base_pose_matrix = concatenate_matrices(rot_matrix,trans_matrix)  # 与np.dot(trans_matrix, rot_matrix)等价
    # baselink2base_pose_matrix = concatenate_matrices(trans_matrix,rot_matrix)

    # camera2base_pose_matrix = np.dot(baselink2base_pose_matrix, camera2baselink_pose_matrix )






    # xyzrpy = [xyz[0],xyz[1],xyz[2], rpy[0],rpy[1],rpy[2]]

    # print(baselink2base_pose_matrix)
    # # print(xyzrpy)

    # print(euler_from_matrix(camera2base_pose_matrix,axes='sxyz'))
    trans_matrix = translation_matrix([0,0,0])

    rot_matrix = euler_matrix(0,0,25/180*np.pi,axes='sxyz')
    # camera2base_pose_matrix = concatenate_matrices(rot_matrix,trans_matrix)  # 与np.dot(trans_matrix, rot_matrix)等价
    baselink2base_pose_matrix = concatenate_matrices(trans_matrix,rot_matrix)


    print(quaternion_from_matrix(baselink2base_pose_matrix))