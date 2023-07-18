#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入基本ros和moveit库
import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from moveit_msgs.msg import  PlanningScene, ObjectColor,CollisionObject, AttachedCollisionObject,Constraints,OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import quaternion_from_euler,euler_from_quaternion, euler_matrix, euler_from_matrix, translation_matrix,concatenate_matrices
from copy import deepcopy
import numpy as np
import math
from copy import deepcopy
from tf_utils import get_trans
# import pointcloud_calibration




class MoveIt_Control:
    # 初始化程序
    def __init__(self):
        # Init ros config
        moveit_commander.roscpp_initialize(sys.argv)

        # 初始化ROS节点
        
        self.arm = moveit_commander.MoveGroupCommander('arm')
        self.arm.set_goal_joint_tolerance(0.001)
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)

        self.end_effector_link = self.arm.get_end_effector_link()
        # 设置机械臂基座的参考系
        self.reference_frame = 'base'
        self.arm.set_pose_reference_frame(self.reference_frame)

        # 设置最大规划时间和是否允许重新规划
        self.arm.set_planning_time(5)
        self.arm.allow_replanning(True)
        self.arm.set_planner_id("RRTConnect")

        # 设置允许的最大速度和加速度（范围：0~1）
        self.arm.set_max_acceleration_scaling_factor(1)
        self.arm.set_max_velocity_scaling_factor(1)

        # 手眼标定结果存储
        self.calibration()

        # # 机械臂初始姿态
        # self.go_home()
        self.prepare()

        rospy.sleep(1.5)
        # 发布场景
        self.set_scene()  # set table
        #self.arm.set_workspace([-2,-2,0,2,2,2])  #[minx miny minz maxx maxy maxz]
        # # 测试专用
        # self.testRobot()

        self.glue_operation()
        # 抓取服务端，负责接收抓取位姿并执行运动
        # server = rospy.Service("moveit_grasp",grasp_pose,self.grasp_callback )
        # rospy.spin()

    def glue_operation(self):
        global object_detected
        print(object_detected)
        while object_detected:
            print("Test for robot...")

            target_xyz = self.camera_detect()
            target_pos = []
            for xyz in target_xyz:
                target_pos.append([xyz[0],xyz[1],xyz[2], np.pi, -np.pi/4, np.pi])

            target_pos = list(np.array(target_pos).flatten())
            self.move_l(target_pos, waypoints_number=len(target_xyz), speed_scale = 1)
            rospy.sleep(2)

            object_detected = False


    def calibration(self):
        xyz = [-0.5,
               0,
               0.3]
        rpy = [0,
               0,
               0]
        
        trans_matrix = translation_matrix(xyz)
        rot_matrix = euler_matrix(rpy[0],rpy[1],rpy[2])
        self.camera2base_pose_matrix = concatenate_matrices(trans_matrix,rot_matrix)  # 与np.dot(trans_matrix, qua_matrix)等价

        xyzrpy = [xyz[0],xyz[1],xyz[2], rpy[0],rpy[1],rpy[2]]

        print(self.camera2base_pose_matrix)
        print(xyzrpy)

        return self.camera2base_pose_matrix, xyzrpy
    


    def close(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


    def scale_trajectory_speed(self, plan,scale):
        n_joints = len(plan.joint_trajectory.joint_names)
        n_points = len(plan.joint_trajectory.points)
        # print(n_joints)
        # print(n_points)
        for i in range(n_points):
            plan.joint_trajectory.points[i].time_from_start *= 1/scale
            plan.joint_trajectory.points[i].velocities = list(np.array(plan.joint_trajectory.points[i].velocities)*scale)
            plan.joint_trajectory.points[i].velocities = list(np.array(plan.joint_trajectory.points[i].accelerations)*scale*scale)
        return plan
    
    def camera_detect(self):
        xyz=[]
        target_xyz = []
        xyz.append([-0.2,0.2,0.001])
        xyz.append([-0.2,-0.2,0.001])

        for pos in xyz:
            trans_matrix = translation_matrix(pos)
            object_matrix = np.dot(self.camera2base_pose_matrix, trans_matrix)
            target_xyz.append(object_matrix[:3,3])
            print(object_matrix)

        print(target_xyz)
        return target_xyz


                
    # 测试程序用
    def testRobot(self):
        try:
            print("Test for robot...")

            target_xyz = self.camera_detect()
            target_pos = []
            for xyz in target_xyz:
                target_pos.append([xyz[0],xyz[1],xyz[2], np.pi, -np.pi/4, np.pi])

            target_pos = list(np.array(target_pos).flatten())
            self.move_l(target_pos, waypoints_number=len(target_xyz), speed_scale = 1)
            rospy.sleep(2)
            




            # self.move_l([-0.7, 0.2, 0.301, np.pi, -np.pi/4, np.pi,
            #              -0.7, -0.2, 0.301, np.pi, -np.pi/4, np.pi,],waypoints_number=2,a=0.005,v=0.005)
            # rospy.sleep(2)


        except:
            print("Test fail! ")

    # 在机械臂下方添加一个table，使得机械臂只能够在上半空间进行规划和运动
    # 避免碰撞到下方的桌子等其他物体
    def set_scene(self):
        ## set table
        self.scene = PlanningSceneInterface()
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        self.colors = dict()
        rospy.sleep(1)
        ground_id = 'ground'
        self.scene.remove_world_object(ground_id)
        rospy.sleep(1)
        ground_size = [2, 2, 0.01]
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = 'world'
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        ground_pose.pose.position.z = -ground_size[2]/2
        ground_pose.pose.orientation.w = 1.0
        self.scene.add_box(ground_id, ground_pose, ground_size)
        self.setColor(ground_id, 0.5, 0.5, 0.5, 1.0)
        self.sendColors()

        base_table_id = 'ground'
        self.scene.remove_world_object(base_table_id)
        rospy.sleep(1)
        base_table_size = [0.5, 0.5, 0.5]
        base_table_pose = PoseStamped()
        base_table_pose.header.frame_id = 'world'
        base_table_pose.pose.position.x = 0.0
        base_table_pose.pose.position.y = 0.0
        base_table_pose.pose.position.z = base_table_size[2]/2
        base_table_pose.pose.orientation.w = 1.0
        self.scene.add_box(base_table_id, base_table_pose, base_table_size)
        self.setColor(base_table_id, 1.0, 0.5, 0.5, 1.0)
        self.sendColors()

        desk_id = 'desk'
        self.scene.remove_world_object(desk_id)
        rospy.sleep(1)
        desk_size = [0.4, 0.8, 0.8]
        desk_pose = PoseStamped()
        desk_pose.header.frame_id = 'world'
        desk_pose.pose.position.x = 0.6+desk_size[0]/2
        desk_pose.pose.position.y = 0.0
        desk_pose.pose.position.z = desk_size[2]/2
        desk_pose.pose.orientation.w = 1.0
        self.scene.add_box(desk_id, desk_pose, desk_size)
        self.setColor(desk_id, 0.5, 0.5, 1.0, 1.0)
        self.sendColors()


    # 关节规划，输入6个关节角度（单位：弧度）
    def move_j(self, joint_configuration=None,a=1,v=1):
        # 设置机械臂的目标位置，使用六轴的位置数据进行描述（单位：弧度）
        if joint_configuration==None:
            joint_configuration = [0, -1.5707, 0, -1.5707, 0, 0]
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)
        self.arm.set_joint_value_target(joint_configuration)
        rospy.loginfo("move_j:"+str(joint_configuration))
        self.arm.go()
        rospy.sleep(1)

    # 空间规划，输入xyzRPY
    def move_p(self, tool_configuration=None,a=1,v=1):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = tool_configuration[0]
        target_pose.pose.position.y = tool_configuration[1]
        target_pose.pose.position.z = tool_configuration[2]
        q = quaternion_from_euler(tool_configuration[3],tool_configuration[4],tool_configuration[5])
        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]

        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        rospy.loginfo("move_p:" + str(tool_configuration))
        traj = self.arm.plan()
        self.arm.execute(traj)
        rospy.sleep(1)
    


    # 空间直线运动，输入(x,y,z,R,P,Y,x2,y2,z2,R2,...)
    # 默认仅执行一个点位，可以选择传入多个点位
    def move_l(self, tool_configuration, waypoints_number=1, speed_scale=1):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]

        # 设置路点
        waypoints = []
        for i in range(waypoints_number):
            target_pose = PoseStamped()
            target_pose.header.frame_id = self.reference_frame
            target_pose.header.stamp = rospy.Time.now()
            target_pose.pose.position.x = tool_configuration[6*i+0]
            target_pose.pose.position.y = tool_configuration[6*i+1]
            target_pose.pose.position.z = tool_configuration[6*i+2]
            q = quaternion_from_euler(tool_configuration[6*i+3],tool_configuration[6*i+4],tool_configuration[6*i+5])
            target_pose.pose.orientation.x = q[0]
            target_pose.pose.orientation.y = q[1]
            target_pose.pose.orientation.z = q[2]
            target_pose.pose.orientation.w = q[3]
            waypoints.append(target_pose.pose)
        rospy.loginfo("move_l:" + str(tool_configuration))
        self.arm.set_start_state_to_current_state()
        fraction = 0.0  # 路径规划覆盖率
        maxtries = 100  # 最大尝试规划次数
        attempts = 0  # 已经尝试规划次数

        # 设置机器臂当前的状态作为运动初始状态
        self.arm.set_start_state_to_current_state()

        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,  # waypoint poses，路点列表
                0.001,  # eef_step，终端步进值
                0.00,  # jump_threshold，跳跃阈值
                True)  # avoid_collisions，避障规划
            attempts += 1
            # print(fraction)
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            
            plan = self.scale_trajectory_speed(plan,speed_scale)

            self.arm.execute(plan)
            rospy.loginfo("Path execution complete.")
        else:
            rospy.loginfo(
                "Path planning failed with only " + str(fraction) +
                " success after " + str(maxtries) + " attempts.")
        rospy.sleep(1)

    def go_home(self,a=1,v=1):
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)
        # “up”为自定义姿态，你可以使用“home”或者其他姿态
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(1)
    
    def prepare(self,a=1,v=1):
        self.move_j([-3.1151271468491033, -2.164855869763364, -0.6071655712046987, -0.5104477333719127, 1.5312004086669155, 0.20760004604457102],a=0.1,v=0.1)

    def setColor(self, name, r, g, b, a=0.9):
        # 初始化moveit颜色对象
        color = ObjectColor()
        # 设置颜色值
        color.id = name
        color.color.r = r
        color.color.g = g
        color.color.b = b
        color.color.a = a
        # 更新颜色字典
        self.colors[name] = color

    # 将颜色设置发送并应用到moveit场景当中
    def sendColors(self):
        # 初始化规划场景对象
        p = PlanningScene()
        # 需要设置规划场景是否有差异
        p.is_diff = True
        # 从颜色字典中取出颜色设置
        for color in self.colors.values():
            p.object_colors.append(color)
        # 发布场景物体颜色设置
        self.scene_pub.publish(p)
    


    def some_useful_function_you_may_use(self):
        # return the robot current pose
        current_pose = self.arm.get_current_pose()
        # rospy.loginfo('current_pose:',current_pose)
        # return the robot current joints
        current_joints = self.arm.get_current_joint_values()
        # rospy.loginfo('current_joints:',current_joints)

        #self.arm.set_planner_id("RRTConnect")
        self.arm.set_planner_id("TRRT")
        plannerId = self.arm.get_planner_id()
        rospy.loginfo(plannerId)

        planning_frame = self.arm.get_planning_frame()
        rospy.loginfo(planning_frame)

        # stop the robot
        self.arm.stop()
  

def glue_moveit():
    moveit_server = MoveIt_Control()


if __name__ =="__main__":
    rospy.init_node('moveit_control_server', anonymous=False)
    moveit_server = MoveIt_Control()
    

