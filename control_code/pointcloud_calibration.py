# from tf_utils import publish_calibration_tf

import rospy,sys
from pynput import keyboard
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, translation_matrix,concatenate_matrices
from geometry_msgs.msg import Vector3, Quaternion, Transform
import geometry_msgs.msg
import tf
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading
from tf_utils import get_trans
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from pointcloud_reconstruction import get_pointcloud
from moveitServer import glue_moveit
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from tf.transformations import quaternion_from_euler,euler_from_quaternion, euler_matrix, euler_from_matrix, translation_matrix,concatenate_matrices
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import  PlanningScene, ObjectColor,CollisionObject, AttachedCollisionObject,Constraints,OrientationConstraint
from camera_capture import color_pointcloud_capture,capture_predict
import open3d as o3d
# import gol

# gol._init()#先必须在主模块初始化（只在Main模块需要一次即可）

# global min_align
# global max_align
# min_align = [0,0,0]
# max_align = [0,0,0]



trajectory_points = []

# global cut_flag
# cut_flag = False

global object_detected
object_detected = False

global capture_flag
capture_flag = False
# gol.set_value('object_detected',object_detected)

global calibration_xyzrpy
# calibration_xyzrpy = [0.766037,	-0.295099,	1.10878-0.512,	-2.13701,	-0.142609,	0.261697]  # base_link
# calibration_xyzrpy = [-0.766037,	0.295099,	1.10878-0.512,	-2.13701,	-0.142609,	0.261697-np.pi]  # base

# calibration_xyzrpy = [-0.795037,   0.265924,   1.0883-0.512,   -2.12718,   -0.131423,   0.268786-np.pi]
# calibration_xyzrpy = [-0.751587, 0.175951, 1.15511-0.512, -2.13151, -0.152176, 0.243722-np.pi]
calibration_xyzrpy = [-0.77723, 0.420131, 1.15564-0.512, -2.13734, -0.156256, 0.260625-np.pi]
publish_flag = True


def callback(data):
    points = np.frombuffer(data.data, dtype=np.float32)
    # print(points.shape)
    points = points.reshape(-1,3)

    global trajectory_points
    xyz = []

    for p in points:
        xyz.append(list(p[:3]))

    trajectory_points = xyz



    # global min_align
    # global max_align
    # min_align = points[0:3]
    # max_align = points[6:9]




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

        

        # # 机械臂初始姿态
        # self.go_home()
        self.prepare(a=0.2,v=0.2)

        rospy.sleep(1.5)
        # 发布场景
        self.set_scene()  # set table
        #self.arm.set_workspace([-2,-2,0,2,2,2])  #[minx miny minz maxx maxy maxz]
        # # 测试专用
        # self.testRobot()
        # self.move_l([-0.7, 0.2, 0.301, np.pi, -np.pi/4, np.pi,
        #                  -0.7, -0.2, 0.301, np.pi, -np.pi/4, np.pi,],waypoints_number=2)
        
        # self.move_l([-0.7, 0.3, 0.4, np.pi/6, np.pi, 0,
        #              -0.7, -0.1, 0.4, np.pi/6, np.pi, 0,],waypoints_number=2)


        # self.prepare(a=0.2,v=0.2)
        # rospy.sleep(2)

        # self.move_l([-0.51750433, -0.2190545 ,  0.25689765, np.pi/6, np.pi, 0,]



        while True:
            self.glue_operation()




        # 抓取服务端，负责接收抓取位姿并执行运动
        # server = rospy.Service("moveit_grasp",grasp_pose,self.grasp_callback )
        # rospy.spin()

    def glue_operation(self):
        global object_detected
        # print(object_detected)
        while object_detected:
            print("Test for robot...")
            try:
                # 手眼标定结果存储
                self.calibration()
                print(1)
                all_target_xyz = self.camera_detect()
                print(2)
                target_pos = []
                for target_xyz in all_target_xyz:
                    for xyz in target_xyz:
                        xyz[2] = xyz[2]

                        target_pos.append([xyz[0],xyz[1],xyz[2], np.pi/6, np.pi, 0])
                        # target_pos.append([xyz[0],xyz[1],xyz[2], 0, np.pi, 0])
                    
                    
                    # target_pos = target_pos[0]


                    target_pos = list(np.array(target_pos,dtype=object).flatten())
                    print(target_pos)
                    print(len(target_xyz))

                    self.move_l(target_pos[:6], waypoints_number=1, speed_scale = 0.5)
                    self.move_l(target_pos[6:], waypoints_number=len(target_xyz)-1, speed_scale = 0.05)

                    end_pos = target_pos[-6:]
                    end_pos[2] = end_pos[2]+0.05
                    self.move_l(end_pos, waypoints_number=1, speed_scale = 0.5)
                    rospy.sleep(2)
                    self.prepare(a=0.5,v=0.5)
                    rospy.sleep(1.5)
                    target_pos = []

                object_detected = False
                
            except:
                object_detected = False
                
                continue


    def calibration(self):
        global calibration_xyzrpy
        xyz = calibration_xyzrpy[:3]
        rpy = calibration_xyzrpy[3:]
        # xyz = [-0.5,
        #        0,
        #        0.3]
        # rpy = [0,
        #        0,
        #        0]
        
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
            plan.joint_trajectory.points[i].accelerations = list(np.array(plan.joint_trajectory.points[i].accelerations)*scale*scale)
        return plan
    
    def camera_detect(self):
        # global min_align
        # global max_align
        rospy.Subscriber('/seam_points', PointCloud2,callback)
        # print(444)

        global trajectory_points
        
        

        # xyz=[]
        target_xyz_part = []
        target_xyz = []
        # xyz.append([-0.2,0.2,0.001])
        # xyz.append([-0.2,-0.2,0.001])


        # xyz.append(min_align)
        # xyz.append(max_align)
        # print(trajectory_points)

        for pos in trajectory_points:
            if pos[0]!=0:
                trans_matrix = translation_matrix(pos)
                object_matrix = np.dot(self.camera2base_pose_matrix, trans_matrix)
                target_xyz_part.append(object_matrix[:3,3])
            else:
                target_xyz.append(np.array(target_xyz_part))
                target_xyz_part = []
            # print(object_matrix)
        
        

        # print(target_xyz)
        
        trajectory_points = []
        return target_xyz


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
        ground_pose.pose.orientation.z = 0.21643961
        ground_pose.pose.orientation.w = 0.97629601
        self.scene.add_box(ground_id, ground_pose, ground_size)
        self.setColor(ground_id, 0.5, 0.5, 0.5, 1.0)
        self.sendColors()

        base_table_id = 'base_table'
        self.scene.remove_world_object(base_table_id)
        rospy.sleep(1)
        base_table_size = [0.5, 0.5, 0.512]
        base_table_pose = PoseStamped()
        base_table_pose.header.frame_id = 'world'
        base_table_pose.pose.position.x = 0.0
        base_table_pose.pose.position.y = 0.0
        base_table_pose.pose.position.z = base_table_size[2]/2
        base_table_pose.pose.orientation.z = 0.21643961
        base_table_pose.pose.orientation.w = 0.97629601
        self.scene.add_box(base_table_id, base_table_pose, base_table_size)
        self.setColor(base_table_id, 1.0, 0.5, 0.5, 1.0)
        self.sendColors()

        desk_id = 'desk'
        self.scene.remove_world_object(desk_id)
        rospy.sleep(1)
        desk_size = [0.8, 1.8, 0.760+0.044]  # 桌板高：0.044 亚克力：0.003
        desk_pose = PoseStamped()
        desk_pose.header.frame_id = 'world'
        desk_pose.pose.position.x = 0.3+desk_size[0]/2
        desk_pose.pose.position.y = 0.0
        desk_pose.pose.position.z = desk_size[2]/2
        desk_pose.pose.orientation.z = 0.21643961
        desk_pose.pose.orientation.w = 0.97629601
        self.scene.add_box(desk_id, desk_pose, desk_size)
        self.setColor(desk_id, 0.5, 0.5, 1.0, 1.0)
        self.sendColors()



        wall_id = 'wall'
        self.scene.remove_world_object(wall_id)
        rospy.sleep(1)
        wall_size = [0.01, 2, 2]
        wall_pose = PoseStamped()
        wall_pose.header.frame_id = 'world'
        wall_pose.pose.position.x = -0.3
        wall_pose.pose.position.y = 0.0
        wall_pose.pose.position.z = wall_size[2]/2
        wall_pose.pose.orientation.z = 0.21643961
        wall_pose.pose.orientation.w = 0.97629601
        self.scene.add_box(wall_id, wall_pose, wall_size)
        self.setColor(wall_id, 0.5, 1, 0.5, 1.0)
        self.sendColors()

        camera_id = 'camera'
        self.scene.remove_world_object(camera_id)
        rospy.sleep(1)
        camera_size = [0.2, 0.2, 1.5]
        camera_pose = PoseStamped()
        camera_pose.header.frame_id = 'world'
        camera_pose.pose.position.x = 0.766037
        camera_pose.pose.position.y = -0.295099
        camera_pose.pose.position.z = camera_size[2]/2
        camera_pose.pose.orientation.z = 0.21643961
        camera_pose.pose.orientation.w = 0.97629601
        self.scene.add_box(camera_id, camera_pose, camera_size)
        self.setColor(camera_id, 1, 1, 1, 0.5)
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
        self.move_j([-0.12174417293871631+25/180*np.pi, -1.548835405419073, 1.0568126924397783, -2.693364369465602, -2.956528061980836, -1.6631575702179635],a=a,v=v)

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
  

def glue_moveit():
    moveit_server = MoveIt_Control()



def publish_calibration_tf(parent_frame,child_frame):
    # rospy.init_node(parent_frame+"2"+child_frame+"_publisher")
    global publish_flag
    while publish_flag:
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

        # print(static_transformStamped)
        # print(calibration_xyzrpy)

        # print(euler_from_quaternion([trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w]))
        # print([trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w])

        broadcaster.sendTransform(static_transformStamped)
    # rospy.spin()





def on_press(key):
    global calibration_xyzrpy
    global key_listener
    global publish_flag
    global object_detected
    global capture_flag

    # dir_keys = ['q','a',  # x
    #             'w','s',  # y
    #             'e','d',  # z
    #             'r','f',  # rx
    #             't','g',  # ry
    #             'y','h',  # rz
    #             ]
    # pos_name = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    # for i in range(6):
    #     if key == KeyCode.from_char(dir_keys[i*2]):
    #         calibration_xyzrpy[i] += 0.01
    #         print(pos_name[i]+'+')
    #         print(calibration_xyzrpy)

    #     if key == KeyCode.from_char(dir_keys[i*2+1]):
    #         calibration_xyzrpy[i] -= 0.01
    #         print(pos_name[i]+'-')
    #         print(calibration_xyzrpy)
    
    if key == Key.tab:
        object_detected = True
        print("glue!")


    if key == Key.space:
        capture_flag = True
        print("capture!")


    # global cut_flag

    # if key == KeyCode.from_char('m'):
    #     cut_flag = True
    #     print("cut!")


    # if key == Key.esc:
    #     print('stop')
    #     key_listener.stop()
    #     publish_flag = False
    #     print(calibration_xyzrpy)





def publish_colored_point_cloud1111():
    pub = rospy.Publisher('/colored_point_cloud', PointCloud2, queue_size=1)
    points = np.array([[1.0, 0, 0], [0, 0, 1.0]])  # 点的位置
    colors = np.array([[1, 0, 0], [0, 0, 1]]).astype(np.float32)  # 点的颜色
    assert len(points) == len(colors), "Points and colors must have the same length"

    data = get_pointcloud()
    print(data)
    while not rospy.is_shutdown():
        # 创建PointCloud2消息
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_base"
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * points.shape[0]



        # 将点和颜色信息填充到消息中
        point_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            # rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            point_data.append([x, y, z, r, g, b])
        
        msg.data = np.asarray(point_data, dtype=np.float32).tobytes()

        # 发布消息
        pub.publish(msg)
        # print("published...")
        rospy.sleep(1.0)





def publish_colored_point_cloud2():
    pub_pcd = rospy.Publisher('/colored_point_cloud', PointCloud2, queue_size=1)
    
    # data = get_pointcloud()
    # data = get_pointcloud_rvbust()
    _,_,data = color_pointcloud_capture(ifvalid = True,add_mask = False)

    print(data.shape)

    while not rospy.is_shutdown():
        # 创建PointCloud2消息
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_base"
        msg.height = 1
        msg.width = len(data)
        msg.is_dense = True

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * data.shape[0]

        # 将点和颜色信息填充到消息中

        msg.data = np.asarray(data, dtype=np.float32).tobytes()

        # 发布消息
        pub_pcd.publish(msg)
        rospy.sleep(1.0)



# 实时拍摄预测并发布

def publish_colored_point_cloud_capture():
    global capture_flag
    pub_pcd = rospy.Publisher('/colored_point_cloud_captured', PointCloud2, queue_size=1)
    
    # data = get_pointcloud()
    # data = get_pointcloud_rvbust()

    # _,_,data = color_pointcloud_capture(ifvalid = True,add_mask = True)
    _,_,data = capture_predict()
    
    # print(data.shape)

    while not rospy.is_shutdown():
        if capture_flag:
            # _,_,data = color_pointcloud_capture(ifvalid = True,add_mask = True)
            _,_,data = capture_predict()
            capture_flag = False
        # 创建PointCloud2消息
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_base"
        msg.height = 1
        msg.width = len(data)
        msg.is_dense = True

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * data.shape[0]

        # 将点和颜色信息填充到消息中

        msg.data = np.asarray(data, dtype=np.float32).tobytes()

        # 发布消息
        pub_pcd.publish(msg)
        rospy.sleep(1.0)
            


# 对RGBD图像进行预测并发布
from predict_code.predict import predict_seam, model
import cv2
def publish_predict_picture():
    global capture_flag
    pub_pcd = rospy.Publisher('/colored_point_cloud_captured', PointCloud2, queue_size=1)

    idx = 90
    color_img = cv2.imread('gluing_dataset/color/color'+ str(idx) +'.png')
    depth_img = cv2.imread('gluing_dataset/depth/depth'+ str(idx) +'.tiff',-1)

    data = predict_seam(color_img, depth_img, model, vis=False, device="cuda")
    
    # print(data.shape)

    while not rospy.is_shutdown():
        
        # 创建PointCloud2消息
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_base"
        msg.height = 1
        msg.width = len(data)
        msg.is_dense = True

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)
        ]
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * data.shape[0]

        # 将点和颜色信息填充到消息中

        msg.data = np.asarray(data, dtype=np.float32).tobytes()

        # 发布消息
        pub_pcd.publish(msg)
        rospy.sleep(1.0)





def thread_job1():
    key_listener = keyboard.Listener(
            on_press=on_press
        )
    key_listener.start()


def talker():

    pub = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=5)
    rate = rospy.Rate(1)

    points=np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
    colors=np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255]])

    while not rospy.is_shutdown():

        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "camera_base"

        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tostring()

        pub.publish(msg)
        print("published...")
        rate.sleep()





# from seam_point_extract.pcd_cut_save import vis_cut_save
# def cut_callback(data):
#     print(data)
#     captured_pointcloud = np.frombuffer(data.data, dtype=np.float32)
#     captured_pointcloud = captured_pointcloud.reshape(-1,6)
#     points = captured_pointcloud[:,:3]
#     colors = captured_pointcloud[:,3:]

#     index = (colors[:, 0] == 1) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
#     points = points[index]
#     valid_pcd_data = points[~np.isnan(points[:, 2])]

#     seam_pcd = o3d.geometry.PointCloud()
#     seam_pcd.points = o3d.utility.Vector3dVector(valid_pcd_data)
#     folder_path = "seam_point_extract/extract_pcd/"
#     print(valid_pcd_data.shape)
#     # o3d.visualization.draw_geometries_with_editing([seam_pcd])
#     vis_cut_save(folder_path, seam_pcd)



# def cut_pcd():
#     while True:
#         global cut_flag
#         if cut_flag:
#             print(cut_flag)
#             rospy.Subscriber('/colored_point_cloud_captured', PointCloud2,cut_callback)
#             cut_flag = False






if __name__ =="__main__":
    rospy.init_node('pointcloud_calibration_publisher', anonymous=False)
    # pub = rospy.Publisher('/colored_point_cloud', PointCloud2, queue_size=1)

    # talker()
    # publish_colored_point_cloud1111()

    # key_listener = keyboard.Listener(
    #         on_press=on_press
    #     )
    # key_listener.start()
    


    add_thread1 = threading.Thread(target = thread_job1)
    add_thread1.start()

    add_thread2 = threading.Thread(target = publish_calibration_tf,args=["base","camera_base"])
    add_thread2.start()

    add_thread3 = threading.Thread(target = publish_colored_point_cloud_capture)
    add_thread3.start()
    # add_thread3 = threading.Thread(target = publish_predict_picture)
    # add_thread3.start()

    

    add_thread4 = threading.Thread(target = glue_moveit)
    add_thread4.start()


    


    