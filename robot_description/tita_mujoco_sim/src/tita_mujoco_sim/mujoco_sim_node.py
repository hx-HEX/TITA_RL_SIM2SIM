import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
import mujoco
import mujoco.viewer as viewer
import time
from ament_index_python.packages import get_package_share_directory
import os
import sys

class MujocoSimNode(Node):
    def __init__(self, asset_path, joint_sensor_names):
        super().__init__('mujoco_sim_node')
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)

        # ROS2 pubs/subs
        self.joint_pub = self.create_publisher(JointState, "/tita/joint_states", 10)
        self.imu_pub   = self.create_publisher(Imu, "/tita/imu_sensor_broadcaster/imu", 10)
        self.effort_sub = self.create_subscription(
            Float64MultiArray,
            "/tita/effort_controller/commands",
            self.effort_callback,
            10
        )

        # MuJoCo init
        self.model = mujoco.MjModel.from_xml_path(asset_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)

        self.ctrl = [0.0] * self.joint_num  # 力矩控制缓存
        self.dt = self.model.opt.timestep
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.frame_count = 0

    def effort_callback(self, msg: Float64MultiArray):
        """接收C++发来的力矩控制指令"""
        self.ctrl = msg.data

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_sensor_names
        msg.position = [self.data.qpos[i + 7] for i in range(self.joint_num)]
        msg.velocity = [self.data.qvel[i + 6] for i in range(self.joint_num)]
        msg.effort   = list(self.ctrl)
        self.joint_pub.publish(msg)

    def publish_imu(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()

        # orientation (四元数)
        quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        quat_addr = self.model.sensor_adr[quat_id]
        msg.orientation.x = self.data.sensordata[quat_addr + 1]
        msg.orientation.y = self.data.sensordata[quat_addr + 2]
        msg.orientation.z = self.data.sensordata[quat_addr + 3]
        msg.orientation.w = self.data.sensordata[quat_addr + 0]

        # angular velocity
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        gyro_addr = self.model.sensor_adr[gyro_id]
        msg.angular_velocity.x = self.data.sensordata[gyro_addr + 0]
        msg.angular_velocity.y = self.data.sensordata[gyro_addr + 1]
        msg.angular_velocity.z = self.data.sensordata[gyro_addr + 2]

        # linear acceleration
        acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
        acc_addr = self.model.sensor_adr[acc_id]
        msg.linear_acceleration.x = self.data.sensordata[acc_addr + 0]
        msg.linear_acceleration.y = self.data.sensordata[acc_addr + 1]
        msg.linear_acceleration.z = self.data.sensordata[acc_addr + 2]

        self.imu_pub.publish(msg)

    def timer_callback(self):
        # 设置控制输入
        for i in range(self.joint_num):
            self.data.ctrl[i] = self.ctrl[i]

        # 物理仿真步进
        mujoco.mj_step(self.model, self.data)

        # 发布关节和 IMU 状态
        self.publish_joint_state()
        self.publish_imu()

        # 渲染器同步,
        if self.frame_count % 20 == 0:#for smoother visualization
            self.viewer.sync()

        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    joint_sensor_names = [
        "joint_left_hip", "joint_left_thigh", "joint_left_calf","joint_left_wheel",
        "joint_right_hip", "joint_right_thigh", "joint_right_calf","joint_right_wheel"
    ]
    
    package_share = get_package_share_directory('tita_description')
    model_path = os.path.join(package_share, 'xml', 'robot.xml')
    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist. Please ensure the ROBOT_TYPE is set correctly.")
        sys.exit(1)

    print("Robot XML Path:", model_path)

    node = MujocoSimNode(model_path, joint_sensor_names)
    try:
        rclpy.spin(node)  # ROS2 会自动处理定时器
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
