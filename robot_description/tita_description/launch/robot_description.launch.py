import os
import launch
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

prefix="tita"

def generate_launch_description():
    wname = "stairs"
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "sim_env",
            default_value = "gazebo",
            choices = ["gazebo","none"],
            description = "Whether to use simulation",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "sim_ctrl",
            default_value = "default",
            choices = ["default","customer"],
            description = "Select simulation control interface",
        )
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments = {'verbose': 'false',
                            "world": os.path.join(get_package_share_directory("robot_rl_sim"), "worlds", wname + ".world"),
                            }.items()
    )

    # 将机器人模型加载到 Gazebo 仿真环境中
    spawn_entity = Node(package='gazebo_ros', 
                        executable='spawn_entity.py',
                        arguments=['-topic', f'{prefix}/robot_description',
                                   '-entity', f'{prefix}',
                                   '-x', '0.',
                                   '-y', '0.',
                                   '-z', '0.65'], 
                        output='screen')

    robot_description_content_dir = PathJoinSubstitution(
        [FindPackageShare("tita_description"), "xacro", "robot.xacro"]
    )
    xacro_executable = FindExecutable(name="xacro")

    robot_description_content = Command(
        [
            PathJoinSubstitution([xacro_executable]),
            " ",
            robot_description_content_dir,
            " ",
            "sim_env:=",
            LaunchConfiguration('sim_env'),
            " ",
            "sim_ctrl:=",
            LaunchConfiguration('sim_ctrl'),
        ]
    )

    robot_description = {"robot_description": robot_description_content}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            robot_description,
            {'use_sim_time': True},
            {"frame_prefix": prefix+"/"},
        ],
        namespace=prefix, 
    )

    joint_state_broadcaster_node = Node(
        package = "controller_manager",
        executable = 'spawner' ,
        arguments = ["joint_state_broadcaster","--controller-manager",prefix + "/controller_manager",],
        output = 'screen',
    )

    imu_sensor_broadcaster_spawner = Node(
        package = "controller_manager",
        executable = "spawner",
        arguments = ["imu_sensor_broadcaster","--controller-manager",prefix + "/controller_manager",],
        output = 'screen',
    )

    robot_joint_controller_node = Node(
        package = "controller_manager",
        executable= 'spawner' ,
        arguments = ["effort_controller","--controller-manager",prefix + "/controller_manager",]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        respawn=True,
        output='screen'
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    nodes = [
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster_node,
        imu_sensor_broadcaster_spawner,
        robot_joint_controller_node,
        # rviz_node,
        # joint_state_publisher_gui,
    ]

    return LaunchDescription(declared_arguments + nodes)