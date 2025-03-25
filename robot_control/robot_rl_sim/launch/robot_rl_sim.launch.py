import os
from launch import LaunchDescription
from launch.actions import  DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    rname = LaunchConfiguration("rname")
    framework = LaunchConfiguration("framework")

    robot_name = ParameterValue(Command(["echo -n ", rname, "_", framework]), value_type=str)

    #调用官方的demo_nodes_cpp功能包中的parameter_blackboard进行参数节点的设置,demo_nodes_cpp主要作用是提供 C++ 版的示例节点
    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
        }],
    )

    # 定义节点
    tita_drl_node = Node(
        package='robot_rl_sim',          # 包名
        executable='robot_rl_sim', # 可执行文件名
        name='tita_rl_node',  # 节点名，需与 YAML 中一致
        output='screen',             # 输出到屏
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rname",
            description="Robot name (e.g., a1, go2)",
            default_value = "tita",
        ),
        DeclareLaunchArgument(
            "framework",
            description="Framework (isaacgym or isaacsim)",
            default_value = "isaacgym",
        ),
        param_node,
        tita_drl_node
    ])