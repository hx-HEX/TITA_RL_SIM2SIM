from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 默认使用 tita_description 包下的 xml/robot.xml
    # tita_desc_share = get_package_share_directory('tita_description')
    # default_model = os.path.join(tita_desc_share, 'xml', 'robot.xml')


    return LaunchDescription([
    Node(
    package='tita_mujoco_sim',
    executable='mujoco_sim_node',
    name='mujoco_sim_node',
    output='screen',
    # 若需要通过参数传模型路径，可以在这里添加 parameters 或 remappings
    )
    ])