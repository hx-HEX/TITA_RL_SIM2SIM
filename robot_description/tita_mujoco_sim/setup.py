from setuptools import setup
import os
from glob import glob

package_name = 'tita_mujoco_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': 'src'},
    install_requires=['setuptools','mujoco'],
    zip_safe=True,
    maintainer='Xing He',
    maintainer_email='1499217911@qq.com',
    description='MuJoCo simulation node for TITA robot',
    license='Apache-2.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    entry_points={
        'console_scripts': [
            'mujoco_sim_node = tita_mujoco_sim.mujoco_sim_node:main',
            'keyboard_teleop = tita_mujoco_sim.keyboard_teleop:main',
        ],
    },
)
