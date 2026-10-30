import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('cone_detector'), 'config', 'cone_finder.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('cone_poses', default_value='/cone_detector/cone_poses'),
        DeclareLaunchArgument('operator_goal', default_value='/operator_goal',
                              description="crl_commander's GOTO goal topic"),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # Drives crl_commander (explore / goto / stop), which must already be running.
        Node(
            package='cone_detector', executable='cone_finder', name='cone_finder',
            parameters=[config, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('cone_poses', LaunchConfiguration('cone_poses')),
                        ('operator_goal', LaunchConfiguration('operator_goal'))],
            output='screen',
        ),
    ])
