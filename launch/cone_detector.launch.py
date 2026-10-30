import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('cone_detector'), 'config', 'cone_detector.yaml')
    image = LaunchConfiguration('image')

    return LaunchDescription([
        DeclareLaunchArgument('image', default_value='/odin1/image',
                              description='Raw sensor_msgs/Image topic to detect on'),
        DeclareLaunchArgument('camera_info', default_value='/odin1/camera_info',
                              description='CameraInfo of the image topic, for cone positions'),
        DeclareLaunchArgument('detections', default_value='/cone_detector/detections'),
        DeclareLaunchArgument('cone_poses', default_value='/cone_detector/cone_poses'),
        DeclareLaunchArgument('markers', default_value='/cone_detector/markers'),
        DeclareLaunchArgument('debug_image', default_value='/cone_detector/debug_image'),
        DeclareLaunchArgument('model_path', default_value='cone_yolo.onnx',
                              description='.engine -> TensorRT, .onnx -> ONNX Runtime '
                                          '(CUDA, else CPU); '
                                          'bare file names are looked up in the models/ dir'),
        DeclareLaunchArgument('decompress', default_value='true',
                              description='Republish <image>/compressed as raw <image>'),

        # The bag only contains CompressedImage, so decode it to a raw Image first.
        Node(
            package='image_transport', executable='republish', name='image_decompressor',
            condition=IfCondition(LaunchConfiguration('decompress')),
            parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
            remappings=[('in/compressed', [image, '/compressed']), ('out', image)],
        ),
        Node(
            package='cone_detector', executable='cone_detector', name='cone_detector',
            parameters=[config, {'model_path': LaunchConfiguration('model_path')}],
            remappings=[('image', image),
                        ('camera_info', LaunchConfiguration('camera_info')),
                        ('detections', LaunchConfiguration('detections')),
                        ('cone_poses', LaunchConfiguration('cone_poses')),
                        ('markers', LaunchConfiguration('markers')),
                        ('debug_image', LaunchConfiguration('debug_image'))],
            output='screen',
        ),
    ])
