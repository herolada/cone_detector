from glob import glob

from setuptools import setup

package_name = 'cone_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.onnx') + glob('models/*.engine')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='herolada',
    maintainer_email='herold.adam7@gmail.com',
    description='YOLO traffic cone detection node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cone_detector = cone_detector.cone_detector_node:main',
            'cone_finder = cone_detector.cone_finder:main',
        ],
    },
)
