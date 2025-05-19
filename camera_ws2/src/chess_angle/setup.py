from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'chess_angle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tecnalia',
    maintainer_email='tecnalia@todo.todo',
    description='Estimación del ángulo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'angle_pub = chess_angle.angle_pub:main',
        'angle_pub2 = chess_angle.angle_pub2:main',
        'angle_sub = chess_angle.angle_sub:main',
        'real_pub = chess_angle.real_pub:main',
        'camera_frame_pub = chess_angle.camera_frame_pub:main',
        ],
    },
)
