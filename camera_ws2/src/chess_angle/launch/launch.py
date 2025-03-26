import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Lanzar 'real_pub' en una nueva terminal
        ExecuteProcess(
            cmd=['gnome-terminal', '--', 'ros2', 'run', 'chess_angle', 'real_pub'],
            output='screen'
        ),
        # Lanzar 'angle_pub' en una nueva terminal
        ExecuteProcess(
            cmd=['gnome-terminal', '--', 'ros2', 'run', 'chess_angle', 'angle_pub'],
            output='screen'
        ),
        # Lanzar 'angle_sub' en una nueva terminal
        ExecuteProcess(
            cmd=['gnome-terminal', '--', 'ros2', 'run', 'chess_angle', 'angle_sub'],
            output='screen'
        ),
    ])

