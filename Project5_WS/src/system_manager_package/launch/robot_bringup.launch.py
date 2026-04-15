from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from datetime import datetime

VENV_PATH = '/home/ros_setup/ros2_ws/venv/bin/python'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_launch_description():

    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value=f'/home/ros_setup/ros2_ws/node_logs/{timestamp}_robot_bringup_run',
    )
    log_dir = LaunchConfiguration('log_dir')

    venv_arg = DeclareLaunchArgument(
        'venv',
        default_value=VENV_PATH,
    )
    venv = LaunchConfiguration('venv')

    set_log_dir = SetEnvironmentVariable(
        name='ROS_LOG_DIR',
        value=log_dir
    )

    usb_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('usb_cam'),
                'launch',
                'camera.launch.py'
            )
        )
    )

    state_manager = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(os.getcwd(), 'src/system_manager_package/launch/sys_manager.launch.py')
        ),
        launch_arguments={
            'log_dir': log_dir,
            'venv': venv
        }.items()
    )

    vision_processing = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(os.getcwd(), 'src/vision_processing_package/launch/vision_processing.launch.py')
        ),
        launch_arguments={
            'log_dir': log_dir,
            'venv': venv
        }.items()
    )

    xarm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(os.getcwd(), 'src/xarm_object_collector_package/launch/xarm_object_collector_ga.launch.py')
        ),
        launch_arguments={
            'log_dir': log_dir,
            'venv': venv
        }.items()
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(os.getcwd(), 'src/navigation_package/launch/navigation.launch.py')
        ),
        launch_arguments={
            'log_dir': log_dir,
            'venv': venv
        }.items()
    )

    return LaunchDescription([
        log_dir_arg,
        venv_arg,
        set_log_dir,
        usb_cam,
        state_manager,
        vision_processing,
        xarm,
        navigation
    ])
