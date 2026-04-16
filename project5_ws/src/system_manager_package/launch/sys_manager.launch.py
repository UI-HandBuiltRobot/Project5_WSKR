from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def generate_launch_description():

    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value='/home/ros_setup/ros2_ws/node_logs/'+timestamp+'_sys_manager',
    )

    log_dir = LaunchConfiguration('log_dir')

    set_log_dir = SetEnvironmentVariable(
        name='ROS_LOG_DIR',
        value=log_dir
    )

    prefix_arg = DeclareLaunchArgument(
        'venv',
        default_value='/home/ros_setup/ros2_ws/venv/bin/python'
    )
    
    venv = LaunchConfiguration('venv')

    sys_manager = Node(
        package='system_manager_package',
        executable='state_manager.py',
        name='state_manager_node',
        prefix=[venv],
        output='both',
        arguments=[
                '--ros-args',
                '--log-file-name', 'state_manager_node']  
    )

    return LaunchDescription([
        prefix_arg,
        log_dir_arg,
        set_log_dir,
        sys_manager, 
    ])

