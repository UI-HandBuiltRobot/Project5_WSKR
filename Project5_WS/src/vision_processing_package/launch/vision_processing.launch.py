from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def generate_launch_description():

    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value='/home/ros_setup/ros2_ws/node_logs/'+timestamp+'_vision_processing',
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

    obj_det_node = Node(
        package='vision_processing_package',
        executable='process_object_vision.py',
        name='object_detection_node',
        prefix=[venv],
        output='both',
        arguments=[
                '--ros-args',
                '--log-file-name', 'object_detection_node']  
    )

    obj_select_node = Node(
        package='vision_processing_package',
        executable='object_selection.py',
        name='object_selection_node',
        prefix=[venv],
        output='both',
        arguments=[
                '--ros-args',
                '--log-file-name', 'object_selection_node']  
    )

    return LaunchDescription([
        prefix_arg,
        log_dir_arg,
        set_log_dir,
        obj_det_node,
        obj_select_node 
    ])

