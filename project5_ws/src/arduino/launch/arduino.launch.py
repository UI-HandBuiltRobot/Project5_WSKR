"""Launch the mecanum serial bridge with parameters from config/serial.yaml."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('arduino')
    default_params = PathJoinSubstitution([pkg_share, 'config', 'serial.yaml'])

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file with mecanum_serial_bridge parameters.',
    )

    bridge_node = Node(
        package='arduino',
        executable='mecanum_serial_bridge',
        name='mecanum_serial_bridge',
        parameters=[LaunchConfiguration('params_file')],
        output='screen',
    )

    return LaunchDescription([params_arg, bridge_node])
