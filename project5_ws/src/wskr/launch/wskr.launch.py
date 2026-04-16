from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    wskr_share = Path(get_package_share_directory('wskr'))
    floor_params = str(wskr_share / 'config' / 'floor_params.yaml')
    lens_params = str(wskr_share / 'config' / 'lens_params.yaml')
    arduino_launch = str(
        Path(get_package_share_directory('arduino')) / 'launch' / 'arduino.launch.py'
    )

    camera_rate_arg = DeclareLaunchArgument(
        'camera_rate_hz',
        default_value='10.0',
        description='gstreamer_camera publish rate (Hz). 0.0 disables throttling.',
    )

    return LaunchDescription([
        camera_rate_arg,
        Node(
            package='gstreamer_camera',
            executable='gstreamer_camera_node',
            name='gstreamer_camera',
            output='screen',
            parameters=[{
                'publish_rate_hz': ParameterValue(
                    LaunchConfiguration('camera_rate_hz'), value_type=float),
            }],
            # Default pipeline inside the node uses nvv4l2decoder + nvvidconv
            # for Jetson HW-accelerated MJPEG decode and color conversion.
        ),
        Node(
            package='wskr',
            executable='wskr_floor.py',
            name='wskr_floor',
            output='screen',
            parameters=[floor_params],
        ),
        Node(
            package='wskr',
            executable='wskr_range.py',
            name='wskr_range',
            output='screen',
            parameters=[lens_params],
        ),
        Node(
            package='wskr',
            executable='wskr_approach_action.py',
            name='wskr_approach_action',
            output='screen',
            parameters=[lens_params],
        ),
        Node(
            package='wskr',
            executable='wskr_dead_reckoning.py',
            name='wskr_dead_reckoning',
            output='screen',
        ),
        Node(
            package='wskr',
            executable='wskr_autopilot.py',
            name='wskr_autopilot',
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(arduino_launch),
        ),
    ])
